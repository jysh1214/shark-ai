# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""FP8 E4M3FN native matmul implementations using hardware acceleration."""

from typing import Optional
import torch
from torch import Tensor

from sharktank.types import QuantizedTensor, BlockScaledLayout
from .signatures import *


def quantize_dynamic_fp8_e4m3fn(
    x: Tensor, *, per_tensor: bool = True
) -> tuple[Tensor, Tensor]:
    """Dynamically quantize tensor to FP8 E4M3FN with per-tensor scale.

    Args:
        x: Input tensor (fp16 or fp32)
        per_tensor: Use per-tensor scale (True) or per-token scale (False)

    Returns:
        (x_fp8, scale): FP8 quantized tensor and scale(s)
    """
    if per_tensor:
        # Per-tensor symmetric quantization
        max_abs = torch.max(torch.abs(x))
        # FP8 E4M3FN range: [-448, 448]
        scale = max_abs / 448.0
        x_normalized = x / scale.clamp(min=1e-12)  # Avoid division by zero
        x_fp8 = x_normalized.to(torch.float8_e4m3fn)
        return x_fp8, scale
    else:
        # Per-token scale (TODO: implement if needed)
        raise NotImplementedError("Per-token FP8 quantization not yet implemented")


@matmul.override(Tensor, QuantizedTensor, impl_name="sharktank.fp8_native")
def matmul_fp8_native(
    lhs: Tensor, rhs: QuantizedTensor, *, transpose_rhs: bool
) -> Tensor:
    """Native FP8 matmul using hardware acceleration.

    This implementation:
    1. Dynamically quantizes LHS (activations) to FP8
    2. Uses native FP8 matmul: torch.mm(fp8, fp8) -> fp8
    3. Casts result to FP32 and applies scales
    4. Returns FP16 result

    Only works when:
    - RHS is BlockScaledLayout with m=None (scale-only, per-tensor)
    - RHS qs dtype is float8_e4m3fn
    - transpose_rhs is True
    """
    if not transpose_rhs:
        return NotImplemented

    # Check if RHS is compatible
    layout = rhs.layout_type
    if layout is not BlockScaledLayout:
        return NotImplemented

    rhs_unpacked = rhs.unpack()

    # Only use FP8 native matmul for scale-only quantization (no offset)
    if rhs_unpacked.m is not None:
        return NotImplemented

    # Check if weights are FP8
    if not hasattr(torch, 'float8_e4m3fn') or rhs_unpacked.qs.dtype != torch.float8_e4m3fn:
        return NotImplemented

    # Get weight scale (per-tensor, all blocks have same scale)
    # Shape: [N, K//32, 1] but all values are identical
    w_scale = rhs_unpacked.d[..., 0, 0]  # Shape: [N]

    # Dynamically quantize input activations to FP8
    # lhs shape: [bs, seq_len, K]
    original_shape = lhs.shape
    lhs_2d = lhs.reshape(-1, lhs.shape[-1])  # [bs*seq_len, K]

    lhs_fp8, lhs_scale = quantize_dynamic_fp8_e4m3fn(lhs_2d, per_tensor=True)

    # Prepare weights for matmul: [N, K//32, 32] -> [N, K]
    w_fp8 = rhs_unpacked.qs.reshape(rhs_unpacked.qs.shape[0], -1)  # [N, K]

    # Native FP8 matmul: (fp8 @ fp8.T) -> fp8
    # lhs_fp8: [bs*seq_len, K], w_fp8.T: [K, N] -> result: [bs*seq_len, N]
    result_fp8 = torch.mm(lhs_fp8, w_fp8.t())  # fp8 output

    # Cast to FP32 for scaling
    result_fp32 = result_fp8.to(torch.float32)

    # Apply scales: result = (fp8_result * lhs_scale * w_scale)
    # lhs_scale: scalar, w_scale: [N]
    result_scaled = result_fp32 * lhs_scale.to(torch.float32) * w_scale.to(torch.float32)

    # Cast to original dtype (fp16)
    result = result_scaled.to(lhs.dtype)

    # Reshape back to original batch dimensions
    result = result.reshape(*original_shape[:-1], -1)

    return result
