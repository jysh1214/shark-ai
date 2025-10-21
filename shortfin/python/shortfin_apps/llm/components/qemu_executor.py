# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
QEMU-based VMFB executor for cross-architecture inference.

This module allows running RISC-V compiled VMFB files on x86 hosts using QEMU
user-mode emulation, while keeping the LLM server's pre/post-processing on the
native architecture.
"""

import logging
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Optional
import shortfin as sf

logger = logging.getLogger(__name__)


class QemuVmfbExecutor:
    """
    Executes VMFB functions via iree-run-module under QEMU user-mode emulation.

    This executor:
    1. Serializes input tensors to temporary files
    2. Invokes iree-run-module under QEMU with the RISC-V VMFB
    3. Deserializes output tensors from result files

    This allows running cross-architecture VMFB (e.g., RISC-V on x86) while
    keeping the rest of the LLM server running natively.
    """

    def __init__(
        self,
        qemu_executable: str,
        qemu_args: List[str],
        iree_run_module_path: str,
        vmfb_path: str,
        parameters_path: Optional[str] = None,
        device: str = "local-task",
    ):
        """
        Initialize the QEMU VMFB executor.

        Args:
            qemu_executable: Path to QEMU user-mode executable (e.g., qemu-riscv64)
            qemu_args: Additional arguments for QEMU (e.g., ["-L", "/usr/riscv64-linux-gnu"])
            iree_run_module_path: Path to RISC-V compiled iree-run-module binary
            vmfb_path: Path to the RISC-V compiled VMFB file
            parameters_path: Optional path to model parameters (IRPA file)
            device: IREE HAL device to use (default: local-task)
        """
        self.qemu_executable = qemu_executable
        self.qemu_args = qemu_args
        self.iree_run_module_path = iree_run_module_path
        self.vmfb_path = vmfb_path
        self.parameters_path = parameters_path
        self.device = device

        # Validate that QEMU executable exists
        if not Path(qemu_executable).exists():
            raise FileNotFoundError(f"QEMU executable not found: {qemu_executable}")

        # Validate that iree-run-module exists
        if not Path(iree_run_module_path).exists():
            raise FileNotFoundError(f"iree-run-module not found: {iree_run_module_path}")

        # Validate that VMFB exists
        if not Path(vmfb_path).exists():
            raise FileNotFoundError(f"VMFB file not found: {vmfb_path}")

        logger.info(f"Initialized QemuVmfbExecutor:")
        logger.info(f"  QEMU: {qemu_executable} {' '.join(qemu_args)}")
        logger.info(f"  iree-run-module: {iree_run_module_path}")
        logger.info(f"  VMFB: {vmfb_path}")
        logger.info(f"  Device: {device}")

    def _serialize_tensor(self, tensor: sf.array, output_path: Path) -> str:
        """
        Serialize a shortfin tensor to a numpy file and return the input spec.

        Args:
            tensor: The shortfin array to serialize
            output_path: Path to save the numpy file

        Returns:
            Input specification string for iree-run-module
        """
        # Convert shortfin array to numpy
        # First map to host memory
        mapped = tensor.map(read=True)
        np_array = np.array(mapped.view, copy=True)
        mapped.unmap()

        # Save to file
        np.save(output_path, np_array)

        # Generate input spec for iree-run-module
        # Format: --input=<shape>x<dtype>=@<file>
        shape_str = "x".join(str(d) for d in np_array.shape)
        dtype_str = self._numpy_dtype_to_iree_dtype(np_array.dtype)

        return f"{shape_str}x{dtype_str}=@{output_path}.npy"

    def _numpy_dtype_to_iree_dtype(self, dtype: np.dtype) -> str:
        """Convert numpy dtype to IREE dtype string."""
        dtype_map = {
            np.float32: "f32",
            np.float16: "f16",
            np.int32: "si32",
            np.int64: "si64",
            np.int8: "si8",
            np.int16: "si16",
            np.uint8: "u8",
            np.uint16: "u16",
            np.uint32: "u32",
            np.uint64: "u64",
            np.bool_: "i1",
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return dtype_map[dtype]

    def _iree_dtype_to_numpy_dtype(self, iree_dtype: str) -> np.dtype:
        """Convert IREE dtype string to numpy dtype."""
        dtype_map = {
            "f32": np.float32,
            "f16": np.float16,
            "si32": np.int32,
            "si64": np.int64,
            "si8": np.int8,
            "si16": np.int16,
            "u8": np.uint8,
            "u16": np.uint16,
            "u32": np.uint32,
            "u64": np.uint64,
            "i1": np.bool_,
        }

        if iree_dtype not in dtype_map:
            raise ValueError(f"Unsupported IREE dtype: {iree_dtype}")

        return dtype_map[iree_dtype]

    def _deserialize_output(self, output_file: Path, device: sf.Device) -> sf.array:
        """
        Deserialize a numpy output file to a shortfin array.

        Args:
            output_file: Path to the numpy output file
            device: Target shortfin device

        Returns:
            A shortfin array on the target device
        """
        # Load numpy array
        np_array = np.load(output_file)

        # Create shortfin array from numpy
        # This assumes the device is a CPU device
        result = sf.array.for_device(device, np_array.shape, np_array.dtype)
        mapped = result.map(write=True)
        np.copyto(np.array(mapped.view, copy=False), np_array)
        mapped.unmap()

        return result

    async def invoke(
        self,
        function_name: str,
        args: List[sf.array],
        device: sf.Device,
    ) -> List[sf.array]:
        """
        Invoke a VMFB function under QEMU.

        Args:
            function_name: Name of the function to invoke (e.g., "prefill_bs1")
            args: List of input tensors as shortfin arrays
            device: Target device for output tensors

        Returns:
            List of output tensors as shortfin arrays
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Serialize input tensors
            input_specs = []
            for i, arg in enumerate(args):
                input_file = tmpdir_path / f"input_{i}"
                input_spec = self._serialize_tensor(arg, input_file)
                input_specs.append(f"--input={input_spec}")

            # Prepare output file path
            output_dir = tmpdir_path / "outputs"
            output_dir.mkdir()

            # Build iree-run-module command
            iree_cmd = [
                self.iree_run_module_path,
                f"--module={self.vmfb_path}",
                f"--device={self.device}",
                f"--function={function_name}",
                f"--output=@{output_dir}/",
            ]

            if self.parameters_path:
                iree_cmd.append(f"--parameters=model={self.parameters_path}")

            iree_cmd.extend(input_specs)

            # Build full QEMU command
            qemu_cmd = [self.qemu_executable] + self.qemu_args + iree_cmd

            logger.debug(f"Executing: {' '.join(qemu_cmd)}")

            # Execute under QEMU
            try:
                result = subprocess.run(
                    qemu_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=600,  # 10 minute timeout
                )

                if result.stdout:
                    logger.debug(f"QEMU stdout: {result.stdout}")
                if result.stderr:
                    logger.debug(f"QEMU stderr: {result.stderr}")

            except subprocess.CalledProcessError as e:
                logger.error(f"QEMU execution failed with return code {e.returncode}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                raise RuntimeError(f"QEMU VMFB execution failed: {e.stderr}") from e
            except subprocess.TimeoutExpired as e:
                logger.error("QEMU execution timed out")
                raise RuntimeError("QEMU VMFB execution timed out") from e

            # Deserialize outputs
            output_files = sorted(output_dir.glob("*.npy"))
            if not output_files:
                raise RuntimeError("No output files generated by iree-run-module")

            outputs = []
            for output_file in output_files:
                output_tensor = self._deserialize_output(output_file, device)
                outputs.append(output_tensor)

            logger.debug(f"Successfully executed {function_name}, got {len(outputs)} outputs")

            return outputs


class QemuProgramFunction:
    """
    Wrapper that makes QemuVmfbExecutor look like a shortfin ProgramFunction.

    This allows it to be used as a drop-in replacement in the LLM service.
    """

    def __init__(self, executor: QemuVmfbExecutor, function_name: str):
        """
        Initialize the wrapper.

        Args:
            executor: The QEMU executor instance
            function_name: Name of the function this wrapper represents
        """
        self.executor = executor
        self.function_name = function_name

    async def __call__(self, *args, fiber=None):
        """
        Call the function via QEMU executor.

        This mimics the shortfin ProgramFunction calling convention.

        Args:
            *args: Input tensors (as shortfin device arrays)
            fiber: The fiber (used to determine the device)

        Returns:
            Output tensors as shortfin arrays
        """
        if fiber is None:
            raise ValueError("fiber argument is required for QemuProgramFunction")

        # Get the device from the fiber
        device = list(fiber.devices_dict.values())[0]

        # Convert args to list of arrays
        arg_list = []
        for arg in args:
            if hasattr(arg, '__iter__') and not isinstance(arg, sf.array):
                arg_list.extend(arg)
            else:
                arg_list.append(arg)

        # Invoke via QEMU executor
        results = await self.executor.invoke(
            self.function_name,
            arg_list,
            device,
        )

        # Return results in the same format as shortfin ProgramFunction
        # Typically returns a single array or tuple of arrays
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
