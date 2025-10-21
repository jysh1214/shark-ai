# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Implements a context manager that configures a shortfin llm server from a namespace mirroring server.py's commandline args, and exposes a context manager interface such that we can do:

```python
def lifecycle(app: FastApi):
    with lifecycle_manager(args) as man:
        yield
```
"""

from .config_struct import ModelParams, ServerParams
from .decode_config import DecodeConfig
from .manager import LlmSystemManager
from .service import LlmGenerateService
from .tokenizer import Tokenizer
from .qemu_executor import QemuVmfbExecutor
from typing import TYPE_CHECKING, Optional
from fastapi import FastAPI
from pathlib import Path


from contextlib import asynccontextmanager
import logging


def get_eos_from_tokenizer_config(json_path):
    import json

    with open(json_path, "rt") as f:
        json_text = f.read()
    config = json.loads(json_text)
    return config["eos_token"]


class ShortfinLlmLifecycleManager:
    """
    Manages the lifecycle of a shortfin llm server, including config loading and parameter setup.

    There are generally two ways to use this.

    To start a full shortfin server, use the context manager or the fastapi_lifespan method.

    To initialize a shortfin server but not start it, use the constructor, then manipulate the services and sysman attributes directly.
    """

    def __init__(self, args):
        # Load server configuration with priority: command line > config file > defaults
        model_params = ModelParams.load_json(args.model_config)
        server_params = ServerParams.load(
            args.server_config if hasattr(args, "server_config") else None
        )
        server_params.update_from_args(args)

        if server_params.decode_config is None:
            decode_config = DecodeConfig(
                num_beams=args.num_beams,
                use_beam_search=args.use_beam_search,
                logits_normalization=model_params.logits_normalization,
            )
            server_params.decode_config = decode_config

        # Setup system (configure devices, etc).
        sysman = LlmSystemManager(
            device=args.device,
            device_ids=server_params.device_ids,
            async_allocs=server_params.amdgpu_async_allocations,
            async_caching=server_params.amdgpu_async_caching,
            amdgpu_allocators=server_params.amdgpu_allocators,
            amdgpu_allow_device_reuse=server_params.amdgpu_allow_device_reuse,
        )

        # Setup QEMU executor if configured
        qemu_executor = None
        if hasattr(args, "qemu_executable") and args.qemu_executable is not None:
            if args.iree_run_module_path is None:
                raise ValueError(
                    "--iree_run_module_path is required when using --qemu_executable"
                )

            # Parse QEMU arguments from file
            qemu_args = []
            if hasattr(args, "qemu_args") and args.qemu_args is not None:
                qemu_args_path = Path(args.qemu_args)
                if not qemu_args_path.exists():
                    raise FileNotFoundError(f"QEMU args file not found: {qemu_args_path}")

                # Read args from file, one per line, skip comments and empty lines
                with open(qemu_args_path, "r") as f:
                    # Parse each line, skipping comments (#) and empty lines
                    # Example file content:
                    #   -L
                    #   /usr/riscv64-linux-gnu
                    #   # This is a comment
                    #   -cpu
                    #   rv64
                    qemu_args = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.strip().startswith('#')
                    ]
                logging.info(f"Loaded {len(qemu_args)} QEMU arguments from {qemu_args_path}")

            # Get parameters path (first one if multiple provided)
            parameters_path = None
            if args.parameters and len(args.parameters) > 0:
                parameters_path = str(args.parameters[0])

            qemu_executor = QemuVmfbExecutor(
                qemu_executable=str(args.qemu_executable),
                qemu_args=qemu_args,
                iree_run_module_path=str(args.iree_run_module_path),
                vmfb_path=str(args.vmfb),
                parameters_path=parameters_path,
                device=args.device,
            )
            logging.info("QEMU executor configured for cross-architecture VMFB execution")

        # Setup each service we are hosting.
        eos_token = get_eos_from_tokenizer_config(args.tokenizer_config_json)
        tokenizer = Tokenizer.from_tokenizer_json_file(
            args.tokenizer_json, eos_token=eos_token
        )
        service = LlmGenerateService(
            name="default",
            sysman=sysman,
            tokenizer=tokenizer,
            model_params=model_params,
            server_params=server_params,
            program_isolation=server_params.program_isolation,
            qemu_executor=qemu_executor,
        )

        # Only load modules if not using QEMU executor
        if qemu_executor is None:
            service.load_inference_module(args.vmfb)
            service.load_inference_parameters(*args.parameters, parameter_scope="model")
        else:
            logging.info("Skipping in-process VMFB loading (using QEMU executor)")

        self.sysman = sysman
        self.services = {"default": service}

    def __enter__(self):
        self.sysman.start()
        for service_name, service in self.services.items():
            logging.info("Initializing service '%s': %r", service_name, service)
            service.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for service_name, service in self.services.items():
            logging.info("Shutting down service '%s'", service_name)
            service.shutdown()
        self.sysman.shutdown()
        return False

    @asynccontextmanager
    async def fastapi_lifespan(self, app: FastAPI):
        """
        Context manager for FastAPI lifespan events.

        Initializes the system manager and services when the app starts, and shuts them down when the app stops.
        Also provides the services via app.state, which can be accessed from route handlers via
        request.app.state.services.

        Implements API described in https://fastapi.tiangolo.com/advanced/events/#lifespan

        See `server.py` for a usage example.
        """
        with self:
            app.state.services = self.services
            yield
