"""
Copyright (C) 2026 The HYPERONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Profile-guided autotuning for CUTLASS kernels.

Benchmarks multiple tile configurations on the target GPU using CUDA events
for timing. No PyTorch dependency — pure CUDA Driver API.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from .config import CutlassConfig


def autotune_kernel(
    generate_fn: Callable[..., tuple[bytes, CutlassConfig, dict]],
    args: list[dict],
    output: dict,
    buffers: dict,
    arch: str,
    configs: list[CutlassConfig],
    warmup: int = 5,
    iterations: int = 100,
) -> CutlassConfig:
    """Benchmark configs and return the fastest.

    For each config, compiles the kernel via generate_fn, loads the cubin,
    and times *iterations* launches using CUDA events.

    Args:
        generate_fn: kernel generator callable with signature
            (args, output, buffers, arch, config, autotune)
            -> (bytes, CutlassConfig, dict)
        args: KernelArgDescriptor list from the manifest step.
        output: output descriptor from the manifest step.
        buffers: buffer table from the manifest.
        arch: GPU arch string (e.g. "sm_90").
        configs: list of CutlassConfig to benchmark.
        warmup: number of warmup iterations (not timed).
        iterations: number of timed iterations.

    Returns:
        The CutlassConfig with the lowest median latency.

    Raises:
        RuntimeError: if all configs fail to compile.
    """
    if len(configs) == 1:
        return configs[0]

    drv = importlib.import_module("cuda.bindings.driver")

    start_event = drv.cuEventCreate(0)
    end_event = drv.cuEventCreate(0)
    stream = drv.cuStreamCreate(0)

    results: list[tuple[CutlassConfig, float]] = []

    for cfg in configs:
        try:
            cubin_bytes, _, launch = generate_fn(
                args, output, buffers, arch, config=cfg, autotune=False
            )
        except Exception:
            continue

        try:
            module = drv.cuModuleLoadData(cubin_bytes)
            func = drv.cuModuleGetFunction(module, "cutlass_kernel")
        except Exception:
            continue

        grid = launch.get("captured_grid", [1, 1, 1])
        block_x = launch.get("num_warps", 4) * 32
        shared = launch.get("shared_mem_bytes", 0)

        param_values: list[Any] = []
        param_types: list[Any] = []
        for a in args:
            if a.get("kind") == "tensor":
                param_values.append(0)
                param_types.append(drv.cuDeviceptr)
            elif a.get("kind") == "scalar":
                val = a.get("value", 0)
                param_values.append(int(val) if isinstance(val, (int, float)) else 0)
                param_types.append(drv.cuInt32)
            elif a.get("value") is not None:
                param_values.append(int(a["value"]))
                param_types.append(drv.cuInt32)

        param_values.append(0)
        param_types.append(drv.cuDeviceptr)

        kernel_params = (tuple(param_values), tuple(param_types))

        for _ in range(warmup):
            drv.cuLaunchKernel(
                func,
                grid[0],
                grid[1],
                grid[2],
                block_x,
                1,
                1,
                shared,
                stream,
                kernel_params,
                0,
            )
        drv.cuStreamSynchronize(stream)

        drv.cuEventRecord(start_event, stream)
        for _ in range(iterations):
            drv.cuLaunchKernel(
                func,
                grid[0],
                grid[1],
                grid[2],
                block_x,
                1,
                1,
                shared,
                stream,
                kernel_params,
                0,
            )
        drv.cuEventRecord(end_event, stream)

        drv.cuEventSynchronize(end_event)
        elapsed_ms = drv.cuEventElapsedTime(start_event, end_event)
        avg_us = (elapsed_ms * 1000) / iterations
        results.append((cfg, avg_us))

        drv.cuModuleUnload(module)

    drv.cuEventDestroy(start_event)
    drv.cuEventDestroy(end_event)
    drv.cuStreamDestroy(stream)

    if not results:
        raise RuntimeError("All CUTLASS configs failed to compile")

    results.sort(key=lambda x: x[1])
    return results[0][0]
