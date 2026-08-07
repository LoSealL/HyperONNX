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

CUTLASS GEMM config tuner using CuTe DSL.
Only tunes and returns the best config — no cubin export.
"""

from __future__ import annotations

import torch

from .config import MM_CONFIGS, CutlassConfig


def _extract_matmul_shapes(
    args: list[dict], buffers: dict
) -> tuple[int, int, int, str]:
    """Extract M, N, K dimensions and dtype from manifest args."""
    tensor_args = [a for a in args if a.get("kind") == "tensor"]
    if len(tensor_args) < 2:
        raise ValueError(f"Expected >=2 tensor args for mm, got {len(tensor_args)}")

    def _shape_of(arg: dict) -> list[int]:
        shape = arg.get("shape")
        if shape:
            return [int(s) for s in shape]
        bid = arg.get("buffer_id")
        if bid is not None:
            for buf_meta in buffers.values():
                if buf_meta.get("buffer_id") == bid and buf_meta.get("shape"):
                    return [int(s) for s in buf_meta["shape"]]
        name = arg.get("name")
        if name and name in buffers:
            meta = buffers[name]
            if meta.get("shape"):
                return [int(s) for s in meta["shape"]]
        raise ValueError(f"Cannot determine shape for arg {arg}")

    shape_a = _shape_of(tensor_args[0])
    shape_b = _shape_of(tensor_args[1])

    if len(shape_a) == 2:
        M, K = shape_a
    elif len(shape_a) == 3:
        M, K = shape_a[-2], shape_a[-1]
    else:
        raise ValueError(f"Unexpected A shape: {shape_a}")

    if len(shape_b) == 2:
        K2, N = shape_b
    elif len(shape_b) == 3:
        K2, N = shape_b[-2], shape_b[-1]
    else:
        raise ValueError(f"Unexpected B shape: {shape_b}")

    if K != K2:
        raise ValueError(f"K mismatch: A has {K}, B has {K2}")

    dtype = (args[-1].get("dtype") if args else None) or "float16"
    return M, N, K, dtype


def _compile_and_bench_mm(
    M: int,
    N: int,
    K: int,
    arch: str,
    config: CutlassConfig,
    warmup: int = 3,
    iters: int = 20,
) -> float:
    """Compile a tiled GEMM with CuTe DSL and benchmark it. Returns avg ms."""
    import cutlass.cute as cute
    import cutlass.cute.arch as cute_arch
    from cutlass.cute import Float16, Float32, Int32
    from cutlass.cute.runtime import make_fake_tensor

    tile_m = config.tile_m
    tile_n = config.tile_n

    @cute.kernel
    def gemm_kernel(
        A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: Int32, N: Int32, K: Int32
    ):
        row = cute_arch.block_idx()[0] * tile_m + cute_arch.thread_idx()[0]
        col = cute_arch.block_idx()[1] * tile_n + cute_arch.thread_idx()[1]
        if row < M and col < N:
            acc = Float32(0.0)
            for k in range(K):
                acc = acc + A[row, k].to(Float32) * B[k, col].to(Float32)
            C[row, col] = acc.to(Float16)

    @cute.jit
    def gemm_jit(
        A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: Int32, N: Int32, K: Int32
    ):
        grid_x = (M + tile_m - 1) // tile_m
        grid_y = (N + tile_n - 1) // tile_n
        gemm_kernel(A, B, C, M, N, K).launch(
            grid=(grid_x, grid_y, 1), block=(tile_m, tile_n, 1)
        )

    fake_a = make_fake_tensor(Float16, (M, K), (K, 1))
    fake_b = make_fake_tensor(Float16, (K, N), (N, 1))
    fake_c = make_fake_tensor(Float16, (M, N), (N, 1))

    compiled = cute.compile(
        gemm_jit,
        fake_a,
        fake_b,
        fake_c,
        M,
        N,
        K,
        options=f"--gpu-arch {arch}",
    )

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.empty(M, N, dtype=torch.float16, device="cuda")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        compiled(A, B, C, M, N, K)

    start.record()
    for _ in range(iters):
        compiled(A, B, C, M, N, K)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def tune_mm(
    args: list[dict],
    buffers: dict,
    arch: str,
    configs: list[CutlassConfig] | None = None,
) -> CutlassConfig:
    """Autotune GEMM configs and return the best one.

    Args:
        args: KernelArgDescriptor list from the manifest step.
        buffers: buffer table from the manifest.
        arch: GPU arch string (e.g. "sm_120").
        configs: list of CutlassConfig to benchmark. Defaults to MM_CONFIGS.

    Returns:
        The CutlassConfig with the lowest average latency.
    """
    if configs is None:
        configs = MM_CONFIGS

    M, N, K, _dtype = _extract_matmul_shapes(args, buffers)

    if len(configs) == 1:
        return configs[0]

    best = None
    best_ms = float("inf")

    for cfg in configs:
        try:
            ms = _compile_and_bench_mm(M, N, K, arch, cfg)
            if ms < best_ms:
                best_ms = ms
                best = cfg
        except Exception:
            continue

    if best is None:
        best = configs[0]

    return best
