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

CUTLass GEMM execution for extern_kernel replay.

Compiles and runs CuTe DSL GEMM kernels for mm / bmm / addmm / convolution
extern steps that carry a ``cutlass_config`` (produced by
``annotate_cutlass_config``). Convolution uses im2col expansion then the
same GEMM path.
"""

# pyright: reportMissingImports=none

from typing import Any

import torch
from cutlass.cute import Float16, Float32

from .config import CutlassConfig
from .mm import _compile_gemm

_CUTLASS_RUNNER_OPS = frozenset({
    "mm",
    "bmm",
    "addmm",
    "convolution",
    "cudnn_convolution",
})

_compiled_cache: dict[tuple, Any] = {}


def _torch_to_cute(torch_dt: torch.dtype):
    """Map torch dtype to (in_type, out_type) CuTe DSL types."""
    if torch_dt == torch.float16:
        return Float16, Float16
    return Float32, Float32


def _norm2d(v: int | list | tuple) -> list[int]:
    """Normalize a scalar or pair to a 2-element int list."""
    if isinstance(v, int):
        return [v, v]
    return [int(v[0]), int(v[1])]


def _get_compiled_gemm(
    M: int,
    N: int,
    K: int,
    arch: str,
    cfg: CutlassConfig,
    in_type,
    out_type,
):
    """Return a cached compiled GEMM for the given dimensions + config.

    ponytail: block dims capped at 32 to stay within CUDA's 1024
    threads-per-block limit. A real tiled GEMM would use the config's
    tile_m/tile_n for cooperative tile sizes, not raw thread count.
    """
    block_m = min(cfg.tile_m, 32)
    block_n = min(cfg.tile_n, 32)
    dtype_key = "fp16" if in_type is Float16 else "fp32"
    key = (M, N, K, arch, block_m, block_n, dtype_key)
    compiled = _compiled_cache.get(key)
    if compiled is None:
        compiled = _compile_gemm(M, N, K, arch, block_m, block_n, in_type, out_type)
        _compiled_cache[key] = compiled
    return compiled


def _run_conv(
    args: list, config: dict, arch: str, kwargs: dict | None = None
) -> torch.Tensor:
    """Run 2D convolution via im2col + CUTLass GEMM.

    Tensor args (input, weight) come from ``args``. Scalar params
    (stride, padding, dilation, groups) come from ``kwargs`` when
    available (aten.convolution passes them as keyword args).
    """
    kwargs = kwargs or {}
    x = args[0]
    weight = args[1]
    bias = kwargs.get("bias")
    if bias is not None and not isinstance(bias, torch.Tensor):
        bias = None

    stride = _norm2d(kwargs.get("stride", [1, 1]))
    padding = _norm2d(kwargs.get("padding", [0, 0]))
    dilation = _norm2d(kwargs.get("dilation", [1, 1]))
    groups = kwargs.get("groups", 1)

    if groups != 1:
        raise NotImplementedError(f"grouped conv (groups={groups}) not supported")

    N, C_in, H, W = x.shape
    K_out = weight.shape[0]
    R, S = weight.shape[2], weight.shape[3]

    H_out = (H + 2 * padding[0] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1
    L = H_out * W_out

    col = torch.nn.functional.unfold(
        x,
        (R, S),
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
    )
    # col: (N, C_in*R*S, L)

    weight_2d = weight.contiguous().view(K_out, -1)  # (K_out, C_in*R*S)

    cfg = CutlassConfig.from_dict(config)
    in_type, out_type = _torch_to_cute(x.dtype)

    # GEMM: weight_2d (M=K_out, K=C_in*R*S) @ col[n] (K, N=L) → (K_out, L)
    compiled = _get_compiled_gemm(K_out, L, C_in * R * S, arch, cfg, in_type, out_type)

    out = torch.empty(N, K_out, L, dtype=x.dtype, device=x.device)
    for n in range(N):
        compiled(weight_2d, col[n], out[n], K_out, L, C_in * R * S)

    if bias is not None:
        out += bias.view(1, -1, 1)

    return out.view(N, K_out, H_out, W_out)


def run_cutlass_extern(
    op_name: str,
    args: list,
    config: dict,
    arch: str,
    *,
    kwargs: dict | None = None,
) -> torch.Tensor:
    """Run a CUTLass kernel for an extern_kernel step.

    Args:
        op_name: short op name (``mm``, ``bmm``, ``addmm``,
            ``convolution``, ``cudnn_convolution``).
        args: resolved positional args (tensors + scalars) already on CUDA.
        config: ``cutlass_config`` dict from the manifest step.
        arch: GPU arch string (e.g. ``"sm_90"``).
        kwargs: resolved keyword args (used by convolution for stride etc.).

    Returns:
        Output tensor.

    Raises:
        NotImplementedError: for ops without a CUTLass runner.
    """
    if op_name in ("convolution", "cudnn_convolution"):
        return _run_conv(args, config, arch, kwargs=kwargs)

    if op_name not in _CUTLASS_RUNNER_OPS:
        raise NotImplementedError(
            f"No CUTLass runner for '{op_name}'. Supported: {_CUTLASS_RUNNER_OPS}"
        )

    cfg = CutlassConfig.from_dict(config)

    if op_name == "addmm":
        a, b = args[1], args[2]
    else:
        a, b = args[0], args[1]

    a = a.contiguous()
    b = b.contiguous()

    in_type, out_type = _torch_to_cute(a.dtype)

    if a.dim() == 2:
        M, K = a.shape
        N = b.shape[1]
        compiled = _get_compiled_gemm(M, N, K, arch, cfg, in_type, out_type)
        c = torch.empty(M, N, dtype=a.dtype, device=a.device)
        compiled(a, b, c, M, N, K)
    else:
        batch_shape = a.shape[:-2]
        M, K = a.shape[-2:]
        N = b.shape[-1]
        compiled = _get_compiled_gemm(M, N, K, arch, cfg, in_type, out_type)
        c = torch.empty(*batch_shape, M, N, dtype=a.dtype, device=a.device)
        a_3d = a.reshape(-1, M, K)
        b_3d = b.reshape(-1, K, N)
        c_3d = c.reshape(-1, M, N)
        for i in range(a_3d.shape[0]):
            compiled(a_3d[i], b_3d[i], c_3d[i], M, N, K)

    if op_name == "addmm":
        c = c + args[0]

    return c
