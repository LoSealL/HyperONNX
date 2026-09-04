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

CUTLASS convolution config tuner.
Only tunes and returns the best config — no cubin export.
"""

import ast
from typing import Any

from .config import CONV_CONFIGS, CutlassConfig
from .mm import tune_mm


def _parse_kwargs(kwarg_strs: list[str] | None) -> dict[str, Any]:
    """Parse ``["stride=(2, 2)", "bias=None"]`` kwarg strings into values."""
    out: dict[str, Any] = {}
    for kw in kwarg_strs or []:
        key, _, value = kw.partition("=")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def _norm2(v: int | list | tuple) -> list[int]:
    """Normalize a scalar or pair to a 2-element int list."""
    if isinstance(v, int):
        return [v, v]
    return [int(v[0]), int(v[1])]


def _extract_conv_shapes(args: list[dict], buffers: dict) -> dict:
    """Extract convolution parameters from manifest args."""
    tensor_args = [a for a in args if a.get("kind") == "tensor"]

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

    if len(tensor_args) < 2:
        raise ValueError(f"Expected >=2 tensor args for conv, got {len(tensor_args)}")

    return {
        "input_shape": _shape_of(tensor_args[0]),
        "weight_shape": _shape_of(tensor_args[1]),
    }


def _conv_to_gemm_args(
    args: list[dict], buffers: dict, kwargs: list[str] | None = None
) -> list[dict]:
    """Convert convolution args to the per-group im2col GEMM replay runs.

    Replay (``run._run_conv``) launches one GEMM per (batch, group):
    weight_g (K_g, Kk) @ col_g (Kk, L), so tune exactly that shape —
    including groups, stride, padding and dilation from the step kwargs.
    """
    params = _extract_conv_shapes(args, buffers)
    input_shape = params["input_shape"]
    weight_shape = params["weight_shape"]

    kw = _parse_kwargs(kwargs)
    stride = _norm2(kw.get("stride", 1))
    padding = _norm2(kw.get("padding", 0))
    dilation = _norm2(kw.get("dilation", 1))
    groups = int(kw.get("groups", 1))

    K_out = weight_shape[0]
    R, S = weight_shape[2], weight_shape[3]
    H_in, W_in = input_shape[2], input_shape[3]
    H_out = (H_in + 2 * padding[0] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1
    L = H_out * W_out

    M = K_out // groups
    K = weight_shape[1] * R * S

    return [
        {"kind": "tensor", "shape": [M, K], "dtype": args[0].get("dtype", "float16")},
        {"kind": "tensor", "shape": [K, L], "dtype": args[0].get("dtype", "float16")},
    ]


def tune_conv(
    args: list[dict],
    buffers: dict,
    arch: str,
    configs: list[CutlassConfig] | None = None,
    kwargs: list[str] | None = None,
) -> tuple[CutlassConfig | None, dict]:
    """Autotune convolution configs against the cuDNN/cuBLAS baseline.

    Delegates to tune_mm with the per-group im2col-expanded shapes; returns
    ``(None, bench)`` when the vendor library is faster (the common case).
    """
    gemm_args = _conv_to_gemm_args(args, buffers, kwargs)
    return tune_mm(gemm_args, buffers, arch, configs or CONV_CONFIGS)
