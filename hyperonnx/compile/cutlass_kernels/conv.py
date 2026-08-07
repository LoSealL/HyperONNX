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

from __future__ import annotations

from .config import CONV_CONFIGS, CutlassConfig
from .mm import tune_mm


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


def _conv_to_gemm_args(args: list[dict], buffers: dict) -> list[dict]:
    """Convert convolution args to equivalent GEMM args for tuning.

    ponytail: v1 uses im2col expansion — conv(M,N,K) maps to GEMM with
    M=N*H_out*W_out, N=K_out, K=C_in*R*S.
    """
    params = _extract_conv_shapes(args, buffers)
    input_shape = params["input_shape"]
    weight_shape = params["weight_shape"]

    N_batch = input_shape[0]
    C_in = input_shape[1]
    K_out = weight_shape[0]
    R, S = weight_shape[2], weight_shape[3]
    # ponytail: assume stride=1, pad=0, dilation=1 for v1
    H_out = input_shape[2] - R + 1
    W_out = input_shape[3] - S + 1

    M = N_batch * H_out * W_out
    K = C_in * R * S
    N = K_out

    return [
        {"kind": "tensor", "shape": [M, K], "dtype": args[0].get("dtype", "float16")},
        {"kind": "tensor", "shape": [K, N], "dtype": args[0].get("dtype", "float16")},
    ]


def tune_conv(
    args: list[dict],
    buffers: dict,
    arch: str,
    configs: list[CutlassConfig] | None = None,
) -> CutlassConfig:
    """Autotune convolution configs and return the best one.

    Delegates to tune_mm with im2col-expanded shapes.
    """
    gemm_args = _conv_to_gemm_args(args, buffers)
    return tune_mm(gemm_args, buffers, arch, configs or CONV_CONFIGS)
