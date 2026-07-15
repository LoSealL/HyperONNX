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
"""

from tempfile import TemporaryDirectory

import onnxscript
import torch
import torch.nn.functional as F
from onnxifier.domain.trt.ops.mamba_plugin import causal_conv1d_schema
from onnxifier.logger import info
from torch.library import custom_op
from torch.onnx import symbolic_helper

HYPERONNX_CAUSAL_CONV1D = f"{causal_conv1d_schema.domain}::{causal_conv1d_schema.name}"


@custom_op(HYPERONNX_CAUSAL_CONV1D, mutates_args=())
def _causal_conv1d_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    context_lengths: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive causal conv1d implementation for ONNX tracing."""
    del context_lengths
    kernel_size = weight.shape[-1]
    seq_len = x.shape[-1]
    if seq_len > 1:
        mixed_qkv = F.silu(
            torch.conv1d(x, weight, bias, stride, padding, dilation, groups)
        )
        return mixed_qkv[:, :, :seq_len].clone(), mixed_qkv[..., -kernel_size:].clone()
    else:
        hidden_states_new = torch.cat([conv_state, x], dim=-1)
        conv_state.copy_(hidden_states_new[:, :, -kernel_size:])
        out = torch.conv1d(hidden_states_new, weight, bias, stride, 0, dilation, groups)
        out = F.silu(out)
        return out[..., -1:].clone(), conv_state.clone()


@_causal_conv1d_impl.register_fake
def _causal_conv1d_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    context_lengths: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, bias, context_lengths, stride, padding, dilation, groups
    return torch.zeros_like(x), conv_state.clone()


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i")
def _symbolic_causal_conv1d(
    g, x, weight, bias, conv_state, context_lengths, stride, padding, dilation, groups
):
    output, conv_state_out = g.op(
        HYPERONNX_CAUSAL_CONV1D,
        x,
        weight,
        bias,
        conv_state,
        context_lengths,
        stride_i=stride,
        padding_i=padding,
        dilation_i=dilation,
        groups_i=groups,
        outputs=2,
    )
    output.setType(x.type())
    conv_state_out.setType(conv_state.type())
    return output, conv_state_out


def register_mamba_opsets():
    """Register ONNX symbolic for torchscript export path (dynamo=False)."""

    register = getattr(torch.onnx, "register_custom_op_symbolic", None)
    if register is None:
        from torch.onnx import utils as onnx_utils

        register = onnx_utils.register_custom_op_symbolic
    try:
        register(HYPERONNX_CAUSAL_CONV1D, _symbolic_causal_conv1d, 1)
        info(f"Registered onnx symbolic function for {HYPERONNX_CAUSAL_CONV1D}")
    except RuntimeError:
        # Already registered in current process.
        pass


CAUSAL_OPSET = onnxscript.values.Opset(causal_conv1d_schema.domain, 1)  # type: ignore
ONNX_CAUSAL_CONV1D_SCRIPT = f"""

@onnxscript.script(CAUSAL_OPSET)
def onnx_causal_conv1d(
    x: onnxscript.FLOAT,
    weight: onnxscript.FLOAT,
    bias: onnxscript.FLOAT,
    conv_state: onnxscript.FLOAT,
    context_lengths: onnxscript.INT64,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> tuple[onnxscript.FLOAT, onnxscript.FLOAT]:
    # ONNX implementation of causal conv1d for ONNX export.
    output, conv_state_out = CAUSAL_OPSET.{causal_conv1d_schema.name}(
        x,
        weight,
        bias,
        conv_state,
        context_lengths,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return output, conv_state_out
"""


def mamba_translation_table() -> dict[str, onnxscript.OnnxFunction]:
    """Return custom translation table for dynamo ONNX export."""
    with TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/_onnxscript_mamba.py", "w", encoding="utf-8") as source:
            source.write(ONNX_CAUSAL_CONV1D_SCRIPT)
            code = compile(ONNX_CAUSAL_CONV1D_SCRIPT, source.name, "exec")
        g = globals()
        meta: dict = {}
        # onnxscript requires a physical file readable to getsource
        eval(code, g, meta)  # pylint: disable=eval-used
    onnx_causal_conv1d = meta["onnx_causal_conv1d"]
    return {HYPERONNX_CAUSAL_CONV1D: onnx_causal_conv1d}


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """Causal Conv1d function"""
    return _causal_conv1d_impl(
        x,
        weight,
        bias,
        conv_state,
        torch.empty(0, dtype=torch.int64),
        stride,
        padding,
        dilation,
        groups,
    )
