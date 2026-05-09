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

import onnx
import pytest
import torch
import torch.nn.functional as F
from torch.onnx._constants import ONNX_MAX_OPSET

from hyperonnx.transformers.mamba import (
    HYPERONNX_CAUSAL_CONV1D,
    causal_conv1d_fn,
    mamba_translation_table,
    register_mamba_opsets,
)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = None,
):
    """Copied from causal-conv1d

    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """Copied from causal-conv1d

    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting
        at the index @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def test_causal_conv1d_forward_matches_causal_conv1d_ref():
    batch, dim, seqlen = 2, 16, 10
    kernel_size = 4

    x = torch.randn(batch, dim, seqlen)
    weight = torch.randn(dim, 1, kernel_size)
    bias = torch.randn(dim)
    conv_state = torch.randn(batch, dim, kernel_size)

    actual, _ = causal_conv1d_fn(
        x, weight, bias, conv_state, padding=kernel_size - 1, groups=dim
    )
    expected = causal_conv1d_ref(x, weight.squeeze(1), bias, activation="silu")
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_causal_conv1d_update_matches_causal_conv1d_update_ref():
    batch, dim, seqlen = 2, 16, 1
    kernel_size = 4

    x = torch.randn(batch, dim, seqlen)
    weight = torch.randn(dim, 1, kernel_size)
    bias = torch.randn(dim)
    conv_state = torch.randn(batch, dim, kernel_size)
    conv_state_ref = conv_state.clone()

    actual, actual_state = causal_conv1d_fn(
        x, weight, bias, conv_state, padding=kernel_size - 1, groups=dim
    )
    expected = causal_conv1d_update_ref(
        x.squeeze(-1), conv_state_ref, weight.squeeze(1), bias, activation="silu"
    )
    torch.testing.assert_close(actual.squeeze(-1), expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual_state, conv_state_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_causal_conv1d(dynamo, tmp_path):
    class CausalConv1dModel(torch.nn.Module):
        def forward(self, x, weight, bias, conv_state):
            return causal_conv1d_fn(
                x,
                weight,
                bias,
                conv_state,
                padding=weight.shape[-1] - 1,
                groups=x.shape[1],
            )

    model = CausalConv1dModel().eval()
    x = torch.randn(2, 4, 10)
    weight = torch.randn(4, 1, 4)
    bias = torch.randn(4)
    conv_state = torch.randn(2, 4, 4)

    output_path = tmp_path / f"causal_conv1d_{dynamo}.onnx"
    if not dynamo:
        register_mamba_opsets()
        ctb = None
    else:
        ctb = mamba_translation_table()
    torch.onnx.export(
        model,
        (x, weight, bias, conv_state),
        output_path,
        input_names=["x", "weight", "bias", "conv_state"],
        output_names=["output", "conv_state_out"],
        opset_version=ONNX_MAX_OPSET,
        dynamo=dynamo,
        custom_translation_table=ctb,
    )
    onnx_model = onnx.load_model(str(output_path))
    onnx.checker.check_model(onnx_model, full_check=True)

    causal_conv1d_node_exported = 0
    for node in onnx_model.graph.node:
        if node.op_type == "CausalConv1d":
            causal_conv1d_node_exported += 1
            assert node.domain == HYPERONNX_CAUSAL_CONV1D.split("::")[0]
    assert causal_conv1d_node_exported == 1
