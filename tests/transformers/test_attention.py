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

from typing import TYPE_CHECKING

import onnx
import pytest
import torch
from torch.onnx._constants import ONNX_MAX_OPSET

from hyperonnx.transformers.attention import (
    HYPERONNX_ATTN_IMPL,
    attention_interface,
    attention_translation_table,
    promote_onnx_model_to_opset24,
    register_attention_opsets,
)

if TYPE_CHECKING:
    from torch._C._nn import scaled_dot_product_attention
else:
    from torch.nn.functional import scaled_dot_product_attention


def test_attention_opset24_forward():
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 4, 7, 8)
    value = torch.randn(2, 4, 7, 6)
    attn_mask = torch.ones([5, 7], dtype=torch.bool)
    scale = 0.42
    actual, _ = attention_interface(
        torch.nn.Module(), query, key, value, attn_mask, scale
    )
    expected = scaled_dot_product_attention(query, key, value, attn_mask, scale=scale)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_attention_function_supports_gqa():
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 2, 7, 8)
    value = torch.randn(2, 2, 7, 6)
    actual, _ = attention_interface(torch.nn.Module(), query, key, value, None, 0.125)
    expected = scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        scale=0.125,
        enable_gqa=True,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_transformers_with_custom_attn_implementation(dynamo, tmp_path):
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    except ImportError:
        pytest.skip("transformers/Qwen3.5 not installed, skipping test")

    config = Qwen3_5TextConfig()
    config.vocab_size = 1024
    config.hidden_size = 256
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.intermediate_size = 256 * 3
    config.layer_types = ["full_attention"]
    # pylint: disable=protected-access
    config._attn_implementation = HYPERONNX_ATTN_IMPL
    model = Qwen3_5DecoderLayer(config, 0).eval()
    hidden_states = torch.randn(1, 16, config.hidden_size)
    position_embeddings = (
        torch.randn(1, 16, config.hidden_size),
        torch.randn(1, 16, config.hidden_size),
    )
    attention_mask = torch.ones(16, 16, dtype=torch.bool)
    register_attention_opsets()
    with torch.no_grad():
        model(hidden_states, position_embeddings, attention_mask)

    output_path = tmp_path / "attention_opset24_ts.onnx"
    torch.onnx.export(
        model,
        (hidden_states, position_embeddings, attention_mask),
        output_path,
        input_names=[
            "hidden_states",
            "position_embeddings_cos",
            "position_embeddings_sin",
            "attention_mask",
        ],
        output_names=["last_hidden_state"],
        opset_version=ONNX_MAX_OPSET,
        dynamo=dynamo,
        custom_translation_table=(
            attention_translation_table()  # type: ignore
        ),
    )
    onnx_model = onnx.load_model(str(output_path))
    onnx_model = promote_onnx_model_to_opset24(onnx_model)
    assert any(i.domain == "" and i.version == 24 for i in onnx_model.opset_import)
    onnx.checker.check_model(onnx_model, full_check=True)

    attention_node_exported = 0
    for node in onnx_model.graph.node:
        if node.op_type == "Attention":
            attention_node_exported += 1
            assert node.domain == ""
            for t in node.attribute:
                if t.name == "scale":
                    assert t.f == 0.0625
    assert attention_node_exported == 1
