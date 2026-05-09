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
from torch.onnx._constants import ONNX_MAX_OPSET

from hyperonnx.transformers.recurrent import (
    HYPERONNX_GATED_DELTA_RULE,
    gated_delta_rule,
    recurrent_translation_table,
    register_recurrent_opsets,
)


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_gated_delta_rule(dynamo, tmp_path):
    class GatedDeltaRuleModel(torch.nn.Module):
        def forward(self, q, k, v, g, beta, initial_state):
            return gated_delta_rule(q, k, v, g, beta, initial_state=initial_state)

    model = GatedDeltaRuleModel().eval()
    batch, seq_len, num_heads, head_dim = 2, 10, 4, 8
    q = torch.randn(batch, seq_len, num_heads, head_dim)
    k = torch.randn(batch, seq_len, num_heads, head_dim)
    v = torch.randn(batch, seq_len, num_heads, head_dim)
    g = torch.randn(batch, seq_len, num_heads)
    beta = torch.randn(batch, seq_len, num_heads)
    initial_state = torch.randn(batch, num_heads, head_dim, head_dim)

    output_path = tmp_path / f"gated_delta_rule_{dynamo}.onnx"
    if dynamo:
        ctb = recurrent_translation_table()
    else:
        register_recurrent_opsets()
        ctb = None
    torch.onnx.export(
        model,
        (q, k, v, g, beta, initial_state),
        output_path,
        input_names=["q", "k", "v", "g", "beta", "initial_state"],
        output_names=["output", "final_state"],
        opset_version=ONNX_MAX_OPSET,
        dynamo=dynamo,
        custom_translation_table=ctb,
    )
    onnx_model = onnx.load_model(str(output_path))
    onnx.checker.check_model(onnx_model, full_check=True)

    gated_delta_rule_node_exported = 0
    for node in onnx_model.graph.node:
        if node.op_type == "GatedDeltaRule":
            gated_delta_rule_node_exported += 1
            assert node.domain == HYPERONNX_GATED_DELTA_RULE.split("::")[0]
    assert gated_delta_rule_node_exported == 1
