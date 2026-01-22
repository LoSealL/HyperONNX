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

# pylint: disable=missing-class-docstring,missing-function-docstring

import onnx
import torch

from hyperonnx.exporter.dynamo import replace_with_custom_op
from hyperonnx.hyper_export import trace_module_spec
from hyperonnx.typing import ExportStatus, default_module_spec


class Hiera1(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).neg()


class Hiera2(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y), torch.mul(x, y).mean(dim=1)


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hiera1 = Hiera1()
        self.hiera2 = Hiera2()

    def forward(self, x, y):
        a = self.hiera1(x)
        b, c = self.hiera2(a, y)
        return {"b": b, "c": c}


def test_dynamo_replace_hiera():
    model = TestModel()
    example_inputs = (torch.randn(2, 3), torch.randn(2, 3))
    spec = default_module_spec()
    spec = trace_module_spec(
        model,
        example_inputs,
        {},
        opset_version=19,
        hiera=[Hiera1, Hiera2],
        module_spec=spec,
        dynamo=True,
    )
    spec[model.hiera1]["status"] = ExportStatus.EXPORTED
    spec[model.hiera2]["status"] = ExportStatus.EXPORTED
    with replace_with_custom_op(model, spec) as ctt:
        f = torch.onnx.export(
            model,
            example_inputs,
            None,
            custom_translation_table=ctt,  # type: ignore
            dynamo=True,
        )
    assert f is not None
    try:
        onnx.checker.check_model(f.model_proto, full_check=True)
        assert len(list(f.model.graph.all_nodes())) == 2
        assert all(n.domain == "hyper" for n in f.model.graph.all_nodes())
    except Exception:
        f.save("error_dynamo_replace_hiera.onnx")
        raise
