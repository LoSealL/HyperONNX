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

from collections.abc import Sequence

import onnx
import pytest
import torch

from hyperonnx.exporter.dynamo import make_custom_op
from hyperonnx.hyper_export import trace_module_spec
from hyperonnx.typing import default_module_spec


class TestSingleTensorToTensor(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).neg()


class TestDualTensorToTuple(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y), torch.mul(x, y).mean(dim=1)


class TestReturnDict(torch.nn.Module):
    def forward(self, x, y):
        return {"sum": torch.add(x, y), "prod": torch.mul(x, y).mean(dim=1)}


class TestPythonPODValue(torch.nn.Module):
    def forward(self, x, y=1, z="str", w=2.0, flag=None):
        ty = torch.ones_like(x) * y
        return torch.add(x, ty)


class TestTensorListInput(torch.nn.Module):
    def forward(self, x: list[torch.Tensor]):
        return torch.stack(x).sum(dim=0)


@pytest.mark.parametrize(
    ["model_cls", "example_inputs"],
    [
        (TestSingleTensorToTensor, (torch.rand(3, 4, 5),)),
        (TestDualTensorToTuple, (torch.rand(3, 4, 5), torch.rand(3, 4, 5))),
        (TestReturnDict, (torch.rand(3, 4, 5), torch.rand(3, 4, 5))),
        (TestPythonPODValue, (torch.rand(3, 4), 2, "str", 2.0, None)),
        (TestTensorListInput, ([torch.rand(3, 4), torch.rand(3, 4)],)),
    ],
)
def test_dynamo_make_custom_op(model_cls, example_inputs):
    model = model_cls()
    spec = default_module_spec()
    spec = trace_module_spec(
        model,
        example_inputs,
        {},
        opset_version=19,
        hiera=[model.__class__],
        module_spec=spec,
        dynamo=True,
    )
    custom_fn, ctt = make_custom_op(model, spec[model])
    out = custom_fn(*spec[model]["args"], **spec[model].get("kwargs", {}))
    n_ins = len(spec[model]["args"]) + len(spec[model].get("kwargs", {}))
    # patch
    if model_cls is TestTensorListInput:
        n_ins = 2
    elif model_cls is TestPythonPODValue:
        n_ins = 1
    n_outs = len(out) if isinstance(out, Sequence) else 1

    f = torch.onnx.export(
        custom_fn,
        example_inputs,
        None,
        custom_translation_table=ctt,  # type: ignore
        dynamo=True,
    )
    assert f is not None
    try:
        onnx.checker.check_model(f.model_proto, full_check=True)
        assert len(f.model.graph.inputs) == n_ins
        assert len(f.model.graph.outputs) == n_outs
        assert f.model.graph.node(0).domain == "hyper"
    except Exception:
        f.save(f"error_{model_cls.__name__}.onnx")
        raise
