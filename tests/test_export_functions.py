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

from io import BytesIO
from pathlib import Path

import onnx
import torch
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnxifier import ONNXIFIER_OPSET, OnnxGraph, PassManager

from hyperonnx import export_hyper_onnx


class ModuleLvl2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, attr1, attr2=2):
        return x * attr1 + y * attr2, x + y


class ModuleLvl1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(1, 1, 1)
        self.body = torch.nn.ModuleList([ModuleLvl2(), ModuleLvl2()])

    def forward(self, x):
        y = self.head(x)
        z1, z2 = x, y
        for i, net in enumerate(self.body):
            z1, z2 = net(z1, z2, attr1=i, attr2=i + 1)
        return {
            "z1": z1,
            "z2": z2,
        }


class ModuleTop(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 1, bias=False)
        self.body = ModuleLvl1()

    def forward(self, x):
        x = self.body(self.head(x))
        return x


def build_hyber_onnx():
    HYPER_DOMAIN = "hyper"

    func1_node = make_node("Foo1", ["x"], ["y", "y2"], name="foo1", domain=HYPER_DOMAIN)
    func2_node = make_node("Foo2", ["x"], ["y", "y2"], name="foo2", domain=HYPER_DOMAIN)

    g = make_graph(
        [func1_node],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 1, 4, 4))],
        [
            make_tensor_value_info("y", onnx.TensorProto.FLOAT, (1, 1, 4, 4)),
            make_tensor_value_info("y2", onnx.TensorProto.FLOAT, (1, 1, 4, 4)),
        ],
    )

    func1 = make_function(
        HYPER_DOMAIN,
        "Foo1",
        ["x"],
        ["y", "y2"],
        [func2_node],
        [make_operatorsetid(HYPER_DOMAIN, 1), ONNXIFIER_OPSET],
    )

    func2 = make_function(
        HYPER_DOMAIN,
        "Foo2",
        ["x2"],
        ["y2", "y3"],
        [
            make_node("Relu", ["x2"], ["y"], name="relu1"),
            make_node("Relu", ["y"], ["y2"], name="relu2"),
            make_node("Relu", ["y2"], ["y3"], name="relu3"),
        ],
        [make_operatorsetid(HYPER_DOMAIN, 1), ONNXIFIER_OPSET],
    )

    model = make_model(
        g,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid(HYPER_DOMAIN, 1)],
        functions=[func1, func2],
    )
    onnx.checker.check_model(model)
    return model


def test_export_module_with_2_levels(tmp_path):
    model = ModuleTop()
    with BytesIO() as f:
        export_hyper_onnx(
            model, (torch.randn(1, 1, 1, 1),), f, hiera=[ModuleLvl1, ModuleLvl2]
        )
        onnx_model = onnx.load_from_string(f.getvalue())
    onnx.checker.check_model(onnx_model, True)
    types = set(":".join(k.split(":")[:-1]) for k in OnnxGraph(onnx_model).functions)
    assert "ModuleLvl1" in types
    assert "ModuleLvl2" in types

    graph = OnnxGraph(onnx_model)
    passes = PassManager(
        ["export_functions"],
        configs={
            "export_functions": {"path": tmp_path},
        },
    )
    passes.optimize(graph, True)
    subgraphs = sorted(Path(tmp_path).rglob("*.onnx"))
    # 00.conv, 01.00, 01.01, 01.02, totally 4 parts
    assert len(subgraphs) == 4
    # ensure the subgraph is topologically sorted
    graph_verify = make_graph(
        [],
        "ver",
        graph.input,
        graph.output,
    )
    for g in subgraphs:
        subgraph = onnx.load_model(g, load_external_data=False)
        node = make_node(
            g.stem,
            [j.name for j in subgraph.graph.input],
            [j.name for j in subgraph.graph.output],
            name=g.stem,
            domain="TEST",
        )
        graph_verify.node.append(node)
    model_verify = make_model(
        graph_verify, opset_imports=[make_operatorsetid("TEST", 1), ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model_verify, True)


def test_export_hyber_model_with_same_io(tmp_path):
    onnx_model = build_hyber_onnx()
    graph = OnnxGraph(onnx_model)
    passes = PassManager(
        ["export_functions"],
        configs={
            "export_functions": {"path": tmp_path},
        },
    )
    passes.optimize(graph, True)
    subgraphs = sorted(Path(tmp_path).rglob("*.onnx"))
    # 00.conv, 01.00, 01.01, 01.02, totally 4 parts
    assert len(subgraphs) == 1
