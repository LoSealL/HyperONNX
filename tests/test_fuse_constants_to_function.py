"""
Copyright (C) 2025 The HYPERONNX Authors.

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

import numpy as np
import onnx
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph
from onnxifier.passes.utils import make_constant

from hyperonnx.function_rewriter import FuseConstantsToFunctionRewriter
from hyperonnx.utils import HYPER_DOMAIN


def test_fuse_constants_to_function():
    r"""
    const const
      \   /
       func
    """
    const1 = make_constant("const1", np.ones([3, 8, 1, 1], dtype=np.float32))
    const2 = make_constant("const2", np.ones([3], dtype=np.float32))
    func = make_node(
        "Foo",
        ["x", const1.output[0], const2.output[0]],
        ["y"],
        name="foo",
        domain=HYPER_DOMAIN,
    )
    g = make_graph(
        [const1, const2, func],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 8, 4, 4))],
        [make_tensor_value_info("y", onnx.TensorProto.FLOAT, (1, 3, 4, 4))],
    )
    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid(HYPER_DOMAIN, 1)],
    )
    model.functions.append(
        make_function(
            HYPER_DOMAIN,
            "Foo",
            ["x", "w", "b"],
            ["y"],
            [make_node("Conv", ["x", "w", "b"], ["y"], name="conv")],
            [make_operatorsetid(HYPER_DOMAIN, 1), ONNXIFIER_OPSET],
        )
    )
    onnx.checker.check_model(model, True)
    rewriter = FuseConstantsToFunctionRewriter()
    graph = rewriter(OnnxGraph(model))
    onnx.checker.check_model(graph.model, True)

    assert len(graph) == 1
    assert len(graph.functions["Foo"].node) == 3


def test_fuse_constants_to_function_multi_users():
    r"""
    const const
      \   /
       func
    """
    const1 = make_constant("const1", np.ones([3, 8, 1, 1], dtype=np.float32))
    const2 = make_constant("const2", np.ones([3], dtype=np.float32))
    const3 = make_constant("const3", np.ones([3, 8, 1, 1], dtype=np.float32))
    const4 = make_constant("const4", np.ones([3], dtype=np.float32))
    func1 = make_node(
        "Foo",
        ["x", const1.output[0], const2.output[0]],
        ["y1"],
        name="foo",
        domain=HYPER_DOMAIN,
    )
    func2 = make_node(
        "Foo",
        ["x", const3.output[0], const4.output[0]],
        ["y2"],
        name="bar",
        domain=HYPER_DOMAIN,
    )
    g = make_graph(
        [const1, const2, const3, const4, func1, func2],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 8, 4, 4))],
        [
            make_tensor_value_info("y1", onnx.TensorProto.FLOAT, (1, 3, 4, 4)),
            make_tensor_value_info("y2", onnx.TensorProto.FLOAT, (1, 3, 4, 4)),
        ],
    )
    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid(HYPER_DOMAIN, 1)],
    )
    model.functions.append(
        make_function(
            HYPER_DOMAIN,
            "Foo",
            ["x", "w", "b"],
            ["y"],
            [make_node("Conv", ["x", "w", "b"], ["y"], name="conv")],
            [make_operatorsetid(HYPER_DOMAIN, 1), ONNXIFIER_OPSET],
        )
    )
    onnx.checker.check_model(model, True)
    rewriter = FuseConstantsToFunctionRewriter()
    graph = rewriter(OnnxGraph(model))
    onnx.checker.check_model(graph.model, True)

    assert len(graph) == 2
    assert len(graph.functions) == 2
    assert len(graph.functions["Foo"].node) == 3
    assert len(graph.functions["Foo(foo)"].node) == 3
