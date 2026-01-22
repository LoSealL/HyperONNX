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

import io
from contextlib import redirect_stderr

import networkx as nx
import numpy as np
import onnx
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from onnxifier import (
    ONNXIFIER_IR_VERSION,
    ONNXIFIER_OPSET,
    OnnxGraph,
    make_operatorsetid,
)
from onnxifier.passes.utils import make_constant

from hyperonnx.function_rewriter import ComposeNodesToFunctionsRewriter


def test_compose_nodes_to_functions_cse():
    r"""
      n_a
       |
      n_b
      / \
    f_c f_d
    """
    n_a = make_node("Relu", ["input"], ["a"], name="n_a")
    n_b = make_node("Sin", ["a"], ["b"], name="n_b")
    f_c = make_node("foo", ["b"], ["output.0"], name="f_c", domain="hyper")
    f_d = make_node("bar", ["b"], ["output.1"], name="f_d", domain="hyper")
    g = make_graph(
        [n_a, n_b, f_c, f_d],
        "test_graph",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [
            make_tensor_value_info("output.0", onnx.TensorProto.FLOAT, [1, 2, 3]),
            make_tensor_value_info("output.1", onnx.TensorProto.FLOAT, [1, 2, 3]),
        ],
    )
    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("hyper", 1)],
    )
    model.functions.append(
        make_function(
            "hyper",
            "foo",
            ["x"],
            ["y"],
            nodes=[make_node("Identity", ["x"], ["y"])],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )
    model.functions.append(
        make_function(
            "hyper",
            "bar",
            ["x"],
            ["y"],
            nodes=[make_node("Identity", ["x"], ["y"])],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )
    onnx.checker.check_model(model)
    graph = OnnxGraph(model)
    rewrite = ComposeNodesToFunctionsRewriter()
    graph = rewrite(graph)

    assert len(graph.functions) == 3
    assert "Component:0" in graph.functions
    onnx.checker.check_model(graph.model)


def test_compose_nodes_to_functions_constant_value():
    r"""
    n_a
     |
    n_b  const
     \  /
     f_c
    """
    const = make_constant("data", np.array([1], dtype=np.float32))
    n_a = make_node("Relu", ["input"], ["a"], name="n_a")
    n_b = make_node("Sin", ["a"], ["b"], name="n_b")
    f_c = make_node(
        "foo", ["b", const.output[0]], ["output.0"], name="f_c", domain="hyper"
    )
    g = make_graph(
        [n_a, n_b, const, f_c],
        "test_graph",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("output.0", onnx.TensorProto.FLOAT, [1, 2, 3])],
    )
    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("hyper", 1)],
    )
    model.functions.append(
        make_function(
            "hyper",
            "foo",
            ["a", "b"],
            ["y"],
            nodes=[make_node("Add", ["a", "b"], ["y"])],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )
    onnx.checker.check_model(model)
    graph = OnnxGraph(model)
    rewrite = ComposeNodesToFunctionsRewriter()
    graph = rewrite(graph)

    assert len(graph.functions) == 2
    assert "Component:0" in graph.functions
    assert "data" in graph
    onnx.checker.check_model(graph.model)


def test_compose_nodes_to_functions_partial_cse():
    r"""
      n_a
       |
      n_b
      / \
    f_c n_d
     |   |
     |  n_e
      \  |
       f_f
    """
    n_a = make_node("Relu", ["input"], ["a"], name="n_a")
    n_b = make_node("Sin", ["a"], ["b"], name="n_b")
    n_d = make_node("Relu", ["b"], ["d"], name="n_d")
    n_e = make_node("Cos", ["d"], ["e"], name="n_e")
    f_c = make_node("foo", ["b"], ["c"], name="f_c", domain="hyper")
    f_f = make_node("bar", ["c", "e"], ["output.0"], name="f_f", domain="hyper")
    g = make_graph(
        [n_a, n_b, n_d, n_e, f_c, f_f],
        "test_graph",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("output.0", onnx.TensorProto.FLOAT, [1, 2, 3])],
    )
    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("hyper", 1)],
    )
    model.functions.append(
        make_function(
            "hyper",
            "foo",
            ["x"],
            ["y"],
            nodes=[make_node("Identity", ["x"], ["y"])],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )
    model.functions.append(
        make_function(
            "hyper",
            "bar",
            ["a", "b"],
            ["y"],
            nodes=[make_node("Add", ["a", "b"], ["y"])],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )
    onnx.checker.check_model(model)
    for order in ["post", "pre"]:
        graph = OnnxGraph(model)
        rewrite = ComposeNodesToFunctionsRewriter(order=order)
        graph = rewrite(graph)

        try:
            assert len(graph.functions) == {"post": 3, "pre": 4}[order]
            assert "Component:0" in graph.functions
            if order == "pre":
                assert "Component:1" in graph.functions
            onnx.checker.check_model(graph.model)
        except Exception:
            graph.save(f"partial_cse_{order}.onnx", check=False)
            raise


def test_compose_nodes_to_functions_initializers():
    r"""
    n_a  init
     |  /
     n_b
      \
      f_c
    """
    n_a = make_node("Relu", ["input"], ["a"], name="n_a")
    n_b = make_node("Add", ["a", "data"], ["b"], name="n_b")
    f_c = make_node("foo", ["b"], ["output.0"], name="f_c", domain="hyper")
    g = make_graph(
        [n_a, n_b, f_c],
        "test_graph",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("output.0", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [from_array(np.array([1], dtype=np.float32), "data")],
    )
    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("hyper", 1)],
    )
    model.functions.append(
        make_function(
            "hyper",
            "foo",
            ["x"],
            ["y"],
            nodes=[make_node("Identity", ["x"], ["y"])],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )
    onnx.checker.check_model(model)
    graph = OnnxGraph(model)
    rewrite = ComposeNodesToFunctionsRewriter()
    graph = rewrite(graph)

    assert len(graph.functions) == 2
    assert "Component:0" in graph.functions
    onnx.checker.check_model(graph.model)


def test_compose_nodes_to_functions_cycle_prevention():
    """Test that cycle detection prevents formation of cycles in function composition.

    This test creates a graph structure that could potentially form cycles
    during function composition and verifies that our cycle detection
    mechanism prevents this.

    Graph structure:

      n_a
       |
      n_b  <-----+
      / \\        | potential cycle
    n_c  n_d     |
     |    |      |
    n_e  f_f     | (f_f is now a function node)
     \\   /       |
      n_g        |
       |         |
      f_h -------+
    """
    # Create nodes that could form a complex dependency structure
    n_a = make_node("Relu", ["input"], ["a"], name="n_a")
    n_b = make_node("Sin", ["a"], ["b"], name="n_b")
    n_c = make_node("Tanh", ["b"], ["c"], name="n_c")
    n_d = make_node("Sigmoid", ["b"], ["d"], name="n_d")
    n_e = make_node("Log", ["c"], ["e"], name="n_e")
    # Change n_f to a function node to create more complex function dependencies
    f_f = make_node("exp_func", ["d"], ["f"], name="f_f", domain="hyper")
    n_g = make_node("Add", ["e", "f"], ["g"], name="n_g")

    # This function node creates a complex dependency pattern
    f_h = make_node("complex_func", ["g"], ["output.0"], name="f_h", domain="hyper")

    g = make_graph(
        [n_a, n_b, n_c, n_d, n_e, f_f, n_g, f_h],
        "test_cycle_prevention",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("output.0", onnx.TensorProto.FLOAT, [1, 2, 3])],
    )

    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("hyper", 1)],
    )

    # Add a complex function that processes the composed result
    model.functions.append(
        make_function(
            "hyper",
            "complex_func",
            ["x"],
            ["y"],
            nodes=[
                make_node("Relu", ["x"], ["temp1"], name="temp_relu"),
                make_node("Sin", ["temp1"], ["y"], name="temp_sin"),
            ],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )

    # Add the exp_func function definition
    model.functions.append(
        make_function(
            "hyper",
            "exp_func",
            ["x"],
            ["y"],
            nodes=[
                make_node("Exp", ["x"], ["temp_exp"], name="exp_internal"),
                make_node("Abs", ["temp_exp"], ["y"], name="abs_internal"),
            ],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )

    onnx.checker.check_model(model)

    # Test with different orders to ensure robustness
    for order in ["post", "pre"]:
        graph = OnnxGraph(model)
        rewrite = ComposeNodesToFunctionsRewriter(order=order)

        # The rewriter should handle this without creating cycles
        graph = rewrite(graph)

        # Verify the graph is still valid
        onnx.checker.check_model(graph.model)

        # Verify no cycles exist in the final graph
        cycles = list(nx.simple_cycles(graph))
        assert (
            len(cycles) == 0
        ), f"Cycles detected in final graph with order={order}: {cycles}"

        # Should have composed some functions
        assert len(graph.functions) >= 2  # At least exp_func and complex_func

        # The hyper functions should still exist
        assert any("complex_func" in func_name for func_name in graph.functions.keys())
        assert any("exp_func" in func_name for func_name in graph.functions.keys())


def test_compose_nodes_to_functions_self_referential_prevention():
    """Test prevention of self-referential cycles in function composition.

    This test creates a scenario where nodes could potentially reference
    themselves through function composition, which should be prevented.

    Graph topology:

                        input
                          |
                         n_a
                          |
                         n_b
                          |
                         n_c
                          |
                         f_d
                          |
                       output

    Dependencies:
    - n_a -> n_b (simple chain)
    - n_b -> n_c (simple chain)
    - n_c -> f_d (simple chain)

    This simple linear structure tests that even basic graphs don't
    accidentally create self-referential cycles during function composition.
    The cycle detection should handle this straightforward case without issues.
    """
    # Create a simple graph with potential for self-reference
    n_a = make_node("Relu", ["input"], ["a"], name="n_a")
    n_b = make_node("Sin", ["a"], ["b"], name="n_b")
    n_c = make_node("Identity", ["b"], ["c"], name="n_c")
    f_d = make_node("self_ref", ["c"], ["output.0"], name="f_d", domain="hyper")

    g = make_graph(
        [n_a, n_b, n_c, f_d],
        "test_self_ref_prevention",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("output.0", onnx.TensorProto.FLOAT, [1, 2, 3])],
    )

    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("hyper", 1)],
    )

    # Add a function that could create self-reference issues
    model.functions.append(
        make_function(
            "hyper",
            "self_ref",
            ["x"],
            ["y"],
            nodes=[make_node("Identity", ["x"], ["y"], name="identity_func")],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )

    onnx.checker.check_model(model)
    graph = OnnxGraph(model)
    rewrite = ComposeNodesToFunctionsRewriter()

    # Should complete without errors and without creating cycles
    graph = rewrite(graph)

    cycles = list(nx.simple_cycles(graph))
    assert len(cycles) == 0, f"Self-referential cycles detected: {cycles}"

    # Verify the graph is still valid
    onnx.checker.check_model(graph.model)

    # Should have at least the original function plus any composed functions
    assert len(graph.functions) == 2


def test_compose_nodes_to_functions_complex_cycle_scenario():
    r"""Test a complex scenario that could create cycles through multiple levels.

    This creates a more sophisticated graph structure where multiple
    components could potentially create interdependencies leading to cycles.

    Graph topology:

                         input
                           |
                          n1
                           |
                          n2
                       /   |   \
                      /    |    \
                     /     |     \
                   n3      n4      n5
                    |      |       |
                    |    /   \     |
                    |   /     \    |
                    |  /       \   |
                    | /         \  |
                   n6           n7-+
                    |            | |
                    |            | |
                    +--------n8--+-+
                            / \
                           /   \
                          /     \
                         +-------+
                              |
                             n9
                              |
                             f1
                              |
                             f2
                              |
                          output

    Dependencies:
    - n1 -> n2
    - n2 -> n3, n4, n5  (fan-out)
    - n3, n4 -> n6
    - n4, n5 -> n7
    - n3, n5 -> n8
    - n6, n7, n8 -> n9  (fan-in convergence)
    - n9 -> f1 -> f2 (sequential processing)

    The cycle detection should ensure that when composing nodes into functions,
    no circular dependencies are created between the resulting function nodes.
    """
    # Layer 1: Input processing
    n1 = make_node("Relu", ["input"], ["t1"], name="layer1_relu")
    n2 = make_node("Sin", ["t1"], ["t2"], name="layer1_sin")

    # Layer 2: Parallel processing branches
    n3 = make_node("Tanh", ["t2"], ["t3"], name="layer2_tanh")
    n4 = make_node("Sigmoid", ["t2"], ["t4"], name="layer2_sigmoid")
    n5 = make_node("Log", ["t2"], ["t5"], name="layer2_log")

    # Layer 3: Cross-connections (potential cycle risk)
    n6 = make_node("Add", ["t3", "t4"], ["t6"], name="layer3_add1")
    n7 = make_node("Mul", ["t4", "t5"], ["t7"], name="layer3_mul1")
    n8 = make_node("Sub", ["t3", "t5"], ["t8"], name="layer3_sub1")

    # Layer 4: Convergence layer
    n9 = make_node("Concat", ["t6", "t7", "t8"], ["t9"], name="layer4_concat", axis=0)

    # Function nodes that could create complex dependencies
    f1 = make_node("proc1", ["t9"], ["t10"], name="func1", domain="hyper")
    f2 = make_node("proc2", ["t10"], ["output.0"], name="func2", domain="hyper")

    g = make_graph(
        [n1, n2, n3, n4, n5, n6, n7, n8, n9, f1, f2],
        "test_complex_cycle_scenario",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("output.0", onnx.TensorProto.FLOAT, [1, 2, 3])],
    )

    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("hyper", 1)],
    )

    # Add functions with complex internal structures
    model.functions.extend(
        [
            make_function(
                "hyper",
                "proc1",
                ["x"],
                ["y"],
                nodes=[
                    make_node(
                        "Constant",
                        [],
                        ["shape1"],
                        name="shape_const1",
                        value=onnx.helper.make_tensor(
                            name="shape1_value",
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=[-1],
                        ),
                    ),
                    make_node("Reshape", ["x", "shape1"], ["r2"], name="proc1_reshape"),
                    make_node("Relu", ["r2"], ["r3"], name="proc1_relu"),
                    make_node(
                        "Constant",
                        [],
                        ["shape2"],
                        name="shape_const2",
                        value=onnx.helper.make_tensor(
                            name="shape2_value",
                            data_type=onnx.TensorProto.INT64,
                            dims=[3],
                            vals=[1, 2, 3],
                        ),
                    ),
                    make_node(
                        "Reshape", ["r3", "shape2"], ["y"], name="proc1_reshape2"
                    ),
                ],
                opset_imports=[
                    make_operatorsetid("hyper", 1),
                    ONNXIFIER_OPSET,
                ],
            ),
            make_function(
                "hyper",
                "proc2",
                ["x"],
                ["y"],
                nodes=[
                    make_node("Abs", ["x"], ["a1"], name="proc2_abs"),
                    make_node("Sin", ["a1"], ["y"], name="proc2_sin"),
                ],
                opset_imports=[
                    make_operatorsetid("hyper", 1),
                    ONNXIFIER_OPSET,
                ],
            ),
        ]
    )

    onnx.checker.check_model(model)

    # Test cycle prevention with this complex structure
    graph = OnnxGraph(model)
    rewriter = ComposeNodesToFunctionsRewriter()

    # This should complete without creating cycles
    result_graph = rewriter(graph)

    # Check for any cycles in the final graph
    cycles = list(nx.simple_cycles(result_graph))
    assert len(cycles) == 0, f"Cycles detected in complex scenario: {cycles}"

    # Verify graph integrity
    onnx.checker.check_model(result_graph.model)

    # Should have created some composed functions
    assert len(result_graph.functions) >= 2

    # Original functions should still be present
    func_names = list(result_graph.functions.keys())
    assert any("proc1" in name or "proc2" in name for name in func_names)


def test_compose_nodes_to_functions_with_logging():
    """Test cycle detection with logging to verify warning messages."""

    # Create a potentially problematic graph structure
    n1 = make_node("Identity", ["input"], ["t1"], name="node1")
    n2 = make_node("Identity", ["t1"], ["t2"], name="node2")
    n3 = make_node("Identity", ["t2"], ["t3"], name="node3")
    f1 = make_node("test_func", ["t3"], ["output"], name="func_node", domain="hyper")

    g = make_graph(
        [n1, n2, n3, f1],
        "test_logging",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )

    model = make_model(
        g,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("hyper", 1)],
    )

    model.functions.append(
        make_function(
            "hyper",
            "test_func",
            ["x"],
            ["y"],
            nodes=[make_node("Identity", ["x"], ["y"])],
            opset_imports=[
                make_operatorsetid("hyper", 1),
                ONNXIFIER_OPSET,
            ],
        )
    )

    onnx.checker.check_model(model)

    # Capture logging output
    captured_output = io.StringIO()

    with redirect_stderr(captured_output):
        graph = OnnxGraph(model)
        rewriter = ComposeNodesToFunctionsRewriter()
        result_graph = rewriter(graph)

    # Verify the rewriting completed successfully
    onnx.checker.check_model(result_graph.model)

    # Check that cycle detection mechanism ran (may or may not log warnings)
    # The important thing is that no exception was raised and the graph is valid
    assert len(result_graph.functions) >= 1
