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

# pylint: disable=arguments-differ

from collections import defaultdict
from contextlib import suppress
from copy import deepcopy
from hashlib import md5
from itertools import chain
from typing import Dict, List, Literal, Sequence, Set, Tuple, cast

import networkx as nx
import onnx
from onnx import ModelProto, NodeProto
from onnx.helper import make_function, make_node, make_operatorsetid
from onnx.numpy_helper import from_array
from onnxifier import OnnxGraph
from onnxifier.logger import debug, error, warning
from onnxifier.passes import PASSES
from onnxifier.passes.globals.infer_shape import infer_shape
from onnxifier.passes.pattern import SingleNodePattern
from onnxifier.passes.rewriter import Rewriter

from .typing import ModuleSpec
from .utils import HYPER_DOMAIN


class ComposeOnnxAsFunctionRewriter(Rewriter):
    """Compose a sub onnx as a local function to replace the custom op marked by a
    specific domain.

    Args:
        domain (str): The domain name of the custom op. Used to match the nodes.
        module_spec (Sequence[ModuleSpec]): A sequence of ModuleSpec, which contains
            the type_name and the onnx of the custom op.
    """

    def __init__(self, domain: str, module_spec: Sequence[ModuleSpec]):
        super().__init__(SingleNodePattern().with_domain(domain))
        self._specs: Dict[str, ModuleSpec] = {
            spec["type_name"]: spec for spec in module_spec
        }

    def _is_dangle_or_constant_input(self, node, input_index):
        inode = self.get_input_node(node, input_index)
        if inode is None:
            return True
        if inode.op_type in ("Constant",):
            # WA: remove the constant and turn it to initializer
            # If it is not used, then can be safely removed.
            if inode.name not in self.graph.initializers:
                self.graph.initializer.append(
                    from_array(self.get_value_or_die(inode), inode.name)
                )
            if len(self.get_output_node(inode)) <= 1:
                self -= inode
            return True
        return False

    def _get_onnx_function(self, spec: ModuleSpec):
        func = spec.get("onnx")
        if func is None:
            raise ValueError(f"No function found for {spec['name']}")
        if not isinstance(func, ModelProto):
            func = onnx.load_model(func, load_external_data=False)
        return func

    def _expand_namespace(self, namespace: str, func: ModelProto):
        if not namespace:
            return
        # encode namespace if it is too long
        if len(namespace) >= 20:
            namespace = md5(namespace.encode()).hexdigest()[9:17]
        func_map = {f.name: f for f in func.functions}

        def _rewrite_node(nodes: Sequence[NodeProto]):
            for node in nodes:
                node.name = f"{namespace}.{node.name}"
                for i, j in enumerate(node.input):
                    if j != "":
                        # can't expand empty IO name
                        node.input[i] = f"{namespace}.{j}"
                for i, j in enumerate(node.output):
                    if j != "":
                        # can't expand empty IO name
                        node.output[i] = f"{namespace}.{j}"
                if node.op_type in func_map:
                    f = func_map[node.op_type]
                    _rewrite_node(f.node)
                    for i, j in enumerate(f.input):
                        f.input[i] = f"{namespace}.{j}"
                    for i, j in enumerate(f.output):
                        f.output[i] = f"{namespace}.{j}"

        _rewrite_node(func.graph.node)
        for graph_io in chain(func.graph.input, func.graph.output):
            graph_io.name = f"{namespace}.{graph_io.name}"

    def _get_nodes_to_graph_input(self, graph: OnnxGraph, input_name: str):
        for node in graph.nodes:
            for i in graph.nodes[node]["pb"].input:
                if i == input_name:
                    yield node

    def _remove_unused_inputs(
        self, graph: OnnxGraph, unused: int, node: NodeProto, spec: ModuleSpec
    ):
        # find the actual input with edges in the graph
        # and remove dangled inputs
        debugable_input_names = spec["input_names"]
        dangled_input: List[str] = []
        dangled_or_constant_unsafe: List[str] = []
        if len(node.input) != len(debugable_input_names):
            warning(
                f"Number of inputs on {node.name} is different from spec. "
                f"Recorded inputs are: [{','.join(debugable_input_names)}]. "
                "This may be caused by a forward function with *args and **kwargs "
                "signature. (`def forward(self, *args, **kwargs)`)"
            )
            debugable_input_names = list(node.input)
        for rev_i, (input_name, debugable_name) in enumerate(
            zip(reversed(node.input), reversed(debugable_input_names))
        ):
            i = len(node.input) - rev_i - 1
            if debugable_name in spec["unused_inputs"]:
                debug(f"input in unused set: {input_name}")
                dangled_input.append(input_name)
            elif self._is_dangle_or_constant_input(node, i):
                debug(f"input unconnected to graph node: {input_name}")
                dangled_or_constant_unsafe.append(input_name)
        if len(dangled_input) < unused:
            error(
                f"Expect {unused} unused inputs but "
                f"enumerated {len(dangled_input)}."
            )
            reminder = unused - len(dangled_input)
            dangled_input.extend(dangled_or_constant_unsafe[:reminder])
        for j in dangled_input:
            debug(f"remove dangled input: {j}")
            if j in graph.inputs:
                if len(list(self._get_nodes_to_graph_input(graph, j))) == 1:
                    graph.remove_input(j)
            node.input.remove(j)

    def _remove_unused_outputs(
        self, graph: OnnxGraph, unused: int, node: NodeProto, spec: ModuleSpec
    ):
        dangled_output: List[str] = []
        dangled_output_unsafe: List[str] = []
        for i, output_name in enumerate(node.output):
            if output_name in spec["unused_outputs"]:
                dangled_output.append(output_name)
            elif output_name not in graph.outputs and not self.get_output_node(node, i):
                dangled_output.append(output_name)
            elif not self.get_output_node(node, i):
                dangled_output_unsafe.append(output_name)
        if len(dangled_output) < unused:
            error(
                f"Expect {unused} unused outputs but "
                f"enumerated {len(dangled_output)}."
            )
            reminder = unused - len(dangled_output)
            # try to drop `reminder` outputs from the last one
            dangled_output.extend(dangled_output_unsafe[-reminder:])
        for j in dangled_output:
            debug(f"remove dangled output: {j}")
            node.output.remove(j)
            if j in graph.outputs:
                graph.remove_output(j)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        if node.op_type not in self._specs:
            raise KeyError(f"node spec is not found for {node.name}")
        spec = self._specs[node.op_type]
        func = self._get_onnx_function(spec)
        # legalize inputs
        if len(node.input) > len(func.graph.input):
            debug(
                f"{node.op_type} has {len(node.input)} inputs but it actually "
                f"has {len(func.graph.input)} inputs in function."
            )
            unused = len(node.input) - len(func.graph.input)
            self._remove_unused_inputs(graph, unused, node, spec)
        elif len(node.input) != len(func.graph.input):
            warning(
                f"{node.op_type} has {len(node.input)} inputs less than "
                f"{len(func.graph.input)} inputs in its function."
            )
        # legalize outputs
        if len(node.output) != len(func.graph.output):
            debug(
                f"{node.op_type} has {len(node.output)} outputs but it actually "
                f"has {len(func.graph.output)} outputs in function."
            )
            unused = len(node.output) - len(func.graph.output)
            self._remove_unused_outputs(graph, unused, node, spec)
        self._expand_namespace(node.name, func)
        self += node
        ext_functions = list(func.functions) + [
            make_function(
                node.domain,
                node.op_type,
                inputs=[j.name for j in func.graph.input],
                outputs=[j.name for j in func.graph.output],
                nodes=func.graph.node,
                opset_imports=[
                    make_operatorsetid(node.domain, 1),
                    make_operatorsetid("", graph.opset_version),
                ],
            )
        ]
        for f in ext_functions:
            graph.onnx_add_function(f)
        for v in func.graph.initializer:
            if v.name not in graph.initializers:
                graph.initializer.append(v)


@PASSES.register("compose_nodes_to_functions", deps=["initializer_unique"])
class ComposeNodesToFunctionsRewriter(Rewriter):
    """Compose individual nodes into local functions.

    The composing nodes is searching based on connectivity and stops on
    another local function node.

    Before:

        conv -> relu -> Func -> slice -> conv

    After:

        Func(conv, relu) -> Func(...) -> Func(slice, conv)
    """

    def __init__(self, name: str = "Component", order: Literal["pre", "post"] = "post"):
        pattern = SingleNodePattern().with_domain(HYPER_DOMAIN).with_order(order)
        super().__init__(pattern=pattern)
        self.register_pre_hook(self._mark_isolated_nodes)
        self._name = name
        self._max_level = 0
        self._level_processed: Set[int] = set()

    def _mark_isolated_nodes(self, graph: OnnxGraph) -> OnnxGraph:
        # make a shallow copy of the graph
        h: nx.DiGraph = graph.copy().to_directed()
        for n in graph:
            if graph.nodes[n]["pb"].op_type in graph.functions:
                h.remove_node(n)
        # remove all functions in the graph and find isolated components
        for k, c in enumerate(nx.weakly_connected_components(h)):
            for i in c:
                # assign each component a different level `k`
                graph.nodes[i]["component"] = k
            self._max_level = max(self._max_level, k + 1)
        return graph

    def _detect_cycles_in_graph(self, graph: OnnxGraph) -> List[List[str]]:
        """Comprehensive cycle detection in the entire graph."""
        try:
            cycles = list(nx.simple_cycles(graph))
            return cycles
        except Exception:
            return []

    def _would_create_cycle(
        self, graph: OnnxGraph, new_node: NodeProto, nodes_to_remove: List[NodeProto]
    ) -> bool:
        """Check if adding a new node and removing old nodes would create a cycle."""
        # Create a temporary graph by copying the original and removing nodes
        temp_graph: OnnxGraph = cast(OnnxGraph, graph.copy(onnx_copy=True))

        # Remove nodes that will be replaced
        remove_names = {node.name for node in nodes_to_remove}
        for node_name in remove_names:
            temp_graph.remove_node(node_name)

        # Add the new composed node
        temp_graph.add_onnx_node(new_node)

        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(temp_graph))
            return len(cycles) > 0
        except nx.NetworkXError as e:
            error(f"NetworkXError encountered during cycle detection: {e}")
            return True  # Assume cycle exists for safety
        except nx.NetworkXUnfeasible as e:
            error(f"NetworkXUnfeasible error encountered: {e}")
            return True  # Assume cycle exists for safety
        except Exception as e:
            error(f"Unexpected error during cycle detection: {e}")
            return True  # Assume cycle exists for safety

    def _compose_subgraph(
        self, graph: OnnxGraph, same_level_nodes: List[NodeProto], level: int
    ) -> None:
        if len(same_level_nodes) == 0:
            return  # skip empty subgraph

        # Check 1: Create subgraph and detect cycles
        try:
            subgraph = graph.onnx_subgraph(same_level_nodes)
            cycles = self._detect_cycles_in_graph(subgraph)
            if cycles:
                warning(f"Cycles detected in same level nodes: {cycles}")
                return
        except Exception as e:
            node_names_list = [n.name for n in same_level_nodes]
            warning(f"Failed to check cycles for nodes {node_names_list}: {e}")
            return

        # Below logic is to break the same level into multiple functions.
        while level in self._level_processed:
            level += self._max_level
        self._level_processed.add(level)

        # turn initializers into constants
        for j in same_level_nodes:
            for v in [v for v in j.input if v in graph.initializers]:
                cst = make_node("Constant", [], [v], value=graph.initializers[v])
                subgraph.add_onnx_node(cst)

        composed_node = make_node(
            f"{self._name}:{level}",
            inputs=list(subgraph.inputs),
            outputs=list(subgraph.outputs),
            domain=HYPER_DOMAIN,
        )

        # Check if the new composed node would create cycles in the main graph
        if self._would_create_cycle(graph, composed_node, same_level_nodes):
            warning(
                f"Composing nodes {[n.name for n in same_level_nodes]} "
                f"would create cycles in the main graph"
            )
            return

        # If all checks pass, proceed with the composition
        try:
            graph.onnx_add_function(
                self.make_function_from_graph(
                    subgraph,
                    domain=HYPER_DOMAIN,
                    type_name=f"{self._name}:{level}",
                )
            )
            self += composed_node
            self -= same_level_nodes

        except Exception as e:
            error(f"Failed to add composed function: {e}")
            return

    def _compose_neighbors(
        self,
        graph: OnnxGraph,
        node: NodeProto,
        direction: Literal["upstream", "downstream"],
    ):
        """Compose neighboring nodes in specified direction (upstream or downstream)."""
        # Get initial neighbors based on direction
        if direction == "upstream":
            initial_neighbors = graph.onnx_predecessors(node)
            get_neighbors = graph.predecessors
        else:  # downstream
            initial_neighbors = graph.onnx_successors(node)
            get_neighbors = graph.successors

        for i in initial_neighbors:
            component_i = graph.nodes[i.name].pop("component", None)
            if component_i is None:
                continue  # i is function or been processed
            if direction == "upstream" and i.op_type == "Constant":
                continue

            same_level_nodes: List[NodeProto] = [i]
            # Use BFS to collect neighbors to avoid recursive dependency issues
            visited = {i.name}
            queue = [i.name]
            while queue:
                current = queue.pop(0)
                for j in get_neighbors(current):
                    if (
                        j not in visited
                        and graph.nodes[j].get("component") == component_i
                    ):
                        # For upstream, skip constants; for downstream, include all
                        if (
                            direction == "upstream"
                            and graph.nodes[j]["pb"].op_type == "Constant"
                        ):
                            continue
                        visited.add(j)
                        queue.append(j)
                        graph.nodes[j].pop("component")
                        same_level_nodes.append(graph.nodes[j]["pb"])
            self._compose_subgraph(graph, same_level_nodes, component_i)

    def _compose_upstreams(self, graph: OnnxGraph, node: NodeProto):
        """Compose upstream nodes into functions."""
        self._compose_neighbors(graph, node, "upstream")

    def _compose_downstreams(self, graph: OnnxGraph, node: NodeProto):
        """Compose downstream nodes into functions."""
        self._compose_neighbors(graph, node, "downstream")

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]

        # Check for cycles before starting composition
        try:
            initial_cycles = list(nx.simple_cycles(graph))
            if initial_cycles:
                warning(f"Initial cycles detected in graph: {initial_cycles}")
        except nx.NetworkXError as e:
            warning(f"Error during initial cycle detection: {e}")

        self._compose_upstreams(graph, node)

        # Check for cycles after upstream composition
        try:
            mid_cycles = list(nx.simple_cycles(graph))
            if mid_cycles:
                warning(f"Cycles detected after upstream composition: {mid_cycles}")
        except (nx.NetworkXError, nx.NetworkXUnfeasible) as e:
            warning(f"Error during mid-cycle detection: {e}")

        self._compose_downstreams(graph, node)

        # Final cycle check
        final_cycles = None
        try:
            final_cycles = list(nx.simple_cycles(graph))
            if final_cycles:
                error(f"Final cycles detected in graph: {final_cycles}")
                # Optionally, you could raise an exception here
                # raise RuntimeError(f"Cycle formation detected: {final_cycles}")
        except (nx.NetworkXError, nx.NetworkXUnfeasible) as e:
            error(f"Error during final cycle detection: {e}")


@PASSES.register("fuse_constants_to_function", deps=["initializer_to_constant"])
class FuseConstantsToFunctionRewriter(Rewriter):
    r"""Fuse constant inputs into the function node.

    If the function has more than two users, this pass will split that function
    into copies for each user.

    Before:

        const1  const2
           \      /
          func(node0)

    After:

        func(const1, const2, node0)
    """

    def __init__(self):
        super().__init__(SingleNodePattern().with_domain(HYPER_DOMAIN))
        self.users: Dict[str, int] = defaultdict(int)
        self.register_pre_hook(self._collect_users)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        func = graph.functions.get(node.op_type)
        if func is None:
            return
        if self.users[node.op_type] > 1:
            # more than 1 users, make a copy of the function
            self.users[node.op_type] -= 1
            func = deepcopy(func)
            func.name += f"({node.name})"
            graph.onnx_add_function(func)
            node.op_type = func.name
            self.users[func.name] += 1

        consts: List[Tuple[int, NodeProto]] = []
        for i, cst in enumerate(self.get_input_nodes(node)):
            if cst is not None and cst.op_type == "Constant":
                consts.append((i, cst))
        for i, cst in reversed(consts):
            cst = deepcopy(cst)
            cst.output[0] = func.input.pop(i)
            node.input.pop(i)
            func.node.insert(0, cst)  # prepend

        self -= [i[-1] for i in consts]

    def _collect_users(self, graph: OnnxGraph) -> OnnxGraph:
        for node in graph:
            func_type = graph.nodes[node]["pb"].op_type
            if func_type in graph.functions:
                self.users[func_type] += 1
        return graph


@PASSES.register("erase_output_types")
class EraseOutputTypesRewriter(Rewriter):
    """Erase output types from the graph."""

    def __init__(self):
        # No need to match any node
        super().__init__(pattern=SingleNodePattern().with_name("__NO_MATCH__"))
        self.register_pre_hook(self._clear_types)
        self.register_post_hook(self._infer_types)
        self.output_types: List[onnx.TypeProto] = []

    def _clear_types(self, graph: OnnxGraph):
        for i in graph.output:
            self.output_types.append(i.type)
            i.ClearField("type")
        return graph

    def _infer_types(self, graph: OnnxGraph):
        with suppress(Exception):
            graph = infer_shape(graph)
        # infer_shape may fail if the graph is not well-formed, restore
        # the old value then.
        for i, old in zip(graph.output, self.output_types):
            if not i.HasField("type"):
                i.type.CopyFrom(old)
            elif not i.type.tensor_type.HasField("shape"):
                i.type.tensor_type.shape.CopyFrom(old.tensor_type.shape)
        return graph

    def rewrite(self, graph, nodes):
        return graph
