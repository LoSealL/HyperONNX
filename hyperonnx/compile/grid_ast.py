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

import ast as pyast
from collections.abc import Mapping


class NotTranslatable(Exception):
    """Raised when a grid expression cannot be translated to the v1 AST."""


# Maps ``ast.BinOp`` operator node types to the v1 AST ``op`` name, or
# ``None`` when the operator is recognised-but-unsupported (used to keep
# Add in the dict so it produces a precise "not in v1 node set" error
# instead of the generic "unsupported BinOp" path). Add a new mapping to
# extend the v1 op set; ``NotTranslatable`` is raised for anything missing.
_ALLOWED_BINOPS = {
    pyast.Add: None,
    pyast.Mult: "mul",
    pyast.FloorDiv: "floordiv",
}


def _translate_expr(node: pyast.AST) -> dict:
    """Translate one Python AST node into a v1 grid-AST dict.

    Walks ``Constant`` / ``Name`` / ``Call`` / ``BinOp`` / ``Subscript``
    only — anything else raises :class:`NotTranslatable` and the caller
    downgrades the whole kernel to ``grid_expr=None``. The accepted node
    set matches the runtime contract in the design doc's "Grid AST" table.
    """
    if isinstance(node, pyast.Constant) and isinstance(node.value, int):
        return {"op": "const", "value": node.value}
    if isinstance(node, pyast.Name):
        return {"op": "meta", "key": node.id}
    if isinstance(node, pyast.Call):
        if not isinstance(node.func, pyast.Name):
            raise NotTranslatable(f"call target not a Name: {pyast.dump(node)}")
        fname = node.func.id
        if fname == "cdiv":
            args = [_translate_expr(a) for a in node.args]
            if len(args) != 2:
                raise NotTranslatable("cdiv needs 2 args")
            return {"op": "cdiv", "a": args[0], "b": args[1]}
        raise NotTranslatable(f"unknown call: {fname}")
    if isinstance(node, pyast.BinOp):
        op_kind = type(node.op)
        if op_kind not in _ALLOWED_BINOPS:
            raise NotTranslatable(f"unsupported BinOp: {op_kind.__name__}")
        op_name = _ALLOWED_BINOPS[op_kind]
        if op_name is None:
            raise NotTranslatable(f"binop {op_kind.__name__} not in v1 node set")
        return {
            "op": op_name,
            "a": _translate_expr(node.left),
            "b": _translate_expr(node.right),
        }
    if isinstance(node, pyast.UnaryOp) and isinstance(node.op, pyast.USub):
        inner = node.operand
        # Inductor's python-mode ceildiv trick: -((a) // -(b)) == cdiv(a, b).
        if (
            isinstance(inner, pyast.BinOp)
            and isinstance(inner.op, pyast.FloorDiv)
            and isinstance(inner.right, pyast.UnaryOp)
            and isinstance(inner.right.op, pyast.USub)
        ):
            return {
                "op": "cdiv",
                "a": _translate_expr(inner.left),
                "b": _translate_expr(inner.right.operand),
            }
        if isinstance(inner, pyast.Constant) and isinstance(inner.value, int):
            return {"op": "const", "value": -inner.value}
        raise NotTranslatable(f"unsupported unary: {pyast.dump(node)}")
    if isinstance(node, pyast.Subscript):
        if isinstance(node.value, pyast.Attribute) and node.value.attr in (
            "shape",
            "size",
        ):
            base = node.value.value
            if not isinstance(base, pyast.Name):
                raise NotTranslatable("shape base not Name")
            idx = node.slice
            if isinstance(idx, pyast.Constant) and isinstance(idx.value, int):
                return {"op": "shape_dim", "input": base.id, "axis": idx.value}
        raise NotTranslatable(f"unsupported subscript: {pyast.dump(node)}")
    raise NotTranslatable(f"unsupported expr: {type(node).__name__}")


def translate_grid(source: str) -> list[dict] | None:
    """Translate the body of a grid lambda into the v1 AST.

    Args:
        source: Python source containing a `return (expr, ...)` statement.

    Returns:
        A list of AST node dicts, one per grid dim, or None on parse error.
        Raises NotTranslatable if a node is outside the v1 set.
    """
    try:
        tree = pyast.parse(source)
    except SyntaxError:
        return None
    returns = [n for n in pyast.walk(tree) if isinstance(n, pyast.Return)]
    if len(returns) != 1:
        return None
    value = returns[0].value
    if value is None:
        return None
    if isinstance(value, pyast.Tuple):
        elems = value.elts
    else:
        elems = [value]
    return [_translate_expr(e) for e in elems]


def evaluate_grid(
    ast_nodes: list[dict],
    io_shapes: Mapping[str, list[int | str]],
    meta: Mapping[str, int],
) -> list[int]:
    """Evaluate a translated AST against concrete shapes and meta values."""

    def _ev(node: dict) -> int:
        op = node["op"]
        if op == "const":
            return int(node["value"])
        if op == "meta":
            return int(meta[node["key"]])
        if op == "shape_dim":
            return int(io_shapes[node["input"]][node["axis"]])
        if op == "mul":
            return _ev(node["a"]) * _ev(node["b"])
        if op == "floordiv":
            return _ev(node["a"]) // _ev(node["b"])
        if op == "cdiv":
            a = _ev(node["a"])
            b = _ev(node["b"])
            return (a + b - 1) // b
        raise ValueError(f"unknown op: {op}")

    return [_ev(n) for n in ast_nodes]
