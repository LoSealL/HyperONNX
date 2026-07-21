"""Unit tests for grid AST translation and evaluation."""

import pytest

from hyperonnx.compile.grid_ast import (
    NotTranslatable,
    evaluate_grid,
    translate_grid,
)


def test_translate_cdiv_only():
    source = "return (cdiv(M, 128),)"
    ast = translate_grid(source)
    assert ast == [
        {
            "op": "cdiv",
            "a": {"op": "meta", "key": "M"},
            "b": {"op": "const", "value": 128},
        }
    ]


def test_translate_shape_subscript():
    source = "return (x.shape[0],)"
    ast = translate_grid(source)
    assert ast == [{"op": "shape_dim", "input": "x", "axis": 0}]


def test_translate_mul_chain():
    source = "return (x.shape[0] * x.shape[1],)"
    ast = translate_grid(source)
    assert ast == [
        {
            "op": "mul",
            "a": {"op": "shape_dim", "input": "x", "axis": 0},
            "b": {"op": "shape_dim", "input": "x", "axis": 1},
        }
    ]


def test_translate_unhandled_raises():
    source = "return (math.sin(x),)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_none_on_parse_error():
    assert translate_grid("not valid python ((") is None


# ---- additional translate_grid branches ------------------------------------


def test_translate_floordiv_binop():
    source = "return (M // 4,)"
    ast = translate_grid(source)
    assert ast == [
        {
            "op": "floordiv",
            "a": {"op": "meta", "key": "M"},
            "b": {"op": "const", "value": 4},
        }
    ]


def test_translate_add_binop_raises():
    """Add is registered as None (not in v1 node set) → NotTranslatable."""
    source = "return (M + 1,)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_unsupported_binop_raises():
    """BinOp ops not in _ALLOWED_BINOPS (e.g. Mod) raise NotTranslatable."""
    source = "return (M % 2,)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_call_with_wrong_argc_raises():
    """cdiv with != 2 args raises NotTranslatable."""
    source = "return (cdiv(M),)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_unknown_call_name_raises():
    """A Call to an unknown function name raises NotTranslatable."""
    source = "return (my_func(M),)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_call_with_attribute_func_raises():
    """A Call whose func is an Attribute (not Name) raises NotTranslatable.
    Covers both math.sin (unhandled) and x.size(N) forms — both parse as
    Call(func=Attribute(...))."""
    source = "return (math.floor(M),)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_subscript_with_non_name_base_raises():
    """A shape subscript whose base isn't a Name raises NotTranslatable."""
    # foo.bar.shape[0] — base is Attribute, not Name
    source = "return (foo.bar.shape[0],)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_subscript_with_non_int_idx_raises():
    """A shape subscript whose index isn't an int constant raises."""
    source = "return (x.shape[y],)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_unsupported_subscript_raises():
    """A subscript that isn't .shape / .size raises NotTranslatable."""
    source = "return (x[0],)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_unsupported_expr_type_raises():
    """A bare string constant (non-int) raises NotTranslatable."""
    source = "return ('hello',)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_none_when_no_return():
    """A function body with no return statement returns None."""
    assert translate_grid("x = 1") is None


def test_translate_none_when_bare_return():
    """A bare `return` (no value) returns None."""
    assert translate_grid("return") is None


def test_translate_none_when_multiple_returns():
    """Multiple return statements returns None (ambiguous)."""
    assert translate_grid("if True:\n  return 1\nelse:\n  return 2") is None


def test_translate_single_non_tuple_return():
    """A non-tuple return (`return M`) is wrapped as a single-element list."""
    ast = translate_grid("return M")
    assert ast == [{"op": "meta", "key": "M"}]


def test_translate_nested_binops():
    """`cdiv(M, 128) * cdiv(N, 64)` translates recursively."""
    source = "return (cdiv(M, 128) * cdiv(N, 64),)"
    ast = translate_grid(source)
    assert ast[0]["op"] == "mul"
    assert ast[0]["a"]["op"] == "cdiv"
    assert ast[0]["b"]["op"] == "cdiv"


# ---- evaluate_grid branches ------------------------------------------------


def test_evaluate_cdiv():
    ast = [
        {
            "op": "cdiv",
            "a": {"op": "meta", "key": "M"},
            "b": {"op": "const", "value": 128},
        }
    ]
    assert evaluate_grid(ast, io_shapes={}, meta={"M": 1000}) == [8]


def test_evaluate_shape_dim():
    ast = [{"op": "shape_dim", "input": "x", "axis": 1}]
    assert evaluate_grid(ast, io_shapes={"x": [4, 7, 3]}, meta={}) == [7]


def test_evaluate_mul():
    ast = [
        {
            "op": "mul",
            "a": {"op": "const", "value": 3},
            "b": {"op": "const", "value": 4},
        }
    ]
    assert evaluate_grid(ast, io_shapes={}, meta={}) == [12]


def test_evaluate_floordiv():
    ast = [
        {
            "op": "floordiv",
            "a": {"op": "const", "value": 17},
            "b": {"op": "const", "value": 5},
        }
    ]
    assert evaluate_grid(ast, io_shapes={}, meta={}) == [3]


def test_evaluate_multi_dim_grid():
    """A full 3-dim grid evaluates in order."""
    ast = [
        {"op": "const", "value": 1},
        {
            "op": "cdiv",
            "a": {"op": "shape_dim", "input": "x", "axis": 0},
            "b": {"op": "const", "value": 4},
        },
        {"op": "const", "value": 1},
    ]
    assert evaluate_grid(ast, io_shapes={"x": [16, 8]}, meta={}) == [1, 4, 1]


def test_evaluate_unknown_op_raises_value_error():
    """An unknown op in the AST raises ValueError."""
    ast = [{"op": "bogus"}]
    with pytest.raises(ValueError):
        evaluate_grid(ast, io_shapes={}, meta={})
