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
