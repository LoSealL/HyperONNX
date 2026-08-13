"""Tests for CUTLASS GEMM shape extraction (mm, bmm, addmm)."""

import pytest

try:
    from hyperonnx.compile.cutlass_kernels.mm import _extract_matmul_shapes
except ImportError:
    pytest.skip("CUTLASS is not available", allow_module_level=True)


def _tensor_arg(shape, name="x", dtype="float16"):
    return {"kind": "tensor", "name": name, "shape": shape, "dtype": dtype}


def test_extract_mm_shapes():
    args = [_tensor_arg([8, 16]), _tensor_arg([16, 32])]
    M, N, K, _ = _extract_matmul_shapes(args, {})
    assert (M, N, K) == (8, 32, 16)


def test_extract_bmm_shapes():
    args = [_tensor_arg([2, 8, 16]), _tensor_arg([2, 16, 32])]
    M, N, K, _ = _extract_matmul_shapes(args, {})
    assert (M, N, K) == (8, 32, 16)


def test_extract_addmm_shapes_skips_1d_bias():
    # addmm(bias[32], mat1[8,16], mat2[16,32]) — bias is the 1D leading arg.
    args = [_tensor_arg([32]), _tensor_arg([8, 16]), _tensor_arg([16, 32])]
    M, N, K, _ = _extract_matmul_shapes(args, {})
    assert (M, N, K) == (8, 32, 16)
