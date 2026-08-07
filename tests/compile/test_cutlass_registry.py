"""Tests for the CUTLASS kernel tuner registry."""


def test_get_tuner_mm():
    from hyperonnx.compile.cutlass_kernels import get_tuner

    tuner = get_tuner("extern_kernels.mm")
    assert callable(tuner)


def test_get_tuner_conv():
    from hyperonnx.compile.cutlass_kernels import get_tuner

    tuner = get_tuner("extern_kernels.convolution")
    assert callable(tuner)


def test_get_tuner_unknown():
    from hyperonnx.compile.cutlass_kernels import get_tuner

    tuner = get_tuner("extern_kernels.some_new_op")
    assert tuner is None


def test_require_cutlass_raises_on_missing():
    from hyperonnx.compile.cutlass_kernels import require_cutlass

    try:
        cute = require_cutlass()
        assert cute is not None
    except RuntimeError as e:
        assert "CUTLASS CuTe DSL required" in str(e)
