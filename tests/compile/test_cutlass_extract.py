"""Tests for GPU detection."""


def test_detect_gpu_arch_format():
    """detect_gpu_arch returns sm_XX format or raises if no GPU."""
    from hyperonnx.compile.cutlass_kernels.extract import detect_gpu_arch

    try:
        arch = detect_gpu_arch()
        assert arch.startswith("sm_")
        assert len(arch) >= 4
    except RuntimeError as e:
        assert "No CUDA device" in str(e)
