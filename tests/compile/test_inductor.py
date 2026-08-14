"""Unit tests for the centralized triton/torch._inductor import layer."""

from types import SimpleNamespace

import pytest

from hyperonnx.compile.inductor import (
    compilation_knobs,
    compiled_kernel_cls,
    extract_grid_constants,
    launcher_constexpr_indices,
    triton_version,
    wrapper_codegen_cls,
)

# triton is an optional dependency (CPU-only torch builds have no triton).
# Skip the whole module when it is absent so the suite degrades gracefully.
pytest.importorskip("triton")


def test_compilation_knobs_exposes_listener():
    # triton.knobs.compilation is a knobs instance (with .listener), not a module.
    assert hasattr(compilation_knobs(), "listener")


def test_wrapper_codegen_cls_has_run_wrapper_ir_passes():
    assert hasattr(wrapper_codegen_cls(), "run_wrapper_ir_passes")


def test_launcher_constexpr_indices_prefers_full_constexprs():
    launcher = SimpleNamespace(
        arg_names=["a", "b", "c"],
        full_constexprs=[1],
        declared_constexprs=[1, 2],
    )
    assert launcher_constexpr_indices(launcher) == {1}


def test_launcher_constexpr_indices_falls_back_to_declared():
    launcher = SimpleNamespace(
        arg_names=["a", "b", "c"],
        declared_constexprs=[2],
    )
    assert launcher_constexpr_indices(launcher) == {2}


def test_launcher_constexpr_indices_empty_when_neither():
    launcher = SimpleNamespace(arg_names=["a", "b"])
    assert launcher_constexpr_indices(launcher) == set()


def test_extract_grid_constants_tuple_keys():
    src = SimpleNamespace(
        fn=SimpleNamespace(arg_names=["xnumel", "XBLOCK", "YBLOCK"]),
        constants={(1,): 512, (2,): 1},
    )
    assert extract_grid_constants(src) == {"XBLOCK": 512, "YBLOCK": 1}


def test_extract_grid_constants_str_keys_and_non_ints_dropped():
    src = SimpleNamespace(
        fn=SimpleNamespace(arg_names=["XBLOCK"]),
        constants={"XBLOCK": 256, "dtype": "fp32", "flag": True},
    )
    assert extract_grid_constants(src) == {"XBLOCK": 256}


def test_extract_grid_constants_no_constants():
    assert extract_grid_constants(SimpleNamespace()) == {}


def test_triton_accessors_raise_readable_error_when_triton_absent(monkeypatch):
    """When triton is not installed, accessors raise RuntimeError with a
    message that explains the cause and the fix — not a raw ModuleNotFoundError."""
    real_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def _block_triton(name, *args, **kwargs):
        if name.startswith("triton"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _block_triton)
    compiled_kernel_cls.cache_clear()
    compilation_knobs.cache_clear()
    triton_version.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="triton is not available"):
            compiled_kernel_cls()
        with pytest.raises(RuntimeError, match="triton is not available"):
            compilation_knobs()
        with pytest.raises(RuntimeError, match="triton is not available"):
            triton_version()
    finally:
        compiled_kernel_cls.cache_clear()
        compilation_knobs.cache_clear()
        triton_version.cache_clear()
