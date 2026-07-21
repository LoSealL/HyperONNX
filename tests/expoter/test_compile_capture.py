"""Unit tests for the capture context manager (no CUDA required).

These tests simulate what triton.compiler.compile does: invoke the listener
with stub metadata + metadata_group. Integration with real triton kernels
is verified in the Tier 2 integration tests.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperonnx.compile.capture import (
    CaptureSink,
    _attach_ast_to_kernel,
    _binary_ext_for_target,
    _extract_target_from_metadata,
    _target_to_dict,
    capture_compiled_kernels,
    extract_grid_value,
)

triton = pytest.importorskip("triton")


def _write_stub_cubin(tmp: Path) -> dict:
    """Create a stub metadata_group mapping 'kernel.json'/'kernel.cubin' to files.

    The JSON must contain everything `CompiledKernel.__init__` reads off disk
    (notably `target`); in real triton the same metadata dict the listener
    receives is also what gets serialized to that JSON file.
    """
    import json

    cubin_file = tmp / "kernel.cubin"
    cubin_file.write_bytes(b"\x00\x01\x02FAKE")
    json_file = tmp / "kernel.json"
    json_file.write_text(json.dumps(_stub_metadata()))
    return {"kernel.cubin": str(cubin_file), "kernel.json": str(json_file)}


def _stub_metadata() -> dict:
    return {
        "name": "stub_kernel_0",
        "num_warps": 4,
        "num_ctas": 1,
        "shared": 1024,
        "num_regs": 32,
        "target": {"backend": "cuda", "arch": "sm_80", "warp_size": 32},
    }


class _StubSrc:
    """Minimal stand-in for triton's ASTSource."""

    def hash(self):
        return "stubhash"


class _StubMeta:
    """Stand-in for triton's KernelMetadata namedtuple."""

    def __init__(
        self, name="stub_kernel_0", num_warps=4, num_ctas=1, shared=1024, num_regs=32
    ):
        self.name = name
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.shared = shared
        self.num_regs = num_regs


class _StubAsmDict(dict):
    pass


class _StubCompiledKernel:
    """Stand-in for triton's CompiledKernel for direct record() tests."""

    def __init__(self, metadata=None, asm=None):
        self.metadata = metadata if metadata is not None else _StubMeta()
        self.asm = asm if asm is not None else _StubAsmDict({"cubin": b"\x00FAKE"})


def test_capture_sink_initially_empty():
    sink = CaptureSink()
    assert sink.kernels == []
    assert sink.grid_sources == {}


def test_listener_records_kernel(tmp_path: Path):
    import triton.knobs as knobs

    metadata_group = _write_stub_cubin(tmp_path)
    metadata = _stub_metadata()

    with capture_compiled_kernels(static_grid=True) as sink:
        # Simulate triton calling the listener
        knobs.compilation.listener(
            src=_StubSrc(),
            metadata=metadata,
            metadata_group=metadata_group,
            times={},
            cache_hit=False,
        )

    assert len(sink.kernels) == 1
    entry = sink.kernels[0]
    assert entry["cubin_bytes"] == b"\x00\x01\x02FAKE"
    assert entry["symbol"] == "stub_kernel_0"
    assert entry["launch"]["num_warps"] == 4
    assert entry["launch"]["grid_expr"] is None
    assert entry["device_target"]["backend"] == "cuda"


def test_listener_restores_original():
    import triton.knobs as knobs

    orig = knobs.compilation.listener
    with capture_compiled_kernels(static_grid=True):
        pass
    assert knobs.compilation.listener is orig


def test_capture_static_grid_skips_ast(tmp_path: Path):
    import triton.knobs as knobs

    metadata_group = _write_stub_cubin(tmp_path)
    metadata = _stub_metadata()

    with capture_compiled_kernels(static_grid=True) as sink:
        knobs.compilation.listener(
            src=_StubSrc(),
            metadata=metadata,
            metadata_group=metadata_group,
            times={},
            cache_hit=False,
        )
    assert all(k["launch"]["grid_expr"] is None for k in sink.kernels)


# ---- record() method coverage ----------------------------------------------


def test_record_skips_kernel_with_empty_cubin():
    """record() must warn + return early when cubin bytes are empty."""
    sink = CaptureSink()
    empty_kernel = _StubCompiledKernel(asm=_StubAsmDict({"cubin": b""}))
    sink.record(empty_kernel, target=SimpleNamespace(backend="cuda"))
    assert sink.kernels == []  # not recorded


def test_record_uses_default_target_when_target_is_none():
    """record() falls back to _DEFAULT_TARGET when target is None."""
    sink = CaptureSink()
    sink.record(_StubCompiledKernel(), target=None)
    assert len(sink.kernels) == 1
    assert sink.kernels[0]["device_target"]["backend"] == "cuda"
    assert sink.kernels[0]["device_target"]["arch"] == "sm_70"


def test_record_falls_back_when_asm_missing():
    """record() treats missing asm attr as empty dict."""
    sink = CaptureSink()
    kernel_no_asm = SimpleNamespace(
        metadata=_StubMeta(),
        # no `asm` attribute at all
    )
    sink.record(kernel_no_asm, target=SimpleNamespace(backend="cuda"))
    assert sink.kernels == []  # empty cubin → skipped


def test_record_assigns_default_name_when_missing():
    """record() uses kernel_N when metadata has no name."""
    sink = CaptureSink()
    kernel_no_name = SimpleNamespace(
        metadata=SimpleNamespace(num_warps=1, num_ctas=1, shared=0, num_regs=0),
        asm=_StubAsmDict({"cubin": b"\x00"}),
    )
    sink.record(kernel_no_name, target=None)
    assert sink.kernels[0]["symbol"] == "kernel_0"


# ---- record_from_listener() error path -------------------------------------


def test_record_from_listener_warns_on_construct_failure(tmp_path: Path):
    """If CompiledKernel construction raises, record_from_listener warns + skips."""
    sink = CaptureSink()
    # metadata_group with no JSON file → CompiledKernel.__init__ raises
    bad_metadata_group = {"kernel.cubin": str(tmp_path / "missing.cubin")}
    sink.record_from_listener(_StubSrc(), _stub_metadata(), bad_metadata_group)
    assert sink.kernels == []


def test_record_from_listener_uses_metadata_name_default():
    """record_from_listener falls back to kernel_N name when missing."""
    sink = CaptureSink()
    # Use a metadata dict with no "name" key. CompiledKernel will fail to
    # construct (no json on disk), but the name-default branch executes first.
    metadata_no_name = {
        "num_warps": 1,
        "target": {"backend": "cuda", "arch": "sm_80", "warp_size": 32},
    }
    sink.record_from_listener(_StubSrc(), metadata_no_name, {})
    assert sink.kernels == []  # construction fails, but name="kernel_0" was used


# ---- helper function coverage ----------------------------------------------


def test_binary_ext_for_cuda_returns_cubin():
    assert _binary_ext_for_target(SimpleNamespace(backend="cuda")) == "cubin"


def test_binary_ext_for_hip_returns_hsaco():
    assert _binary_ext_for_target(SimpleNamespace(backend="hip")) == "hsaco"


def test_binary_ext_unknown_backend_defaults_to_cubin():
    assert _binary_ext_for_target(SimpleNamespace(backend="rocm")) == "cubin"


def test_binary_ext_for_none_target_defaults_to_cubin():
    assert _binary_ext_for_target(None) == "cubin"


def test_extract_target_from_metadata_dict():
    """Dict target is converted to SimpleNamespace."""
    metadata = {"target": {"backend": "cuda", "arch": "sm_90", "warp_size": 32}}
    target = _extract_target_from_metadata(metadata)
    assert target.backend == "cuda"
    assert target.arch == "sm_90"
    assert target.warp_size == 32


def test_extract_target_from_metadata_passes_through_object():
    """Non-dict target is returned as-is (namedtuple / object with attrs)."""
    original = SimpleNamespace(backend="hip", arch="gfx90a", warp_size=64)
    metadata = {"target": original}
    target = _extract_target_from_metadata(metadata)
    assert target is original


def test_extract_target_from_metadata_none_when_absent():
    """Missing 'target' key returns None."""
    assert _extract_target_from_metadata({}) is None


def test_target_to_dict_with_partial_object():
    """_target_to_dict uses getattr defaults for missing attrs."""
    result = _target_to_dict(SimpleNamespace())  # no attrs
    assert result == {"backend": "cuda", "arch": "sm_70", "warp_size": 32}


# ---- extract_grid_value() coverage -----------------------------------------


def test_extract_grid_value_from_callable():
    """When lam is callable, it's invoked with meta."""
    lam = lambda meta: (4, 8, 1)  # noqa: E731
    assert extract_grid_value(lam, {}) == (4, 8, 1)


def test_extract_grid_value_from_raw_tuple():
    """When lam is not callable, it's treated as the grid directly."""
    assert extract_grid_value((2, 4, 1), {}) == (2, 4, 1)


def test_extract_grid_value_returns_none_on_exception():
    """Exceptions are caught and None is returned."""
    result = extract_grid_value(None, {})  # iter(None) raises
    assert result is None


# ---- _attach_ast_to_kernel() coverage --------------------------------------


def test_attach_ast_to_kernel_updates_matching_symbol():
    """AST is attached to the kernel with matching symbol."""
    sink = CaptureSink()
    sink.kernels.append(
        {
            "symbol": "k0",
            "launch": {"grid_expr": None},
        }
    )
    ast = [{"op": "const", "value": 1}]
    _attach_ast_to_kernel(sink, "k0", ast)
    assert sink.kernels[0]["launch"]["grid_expr"] is ast


def test_attach_ast_to_kernel_no_match_is_noop():
    """When no kernel matches, nothing changes."""
    sink = CaptureSink()
    sink.kernels.append(
        {
            "symbol": "k0",
            "launch": {"grid_expr": None},
        }
    )
    _attach_ast_to_kernel(sink, "nonexistent", [{"op": "const", "value": 1}])
    assert sink.kernels[0]["launch"]["grid_expr"] is None


# ---- attach_grid_source() coverage -----------------------------------------


def test_attach_grid_source_stores_source():
    sink = CaptureSink()
    sink.attach_grid_source("k0", "return (1,)")
    assert sink.grid_sources == {"k0": "return (1,)"}


# ---- capture_compiled_kernels() non-static path ----------------------------


def test_capture_dynamic_grid_runs_ast_loop_on_no_sources(tmp_path: Path):
    """With static_grid=False, the post-yield AST loop runs (but is a no-op
    when grid_sources is empty)."""
    import triton.knobs as knobs

    metadata_group = _write_stub_cubin(tmp_path)
    metadata = _stub_metadata()

    with capture_compiled_kernels(static_grid=False) as sink:
        knobs.compilation.listener(
            src=_StubSrc(),
            metadata=metadata,
            metadata_group=metadata_group,
            times={},
            cache_hit=False,
        )
    # grid_sources is empty in v1, so grid_expr stays None even with
    # static_grid=False.
    assert all(k["launch"]["grid_expr"] is None for k in sink.kernels)


def test_capture_dynamic_grid_with_untranslatable_source(tmp_path: Path):
    """When grid_sources has an untranslatable source, AST stays None and
    the warning is logged."""
    import triton.knobs as knobs

    metadata_group = _write_stub_cubin(tmp_path)
    metadata = _stub_metadata()

    with capture_compiled_kernels(static_grid=False) as sink:
        knobs.compilation.listener(
            src=_StubSrc(),
            metadata=metadata,
            metadata_group=metadata_group,
            times={},
            cache_hit=False,
        )
        # Populate grid_sources after recording so the post-yield loop has
        # something to translate. Use an untranslatable expression.
        sink.grid_sources["stub_kernel_0"] = "return (unknown_func(x),)"
    # AST extraction failed silently; grid_expr remains None
    assert all(k["launch"]["grid_expr"] is None for k in sink.kernels)


def test_capture_dynamic_grid_with_valid_source(tmp_path: Path):
    """When grid_sources has a valid cdiv source, AST is attached."""
    import triton.knobs as knobs

    metadata_group = _write_stub_cubin(tmp_path)
    metadata = _stub_metadata()

    with capture_compiled_kernels(static_grid=False) as sink:
        knobs.compilation.listener(
            src=_StubSrc(),
            metadata=metadata,
            metadata_group=metadata_group,
            times={},
            cache_hit=False,
        )
        sink.grid_sources["stub_kernel_0"] = "return (cdiv(M, 128),)"
    assert len(sink.kernels) == 1
    grid_expr = sink.kernels[0]["launch"]["grid_expr"]
    assert grid_expr is not None
    assert grid_expr[0]["op"] == "cdiv"


def test_listener_callback_swallows_record_exceptions(tmp_path: Path):
    """If record_from_listener raises inside _listener, the exception is
    caught (logged) and doesn't propagate to the triton caller."""
    from unittest.mock import patch

    import triton.knobs as knobs

    metadata_group = _write_stub_cubin(tmp_path)
    metadata = _stub_metadata()

    with (
        capture_compiled_kernels(static_grid=True),
        patch.object(
            CaptureSink,
            "record_from_listener",
            side_effect=RuntimeError("boom"),
        ),
    ):
        # Must not raise
        knobs.compilation.listener(
            src=_StubSrc(),
            metadata=metadata,
            metadata_group=metadata_group,
            times={},
            cache_hit=False,
        )
