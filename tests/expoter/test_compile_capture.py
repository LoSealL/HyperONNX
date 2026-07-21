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
    _binary_ext_for_target,
    _extract_target_from_metadata,
    _target_to_dict,
    capture_compiled_kernels,
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


def test_listener_records_kernel(tmp_path: Path):
    import triton.knobs as knobs

    metadata_group = _write_stub_cubin(tmp_path)
    metadata = _stub_metadata()

    with capture_compiled_kernels() as sink:
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
    assert entry["device_target"]["backend"] == "cuda"


def test_listener_restores_original():
    import triton.knobs as knobs

    orig = knobs.compilation.listener
    with capture_compiled_kernels():
        pass
    assert knobs.compilation.listener is orig


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


def test_extract_target_from_metadata_returns_dict_as_is():
    """Dict target is returned as-is (now that _target_to_dict duck-types)."""
    target_dict = {"backend": "cuda", "arch": "sm_90", "warp_size": 32}
    metadata = {"target": target_dict}
    assert _extract_target_from_metadata(metadata) is target_dict


def test_extract_target_from_metadata_passes_through_object():
    """Non-dict target is returned as-is (namedtuple / object with attrs)."""
    original = SimpleNamespace(backend="hip", arch="gfx90a", warp_size=64)
    metadata = {"target": original}
    assert _extract_target_from_metadata(metadata) is original


def test_extract_target_from_metadata_none_when_absent():
    """Missing 'target' key returns None."""
    assert _extract_target_from_metadata({}) is None


def test_target_to_dict_with_partial_object():
    """_target_to_dict uses getattr defaults for missing attrs."""
    result = _target_to_dict(SimpleNamespace())  # no attrs
    assert result == {"backend": "cuda", "arch": "sm_70", "warp_size": 32}


def test_target_to_dict_with_dict_input():
    """_target_to_dict duck-types dict input (from listener metadata)."""
    result = _target_to_dict({"backend": "hip", "arch": "gfx90a", "warp_size": 64})
    assert result == {"backend": "hip", "arch": "gfx90a", "warp_size": 64}


def test_target_to_dict_with_partial_dict():
    """_target_to_dict uses .get defaults for missing keys."""
    result = _target_to_dict({})
    assert result == {"backend": "cuda", "arch": "sm_70", "warp_size": 32}


# ---- capture_compiled_kernels() listener error path -----------------------


def test_listener_callback_swallows_record_exceptions(tmp_path: Path):
    """If record_from_listener raises inside _listener, the exception is
    caught (logged) and doesn't propagate to the triton caller."""
    from unittest.mock import patch

    import triton.knobs as knobs

    metadata_group = _write_stub_cubin(tmp_path)
    metadata = _stub_metadata()

    with (
        capture_compiled_kernels(),
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
