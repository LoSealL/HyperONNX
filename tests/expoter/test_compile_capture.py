"""Unit tests for the capture context manager (no CUDA required).

These tests simulate what triton.compiler.compile does: invoke the listener
with stub metadata + metadata_group. Integration with real triton kernels
is verified in the Tier 2 integration tests.
"""

from pathlib import Path

import pytest

from hyperonnx.compile.capture import CaptureSink, capture_compiled_kernels

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
