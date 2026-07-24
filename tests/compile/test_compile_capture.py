"""Unit tests for the capture context manager (no CUDA required).

These tests simulate what triton.compiler.compile does: invoke the listener
with stub metadata + metadata_group. Integration with real triton kernels
is verified in the Tier 2 integration tests.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hyperonnx.compile.capture import (
    CaptureSink,
    _extract_target_from_metadata,
    attach_grid_exprs,
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
    assert sink.grid_constants == {}


def test_listener_records_kernel(tmp_path: Path):
    metadata_group = _write_stub_cubin(tmp_path)
    metadata = _stub_metadata()

    with capture_compiled_kernels() as sink:
        # Simulate triton calling the listener
        triton.knobs.compilation.listener(
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
    orig = triton.knobs.compilation.listener
    with capture_compiled_kernels():
        pass
    assert triton.knobs.compilation.listener is orig


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


# ---- attach_grid_exprs() coverage -------------------------------------------


def _wrapper_graph_with(grid_type: str, kernel: str = "k0") -> list[dict]:
    return [
        {
            "graph": "",
            "steps": [
                {"type": "triton_kernel", "kernel": kernel, "grid_type": grid_type}
            ],
        }
    ]


def _sink_with_kernel(symbol: str = "k0", cfg: dict | None = None) -> CaptureSink:
    sink = CaptureSink()
    sink.kernels.append({"symbol": symbol, "launch": {"grid_expr": None}})
    if cfg is not None:
        sink.grid_constants[symbol] = cfg
    return sink


def test_attach_grid_exprs_grid1d():
    sink = _sink_with_kernel(cfg={"XBLOCK": 512})
    attach_grid_exprs(sink, _wrapper_graph_with("Grid1D"))
    ast = sink.kernels[0]["launch"]["grid_expr"]
    assert ast is not None
    assert ast[0] == {
        "op": "cdiv",
        "a": {"op": "meta", "key": "xnumel"},
        "b": {"op": "const", "value": 512},
    }
    assert ast[1] == {"op": "const", "value": 1}
    assert ast[2] == {"op": "const", "value": 1}


def test_attach_grid_exprs_grid2d():
    sink = _sink_with_kernel(cfg={"XBLOCK": 16, "YBLOCK": 32})
    attach_grid_exprs(sink, _wrapper_graph_with("Grid2D"))
    ast = sink.kernels[0]["launch"]["grid_expr"]
    assert ast is not None
    assert ast[0]["op"] == "cdiv"
    assert ast[0]["a"] == {"op": "meta", "key": "xnumel"}
    assert ast[0]["b"] == {"op": "const", "value": 16}
    assert ast[1]["a"] == {"op": "meta", "key": "ynumel"}
    assert ast[1]["b"] == {"op": "const", "value": 32}


def test_attach_grid_exprs_no_config_stays_null():
    """No listener-captured config for the symbol → grid_expr stays None."""
    sink = _sink_with_kernel(cfg=None)
    attach_grid_exprs(sink, _wrapper_graph_with("Grid1D"))
    assert sink.kernels[0]["launch"]["grid_expr"] is None


def test_attach_grid_exprs_unknown_grid_type_stays_null():
    sink = _sink_with_kernel(cfg={"XBLOCK": 512})
    attach_grid_exprs(sink, _wrapper_graph_with("NoSuchGrid"))
    assert sink.kernels[0]["launch"]["grid_expr"] is None


def test_attach_grid_exprs_prefix_grid_stays_null():
    """Grid types with prefix assignments (overflow guards) stay null."""
    sink = _sink_with_kernel(cfg={"XBLOCK": 512, "YBLOCK": 512})
    attach_grid_exprs(sink, _wrapper_graph_with("Grid2DWithYZOverflow"))
    assert sink.kernels[0]["launch"]["grid_expr"] is None


def test_listener_callback_swallows_record_exceptions(tmp_path: Path):
    """If record_from_listener raises inside _listener, the exception is
    caught (logged) and doesn't propagate to the triton caller."""
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
        triton.knobs.compilation.listener(
            src=_StubSrc(),
            metadata=metadata,
            metadata_group=metadata_group,
            times={},
            cache_hit=False,
        )
