"""Unit tests for the capture context manager (no CUDA required).

These tests monkey-patch triton.compiler.compile with a stub so the hook
logic can be tested without a real GPU. Integration with real triton
kernels is verified in the Tier 2 integration tests.
"""

from hyperonnx.compile.capture import CaptureSink, capture_compiled_kernels


class StubMetadata:
    def __init__(self):
        self.name = "stub_kernel_0"
        self.num_warps = 4
        self.num_ctas = 1
        self.shared = 1024
        self.num_regs = 32


class StubAsmDict(dict):
    pass


class StubCompiledKernel:
    def __init__(self):
        self.metadata = StubMetadata()
        self.asm = StubAsmDict({"cubin": b"\x00\x01\x02FAKE"})


def test_capture_sink_initially_empty():
    sink = CaptureSink()
    assert sink.kernels == []
    assert sink.grid_sources == {}


def test_capture_hook_spy_passes_through_and_records(monkeypatch):
    import triton.compiler as tc

    recorded = []
    fake_ck = StubCompiledKernel()

    def fake_compile(src, target=None, options=None, **kw):
        recorded.append(src)
        return fake_ck

    monkeypatch.setattr(tc, "compile", fake_compile)

    with capture_compiled_kernels(static_grid=True) as sink:
        result = tc.compile("kernel void foo() {}")
        assert result is fake_ck
        assert len(sink.kernels) == 1
        entry = sink.kernels[0]
        assert entry["cubin_bytes"] == b"\x00\x01\x02FAKE"
        assert entry["symbol"] == "stub_kernel_0"
        assert entry["launch"]["num_warps"] == 4
        assert entry["launch"]["grid_expr"] is None
        assert entry["device_target"]["backend"]  # filled from target or default

    assert recorded == ["kernel void foo() {}"]


def test_capture_hook_restores_original(monkeypatch):
    import triton.compiler as tc

    orig = tc.compile
    with capture_compiled_kernels(static_grid=True):
        pass
    assert tc.compile is orig


def test_capture_static_grid_skips_ast(monkeypatch):
    import triton.compiler as tc

    monkeypatch.setattr(tc, "compile", lambda *a, **k: StubCompiledKernel())
    with capture_compiled_kernels(static_grid=True) as sink:
        tc.compile("foo")
    assert all(k["launch"]["grid_expr"] is None for k in sink.kernels)
