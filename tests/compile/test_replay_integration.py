"""Integration tests for the cubin capture + replay harness.

Requires a real CUDA device + triton. Skipped otherwise.

Two contracts tested:
- **Pure-triton modules** (pointwise/reduction/layernorm) are captured and
  replay bit-close to the torch.compile reference.
- **Vendor-delegated modules** (conv→cuDNN, matmul→cuBLAS) are partially
  captured: triton kernels dump normally, but buffers produced by vendor-lib
  ops are detected via write-direction coverage and marked in the manifest's
  ``vendor_lib`` key. See the cubin-replay design doc ("Vendor-library ops").
"""

import glob
import json
from pathlib import Path

import pytest
import torch

from hyperonnx import export_hyper_onnx
from hyperonnx.compile.testing import verify

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def _isolate_inductor_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    torch._dynamo.reset()
    yield


def _export(model, sample, module_type, tmp_path: Path) -> None:
    export_hyper_onnx(
        model,
        sample,
        str(tmp_path / "m.onnx"),
        compile_hier=[module_type],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )


def _manifests(tmp_path: Path) -> list[dict]:
    bundles = glob.glob(str(tmp_path / "*.kernels"))
    return [json.loads((Path(b) / "manifest.json").read_text()) for b in bundles]


def test_replay_pointwise(tmp_path: Path):
    """A triton-only (pointwise) module replays bit-close to the reference.

    Inductor lowers pure elementwise graphs fully to triton, so every
    producer is captured and no ``vendor_lib`` marker is written.
    """

    class _P(torch.nn.Module):
        def forward(self, x):
            return x * 2.0 + x

    model = _P().cuda().eval()
    sample = (torch.randn(1, 16, 32, 32, device="cuda"),)
    _export(model, sample, _P, tmp_path)
    manifests = _manifests(tmp_path)
    assert len(manifests) == 1
    assert "vendor_lib" not in manifests[0], "pointwise should be fully triton"

    expected = model(*sample).detach().cpu().numpy()
    inputs = [s.detach().cpu().numpy() for s in sample]
    ok = verify(
        glob.glob(str(tmp_path / "*.kernels"))[0],
        inputs,
        expected,
        atol=1e-4,
        rtol=1e-4,
    )
    assert ok, "replay output does not match torch.compile reference"

    # TTIR/TTGIR off by default; wrapper.py absent (pure triton, no gaps).
    bundle = Path(glob.glob(str(tmp_path / "*.kernels"))[0])
    assert not list(bundle.glob("*.ttir")), "TTIR should be off by default"
    assert not list(bundle.glob("*.ttgir")), "TTGIR should be off by default"
    assert not (bundle / "wrapper.py").exists(), "no wrapper for pure-triton"


def test_partial_capture_conv_has_vendor_lib_marker(tmp_path: Path):
    """Conv→cuDNN; triton kernels still dump, manifest marks the gap + op descriptor."""

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = _M().cuda().eval()
    sample = (torch.randn(1, 3, 32, 32, device="cuda"),)
    _export(model, sample, _M, tmp_path)
    manifests = _manifests(tmp_path)
    assert len(manifests) == 1, "bundle should still be written (partial capture)"
    man = manifests[0]
    vl = man.get("vendor_lib", {})
    assert vl.get("unwritten_buffers"), "conv graph should mark vendor-lib gap"

    ops = vl.get("ops", [])
    assert ops, "conv graph should record vendor-op descriptor"
    conv_op = ops[0]
    assert conv_op["type"] == "conv2d"
    assert conv_op["attrs"]["stride"] == [1, 1]
    assert conv_op["attrs"]["padding"] == [1, 1]
    assert conv_op["attrs"]["groups"] == 1
    # 3 operands: input, weight, bias
    assert len(conv_op["operands"]) == 3
    for op in conv_op["operands"]:
        assert op["buffer_id"] is not None, "every operand should link to a buffer"
        assert op["nbytes"] > 0, "every operand should have nbytes"
    # input [1,3,32,32] float32 = 1*3*32*32*4 = 12288 bytes
    assert conv_op["operands"][0]["nbytes"] == 12288
    assert conv_op["output"]["buffer_id"] is not None
    assert conv_op["output"]["nbytes"] > 0

    # Wrapper: inductor's def call() body showing triton↔cuDNN interleaving.
    bundle = Path(glob.glob(str(tmp_path / "*.kernels"))[0])
    wrapper = (bundle / "wrapper.py").read_text(encoding="utf-8")
    assert "def call(" in wrapper
    assert "extern_kernels" in wrapper


def test_partial_capture_matmul_has_vendor_lib_marker(tmp_path: Path):
    """Linear+relu: matmul→cuBLAS gap, relu triton kernel still dumps."""

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(64, 128)

        def forward(self, x):
            return torch.relu(self.lin(x))

    model = _M().cuda().eval()
    sample = (torch.randn(1, 64, device="cuda"),)
    _export(model, sample, _M, tmp_path)
    manifests = _manifests(tmp_path)
    assert len(manifests) == 1, "bundle should still be written (partial capture)"
    man = manifests[0]
    vl = man.get("vendor_lib", {})
    assert vl.get("unwritten_buffers"), "matmul graph should mark vendor-lib gap"

    ops = vl.get("ops", [])
    assert ops, "matmul graph should record vendor-op descriptor"
    lin_op = ops[0]
    assert lin_op["type"] == "linear"
    assert lin_op["attrs"]["alpha"] == 1.0
    assert lin_op["attrs"]["beta"] == 1.0
    assert len(lin_op["operands"]) == 3
    assert lin_op["output"]["buffer_id"] is not None


def test_ttir_dump_when_env_set(tmp_path: Path, monkeypatch):
    """HYPERONNX_TTIR=1 enables TTIR/TTGIR sidecars (off by default)."""

    class _P(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    monkeypatch.setenv("HYPERONNX_TTIR", "1")
    model = _P().cuda().eval()
    sample = (torch.randn(1, 16, device="cuda"),)
    _export(model, sample, _P, tmp_path)
    bundle = Path(glob.glob(str(tmp_path / "*.kernels"))[0])
    assert list(bundle.glob("*.ttir")), "TTIR should be written with env var"
    assert list(bundle.glob("*.ttgir")), "TTGIR should be written with env var"
