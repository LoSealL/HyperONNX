"""Integration tests for the cubin capture + replay harness.

Requires a real CUDA device + triton. Skipped otherwise.

Tested structures span:
- **Pure-triton modules** (pointwise, gather/scatter, cumsum, broadcast).
- **Vendor-delegated modules** (conv→cuDNN, matmul→cuBLAS) as partial
  captures with ``extern_kernel`` steps.
- **Complex multi-op modules** (conv-BN-residual with buffer aliasing,
  SqueezeNet Fire with channel concat, MobileNetV2 inverted residual,
  full ResNet18 BasicBlocks).
All are replayed end-to-end and checked bit-close to the eager reference.
"""

import glob
import json
from pathlib import Path

import pytest
import torch
import torchvision as tv

from hyperonnx import export_hyper_onnx
from hyperonnx.compile.testing import verify

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def _isolate_inductor_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    torch._dynamo.reset()
    yield


def _export(model, sample, module_types, tmp_path: Path) -> None:
    export_hyper_onnx(
        model,
        sample,
        str(tmp_path / "m.onnx"),
        compile_hier=module_types,
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
    _export(model, sample, [_P], tmp_path)
    manifests = _manifests(tmp_path)
    assert len(manifests) == 1
    man = manifests[0]
    assert man["io"]["inputs"] == [
        {"name": "x", "dtype": "float32", "shape": [1, 16, 32, 32]}
    ]
    steps = [s for g in man["pipeline"] for s in g["steps"]]
    triton_steps = [s for s in steps if s["type"] == "triton_kernel"]
    assert triton_steps, "pipeline should record triton kernel calls"
    assert not any(s["type"] == "extern_kernel" for s in steps), (
        "pointwise should be fully triton"
    )
    assert all(s["launch"]["grid_expr"] for s in triton_steps), (
        "grid AST should be attached by default (compile_static_grid=False)"
    )
    # Kernel launch payload is inlined in the pipeline (v2: no kernels[]).
    assert "kernels" not in man
    assert all(s["cubin"] and s["args"] for s in triton_steps)

    expected = model(*sample).detach()
    inputs = list(sample)
    ok = verify(
        glob.glob(str(tmp_path / "*.kernels"))[0],
        inputs,
        expected,
        atol=1e-4,
        rtol=1e-4,
    )
    assert ok, "replay output does not match torch.compile reference"

    # TTIR/TTGIR off by default; source_debug.py written unconditionally.
    bundle = Path(glob.glob(str(tmp_path / "*.kernels"))[0])
    assert not list(bundle.glob("*.ttir")), "TTIR should be off by default"
    assert not list(bundle.glob("*.ttgir")), "TTGIR should be off by default"
    source_debug = (bundle / "source_debug.py").read_text(encoding="utf-8")
    assert "def call(" in source_debug
    assert ".run(" in source_debug, "triton kernel launch should appear in source"
    alloc_steps = [s for s in steps if s["type"] == "allocate"]
    assert alloc_steps, "pipeline should record buffer allocations"
    assert all(s["buffer"] and s["shape"] and s["dtype"] for s in alloc_steps)


def test_partial_capture_conv_records_extern_step(tmp_path: Path):
    """Conv→cuDNN; triton kernels still dump, the vendor call is an extern step."""

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = _M().cuda().eval()
    sample = (torch.randn(1, 3, 32, 32, device="cuda"),)
    _export(model, sample, [_M], tmp_path)
    manifests = _manifests(tmp_path)
    assert len(manifests) == 1, "bundle should still be written (partial capture)"
    man = manifests[0]
    assert man["io"]["inputs"] == [
        {"name": "x", "dtype": "float32", "shape": [1, 3, 32, 32]}
    ]

    # Pipeline: extern (vendor) call recorded in execution order, between
    # allocations and triton kernels.
    steps = [s for g in man["pipeline"] for s in g["steps"]]
    extern_steps = [s for s in steps if s["type"] == "extern_kernel"]
    assert extern_steps, "pipeline should record the cuDNN extern call"
    conv_step = extern_steps[0]
    assert "convolution" in conv_step["kernel"]
    assert any(a["kind"] == "tensor" and a.get("name") for a in conv_step["args"]), (
        "extern args should carry buffer symbols"
    )
    assert conv_step["output"]["name"], "extern step should name its output buffer"
    assert conv_step["output"]["buffer_id"] is not None
    assert any(s["type"] == "triton_kernel" for s in steps)
    # Unified schema: triton tensor args carry both name and buffer_id.
    for s in steps:
        if s["type"] == "triton_kernel":
            assert any(
                a["kind"] == "tensor"
                and a.get("name")
                and a.get("buffer_id") is not None
                for a in s["args"]
            )
            assert s["output"]["buffer_id"] is not None

    # Debug source: inductor's def call() body showing triton↔cuDNN interleaving.
    bundle = Path(glob.glob(str(tmp_path / "*.kernels"))[0])
    wrapper = (bundle / "source_debug.py").read_text(encoding="utf-8")
    assert "def call(" in wrapper
    assert "extern_kernels" in wrapper

    # Partial capture still replays bit-close: cuDNN extern call is run via
    # its aten counterpart, triton kernels via cubin launch.
    expected = model(*sample).detach()
    inputs = list(sample)
    ok = verify(
        glob.glob(str(tmp_path / "*.kernels"))[0],
        inputs,
        expected,
        atol=1e-4,
        rtol=1e-4,
    )
    assert ok, "conv partial-capture replay does not match reference"


def test_partial_capture_matmul_records_extern_step(tmp_path: Path):
    """Linear+relu: matmul→cuBLAS extern step, relu triton kernel still dumps."""

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(64, 128)

        def forward(self, x):
            return torch.relu(self.lin(x))

    model = _M().cuda().eval()
    sample = (torch.randn(1, 64, device="cuda"),)
    _export(model, sample, [_M], tmp_path)
    manifests = _manifests(tmp_path)
    assert len(manifests) == 1, "bundle should still be written (partial capture)"
    man = manifests[0]
    steps = [s for g in man["pipeline"] for s in g["steps"]]
    extern_steps = [s for s in steps if s["type"] == "extern_kernel"]
    assert extern_steps, "pipeline should record the cuBLAS extern call"
    assert any("mm" in s["kernel"] or "addmm" in s["kernel"] for s in extern_steps)
    assert any(s["type"] == "triton_kernel" for s in steps)

    # Debug source: cuBLAS extern call interleaved with the triton relu.
    bundle = Path(glob.glob(str(tmp_path / "*.kernels"))[0])
    source_debug = (bundle / "source_debug.py").read_text(encoding="utf-8")
    assert "def call(" in source_debug
    assert "extern_kernels" in source_debug

    # Partial capture still replays bit-close: cuBLAS extern call is run via
    # its aten counterpart, triton relu via cubin launch.
    expected = model(*sample).detach()
    inputs = list(sample)
    ok = verify(
        glob.glob(str(tmp_path / "*.kernels"))[0],
        inputs,
        expected,
        atol=1e-4,
        rtol=1e-4,
    )
    assert ok, "matmul partial-capture replay does not match reference"


def test_resnet_capture_replay(tmp_path: Path):
    resnet = tv.models.resnet18().cuda().half().eval()
    sample = (torch.randn(1, 3, 224, 224, device="cuda").half(),)
    _export(
        resnet,
        sample,
        [tv.models.resnet.BasicBlock],
        tmp_path,
    )
    bundles = sorted(
        Path(tmp_path).glob("*.kernels"), key=lambda p: int(p.stem.split("_")[1])
    )
    blocks = [b for b in resnet.modules() if isinstance(b, tv.models.resnet.BasicBlock)]
    assert len(bundles) == len(blocks)

    # glob basicblocks inputs and outputs
    inputs = {}
    outputs = {}
    for block in blocks:
        block.register_forward_pre_hook(inputs.setdefault)
        block.register_forward_hook(
            lambda module, input, output: outputs.setdefault(module, output)
        )
    with torch.inference_mode():
        resnet(*sample)
    bundles = iter(bundles)
    results = {}
    for module in resnet.modules():
        if isinstance(module, tv.models.resnet.BasicBlock):
            bundle = next(bundles)
            results[module] = verify(
                bundle, inputs[module], outputs[module], atol=1e-2, rtol=1e-3
            )
    assert all(results.values())


def _export_and_verify(model, sample, module_type, tmp_path, atol=1e-2):
    """Export a single-module model and verify replay matches eager output."""
    _export(model, sample, [module_type], tmp_path)
    bundle = glob.glob(str(tmp_path / "*.kernels"))[0]
    expected = model(*sample).detach()
    assert verify(bundle, list(sample), expected, atol=atol, rtol=atol)


# ---------------------------------------------------------------------------
# Custom structural models — stress specific patterns the replay harness
# must handle: index ops, multi-input reductions, broadcasting,
# channel-cat conv branches, repeated kernels with aliasing.
# ---------------------------------------------------------------------------


def test_replay_gather_scatter(tmp_path: Path):
    """Gather + scatter on dim 1 — exercises indirect indexing and the
    triton index kernels (not vendor-delegated)."""

    class _M(torch.nn.Module):
        def forward(self, x, idx):
            g = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(2)))
            s = torch.zeros_like(x)
            s.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, x.size(2)), g)
            return x * 0.5 + s * 0.5

    model = _M().cuda().eval()
    sample = (
        torch.randn(2, 8, 16, device="cuda"),
        torch.randint(0, 8, (2, 3), device="cuda"),
    )
    _export_and_verify(model, sample, _M, tmp_path)


def test_replay_cumsum_sort(tmp_path: Path):
    """Sort + cumsum — reductions with data-dependent control flow
    (the sort permutation is runtime-dependent)."""

    class _M(torch.nn.Module):
        def forward(self, x):
            s, _ = x.sort(dim=1)
            return s.cumsum(dim=1)

    model = _M().cuda().eval()
    sample = (torch.randn(4, 16, device="cuda"),)
    _export_and_verify(model, sample, _M, tmp_path)


def test_replay_broadcast_clamp(tmp_path: Path):
    """Broadcasting + clamp — exercises triton pointwise kernels with
    rank-mismatched inputs and elementwise clamping."""

    class _M(torch.nn.Module):
        def forward(self, x, y):
            return (x + y.unsqueeze(1)).clamp(min=0, max=1)

    model = _M().cuda().eval()
    sample = (
        torch.randn(4, 8, 16, device="cuda"),
        torch.randn(4, 1, 1, device="cuda"),
    )
    _export_and_verify(model, sample, _M, tmp_path)


def test_replay_repeated_conv_chain(tmp_path: Path):
    """Two 3×3 convs with identical in/out channels — inductor fuses both
    weight-rotation kernels under one symbol. The second call site must
    resolve its weights from the table (not the first site's launch ids).
    Exercises the same repeated-kernel issue that broke ResNet replay."""

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(8, 8, 3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(8)
            self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(8)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            return x

    model = _M().cuda().eval()
    sample = (torch.randn(1, 8, 16, 16, device="cuda"),)
    _export_and_verify(model, sample, _M, tmp_path)


def test_replay_conv_bn_residual_aliasing(tmp_path: Path):
    """Conv-BN-ReLU with residual add — exercises buffer reuse/aliasing
    (the residual input shares storage with the output via reuse steps),
    two conv extern calls, and BN running stats as parameters."""

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(8, 8, 3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(8)
            self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(8)

        def forward(self, x):
            r = x
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)) + r)
            return x

    model = _Block().cuda().eval()
    sample = (torch.randn(1, 8, 16, 16, device="cuda"),)
    _export_and_verify(model, sample, _Block, tmp_path)


def test_replay_depthwise_grouped_conv(tmp_path: Path):
    """Depthwise (groups=channels) + pointwise conv — exercises grouped
    convolution via cuDNN extern calls and two different conv shapes
    in the same module."""

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dw = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8)
            self.pw = torch.nn.Conv2d(8, 16, 1)
            self.bn = torch.nn.BatchNorm2d(16)

        def forward(self, x):
            return torch.relu(self.bn(self.pw(self.dw(x))))

    model = _M().cuda().eval()
    sample = (torch.randn(1, 8, 16, 16, device="cuda"),)
    _export_and_verify(model, sample, _M, tmp_path)


def test_replay_squeezenet_fire_cat(tmp_path: Path):
    """SqueezeNet Fire module — exercises channel concatenation of two
    conv branches (squeeze→expand1x1 cat squeeze→expand3x3), testing
    that the replay correctly handles multi-output graph structure."""

    class _Fire(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.squeeze = torch.nn.Conv2d(16, 8, 1)
            self.expand1x1 = torch.nn.Conv2d(8, 16, 1)
            self.expand3x3 = torch.nn.Conv2d(8, 16, 3, padding=1)

        def forward(self, x):
            s = torch.relu(self.squeeze(x))
            return torch.cat(
                [torch.relu(self.expand1x1(s)), torch.relu(self.expand3x3(s))],
                dim=1,
            )

    model = _Fire().cuda().eval()
    sample = (torch.randn(1, 16, 16, 16, device="cuda"),)
    _export_and_verify(model, sample, _Fire, tmp_path)


def test_replay_mobilenet_inverted_residual(tmp_path: Path):
    """MobileNetV2 InvertedResidual — depthwise conv + pointwise
    projection + residual, exercises the full conv-bn-relu pattern
    in a real torchvision submodule."""

    block = tv.models.mobilenet_v2(weights=None).features[1]
    model = block.cuda().eval()
    sample = (torch.randn(1, 32, 56, 56, device="cuda"),)
    _export_and_verify(model, sample, type(block), tmp_path)


def test_replay_mha_4d_reshape(tmp_path: Path):
    """Multi-head attention with 4D reshape/transpose — exercises the
    output buffer reshape that inductor's return expression applies
    (reinterpret_tensor wrapping the final output buffer to give it the
    model's declared output shape)."""

    class _MHA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(32, 32)
            self.k_proj = torch.nn.Linear(32, 32)
            self.v_proj = torch.nn.Linear(32, 32)
            self.out_proj = torch.nn.Linear(32, 32)

        def forward(self, x):
            B, S, D = x.shape
            q = self.q_proj(x).view(B, S, 4, 8).transpose(1, 2)
            k = self.k_proj(x).view(B, S, 4, 8).transpose(1, 2)
            v = self.v_proj(x).view(B, S, 4, 8).transpose(1, 2)
            attn = torch.softmax((q @ k.transpose(-2, -1)) / (8**0.5), dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).reshape(B, S, D)
            return self.out_proj(out)

    model = _MHA().cuda().eval()
    sample = (torch.randn(2, 8, 32, device="cuda"),)
    _export_and_verify(model, sample, _MHA, tmp_path, atol=1e-2)


def test_ttir_dump_when_env_set(tmp_path: Path, monkeypatch):
    """HYPERONNX_TTIR=1 enables TTIR/TTGIR sidecars (off by default)."""

    class _P(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    monkeypatch.setenv("HYPERONNX_TTIR", "1")
    model = _P().cuda().eval()
    sample = (torch.randn(1, 16, device="cuda"),)
    _export(model, sample, [_P], tmp_path)
    bundle = Path(glob.glob(str(tmp_path / "*.kernels"))[0])
    assert list(bundle.glob("*.ttir")), "TTIR should be written with env var"
    assert list(bundle.glob("*.ttgir")), "TTGIR should be written with env var"
