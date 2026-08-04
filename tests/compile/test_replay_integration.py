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
from hyperonnx.compile.testing import replay, verify

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


# ---------------------------------------------------------------------------
# Known-issue regression tests (document the two replay blockers discovered
# via the MatchStereo model-zoo bundle replay, 2026-08-04).
# ---------------------------------------------------------------------------


def test_multi_output_module_tags_output_buffer(tmp_path: Path):
    """A module returning a tuple must tag output buffers in the registry.

    identify_output receives the compiled forward's return value. When the
    output is a tuple, getattr(tuple, "data_ptr", ...) returns None and the
    call is a no-op — no buffer gets kind="output", so replay() raises
    "no output buffer in manifest".

    Reproduces MatchAttentionLayer which returns (x, self_rpos, field).
    """

    class _Multi(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)

        def forward(self, x):
            h = self.lin(x)
            return h, h * 2

    model = _Multi().cuda().eval()
    sample = (torch.randn(2, 8, 16, device="cuda"),)
    _export(model, sample, [_Multi], tmp_path)
    man = _manifests(tmp_path)[0]
    assert len(man["io"]["outputs"]) == 2, "manifest should list 2 outputs"
    output_bufs = [b for b in man["buffers"] if b["kind"] == "output"]
    assert output_bufs, (
        "at least one registry buffer must be tagged kind=output for replay "
        "to locate the result"
    )


def test_gather_mm_does_not_leak_int64_into_extern(tmp_path: Path):
    """A module with int64 gather indices + float mm must not leak dtype.

    Inductor's memory planner may reuse the same storage for temporaries of
    different dtypes. The launch trace freezes dtype on first sight (int64
    from an index intermediate), but the allocate records float32. Replay
    then resolves a float32 view over int64 storage and the extern mm
    receives mismatched dtypes.

    Reproduces MatchAttentionLayer_705 buffer_id=29 where registry says
    int64 but the allocate is float32.

    NOTE: this minimal model may not trigger inductor's reuse deterministically;
    the test asserts no dtype conflicts exist between registry and allocates.
    If inductor doesn't reuse here, the test passes trivially (no regression).
    """

    class _GatherMM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(32, 32)

        def forward(self, x, offset):
            B, N, C = x.shape
            idx = torch.floor(offset).long().clamp(0, N - 1).squeeze(-1)
            idx_expand = idx.unsqueeze(-1).expand(-1, -1, C)
            gathered = torch.gather(x, 1, idx_expand)
            return self.lin(gathered)

    model = _GatherMM().cuda().eval()
    sample = (
        torch.randn(2, 8, 32, device="cuda"),
        torch.randn(2, 8, 1, device="cuda"),
    )
    _export(model, sample, [_GatherMM], tmp_path)
    man = _manifests(tmp_path)[0]
    # Check for dtype conflicts between registry and graph-table allocates.
    tbl = man["pipeline"][0].get("buffers", {}) if man["pipeline"] else {}
    conflicts = []
    for _name, meta in tbl.items():
        if meta.get("kind") != "allocate":
            continue
        bid = meta.get("buffer_id")
        if bid is None:
            continue
        reg = next((b for b in man["buffers"] if b["id"] == bid), None)
        if reg and reg["dtype"] != meta.get("dtype"):
            conflicts.append((bid, reg["dtype"], meta["dtype"]))
    assert not conflicts, (
        f"registry/allocate dtype conflicts: {conflicts} — "
        "replay would pass int64 storage to a float32 extern mm"
    )


def test_replay_multi_output_selects_correct_buffer(tmp_path: Path):
    """replay() must return the buffer matching io.outputs[0]'s shape, not
    the manifest-order-first output buffer.

    For multi-output modules every tuple element is tagged kind="output" in
    buffer-creation order. We compute the LARGER output first (``big``) but
    return it SECOND, so the manifest-first output buffer is io.outputs[1]
    (shape [2,8,16]) while io.outputs[0] is the smaller [2,8,8]. Without the
    fix, replay picks the [2,8,16] buffer and the final reshape to [2,8,8]
    raises (numel 256 vs 128); with the fix it matches by shape and returns
    the correct tensor.
    """

    class _Multi(torch.nn.Module):
        def forward(self, x):
            big = x * 3.0  # [2,8,16], materialized first
            small = big[:, :, :8] + 7.0  # [2,8,8], depends on big
            return small, big

    model = _Multi().cuda().eval()
    sample = (torch.randn(2, 8, 16, device="cuda"),)
    _export(model, sample, [_Multi], tmp_path)
    man = _manifests(tmp_path)[0]
    io_out = man["io"]["outputs"]
    assert len(io_out) == 2, "manifest should list 2 outputs"
    out_bufs = [b for b in man["buffers"] if b["kind"] == "output"]
    assert out_bufs, "module should tag output buffers"
    # Precondition for the regression: manifest-first output buffer must NOT
    # already match io.outputs[0]'s shape, otherwise the old bug is inert.
    first_out_shape = [int(s) for s in out_bufs[0]["shape"]]
    expected_shape = [int(s) for s in io_out[0]["shape"]]
    assert first_out_shape != expected_shape, (
        "precondition not met: manifest-first output buffer already matches "
        "io.outputs[0]; test would not exercise the bug"
    )

    expected = model(*sample)[0].detach()
    ok = verify(
        glob.glob(str(tmp_path / "*.kernels"))[0],
        list(sample),
        expected,
        atol=1e-4,
        rtol=1e-4,
    )
    assert ok, "replay did not return the io.outputs[0] buffer"


def test_reinterpret_view_captures_offset(tmp_path: Path):
    """A cat whose inputs are written to non-zero-offset slices of a
    pre-allocated buffer (inductor's ReinterpretLine) must record the real
    storage offset in the manifest.

    inductor allocates one buffer for the cat output and reinterprets it at
    each input's slice offset (``reinterpret_tensor(buf, shape, stride,
    offset)``). The capture code must read the offset from the view's layout
    — ``NonOwningLayout.offset`` is always 0 (its ``__init__`` drops the
    view's offset), so reading ``line.layout.offset`` records every view at
    offset 0 and replay aliases all slices to channel 0.

    The module mirrors MatchAttentionLayer's cat phase: the three inputs
    (x, field*scale, rpos) are written by separate triton kernels into
    non-zero-offset slices (offset 256/258 of a 266-wide buffer).
    """

    class _CatSlices(torch.nn.Module):
        def __init__(self, dim, num_head):
            super().__init__()
            self.norm = torch.nn.LayerNorm(dim + 2 + num_head * 2)
            self.num_head = num_head

        def forward(self, x, rpos, field):
            B, H, W, _ = x.shape
            scale = self.norm.weight.new_ones(1, 1, 1, 2)
            x_cat = torch.cat(
                (x, field * scale.to(field.dtype), rpos), dim=-1
            ).contiguous()
            grid = torch.meshgrid(
                torch.arange(H, device=x.device),
                torch.arange(W, device=x.device),
                indexing="ij",
            )
            coords = torch.stack(grid[::-1], dim=-1)[None].repeat(
                1, 1, 1, self.num_head
            )
            off = (rpos + coords).view(B, H * W, self.num_head, 2).contiguous()
            return x_cat, self.norm(x_cat), off

    dim, num_head = 256, 4
    model = _CatSlices(dim, num_head).cuda().eval()
    sample = (
        torch.randn(2, 12, 20, dim, device="cuda"),
        torch.randn(2, 12, 20, num_head * 2, device="cuda"),
        torch.randn(2, 12, 20, 2, device="cuda"),
    )
    _export(model, sample, [_CatSlices], tmp_path)
    manifests = _manifests(tmp_path)
    assert manifests, "no kernel bundle produced"
    # At least one view_of entry must carry a non-zero offset — the cat's
    # non-first slice lives deeper in the pre-allocated buffer.
    nonzero = [
        (name, meta["offset"])
        for man in manifests
        for g in man["pipeline"]
        for name, meta in g.get("buffers", {}).items()
        if meta.get("view_of") and int(meta.get("offset", 0)) != 0
    ]
    assert nonzero, (
        "no non-zero-offset view_of recorded — ReinterpretLine offset "
        "capture is broken (NonOwningLayout.offset is always 0)"
    )


def test_run_extern_writes_to_view_base_storage(tmp_path: Path):
    """An extern_kernel whose output buffer is a ``view_of`` a different base
    allocation must write its result into that BASE storage — the same storage
    ``tensor_for`` reads pull from.

    The launch-trace ``buffer_id`` on the extern output is the VIEW's runtime
    id, not the underlying allocation. Before the fix, ``run_extern`` wrote to
    ``storages[out_bid]`` (the view id) while downstream ``tensor_for`` reads
    followed ``view_of`` to ``storages[base_bid]`` — so reads saw stale data.

    This builds a minimal extern-only bundle (no cubins needed) where an
    ``aten.mm`` output is declared ``view_of`` a separate output allocation,
    then asserts replay returns the correct mm product. Without the fix the
    write misses the output storage and replay returns zeros.
    """

    bundle = tmp_path / "mmview.kernels"
    bundle.mkdir()
    manifest = {
        "buffers": [
            {
                "id": 0,
                "kind": "input",
                "name": "x",
                "dtype": "float32",
                "shape": [2, 2],
            },
            {
                "id": 1,
                "kind": "input",
                "name": "w",
                "dtype": "float32",
                "shape": [2, 2],
            },
            {"id": 2, "kind": "output", "dtype": "float32", "shape": [2, 2]},
            {"id": 3, "kind": "intermediate", "dtype": "float32", "shape": [2, 2]},
        ],
        "io": {
            "inputs": [
                {"name": "x", "dtype": "float32", "shape": [2, 2]},
                {"name": "w", "dtype": "float32", "shape": [2, 2]},
            ],
            "outputs": [{"name": "y", "dtype": "float32", "shape": [2, 2]}],
        },
        "pipeline": [
            {
                "buffers": {
                    "buf_base": {
                        "kind": "allocate",
                        "buffer_id": 2,
                        "shape": [2, 2],
                        "stride": [2, 1],
                        "dtype": "float32",
                    },
                    "buf_view": {
                        "kind": "view",
                        "view_of": "buf_base",
                        "buffer_id": 3,
                        "shape": [2, 2],
                        "stride": [2, 1],
                        "dtype": "float32",
                    },
                },
                "steps": [
                    {
                        "type": "extern_kernel",
                        "kernel": "extern_kernels.mm",
                        "args": [
                            {"kind": "tensor", "name": "x", "buffer_id": 0},
                            {"kind": "tensor", "name": "w", "buffer_id": 1},
                        ],
                        "output": {"name": "buf_view", "buffer_id": 3},
                    }
                ],
            }
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    x = torch.randn(2, 2, device="cuda")
    w = torch.randn(2, 2, device="cuda")
    expected = (x @ w).detach()
    out = replay(bundle, [x, w])
    assert list(out.shape) == [2, 2]
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), (
        f"extern mm output (view_of base) not written to base storage: "
        f"max_abs_diff={((out - expected).abs().max()).item()}"
    )
