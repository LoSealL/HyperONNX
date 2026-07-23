"""Tier 2 integration tests for compile + kernel bundle export.

These require a real CUDA device + triton. Skipped otherwise.
"""

import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def _isolate_inductor_cache(tmp_path, monkeypatch):
    # ponytail: inductor's on-disk cache sits above triton.compile, so a hit
    # short-circuits the listener entirely. Force a fresh cache dir per test
    # so every torch.compile actually reaches triton.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    torch._dynamo.reset()
    yield


class _Compiled(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


class _Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _Compiled()

    def forward(self, x):
        return self.child(x) * 2.0


def test_compile_produces_bundle(tmp_path: Path):
    from hyperonnx import export_hyper_onnx

    model = _Parent().cuda()
    args = (torch.randn(4, 8, device="cuda"),)
    export_hyper_onnx(
        model,
        args,
        str(tmp_path / "model.onnx"),
        hiera=[_Compiled],
        compile_hier=[_Compiled],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    bundles = list(tmp_path.glob("*.kernels"))
    assert len(bundles) == 1
    manifest_path = bundles[0] / "manifest.json"
    assert manifest_path.is_file()
    data = json.loads(manifest_path.read_text())
    assert data["schema_version"] == 1
    assert len(data["kernels"]) >= 1
    cubin_path = bundles[0] / data["kernels"][0]["cubin"]
    assert cubin_path.stat().st_size > 0


def test_compile_subset_of_hiera(tmp_path: Path):
    from hyperonnx import export_hyper_onnx

    class _A(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class _B(torch.nn.Module):
        def forward(self, x):
            return x * 2

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _A()
            self.b = _B()

        def forward(self, x):
            return self.b(self.a(x))

    model = _M().cuda()
    args = (torch.randn(2, 4, device="cuda"),)
    export_hyper_onnx(
        model,
        args,
        str(tmp_path / "m.onnx"),
        hiera=[_A, _B],
        compile_hier=[_A],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    a_bundles = list(tmp_path.glob("*A*.kernels"))
    b_bundles = list(tmp_path.glob("*B*.kernels"))
    assert len(a_bundles) == 1
    assert len(b_bundles) == 0


def test_compile_auto_promotes_into_hiera(tmp_path: Path):
    from hyperonnx import export_hyper_onnx

    model = _Parent().cuda()
    args = (torch.randn(2, 4, device="cuda"),)
    export_hyper_onnx(
        model,
        args,
        str(tmp_path / "m.onnx"),
        compile_hier=[_Compiled],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    bundles = list(tmp_path.glob("*.kernels"))
    assert len(bundles) == 1


def test_compile_static_grid_skips_ast(tmp_path: Path):
    from hyperonnx import export_hyper_onnx

    model = _Parent().cuda()
    args = (torch.randn(2, 4, device="cuda"),)
    export_hyper_onnx(
        model,
        args,
        str(tmp_path / "m.onnx"),
        compile_hier=[_Compiled],
        compile_static_grid=True,
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    bundle = next(tmp_path.glob("*.kernels"))
    data = json.loads((bundle / "manifest.json").read_text())
    assert all(k["launch"]["grid_expr"] is None for k in data["kernels"])


def test_bundle_deletion_leaves_valid_model(tmp_path: Path):
    import onnx

    from hyperonnx import export_hyper_onnx

    model = _Parent().cuda()
    args = (torch.randn(2, 4, device="cuda"),)
    out_onnx = tmp_path / "m.onnx"
    export_hyper_onnx(
        model,
        args,
        str(out_onnx),
        compile_hier=[_Compiled],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    for bundle in tmp_path.glob("*.kernels"):
        for f in bundle.iterdir():
            f.unlink()
        bundle.rmdir()
    onnx.checker.check_model(str(out_onnx))
