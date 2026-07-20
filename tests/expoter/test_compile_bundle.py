"""Unit tests for the bundle writer."""

import json
from pathlib import Path

from hyperonnx.compile.bundle import write_kernel_bundle
from hyperonnx.compile.typing import CompiledKernelInfo


def _fake_kernel(symbol: str) -> CompiledKernelInfo:
    return CompiledKernelInfo(
        cubin_bytes=b"\x00FAKE",
        symbol=symbol,
        device_target={"backend": "cuda", "arch": "sm_80", "warp_size": 32},
        launch={
            "num_warps": 4,
            "num_ctas": 1,
            "shared_mem_bytes": 2048,
            "num_regs": 64,
            "grid_expr": None,
            "captured_grid": [1, 2, 1],
        },
        args=[],
    )


def test_write_bundle_creates_dir_and_files(tmp_path: Path):
    kernels = [_fake_kernel("k0"), _fake_kernel("k1")]
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="Attention:0",
        kernels=kernels,
        io={
            "inputs": [{"name": "x", "dtype": "float16", "shape": [1, 8]}],
            "outputs": [],
        },
        module_meta={
            "type_name": "Attention:0",
            "python_class": "M.Attention",
            "torch_version": "2.10.0",
            "triton_version": "3.5.0",
        },
    )
    assert out == tmp_path / "Attention_0.kernels"
    assert out.is_dir()
    assert (out / "manifest.json").is_file()
    assert (out / "kernel_0000.cubin").read_bytes() == b"\x00FAKE"
    assert (out / "kernel_0001.cubin").read_bytes() == b"\x00FAKE"


def test_manifest_is_well_formed(tmp_path: Path):
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="A",
        kernels=[_fake_kernel("k0")],
        io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
    )
    data = json.loads((out / "manifest.json").read_text())
    assert data["schema_version"] == 1
    assert data["module"]["type_name"] == "A"
    assert data["kernels"][0]["id"] == "kernel_0000"
    assert data["kernels"][0]["cubin"] == "kernel_0000.cubin"
    assert data["kernels"][0]["variants"] == []


def test_bundle_dir_is_legalized_for_unsafe_chars(tmp_path: Path):
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="A/B:0",
        kernels=[],
        io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A/B:0",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
    )
    assert out.exists()


def test_grid_expr_serializes_when_present(tmp_path: Path):
    k = _fake_kernel("k0")
    k["launch"]["grid_expr"] = [{"op": "const", "value": 1}]
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="A",
        kernels=[k],
        io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
    )
    data = json.loads((out / "manifest.json").read_text())
    assert data["kernels"][0]["launch"]["grid_expr"] == [{"op": "const", "value": 1}]
