"""Unit tests for the bundle writer."""

import json
from pathlib import Path

from hyperonnx.compile.bundle import _finalize_pipeline, write_kernel_bundle
from hyperonnx.compile.capture import BufferInfo, LaunchTraceEntry, LaunchTraceSink
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
        module_io={
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
        module_io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
    )
    data = json.loads((out / "manifest.json").read_text())
    assert data["schema_version"] == 2
    assert data["module"]["type_name"] == "A"
    # No wrapper graph passed → fallback pipeline synthesized from entries.
    steps = data["pipeline"][0]["steps"]
    assert [s["type"] for s in steps] == ["triton_kernel"]
    assert steps[0]["kernel"] == "k0"
    assert steps[0]["cubin"] == "kernel_0000.cubin"


def test_bundle_dir_is_legalized_for_unsafe_chars(tmp_path: Path):
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="A/B:0",
        kernels=[],
        module_io={"inputs": [], "outputs": []},
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
        module_io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
    )
    data = json.loads((out / "manifest.json").read_text())
    step = data["pipeline"][0]["steps"][0]
    assert step["launch"]["grid_expr"] == [{"op": "const", "value": 1}]


def test_captured_grid_null_serializes(tmp_path: Path):
    # captured_grid=null when the launch trace is absent (no runtime grid).
    k = _fake_kernel("k0")
    k["launch"]["captured_grid"] = None
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="A",
        kernels=[k],
        module_io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
    )
    data = json.loads((out / "manifest.json").read_text())
    step = data["pipeline"][0]["steps"][0]
    assert step["launch"]["captured_grid"] is None


# ---- _finalize_pipeline() coverage ------------------------------------------


def _rt_buffer(buffer_id: int, kind: str, dtype: str, shape: tuple) -> BufferInfo:
    return BufferInfo(
        data_ptr=buffer_id * 1000,
        kind=kind,
        dtype=dtype,
        shape=shape,
        buffer_id=buffer_id,
    )


def _pipeline_graph() -> dict:
    return {
        "graph": "",
        "buffers": {
            "arg0_1": {
                "shape": ["1", "64"],
                "stride": ["64", "1"],
                "dtype": "float32",
                "kind": "input",
            }
        },
        "steps": [
            {"type": "assert_size_stride", "detail": "noise"},
            {"type": "enter_device_context_manager", "detail": "noise"},
            {
                "type": "allocate",
                "buffer": "buf0",
                "comm_buffer": False,
                "shape": ["1", "128"],
                "stride": ["128", "1"],
                "dtype": "float32",
            },
            {"type": "comment", "detail": "noise"},
            {
                "type": "extern_kernel",
                "kernel": "extern_kernels.mm",
                "output": "buf0",
                "args": ["arg0_1", "reinterpret_tensor(arg0_1, (64, 1), (1, 64), 0)"],
                "kwargs": [],
            },
            {"type": "reuse", "source": "buf0", "reused_as": "buf1"},
            {"type": "kernel_definition", "detail": "noise"},
            {
                "type": "triton_kernel",
                "kernel": "k0",
                "args": ["buf1", 128],
                "grid_type": "Grid1D",
            },
            {"type": "free", "detail": "noise"},
        ],
    }


def _kernel_entries() -> list:
    k = _fake_kernel("k0")
    k["args"] = [
        {"kind": "tensor", "buffer_id": 3, "direction": "out"},
        {"kind": "scalar", "dtype": "int32", "value": 128},
    ]
    # Simulate write_kernel_bundle's entry wrapping.
    return [
        {
            "id": "kernel_0000",
            "cubin": "kernel_0000.cubin",
            "symbol": "k0",
            "device_target": k["device_target"],
            "launch": k["launch"],
            "args": k["args"],
            "variants": [],
        }
    ]


def test_finalize_pipeline_merges_and_validates():
    runtime = [
        _rt_buffer(2, "input", "float32", (1, 64)),
        _rt_buffer(3, "output", "float32", (1, 128)),
    ]
    out = _finalize_pipeline([_pipeline_graph()], _kernel_entries(), runtime)
    assert len(out) == 1
    graph = out[0]

    kept_types = [s["type"] for s in graph["steps"]]
    assert kept_types == ["allocate", "as_strided", "extern_kernel", "triton_kernel"]

    triton = graph["steps"][-1]
    # kernel launch payload inlined; static names merged into descriptors
    assert triton["cubin"] == "kernel_0000.cubin"
    assert triton["launch"]["num_warps"] == 4
    assert triton["device_target"]["arch"] == "sm_80"
    assert triton["args"] == [
        {"kind": "tensor", "buffer_id": 3, "direction": "out", "name": "buf1"},
        {"kind": "scalar", "dtype": "int32", "value": 128},
    ]
    assert triton["output"] == {"name": "buf1", "buffer_id": 3}
    assert "grid_type" not in triton
    assert "call_args" not in triton
    assert "_static_args" not in triton

    # reinterpret_tensor hoisted into a formal as_strided step before the
    # extern call; the extern arg now references the view buffer by name.
    as_strided = graph["steps"][1]
    assert as_strided["args"] == [
        {"kind": "tensor", "name": "arg0_1", "buffer_id": 2},
    ]
    assert as_strided["shape"] == [64, 1]
    assert as_strided["stride"] == [1, 64]
    assert as_strided["offset"] == 0
    assert as_strided["output"]["name"] == "vbuf1"
    assert as_strided["output"]["buffer_id"] is None

    # extern step shares the same descriptor schema; the reinterpreted arg
    # is now a plain name reference (no buffer_id — resolved by name at
    # replay via the as_strided step's output in name_tensors).
    extern = graph["steps"][2]
    assert extern["args"] == [
        {"kind": "tensor", "name": "arg0_1", "buffer_id": 2},
        {"kind": "tensor", "name": "vbuf1", "buffer_id": None},
    ]
    assert extern["output"] == {"name": "buf0", "buffer_id": 3, "direction": "out"}

    table = graph["buffers"]
    # input matched by (shape, dtype) to runtime buffer 2
    assert table["arg0_1"]["buffer_id"] == 2
    # allocate matched to runtime output buffer 3
    assert table["buf0"]["buffer_id"] == 3
    assert table["buf0"]["kind"] == "allocate"
    # reuse folded into the table and inherits the source's buffer_id
    assert table["buf1"] == {"alias_of": "buf0", "buffer_id": 3}
    # hoisted view buffer: carries layout but no buffer_id
    assert table["vbuf1"]["reinterpret_of"] == "arg0_1"
    assert "buffer_id" not in table["vbuf1"]
    # every referenced name is defined
    assert all("undefined" not in m for m in table.values())


def test_finalize_pipeline_flags_undefined_buffer():
    graph = _pipeline_graph()
    # buf9 is referenced by the extern step but defined nowhere (no input
    # registry entry, no allocate, no alias, no launch trace).
    extern = next(s for s in graph["steps"] if s["type"] == "extern_kernel")
    extern["args"] = ["buf9"]
    out = _finalize_pipeline([graph], _kernel_entries(), [])
    table = out[0]["buffers"]
    assert table["buf9"] == {"undefined": True}


def test_finalize_pipeline_without_launch_trace():
    """No runtime buffers and no trace ids in entries: the static table
    stays, no buffer_id attached."""
    entries = _kernel_entries()
    for arg in entries[0]["args"]:
        arg.pop("buffer_id", None)
    out = _finalize_pipeline([_pipeline_graph()], entries, [])
    table = out[0]["buffers"]
    assert "buffer_id" not in table["buf0"]
    assert table["buf1"] == {"alias_of": "buf0"}


def test_finalize_pipeline_repeated_kernel_resolves_second_from_table():
    """A kernel appearing multiple times in the pipeline is seeded from the
    launch trace only on its first occurrence. The second call site resolves
    its buffer ids from the table (alias propagation + shape matching),
    avoiding cross-contamination from the first call site's ids."""
    graph = _pipeline_graph()
    triton = next(s for s in graph["steps"] if s["type"] == "triton_kernel")
    graph["steps"].append(dict(triton))  # second call site, same kernel
    out = _finalize_pipeline([graph], _kernel_entries(), [])
    steps = [s for s in out[0]["steps"] if s["type"] == "triton_kernel"]
    assert len(steps) == 2
    # First occurrence: launch descriptors (direction, buffer_id from trace).
    assert "launch_missing" not in steps[0]
    assert steps[0]["args"][0].get("direction") == "out"
    assert steps[0]["args"][0].get("buffer_id") == 3
    assert steps[0]["args"][0]["name"] == "buf1"
    # Second occurrence: table-resolved (launch_missing set, but cubin
    # still available so replay works).
    assert steps[1]["launch_missing"] is True
    assert steps[1]["args"][0]["name"] == "buf1"


def test_autotune_loser_variants_are_dropped(tmp_path: Path):
    """Two compiled variants share symbol+warps+shared (differ only in
    XBLOCK, invisible to the legacy key filter); only the launched winner's
    hash survives."""
    winner = _fake_kernel("k0")
    winner["src_hash"] = "hash_winner"
    loser = _fake_kernel("k0")
    loser["src_hash"] = "hash_loser"

    trace = LaunchTraceSink()
    trace.entries = [
        LaunchTraceEntry(
            symbol="k0",
            grid=(128, 1, 1),
            shared_mem=2048,
            num_warps=4,
            args=[],
            kernel_hash="hash_winner",
        )
    ]

    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="A",
        kernels=[winner, loser],
        module_io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
        launch_trace=trace,
    )
    data = json.loads((out / "manifest.json").read_text())
    steps = [s for g in data["pipeline"] for s in g["steps"]]
    assert len(steps) == 1
    assert steps[0]["kernel"] == "k0"
    assert len(list(out.glob("*.cubin"))) == 1
