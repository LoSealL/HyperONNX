"""Unit tests for the bundle writer."""

import json
from inspect import signature
from pathlib import Path

import torch

from hyperonnx.compile.bundle import (
    _finalize_pipeline,
    _layout_span,
    _reconcile_registry_with_allocate,
    write_kernel_bundle,
)
from hyperonnx.compile.capture import BufferInfo, LaunchTraceEntry, LaunchTraceSink
from hyperonnx.compile.typing import CompiledKernelInfo
from hyperonnx.hyper_export import _spec_io, _spec_io_from_output, _tensor_meta


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


# ---- _reconcile_registry_with_allocate() coverage --------------------------


def test_layout_span_basic():
    assert _layout_span([2, 3], [3, 1]) == 6
    assert _layout_span(["2", "3"], ["3", "1"]) == 6  # str entries OK
    assert _layout_span([1, 240, 258], [61920, 258, 1]) == 61920
    assert _layout_span([2, 4], [4, 1, 1]) is None  # mismatched lengths


def test_reconcile_grows_registry_to_allocate_span():
    """The view-before-allocate bug: registry froze [1,240,258] (the view a
    kernel saw), but the graph-table allocate says [2,240,258]. Reconcile
    grows the registry so replay allocates enough storage."""
    registry = [
        {
            "id": 38,
            "kind": "intermediate",
            "dtype": "float32",
            "shape": [1, 240, 258],
            "stride": [61920, 258, 1],
        },
    ]
    pipeline = [
        {
            "graph": "g0",
            "buffers": {
                "buf35": {
                    "kind": "allocate",
                    "shape": ["2", "240", "258"],
                    "stride": ["61920", "258", "1"],
                    "dtype": "float32",
                    "buffer_id": 38,
                }
            },
            "steps": [],
        }
    ]
    _reconcile_registry_with_allocate(registry, pipeline)
    assert registry[0]["shape"] == [2, 240, 258]
    assert registry[0]["stride"] == [61920, 258, 1]


def test_reconcile_leaves_registry_unchanged_when_already_large():
    """When the registry span already covers the allocate, nothing changes."""
    registry = [
        {
            "id": 0,
            "kind": "intermediate",
            "dtype": "float32",
            "shape": [2, 240, 258],
            "stride": [61920, 258, 1],
        },
    ]
    pipeline = [
        {
            "graph": "g0",
            "buffers": {
                "buf0": {
                    "kind": "allocate",
                    "shape": ["2", "240", "258"],
                    "stride": ["61920", "258", "1"],
                    "buffer_id": 0,
                }
            },
            "steps": [],
        }
    ]
    _reconcile_registry_with_allocate(registry, pipeline)
    assert registry[0]["shape"] == [2, 240, 258]  # unchanged


def test_reconcile_skips_allocate_without_buffer_id():
    """An allocate whose buffer_id never resolved (no launch match) is left
    alone — the registry cannot be corrected without the id link."""
    registry = [
        {"id": 5, "kind": "intermediate", "dtype": "float32", "shape": [1, 4]},
    ]
    pipeline = [
        {
            "graph": "g0",
            "buffers": {
                "buf0": {
                    "kind": "allocate",
                    "shape": ["2", "4"],
                    "stride": ["4", "1"],
                    # no buffer_id
                }
            },
            "steps": [],
        }
    ]
    _reconcile_registry_with_allocate(registry, pipeline)
    assert registry[0]["shape"] == [1, 4]  # unchanged


def test_reconcile_drops_stale_stride_when_lengths_differ():
    """When the allocate has no stride (or a different rank) the stale
    registry stride is dropped so replay falls back to a contiguous view
    of the new, larger shape."""
    registry = [
        {
            "id": 1,
            "kind": "intermediate",
            "dtype": "float32",
            "shape": [4],
            "stride": [1],
        },
    ]
    pipeline = [
        {
            "graph": "g0",
            "buffers": {
                "buf1": {
                    "kind": "allocate",
                    "shape": ["2", "4"],
                    "dtype": "float32",
                    "buffer_id": 1,
                    # no stride
                }
            },
            "steps": [],
        }
    ]
    _reconcile_registry_with_allocate(registry, pipeline)
    assert registry[0]["shape"] == [2, 4]
    assert "stride" not in registry[0]


def test_reconcile_through_full_write_kernel_bundle(tmp_path: Path):
    """End-to-end: write_kernel_bundle must reconcile the registry against
    the pipeline's allocate entry when the launch trace froze a too-small
    shape (the actual view-before-allocate bug scenario).

    Setup mirrors the real bug: buf35 is allocated at [2,240,258] but the
    only kernel that touches it sees a [1,240,258] view, so the launch trace
    freezes id=0 at [1,240,258]. The allocate's buffer_id is seeded onto
    the table via the kernel's static-arg → trace-id link, so reconcile can
    then grow the registry.
    """
    k = _fake_kernel("k0")
    k["args"] = [
        {
            "kind": "tensor",
            "buffer_id": 0,
            "direction": "out",
            "shape": [1, 240, 258],
            "stride": [61920, 258, 1],
        },
        {"kind": "scalar", "dtype": "int32", "value": 61920},
    ]
    trace = LaunchTraceSink()
    trace.entries = [
        LaunchTraceEntry(
            symbol="k0",
            grid=(1, 1, 1),
            shared_mem=2048,
            num_warps=4,
            args=k["args"],
        )
    ]
    # Launch trace freezes buffer 0 at the half-size view shape [1,240,258].
    trace.get_or_create_buffer(0x26000, "float32", (1, 240, 258), (61920, 258, 1))
    # Graph-table allocate says the true size is [2,240,258]; the kernel's
    # static arg names buf35, which links the allocate to trace buffer 0.
    wrapper_graph = [
        {
            "graph": "",
            "buffers": {},
            "steps": [
                {
                    "type": "allocate",
                    "buffer": "buf35",
                    "comm_buffer": False,
                    "shape": ["2", "240", "258"],
                    "stride": ["61920", "258", "1"],
                    "dtype": "float32",
                },
                {
                    "type": "triton_kernel",
                    "kernel": "k0",
                    "args": ["buf35", "61920"],
                    "grid_type": "Grid1D",
                },
            ],
        }
    ]
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
        launch_trace=trace,
        wrapper_graph=wrapper_graph,
    )
    data = json.loads((out / "manifest.json").read_text())
    buf0 = next(b for b in data["buffers"] if b["id"] == 0)
    assert buf0["shape"] == [2, 240, 258]
    assert buf0["stride"] == [61920, 258, 1]


def test_reconcile_dtype_mismatch():
    """_reconcile_registry_with_allocate must also fix dtype conflicts.

    When inductor's memory planner reuses the same storage for temporaries
    of different dtypes (e.g. an int64 index buffer followed by a float32
    intermediate), the launch trace freezes the first-seen dtype (int64)
    while the graph-table allocate records the true dtype (float32).
    Reconcile should adopt the allocate's dtype so replay allocates the
    correct storage type.

    Reproduces the structure of MatchAttentionLayer_705 buffer_id=29:
    registry dtype=int64 [2,3840,32] vs allocate dtype=float32 [2,3840,32].
    """
    registry = [
        {
            "id": 29,
            "kind": "intermediate",
            "dtype": "int64",
            "shape": [2, 3840, 32],
            "stride": [122880, 32, 1],
        },
    ]
    pipeline = [
        {
            "graph": "g0",
            "buffers": {
                "buf42": {
                    "kind": "allocate",
                    "shape": ["2", "3840", "32"],
                    "stride": ["122880", "32", "1"],
                    "dtype": "float32",
                    "buffer_id": 29,
                }
            },
            "steps": [],
        }
    ]
    _reconcile_registry_with_allocate(registry, pipeline)
    # Shape/span already matches, but dtype must be corrected.
    assert registry[0]["dtype"] == "float32", (
        "registry dtype should be reconciled to the allocate's float32"
    )


def test_buffer_id_of_links_tensor_by_data_ptr():
    """buffer_id_of resolves a tensor to its registered buffer by data_ptr.

    This is the name-agnostic link that lets io entries point into
    ``buffers[]`` without relying on creation order.
    """
    sink = LaunchTraceSink()
    t = torch.zeros(2, 3)
    sink.pre_register(t.data_ptr(), "input", "float32", (2, 3), "x")
    assert sink.buffer_id_of(t) == 0
    # A tensor the trace never sighted resolves to None.
    assert sink.buffer_id_of(torch.zeros(4, 4)) is None


def test_tensor_meta_stamps_buffer_id_only_when_trace_sighted():
    """_tensor_meta adds buffer_id when the trace knows the tensor and
    omits the key entirely when it does not (or when no trace is given)."""
    sink = LaunchTraceSink()
    t = torch.zeros(2, 3, dtype=torch.float16)
    sink.pre_register(t.data_ptr(), "input", "float16", (2, 3), "x")

    stamped = _tensor_meta("x", t, sink)
    assert stamped["buffer_id"] == 0

    plain = _tensor_meta("x", t)
    assert "buffer_id" not in plain

    unknown = torch.zeros(8, dtype=torch.float16)
    missed = _tensor_meta("y", unknown, sink)
    assert "buffer_id" not in missed


def test_spec_io_stamps_buffer_ids_by_data_ptr_not_position():
    """io entries carry buffer_id resolved by data_ptr, so the mapping is
    correct even when buffer[] creation-order differs from io order.

    Registers ``y`` before ``x`` (so y=bid 0, x=bid 1) but declares the
    signature as ``(x, y)``: positional matching against buffers[] would
    wrongly bind x→0, y→1; the data_ptr link must give x→1, y→0.
    """
    sink = LaunchTraceSink()
    y = torch.zeros(4, dtype=torch.float16)
    x = torch.zeros(2, 3, dtype=torch.float16)
    sink.pre_register(y.data_ptr(), "input", "float16", (4,), "y")  # bid 0
    sink.pre_register(x.data_ptr(), "input", "float16", (2, 3), "x")  # bid 1

    def _fn(x, y): ...

    entries = _spec_io([x, y], {}, signature(_fn), sink)
    assert [e["name"] for e in entries] == ["x", "y"]
    assert entries[0]["buffer_id"] == 1
    assert entries[1]["buffer_id"] == 0


def test_spec_io_from_output_stamps_output_buffer_id():
    """_spec_io_from_output stamps the buffer_id of the identified output."""
    sink = LaunchTraceSink()
    out = torch.zeros(2, 2)
    sink.get_or_create_buffer(out.data_ptr(), "float32", (2, 2))
    sink.identify_output(out)
    entries = _spec_io_from_output(out, sink)
    assert len(entries) == 1
    assert entries[0]["name"] == "output"
    assert entries[0]["buffer_id"] == 0
