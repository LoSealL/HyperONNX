"""
Copyright (C) 2026 The HYPERONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# pyright: reportMissingImports=none

import ast as pyast
import importlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from onnxifier.logger import debug, info, warning
from onnxifier.utils import legalize_path_name

from .typing import BufferEntry, CompiledKernelInfo, KernelBundleManifest, KernelEntry

if TYPE_CHECKING:
    from .capture import (
        BufferInfo,
        LaunchTraceEntry,
        LaunchTraceSink,
    )

_SCHEMA_VERSION = 2
"""Kernel-bundle manifest schema version.

Bump when ``KernelBundleManifest`` gains a breaking field; runtimes gate
dispatch on this and must refuse unknown versions rather than guess.
"""

_BUFFER_NAME_RE = re.compile(r"\b(?:buf\d+|arg\d+_\d+|vbuf\d+)\b")
"""Matches inductor's static buffer names (and hoisted view buffers) inside
step arg expressions."""


def _map_graph_inputs_by_source(
    table: dict[str, dict],
    runtime_buffers: "list[BufferInfo]",
    rt_claimed: set[int],
) -> None:
    """Resolve graph inputs to registry buffers by FX placeholder target.

    Graph inputs carry ``source`` (the original parameter/input name from
    the FX placeholder, captured in the wrapper spy); the launch trace
    pre-registers parameters/buffers/inputs under the same names. Exact
    name matching replaces any positional or shape heuristic.
    """
    by_name = {b.name: b.buffer_id for b in runtime_buffers if b.name}
    for meta in table.values():
        if meta.get("kind") != "input" or "buffer_id" in meta:
            continue
        source = meta.get("source")
        if source and source in by_name:
            meta["buffer_id"] = by_name[source]
            rt_claimed.add(by_name[source])


_KEEP_STEP_TYPES = frozenset({
    "allocate",
    "triton_kernel",
    "extern_kernel",
    "as_strided",
})
"""Step types kept in the final pipeline; asserts, device guards, comments,
line contexts, kernel definitions, frees and inductor's own reuse-as views
are codegen noise for replay (reuse views fold into the buffer table like
aliases). ``as_strided`` steps are hoisted from ``reinterpret_tensor(...)``
expressions inside extern call args — they carry real layout transforms
(weight.T for mm, etc.) and must execute formally at replay time."""


def _shape_ints(meta: dict) -> tuple[int, ...] | None:
    """Coerce static shape entries (str/int, possibly symbolic) to ints."""
    out = []
    for s in meta.get("shape", []):
        try:
            out.append(int(s))
        except (TypeError, ValueError):
            return None
    return tuple(out)


def _layout_span(shape: list, stride: list) -> int | None:
    """Storage span of a ``(shape, stride)`` layout, in elements.

    ``None`` when shape/stride are symbolic or mismatched. Accepts str
    or int entries (allocate-table shapes arrive as strings from
    ``_serialize_buffer``).
    """
    if len(shape) != len(stride):
        return None
    try:
        return sum((int(s) - 1) * int(st) for s, st in zip(shape, stride)) + 1
    except (TypeError, ValueError):
        return None


def _contiguous_span(shape: list) -> int:
    """Element count of a contiguous tensor of this shape (>= 1)."""
    n = 1
    try:
        for s in shape:
            n *= int(s)
    except (TypeError, ValueError):
        return 0
    return max(1, n)


def _registry_span(entry: BufferEntry) -> int:
    """Storage span a registry ``BufferEntry`` demands at replay."""
    shape = entry.get("shape")
    if not shape:
        return 0
    stride = entry.get("stride")
    if stride and len(stride) == len(shape):
        span = _layout_span(shape, stride)
        if span is not None:
            return span
    return _contiguous_span(shape)


def _static_arg_descriptor(arg: Any, table: dict[str, dict], owner: str) -> dict:
    """Build a unified ``KernelArgDescriptor`` from a static wrapper arg.

    Buffer names (possibly embedded in compound expressions like
    ``reinterpret_tensor(buf0, ...)``) resolve to ``buffer_id`` via the
    graph's buffer table — the name↔index correspondence lands directly on
    each arg. Undefined names are flagged in the table and warned about.
    Scalars and literals pass through with their values.
    """
    if isinstance(arg, str):
        if _BUFFER_NAME_RE.fullmatch(arg):
            return {
                "kind": "tensor",
                "name": arg,
                "buffer_id": table.get(arg, {}).get("buffer_id"),
            }
        names = _BUFFER_NAME_RE.findall(arg)
        if names:
            return {
                "kind": "tensor",
                "name": names[0],
                "buffer_id": table.get(names[0], {}).get("buffer_id"),
                "expr": arg,
            }
        try:
            return {"kind": "scalar", "dtype": "int32", "value": int(arg)}
        except ValueError:
            try:
                return {"kind": "scalar", "dtype": "float32", "value": float(arg)}
            except ValueError:
                return {"kind": "literal", "value": arg}
    if isinstance(arg, bool):
        return {"kind": "scalar", "dtype": "int32", "value": int(arg)}
    if isinstance(arg, int):
        return {"kind": "scalar", "dtype": "int32", "value": arg}
    if isinstance(arg, float):
        return {"kind": "scalar", "dtype": "float32", "value": arg}
    debug(f"{owner}: non-scalar literal arg kept as-is: {arg!r:.80}")
    return {"kind": "literal", "value": arg}


def _flag_undefined(names: list[str], table: dict[str, dict], owner: str) -> None:
    """Warn + flag every referenced buffer name missing from the table."""
    for name in names:
        if name not in table:
            warning(f"undefined buffer {name} referenced by {owner}")
            table[name] = {"undefined": True}


_REINTERPRET_RE = re.compile(
    r"reinterpret_tensor\("
    r"\s*(\w+)\s*,"
    r"\s*(\([^)]*\))\s*,"
    r"\s*(\([^)]*\))\s*,"
    r"\s*(\d+)\s*"
    r"\)$"
)


def _hoist_reinterpret_views(
    step: dict, table: dict[str, dict], seq: int
) -> tuple[list[dict], int]:
    """Promote ``reinterpret_tensor(name, shape, stride, offset)`` arg
    expressions into first-class ``as_strided`` pipeline steps.

    Inductor codegens layout transforms (e.g. weight.T for mm) inline in
    extern call args as opaque strings. Each is parsed here and emitted as
    a structured step preceding the extern call, with a synthetic ``vbuf``
    buffer registered in the table (carrying shape/stride/dtype but no
    ``buffer_id`` — the step carries the source reference and replay stores
    the view by name so ``tensor_for`` resolves it correctly).

    Returns the list of view steps to insert and the updated sequence
    counter.
    """
    views: list[dict] = []
    args = step["args"]
    for i, arg in enumerate(args):
        if not isinstance(arg, str):
            continue
        m = _REINTERPRET_RE.match(arg)
        if not m:
            continue
        src_name = m.group(1)
        shape = list(pyast.literal_eval(m.group(2)))
        stride = list(pyast.literal_eval(m.group(3)))
        offset = int(m.group(4))
        seq += 1
        vbuf = f"vbuf{seq}"
        src_meta = table.get(src_name, {})
        table[vbuf] = {
            "shape": list(shape),
            "stride": list(stride),
            "dtype": src_meta.get("dtype", "float32"),
            "kind": "view",
            "reinterpret_of": src_name,
        }
        views.append({
            "type": "as_strided",
            "args": [src_name],
            "shape": shape,
            "stride": stride,
            "offset": offset,
            "output": {"name": vbuf},
        })
        args[i] = vbuf
    return views, seq


def _finalize_triton_step(
    step: dict,
    table: dict[str, dict],
    launch_args: list | None,
) -> None:
    """Unify a triton step's args with the launch-trace descriptors.

    Static buffer names merge positionally into the runtime descriptors
    (each tensor arg ends up with both ``name`` and ``buffer_id``); the
    primary output (first ``direction=out`` tensor) surfaces as
    ``output``. ``launch_args`` are this step's own launch descriptors
    (repeated kernels launch once per step); when ``None`` the call site
    wasn't traced and args resolve via the buffer table only.
    """
    static = step.pop("_static_args")
    if launch_args is None:
        # Repeated kernel whose first-occurrence launch was already consumed;
        # resolve names from the table only.
        step["launch_missing"] = True
        owner = f"triton_kernel {step['kernel']}"
        _flag_undefined(
            [a for a in static if isinstance(a, str) and _BUFFER_NAME_RE.fullmatch(a)],
            table,
            owner,
        )
        step["args"] = [_static_arg_descriptor(a, table, owner) for a in static]
        return
    kargs = [dict(a) for a in launch_args]
    if len(static) != len(kargs):
        debug(
            f"{step['kernel']}: static/runtime arg count differs "
            f"({len(static)} vs {len(kargs)}); names skipped"
        )
    merged: list[dict] = []
    for i, ka in enumerate(kargs):
        sa = static[i] if i < len(static) else None
        if isinstance(sa, str) and _BUFFER_NAME_RE.fullmatch(sa):
            ka["name"] = sa
            bid = table.get(sa, {}).get("buffer_id")
            if (
                bid is not None
                and ka.get("buffer_id") is not None
                and bid != ka["buffer_id"]
            ):
                # Allocator reuse: one name can map to different runtime ids
                # across steps. Step's own launch args stay authoritative.
                warning(
                    f"buffer conflict for {step['kernel']} arg {sa}: "
                    f"pipeline={bid} launch_trace={ka.get('buffer_id')} "
                    f"(allocator reuse; step args stay authoritative)"
                )
        merged.append(ka)
    step["args"] = merged
    _flag_undefined(
        [a["name"] for a in merged if a.get("name")],
        table,
        f"triton_kernel {step['kernel']}",
    )
    for ka in merged:
        if ka.get("kind") == "tensor" and ka.get("direction") == "out":
            step["output"] = {"name": ka.get("name"), "buffer_id": ka.get("buffer_id")}
            break


def _finalize_extern_step(step: dict, table: dict[str, dict]) -> None:
    """Unify an extern step: static args become descriptors with resolved
    ``buffer_id`` (same schema as triton steps); ``output`` gains the
    resolved id. Non-empty ``kwargs`` are kept as-is."""
    static = step.pop("args")
    kwargs = step.pop("kwargs", None)
    owner = f"extern_kernel {step['kernel']}"
    _flag_undefined(
        [n for a in static if isinstance(a, str) for n in _BUFFER_NAME_RE.findall(a)],
        table,
        owner,
    )
    step["args"] = [_static_arg_descriptor(a, table, owner) for a in static]
    if kwargs:
        step["kwargs"] = kwargs
    out_name = step.pop("output")
    out_meta = table.get(out_name)
    if out_meta is None:
        # Output known only via a downstream alias.
        for meta in table.values():
            if meta.get("alias_of") == out_name or meta.get("view_of") == out_name:
                out_meta = meta
                break
    step["output"] = {
        "name": out_name,
        "buffer_id": (out_meta or {}).get("buffer_id"),
        "direction": "out",
    }


def _finalize_as_strided_step(step: dict, table: dict[str, dict]) -> None:
    """Resolve the source buffer_id for an ``as_strided`` step's args and
    structure its output. The output buffer_id stays ``None`` — the view
    shares the source's storage but is resolved at replay by name (stored
    in ``name_tensors`` after the step executes), not by pointer."""
    owner = f"as_strided {step['output']['name']}"
    _flag_undefined(
        [
            n
            for a in step["args"]
            if isinstance(a, str)
            for n in _BUFFER_NAME_RE.findall(a)
        ],
        table,
        owner,
    )
    step["args"] = [_static_arg_descriptor(a, table, owner) for a in step["args"]]
    out_name = step["output"]["name"]
    out_meta = table.get(out_name, {})
    step["output"] = {
        "name": out_name,
        "buffer_id": out_meta.get("buffer_id"),
        "direction": "out",
    }


def _finalize_pipeline(
    pipeline: list[dict],
    kernel_entries: list[KernelEntry],
    runtime_buffers: "list[BufferInfo]",
) -> list[dict]:
    """Merge the wrapper pipeline with kernel entries and validate buffers.

    Per graph:
      - folds ``reuse`` steps into the buffer table as ``alias_of`` entries
        and drops noise steps (see ``_KEEP_STEP_TYPES``);
      - inlines the kernel launch payload (``cubin``, ``device_target``,
        ``launch``) into each ``triton_kernel`` step — the pipeline *is*
        the kernel list;
      - unifies step args: both ``triton_kernel`` and ``extern_kernel``
        carry ``args`` as ``KernelArgDescriptor``\\s where tensor args have
        both ``name`` (static buffer symbol) and ``buffer_id`` (runtime
        index), plus an ``output`` entry of the same shape;
      - cross-validates the static buffer table against runtime buffers by
        (shape, dtype), attaching ``buffer_id`` on match; undefined names
        are flagged and warned about.

    Launch descriptors come directly from ``kernel_entries`` — each entry's
    ``args`` were already overlaid with the traced runtime descriptors
    (carrying ``buffer_id``\\s) in :func:`write_kernel_bundle`. A symbol
    appearing multiple times is seeded only on first occurrence.

    Mismatches are logged and annotated, never fatal.
    """
    symbol_to_entry = {k["symbol"]: k for k in kernel_entries}
    rt_by_shape: dict[tuple, list[int]] = {}
    for b in runtime_buffers:
        rt_by_shape.setdefault((tuple(b.shape), b.dtype), []).append(b.buffer_id)
    rt_claimed: set[int] = set()

    def match_runtime(meta: dict) -> int | None:
        shape = _shape_ints(meta)
        if shape is None:
            return None
        for bid in rt_by_shape.get((shape, meta.get("dtype", "")), []):
            if bid not in rt_claimed:
                rt_claimed.add(bid)
                return bid
        return None

    out_graphs: list[dict] = []
    for graph in pipeline:
        table: dict[str, dict] = dict(graph.get("buffers", {}))
        kept: list[dict] = []
        vbuf_seq = 0
        for step in graph["steps"]:
            stype = step["type"]
            if stype == "reuse":
                # Layout-carrying reuse → view edge; plain alias otherwise.
                if "shape" in step or "stride" in step:
                    meta = {
                        k: step[k]
                        for k in ("shape", "stride", "dtype", "offset")
                        if k in step
                    }
                    table[step["reused_as"]] = {
                        **meta,
                        "view_of": step["source"],
                        "kind": "view",
                    }
                else:
                    table[step["reused_as"]] = {"alias_of": step["source"]}
                continue
            if stype == "view":
                meta = {
                    k: step[k]
                    for k in ("shape", "stride", "dtype", "offset")
                    if k in step
                }
                table[step["view"]] = {
                    **meta,
                    "view_of": step["source"],
                    "kind": "view",
                }
                continue
            if stype not in _KEEP_STEP_TYPES:
                continue
            if stype == "allocate":
                meta = {k: step[k] for k in ("shape", "stride", "dtype") if k in step}
                table[step["buffer"]] = {**meta, "kind": "allocate"}
            if stype == "extern_kernel":
                views, vbuf_seq = _hoist_reinterpret_views(step, table, vbuf_seq)
                kept.extend(views)
            if stype == "triton_kernel":
                entry = symbol_to_entry.get(step["kernel"])
                step["_static_args"] = step.pop("args")
                step.pop("grid_type", None)
                if entry is None:
                    warning(f"pipeline kernel {step['kernel']} has no cubin entry")
                else:
                    step["cubin"] = entry["cubin"]
                    step["device_target"] = entry["device_target"]
                    step["launch"] = entry["launch"]
            kept.append(step)

        # Resolve graph inputs by FX placeholder target (exact name match).
        _map_graph_inputs_by_source(table, runtime_buffers, rt_claimed)
        # Seed name→buffer_id from kernel entries. Only the FIRST occurrence
        # of each symbol is seeded — repeated call sites have different
        # buffer ids and must resolve via the table.
        seeded_symbols: set[str] = set()
        for step in kept:
            if step["type"] != "triton_kernel" or "cubin" not in step:
                continue
            entry = symbol_to_entry.get(step["kernel"])
            entry_args = entry["args"] if entry else None
            kernel_sym = step["kernel"]
            if kernel_sym in seeded_symbols:
                step["_launch_args"] = None
                continue
            seeded_symbols.add(kernel_sym)
            step["_launch_args"] = entry_args or None
            if not entry_args:
                continue
            static = step["_static_args"]
            for i, ka in enumerate(entry_args):
                if ka.get("kind") != "tensor":
                    continue
                ka_bid = ka.get("buffer_id")
                if ka_bid is None:
                    continue
                sa = static[i] if i < len(static) else None
                if isinstance(sa, str) and _BUFFER_NAME_RE.fullmatch(sa):
                    meta = table.setdefault(sa, {})
                    if "buffer_id" not in meta:
                        meta["buffer_id"] = ka_bid
                        rt_claimed.add(ka_bid)

        # Alias/view pairs share storage: propagate buffer_id bidirectionally.
        for name, meta in table.items():
            linked = meta.get("alias_of") or meta.get("view_of")
            if linked is None:
                continue
            src = table.get(linked, {})
            if "buffer_id" in meta and "buffer_id" not in src:
                src["buffer_id"] = meta["buffer_id"]
            elif "buffer_id" in src and "buffer_id" not in meta:
                meta["buffer_id"] = src["buffer_id"]

        # Shape-matching fallback for names neither traced nor linked.
        # Marked ``guessed`` because shapes are not unique on real models.
        for name, meta in table.items():
            if (
                "alias_of" in meta
                or "view_of" in meta
                or "reinterpret_of" in meta
                or "buffer_id" in meta
            ):
                continue
            bid = match_runtime(meta)
            if bid is not None:
                meta["buffer_id"] = bid
                meta["guessed"] = True
            elif meta.get("shape"):
                debug(f"pipeline buffer {name} has no runtime buffer match")

        # Unify args/output now that buffer_ids are resolved.
        for step in kept:
            if step["type"] == "triton_kernel":
                launch_args = step.pop("_launch_args", None)
                if "cubin" in step:
                    _finalize_triton_step(step, table, launch_args)
                else:
                    step.pop("_static_args", None)
            elif step["type"] == "extern_kernel":
                _finalize_extern_step(step, table)
            elif step["type"] == "as_strided":
                _finalize_as_strided_step(step, table)
        out_graphs.append({"graph": graph["graph"], "buffers": table, "steps": kept})
    return out_graphs


def _dump_parameter_buffer(buf: "BufferInfo", bundle_dir: Path) -> str:
    """Copy a device parameter buffer to a ``.bin`` file on disk."""
    drv = importlib.import_module("cuda.bindings.driver")
    fname = f"buffer_{buf.buffer_id:04d}.bin"
    np_dt = np.float16 if buf.dtype == "bfloat16" else getattr(np, buf.dtype)
    raw = np.empty(int(np.prod(buf.shape)), dtype=np_dt)
    drv.cuMemcpyDtoH(raw.ctypes.data, buf.data_ptr, raw.nbytes)
    drv.cuStreamSynchronize(drv.CUstream(0))
    (bundle_dir / fname).write_bytes(raw.tobytes())
    return fname


def _step_from_entry(entry: KernelEntry) -> dict:
    """Build a fallback pipeline step from a kernel entry.

    Used when the wrapper-codegen spy produced no graph (e.g. inductor
    served the compile from cache) so the pipeline is always populated.
    No static names exist here, but the shape matches inlined steps:
    ``args`` as descriptors plus ``output`` derived from the trace.
    """
    step = {
        "type": "triton_kernel",
        "kernel": entry["symbol"],
        "cubin": entry["cubin"],
        "device_target": entry["device_target"],
        "launch": entry["launch"],
        "args": entry["args"],
    }
    for a in entry["args"]:
        if a.get("kind") == "tensor" and a.get("direction") == "out":
            step["output"] = {"name": None, "buffer_id": a.get("buffer_id")}
            break
    return step


def _reconcile_registry_with_allocate(
    registry: list[BufferEntry],
    pipeline: list[dict],
) -> None:
    """Grow registry buffer spans to cover graph-table allocations.

    The launch trace freezes a buffer's shape/stride on first sight
    (:meth:`LaunchTraceSink.get_or_create_buffer`). When inductor splits a
    full-size allocation into per-batch views that are *consumed by kernels
    before* the full tensor ever appears as a launch arg, the trace only
    ever observes the half-size view — so ``manifest["buffers"][i]`` records
    a too-small shape. The full-size ``[2,…]`` shape exists only in the
    graph-table ``allocate`` entry (from inductor IR ``buf.get_size()``).

    For every graph-table entry with ``kind=="allocate"`` and a resolved
    ``buffer_id``, compute the allocate's storage span; if the registry
    entry for that id has a smaller span, overwrite it with the allocate's
    shape/stride (and contiguous fallback) so replay allocates enough
    storage for the views the kernels take.

    This is the only position with full information: launch-trace-side
    capture cannot observe allocations that never appear as kernel args.
    Mutates ``registry`` in place.
    """
    by_id: dict[int, BufferEntry] = {b["id"]: b for b in registry}
    for graph in pipeline:
        table = graph.get("buffers", {})
        for meta in table.values():
            if meta.get("kind") != "allocate":
                continue
            bid = meta.get("buffer_id")
            if bid is None:
                continue
            entry = by_id.get(bid)
            if entry is None:
                continue
            alloc_shape = meta.get("shape")
            if not alloc_shape:
                continue
            alloc_dtype = meta.get("dtype")
            if alloc_dtype and entry.get("dtype") != alloc_dtype:
                entry["dtype"] = alloc_dtype
            alloc_stride = meta.get("stride")
            alloc_span: int
            if alloc_stride and len(alloc_stride) == len(alloc_shape):
                s = _layout_span(alloc_shape, alloc_stride)
                alloc_span = s if s is not None else _contiguous_span(alloc_shape)
            else:
                alloc_span = _contiguous_span(alloc_shape)
            if alloc_span <= _registry_span(entry):
                continue
            # Overwrite: registry must hold at least the allocate span.
            entry["shape"] = [int(x) for x in alloc_shape]
            if alloc_stride and len(alloc_stride) == len(alloc_shape):
                entry["stride"] = [int(x) for x in alloc_stride]
            elif "stride" in entry:
                # Old stride was for a smaller shape; drop it so replay
                # falls back to a contiguous view of the new shape.
                del entry["stride"]


def write_kernel_bundle(
    directory: Path,
    type_name: str,
    kernels: list[CompiledKernelInfo],
    module_io: dict,
    module_meta: dict,
    launch_trace: "LaunchTraceSink | None" = None,
    wrapper_text: str | None = None,
    wrapper_graph: list[dict] | None = None,
) -> Path:
    """Write the kernel bundle sidecar directory.

    Creates ``<directory>/<legalized(type_name)>.kernels/`` containing:
      - manifest.json (embeds the ``pipeline`` execution graph when
        wrapper_graph is set)
      - kernel_NNNN.cubin for each captured kernel
      - buffer_NNNN.bin for each parameter buffer (when launch_trace is set)
      - source_debug.py (when wrapper_text is set): the inductor wrapper's
        ``def call()`` source, kept for debugging only

    When ``launch_trace`` is provided (a :class:`LaunchTraceSink`), the
    manifest is enriched with per-kernel ``args``, ``captured_grid``, and a
    top-level ``buffers`` array — enough for a torch-free replay.

    Args:
        directory: parent directory (typically external_directory).
        type_name: module spec type_name (e.g. "Attention:0").
        kernels: list of captured kernels.
        module_io: {"inputs": [...], "outputs": [...]} mirroring the ONNX
            function. Serialized as the `io` key in the manifest JSON.
        module_meta: provenance dict with python_class / torch_version / etc.
        launch_trace: optional LaunchTraceSink from capture_launch_trace.
        wrapper_text: optional inductor wrapper source (debug artifact).
        wrapper_graph: optional structured wrapper execution graph from
            ``capture_wrapper_lines``; becomes the manifest's ``pipeline``
            (when absent, a fallback pipeline is synthesized from the
            kernel entries so ``pipeline`` is always populated).

    Returns:
        The Path to the created bundle directory.
    """
    bundle_name = legalize_path_name(f"{type_name}.kernels")
    bundle_dir = Path(directory) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    symbol_to_trace: dict[str, LaunchTraceEntry] = {}
    winning_keys: set[str] | None = None
    winning_hashes: set[str] = set()
    if launch_trace is not None:
        for e in launch_trace.entries:
            symbol_to_trace[e.symbol] = e
        winning_keys = {
            f"{e.symbol}|{e.shared_mem}|{e.num_warps}" for e in launch_trace.entries
        }
        winning_hashes = {e.kernel_hash for e in launch_trace.entries if e.kernel_hash}

    entries: list[KernelEntry] = []
    for k in kernels:
        key = (
            f"{k['symbol']}|{k['launch']['shared_mem_bytes']}"
            f"|{k['launch']['num_warps']}"
        )

        # Autotune: keep only the variant that actually launched. Hash match
        # is exact (variants can share symbol+warps+shared); fall back to
        # the coarser key when no hashes were captured.
        if winning_hashes:
            if k.get("src_hash") not in winning_hashes:
                continue
        elif winning_keys is not None and key not in winning_keys:
            continue

        cubin_idx = len(entries)
        cubin_filename = f"kernel_{cubin_idx:04d}.cubin"
        (bundle_dir / cubin_filename).write_bytes(k["cubin_bytes"])

        stem = cubin_filename.removesuffix(".cubin")
        ttir = k.get("ttir")
        if ttir:
            (bundle_dir / f"{stem}.ttir").write_text(ttir, encoding="utf-8")
        ttgir = k.get("ttgir")
        if ttgir:
            (bundle_dir / f"{stem}.ttgir").write_text(ttgir, encoding="utf-8")

        launch = dict(k["launch"])
        args = list(k["args"])

        trace: LaunchTraceEntry | None = symbol_to_trace.get(k["symbol"])
        if trace is not None:
            launch["captured_grid"] = list(trace.grid)
            launch["shared_mem_bytes"] = trace.shared_mem
            launch["num_warps"] = trace.num_warps
            launch["num_scratch_args"] = trace.num_scratch_args
            args = trace.args

        entries.append(
            KernelEntry(
                id=cubin_filename.removesuffix(".cubin"),
                cubin=cubin_filename,
                symbol=k["symbol"],
                device_target=k["device_target"],
                launch=launch,  # type: ignore[typeddict-item]
                args=args,
                variants=[],
            )
        )

    manifest = KernelBundleManifest(
        schema_version=_SCHEMA_VERSION,
        module=module_meta,
        io=module_io,
        pipeline=[],
    )

    if launch_trace is not None and launch_trace.buffers:
        buffer_entries: list[BufferEntry] = []
        for buf in launch_trace.buffers:
            entry: BufferEntry = BufferEntry(
                id=buf.buffer_id,
                kind=buf.kind,
                dtype=buf.dtype,
                shape=list(buf.shape),
            )
            if buf.stride:
                entry["stride"] = list(buf.stride)
            if buf.name:
                entry["name"] = buf.name
            if buf.kind == "parameter":
                entry["file"] = _dump_parameter_buffer(buf, bundle_dir)
            buffer_entries.append(entry)
        manifest["buffers"] = buffer_entries

    # Kernel launch payloads live inlined in triton steps (no separate list).
    if wrapper_graph:
        manifest["pipeline"] = _finalize_pipeline(
            wrapper_graph,
            entries,
            launch_trace.buffers if launch_trace is not None else [],
        )
    else:
        manifest["pipeline"] = [
            {
                "graph": None,
                "buffers": {},
                "steps": [_step_from_entry(e) for e in entries],
            }
        ]

    # Reconcile the registry against graph-table allocations: the launch
    # trace freezes shapes on first sight, which is too small when kernels
    # only ever see per-batch views of a larger allocation. The graph-table
    # allocate entry carries the true full-size shape from inductor IR.
    registry = manifest.get("buffers")
    if registry and manifest["pipeline"]:
        _reconcile_registry_with_allocate(registry, manifest["pipeline"])

    if wrapper_text:
        (bundle_dir / "source_debug.py").write_text(wrapper_text, encoding="utf-8")

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    info(f"wrote kernel bundle: {bundle_dir} ({len(entries)} kernels)")
    return bundle_dir
