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

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from onnxifier.logger import debug, info
from onnxifier.utils import legalize_path_name

from .typing import BufferEntry, CompiledKernelInfo, KernelBundleManifest, KernelEntry

if TYPE_CHECKING:
    from .capture import (
        BufferInfo,
        LaunchTraceEntry,
        LaunchTraceSink,
        VendorOpDescriptor,
    )

_SCHEMA_VERSION = 1
"""Kernel-bundle manifest schema version.

Bump when ``KernelBundleManifest`` gains a breaking field; runtimes gate
dispatch on this and must refuse unknown versions rather than guess.
"""


def _dump_parameter_buffer(buf: "BufferInfo", bundle_dir: Path) -> str:
    drv = importlib.import_module("cuda.bindings.driver")
    fname = f"buffer_{buf.buffer_id:04d}.bin"
    # bfloat16 has no numpy equivalent; fall back to float16 (matches the
    # replay-side dtype map in testing._NP_DTYPES). All other torch dtypes
    # map 1:1 onto numpy via getattr.
    np_dt = np.float16 if buf.dtype == "bfloat16" else getattr(np, buf.dtype)
    raw = np.empty(int(np.prod(buf.shape)), dtype=np_dt)
    drv.cuMemcpyDtoH(raw.ctypes.data, buf.data_ptr, raw.nbytes)
    drv.cuStreamSynchronize(drv.CUstream(0))
    (bundle_dir / fname).write_bytes(raw.tobytes())
    return fname


def write_kernel_bundle(
    directory: Path,
    type_name: str,
    kernels: list[CompiledKernelInfo],
    module_io: dict,
    module_meta: dict,
    launch_trace: "LaunchTraceSink | None" = None,
    vendor_ops: "list[VendorOpDescriptor] | None" = None,
    wrapper_text: str | None = None,
) -> Path:
    """Write the kernel bundle sidecar directory.

    Creates ``<directory>/<legalized(type_name)>.kernels/`` containing:
      - manifest.json
      - kernel_NNNN.cubin for each captured kernel
      - buffer_NNNN.bin for each parameter buffer (when launch_trace is set)

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

    Returns:
        The Path to the created bundle directory.
    """
    bundle_name = legalize_path_name(f"{type_name}.kernels")
    bundle_dir = Path(directory) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    symbol_to_trace: dict[str, LaunchTraceEntry] = {}
    winning_keys: set[str] | None = None
    if launch_trace is not None:
        for e in launch_trace.entries:
            symbol_to_trace[e.symbol] = e
        winning_keys = {
            f"{e.symbol}|{e.shared_mem}|{e.num_warps}" for e in launch_trace.entries
        }

    entries: list[KernelEntry] = []
    for k in kernels:
        key = (
            f"{k['symbol']}|{k['launch']['shared_mem_bytes']}"
            f"|{k['launch']['num_warps']}"
        )

        if winning_keys is not None and key not in winning_keys:
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
        kernels=entries,
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
            if buf.name:
                entry["name"] = buf.name
            if buf.kind == "parameter":
                entry["file"] = _dump_parameter_buffer(buf, bundle_dir)
            buffer_entries.append(entry)
        manifest["buffers"] = buffer_entries

        gaps = launch_trace.vendor_lib_gaps()
        if gaps or vendor_ops:
            vendor_entry: dict = {}
            if gaps:
                vendor_entry["unwritten_buffers"] = [b.buffer_id for b in gaps]

            # Build shape→buffer_id map for operand linkage.
            shape_to_buf: dict[tuple[tuple[int, ...], str], int] = {}
            for b in launch_trace.buffers:
                buf_key: tuple[tuple[int, ...], str] = (tuple(b.shape), b.dtype)
                if buf_key not in shape_to_buf:
                    shape_to_buf[buf_key] = b.buffer_id

            # Gap shapes — a candidate FX node is confirmed vendor-delegated
            # iff its output shape matches a gap buffer (not triton-written).
            # This replaces any op-name whitelist with ground-truth coverage.
            gap_keys: set[tuple[tuple[int, ...], str]] = {
                (tuple(b.shape), b.dtype) for b in gaps
            }

            # Filter: keep only candidates whose output is a gap buffer.
            confirmed = [d for d in (vendor_ops or []) if d.output in gap_keys]
            if confirmed:

                def _link(sd: tuple[tuple[int, ...], str]) -> dict:
                    shape, dtype = sd
                    bid = shape_to_buf.get((shape, dtype))
                    if bid is None:
                        debug(f"vendor op operand {shape} {dtype} has no buffer match")
                    return {
                        "shape": list(shape),
                        "dtype": dtype,
                        "buffer_id": bid,
                        "nbytes": int(np.prod(shape))
                        * (
                            2
                            if dtype == "bfloat16"
                            else np.dtype(np.float32 if dtype == "" else dtype).itemsize
                        ),
                    }

                vendor_entry["ops"] = [
                    {
                        "type": d.type,
                        "operands": [_link(sd) for sd in d.operands],
                        "output": _link(d.output),
                        "attrs": d.attrs,
                    }
                    for d in confirmed
                ]
            if vendor_entry:
                manifest["vendor_lib"] = vendor_entry

    if wrapper_text:
        (bundle_dir / "wrapper.py").write_text(wrapper_text, encoding="utf-8")

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    info(f"wrote kernel bundle: {bundle_dir} ({len(entries)} kernels)")
    return bundle_dir
