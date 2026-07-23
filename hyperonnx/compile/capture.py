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
# pylint: disable=import-outside-toplevel
# triton and torch._inductor are optional runtime dependencies (triton is
# absent on linux cpu installs — see pyproject ``[project.optional-dependencies]``),
# so they are imported lazily inside the functions that use them and
# feature-checked at call time. Do not move these to module scope.

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
from onnxifier.logger import debug, warning

from .grid_ast import NotTranslatable, translate_grid
from .typing import (
    CompiledKernelInfo,
    GPUTarget,
    KernelArgDescriptor,
    LaunchDescriptor,
)

_DEFAULT_TARGET: GPUTarget = {"backend": "cuda", "arch": "sm_70", "warp_size": 32}
"""Fallback ``GPUTarget`` when the listener does not surface one.

Used only on the ``record_from_listener`` path when the upstream metadata
dict has no ``target`` key; production runs on a real GPU always carry a
populated target, so this is a debug/CI safety net rather than a default
that ships in a real bundle.
"""


@dataclass
class CaptureSink:
    """Accumulator for kernels seen by the triton compilation listener.

    A fresh instance is created per ``capture_compiled_kernels`` context.
    ``record_from_listener`` is the entry point invoked by triton's hook;
    the recorded entries are drained by ``write_kernel_bundle`` once the
    context exits. ``grid_sources`` is the seam for the v1.1 grid-AST
    pipeline (the inductor wrapper-codegen hook that feeds it is not yet
    wired in, so it stays empty in v1).
    """

    kernels: list[CompiledKernelInfo] = field(default_factory=list)
    grid_sources: dict[str, str] = field(default_factory=dict)

    def record(self, compiled_kernel: Any, target: Any = None) -> None:
        """Extract cubin + metadata from a triton ``CompiledKernel``.

        Args:
            compiled_kernel: a ``triton.compiler.CompiledKernel`` (or any
                duck-typed object exposing ``asm`` and ``metadata``).
            target: optional ``GPUTarget``-like; falls back to
                ``_DEFAULT_TARGET`` when omitted (e.g. listener path with
                no target dict).

        Returns ``None``; on success the new ``CompiledKernelInfo`` is
        appended to :attr:`kernels`. Kernels with no capturable binary are
        logged and skipped (the ONNX function stays a valid fallback).
        """
        meta = getattr(compiled_kernel, "metadata", None)
        name = getattr(meta, "name", f"kernel_{len(self.kernels)}")
        asm = getattr(compiled_kernel, "asm", {}) or {}
        binary_ext = _binary_ext_for_target(target)
        cubin_bytes = asm.get(binary_ext, b"")
        if not cubin_bytes:
            warning(f"no {binary_ext} bytes captured for {name}")
            return
        gpu_target = _target_to_dict(target) if target else _DEFAULT_TARGET
        launch = LaunchDescriptor(
            num_warps=int(getattr(meta, "num_warps", 1)),
            num_ctas=int(getattr(meta, "num_ctas", 1)),
            shared_mem_bytes=int(getattr(meta, "shared", 0)),
            num_regs=int(getattr(meta, "num_regs", 0)),
            grid_expr=None,
            captured_grid=None,
        )
        args = _infer_args(meta)
        info = CompiledKernelInfo(
            cubin_bytes=cubin_bytes,
            symbol=name,
            device_target=gpu_target,
            launch=launch,
            args=args,
        )
        self.kernels.append(info)
        if os.environ.get("HYPERONNX_TTIR"):
            for key in ("ttir", "ttgir"):
                val = asm.get(key)
                if val:
                    info[key] = val.decode() if isinstance(val, bytes) else val

    def attach_grid_source(self, kernel_name: str, source: str) -> None:
        """Stash a grid-lambda source body for later AST translation.

        Called by the inductor wrapper-codegen hook (not yet wired in v1).
        ``capture_compiled_kernels`` drains this dict on exit, translating
        each source via ``grid_ast.translate_grid`` and attaching the
        result to the matching kernel entry.
        """
        self.grid_sources[kernel_name] = source

    def record_from_listener(
        self, src: Any, metadata: dict, metadata_group: dict
    ) -> None:
        """Build a CompiledKernel from listener data and record it.

        The listener passes raw metadata dict + metadata_group (filename->path).
        We construct a CompiledKernel the same way triton.compiler.compile does,
        so the existing record() method works unchanged.
        """
        from triton.compiler import CompiledKernel

        name = metadata.get("name", f"kernel_{len(self.kernels)}")
        # ponytail: hash is not surfaced by the listener API; our capture path
        # never reads ck.hash, so a placeholder is fine. If a future consumer
        # needs the real hash, plumb it through the listener upstream.
        try:
            ck = CompiledKernel(src, metadata_group, hash="listener_captured")
        except Exception as exc:  # pylint: disable=broad-except
            warning(f"failed to construct CompiledKernel for {name}: {exc}")
            return
        target = _extract_target_from_metadata(metadata)
        self.record(ck, target=target)


def _binary_ext_for_target(target: Any) -> str:
    """Pick the ``asm`` key holding the kernel binary for a backend.

    CUDA → ``"cubin"``, HIP → ``"hsaco"``. Any other backend defaults to
    ``"cubin"``; add a branch when a third backend ships.
    """
    backend = getattr(target, "backend", None) if target else None
    return "hsaco" if backend == "hip" else "cubin"


def _extract_target_from_metadata(metadata: dict) -> Any:
    """Reconstruct a GPUTarget-like object from a listener metadata dict.

    The listener serializes metadata via `namedtuple._asdict()`, which only
    converts the top level — nested namedtuples stay as namedtuples. So
    `metadata['target']` may be a dict (JSON round-tripped) or a real
    GPUTarget. Both expose the attributes `record()` needs, so we just pass
    the value through when it isn't a dict.
    """
    target = metadata.get("target")
    if target is None:
        return None
    if isinstance(target, dict):
        return SimpleNamespace(
            backend=target.get("backend", "cuda"),
            arch=target.get("arch", "sm_70"),
            warp_size=int(target.get("warp_size", 32)),
        )
    # Already a GPUTarget-like (namedtuple / object with attrs).
    return target


def _target_to_dict(target: Any) -> GPUTarget:
    """Coerce a triton ``GPUTarget`` (namedtuple) into the manifest TypedDict.

    ``getattr`` defaults keep capture alive when triton adds new required
    fields without bumping the listener contract.
    """
    return {
        "backend": getattr(target, "backend", "cuda"),
        "arch": getattr(target, "arch", "sm_70"),
        "warp_size": int(getattr(target, "warp_size", 32)),
    }


def _infer_args(meta: Any) -> list[KernelArgDescriptor]:  # pylint: disable=unused-argument
    # ponytail: v1 records minimal arg metadata. A complete args list
    # requires parsing inductor's wrapper code, deferred to v1.1. ``meta``
    # is kept in the signature for the v1.1 implementation; suppress the
    # unused warning until then.
    return []


def extract_grid_value(lam: Any, meta: dict) -> tuple[int, ...] | None:
    """Best-effort evaluation of a grid lambda against autotuner ``meta``.

    Used to populate ``launch.captured_grid``. Returns ``None`` on any
    exception (callable raises, non-integral result, etc.); the caller
    then leaves ``captured_grid`` null and the runtime falls back to the
    ONNX function per the static-match contract.
    """
    try:
        out: Any = lam(meta) if callable(lam) else lam
        return tuple(int(x) for x in out)
    except Exception as exc:  # pylint: disable=broad-except
        # Best-effort: any failure (callable error, non-integral result, ...)
        # means we leave captured_grid null and fall back to the ONNX function.
        debug(f"grid extraction failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Vendor-library op metadata capture (cuDNN / cuBLAS)
#
# Inductor delegates some ops to vendor libraries (conv → cuDNN, BLAS →
# cuBLAS). Those produce no cubin, so the kernel-replay harness can't cover
# them. To enable a future raw-vendor-API replay, we capture the *static*
# descriptor info (op type, operand shapes/dtypes, hyperparameters) from
# every FX compute node at the compile_fx boundary.
#
# We do NOT use an op-name whitelist to decide which nodes are vendor ops.
# Instead, we record ALL call_function nodes as candidates, then let
# buffer-write coverage (LaunchTraceSink.vendor_lib_gaps) confirm which
# ones are actually vendor-delegated: a node is vendor iff its output
# buffer is not written by any captured triton kernel. This is ground
# truth, immune to inductor version changes or new delegation heuristics.
# ---------------------------------------------------------------------------


@dataclass
class VendorOpDescriptor:
    """Static metadata for one FX compute node, captured from the graph.

    Carries enough info to build raw cuDNN/cuBLAS descriptors later:
    operand shapes+dtypes, output shape+dtype, and op-specific attrs
    (stride/padding/dilation/groups for conv; alpha/beta for GEMM).
    ``buffer_id`` linkage is filled at bundle-write time by shape-matching
    against the launch trace's buffer registry.

    Not every recorded descriptor is a vendor op — the caller filters by
    buffer coverage to keep only those whose output is not triton-written.
    """

    type: str
    operands: list[tuple[tuple[int, ...], str]]
    output: tuple[tuple[int, ...], str]
    attrs: dict[str, Any]


def _fx_node_shape_dtype(node: Any) -> tuple[tuple[int, ...], str] | None:
    """Extract ``(shape, dtype_str)`` from an FX node's fake/example value."""
    meta = getattr(node, "meta", {})
    if not meta:
        return None
    val = meta.get("example_value")
    if val is None:
        val = meta.get("val")
    if val is None:
        return None
    shape = tuple(int(s) for s in getattr(val, "shape", ()))
    dtype = str(getattr(val, "dtype", "")).replace("torch.", "")
    return (shape, dtype)


def _json_safe(v: Any) -> Any:
    """Coerce an FX literal arg to a JSON-serializable primitive.

    FX node literal args can hold non-serializable objects (``Ellipsis``
    from slice/index ops, ``torch.dtype``, ``torch.device``); these break
    ``json.dumps`` at bundle-write time. Primitives pass through; containers
    recurse; anything else becomes its ``repr``.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return repr(v)


def _extract_vendor_attrs(op: str, scalar_args: dict[int, Any]) -> dict[str, Any]:
    """Extract op-specific replay attributes from the FX node's literal args.

    Known op patterns get structured attrs (conv stride/padding/…, GEMM
    alpha/beta). Unknown ops get a generic ``scalar_args`` dict so no info
    is lost; a future replay path can interpret them by ``op`` type.
    """
    if op.startswith("conv"):
        return {
            "stride": list(scalar_args.get(3, (1, 1))),
            "padding": list(scalar_args.get(4, (0, 0))),
            "dilation": list(scalar_args.get(5, (1, 1))),
            "groups": scalar_args.get(6, 1),
        }
    if op in ("mm", "bmm", "matmul"):
        return {"alpha": 1.0, "beta": 0.0}
    if op == "addmm":
        return {
            "alpha": float(scalar_args.get(3, 1.0)),
            "beta": float(scalar_args.get(4, 1.0)),
        }
    if op == "linear":
        return {"alpha": 1.0, "beta": 1.0}
    # Generic fallback: record raw scalar args so no info is lost.
    if scalar_args:
        return {"scalar_args": dict(scalar_args)}
    return {}


@contextmanager
def capture_vendor_ops():
    """Wrap ``compile_fx`` to extract compute-node descriptors from the FX graph.

    Records :class:`VendorOpDescriptor` for **every** ``call_function`` node
    that produces a tensor output — not just a whitelist of known vendor ops.
    The caller filters these candidates by buffer-write coverage after the
    forward: a node is confirmed vendor-delegated iff its output buffer is
    not written by any captured triton kernel (see
    :meth:`LaunchTraceSink.vendor_lib_gaps`). This is ground truth, immune
    to inductor version changes.

    The descriptors carry static graph info — operand shapes/dtypes and
    op-specific hyperparameters — enough to build raw cuDNN/cuBLAS
    descriptors for a future vendor-API replay path.

    Does not modify or abort compilation; the graph compiles normally.

    Yields:
        ``list[VendorOpDescriptor]`` — all FX compute-node candidates; the
        caller filters by coverage to find actual vendor ops.
    """
    import torch._inductor.compile_fx as cfx

    orig = cfx.compile_fx
    descriptors: list[VendorOpDescriptor] = []

    def _record(gm: Any, example_inputs: Any, **kwargs: Any) -> Any:
        for n in gm.graph.nodes:
            if getattr(n, "op", None) != "call_function":
                continue
            out_sd = _fx_node_shape_dtype(n)
            if out_sd is None:
                continue
            t = n.target
            name = getattr(t, "__name__", None)
            if name is None:
                name = t if isinstance(t, str) else ""
            name = str(name)
            operands: list[tuple[tuple[int, ...], str]] = []
            scalar_args: dict[int, Any] = {}
            for i, a in enumerate(n.args):
                if hasattr(a, "meta"):
                    sd = _fx_node_shape_dtype(a)
                    if sd is not None:
                        operands.append(sd)
                else:
                    scalar_args[i] = a
            descriptors.append(
                VendorOpDescriptor(
                    type=name,
                    operands=operands,
                    output=out_sd,
                    attrs=_json_safe(_extract_vendor_attrs(name, scalar_args)),
                )
            )
        return orig(gm, example_inputs, **kwargs)

    cfx.compile_fx = _record
    try:
        yield descriptors
    finally:
        cfx.compile_fx = orig


@contextmanager
def capture_compiled_kernels(static_grid: bool = False):
    """Install a triton compilation listener to capture every compiled kernel.

    Uses triton's official `knobs.compilation.listener` hook. The listener is
    called for both cache hits and misses, with full metadata + metadata_group
    (which maps filenames like 'kernel.cubin' to local filesystem paths).

    Args:
        static_grid: if True, leave grid_expr=None for every captured kernel.

    Yields:
        CaptureSink populated as kernels compile.
    """
    from triton.knobs import compilation as kc

    # ponytail: grid AST extraction (compile_static_grid=False path) is a no-op
    # in v1 because the inductor wrapper-codegen hook is not yet implemented.
    # All kernels get grid_expr=null. The grid_sources dict stays empty, so the
    # post-yield translate_grid loop never runs. The translate_grid/evaluate_grid
    # functions in grid_ast.py are tested and ready for v1.1 — see spec §"Grid AST".
    sink = CaptureSink()
    if not hasattr(kc, "listener"):
        # ponytail: triton < 3.7 (torch < 2.10) has no compilation listener hook.
        # Compile capture is silently disabled; the ONNX function still exports.
        warning(
            "triton.knobs.compilation.listener not available "
            "(triton < 3.7 / torch < 2.10); compile capture disabled."
        )
        yield sink
        return

    orig_listener = kc.listener

    def _listener(  # pylint: disable=unused-argument
        *, src, metadata, metadata_group, times, cache_hit
    ):
        try:
            sink.record_from_listener(src, metadata, metadata_group)
        except Exception as exc:  # pylint: disable=broad-except
            warning(f"capture failed for kernel: {exc}")

    kc.listener = _listener
    try:
        yield sink
    finally:
        kc.listener = orig_listener

    if not static_grid:
        for name, source in sink.grid_sources.items():
            try:
                ast = translate_grid(source)
            except NotTranslatable as exc:
                debug(f"grid AST untranslatable for {name}: {exc}")
                continue
            except Exception as exc:  # pylint: disable=broad-except
                warning(f"grid AST failed for {name}: {exc}")
                continue
            _attach_ast_to_kernel(sink, name, ast)


def _attach_ast_to_kernel(sink: CaptureSink, name: str, ast: list[dict] | None) -> None:
    """Write a translated grid AST back onto the matching kernel entry.

    Lookup is by ``symbol`` (== ``metadata.name`` from triton). No-op when
    no kernel with that symbol was captured; the corresponding grid stays
    ``null`` and the runtime falls back to the ONNX function per contract.
    """
    for k in sink.kernels:
        if k["symbol"] == name:
            k["launch"]["grid_expr"] = ast
            return


@dataclass
class BufferInfo:
    """A device buffer seen during the launch trace.

    ``data_ptr`` is the GPU address at capture time; at replay the buffer
    is re-allocated and the mapping is by ``buffer_id``.
    """

    data_ptr: int
    kind: str
    dtype: str
    shape: tuple[int, ...]
    name: str | None = None
    buffer_id: int = -1


@dataclass
class LaunchTraceEntry:
    """One kernel launch (post-autotuning, the winning config)."""

    symbol: str
    grid: tuple[int, int, int]
    shared_mem: int
    num_warps: int
    args: list[KernelArgDescriptor]
    num_scratch_args: int = 0


class LaunchTraceSink:
    """Accumulator for launch + buffer metadata captured during one forward.

    Built by :func:`capture_launch_trace`. After the forward, call
    :meth:`identify_output` to mark the output buffer, then pass the whole
    sink to :func:`write_kernel_bundle` for serialization.
    """

    def __init__(self) -> None:
        self.entries: list[LaunchTraceEntry] = []
        self.buffers: list[BufferInfo] = []
        self._ptr_to_buf: dict[int, BufferInfo] = {}
        self._next_id = 0
        self.all_launches: list[LaunchTraceEntry] = []

    def get_or_create_buffer(
        self, data_ptr: int, dtype: str, shape: tuple[int, ...]
    ) -> int:
        """Return the buffer id for ``data_ptr``, allocating one if unseen.

        Unseen pointers are registered as ``"intermediate"`` buffers; the
        caller rewrites them to ``"output"`` later via :meth:`identify_output`.
        """
        if data_ptr not in self._ptr_to_buf:
            buf = BufferInfo(
                data_ptr=data_ptr,
                kind="intermediate",
                dtype=dtype,
                shape=shape,
                buffer_id=self._next_id,
            )
            self._ptr_to_buf[data_ptr] = buf
            self.buffers.append(buf)
            self._next_id += 1
        return self._ptr_to_buf[data_ptr].buffer_id

    def pre_register(
        self,
        data_ptr: int,
        kind: str,
        dtype: str,
        shape: tuple[int, ...],
        name: str,
    ) -> None:
        """Register a known buffer (input/parameter) before kernels fire.

        No-op if ``data_ptr`` is already registered (e.g. a parameter alias),
        so it is safe to call for every input and module parameter.
        """
        if data_ptr in self._ptr_to_buf:
            return
        buf = BufferInfo(
            data_ptr=data_ptr,
            kind=kind,
            dtype=dtype,
            shape=tuple(shape),
            name=name,
            buffer_id=self._next_id,
        )
        self._ptr_to_buf[data_ptr] = buf
        self.buffers.append(buf)
        self._next_id += 1

    def identify_output(self, output_tensor: Any) -> None:
        """Mark the buffer backing ``output_tensor`` as kind ``"output"``.

        No-op when the output is a fresh allocation whose pointer was never
        passed to a kernel (then no buffer is marked and replay raises); the
        caller is expected to pass the same tensor object the traced forward
        produced.
        """
        ptr = getattr(output_tensor, "data_ptr", lambda: None)()
        if ptr is not None and ptr in self._ptr_to_buf:
            self._ptr_to_buf[ptr].kind = "output"

    def finalize(self) -> None:
        """Deduplicate launches, keeping the last-seen config per symbol.

        Triton's autotuner may fire the same kernel several times; replay
        wants one entry per symbol (the winning config), so we walk
        ``all_launches`` in reverse and keep the first occurrence of each.
        """
        seen: set[str] = set()
        winners: list[LaunchTraceEntry] = []
        for entry in reversed(self.all_launches):
            if entry.symbol not in seen:
                seen.add(entry.symbol)
                winners.append(entry)
        winners.reverse()
        self.entries = winners

    def vendor_lib_gaps(self) -> list[BufferInfo]:
        """Return intermediate/output buffers no triton kernel wrote.

        After :meth:`finalize`, scans every captured launch's tensor args
        flagged ``direction="out"`` (inductor names write-back pointers
        ``out_ptr*``) to build the set of buffer_ids produced by triton.
        Any intermediate or output buffer outside that set was produced by
        a vendor-library call (cuDNN/cuBLAS) that bypasses the triton
        launcher — it has no cubin and stays unwritten at replay.

        Returns:
            List of :class:`BufferInfo` for the gap buffers (empty when the
            graph is fully triton-covered).
        """
        written: set[int] = set()
        for entry in self.all_launches:
            for arg in entry.args:
                if arg.get("kind") == "tensor" and arg.get("direction") == "out":
                    bid = arg.get("buffer_id")
                    if bid is not None:
                        written.add(bid)
        return [
            b
            for b in self.buffers
            if b.kind in ("intermediate", "output") and b.buffer_id not in written
        ]


@contextmanager
def capture_launch_trace(
    module: Any, input_args: tuple, input_kwargs: dict | None = None
):
    """Hook ``StaticallyLaunchedCudaKernel.run`` to capture launch metadata.

    Pre-registers module parameters and input tensors as known buffers so
    they can be classified at serialization time. Yields a
    :class:`LaunchTraceSink` populated as kernels fire.

    Args:
        module: the ``nn.Module`` being compiled (used only to enumerate
            parameters — not needed at replay time).
        input_args: positional input tensors.
        input_kwargs: keyword input tensors (optional).

    Yields:
        LaunchTraceSink with ``.entries`` (deduped launch records) and
        ``.buffers`` (all device buffers seen).

    Limitation: only triton launches via ``StaticallyLaunchedCudaKernel.run``
    are observed. Ops inductor delegates to a vendor library (cuDNN convs,
    cuBLAS matmuls) produce no cubin and no launch is captured here, so their
    output buffers stay unwritten at replay. :meth:`LaunchTraceSink.vendor_lib_gaps`
    detects such buffers after the forward by comparing write-direction arg
    names against the full buffer set. See the cubin-replay design doc
    ("Vendor-library ops are invisible").
    """
    from torch._inductor.runtime.static_triton_launcher import (
        StaticallyLaunchedCudaKernel,
    )

    sink = LaunchTraceSink()

    for name, param in module.named_parameters():
        if param.is_cuda and param.data_ptr() != 0:
            sink.pre_register(
                param.data_ptr(),
                "parameter",
                str(param.dtype).replace("torch.", ""),
                tuple(param.shape),
                name,
            )

    for i, arg in enumerate(input_args):
        if torch.is_tensor(arg) and arg.is_cuda:
            sink.pre_register(
                arg.data_ptr(),
                "input",
                str(arg.dtype).replace("torch.", ""),
                tuple(arg.shape),
                f"input_{i}",
            )
    if input_kwargs:
        for name, arg in input_kwargs.items():
            if torch.is_tensor(arg) and arg.is_cuda:
                sink.pre_register(
                    arg.data_ptr(),
                    "input",
                    str(arg.dtype).replace("torch.", ""),
                    tuple(arg.shape),
                    name,
                )

    orig_run = StaticallyLaunchedCudaKernel.run

    def spy_run(self, grid_x, grid_y, grid_z, stream, *args):
        # Inductor's launcher names write-back pointers "out_ptr*" and
        # read pointers "in_ptr*". Align arg_names to runtime args by
        # excluding declared_constexprs indices (those are compile-time
        # constants absent from the runtime arg list).
        names = list(getattr(self, "arg_names", []) or [])
        constexprs = set(getattr(self, "declared_constexprs", []) or [])
        runtime_names = [names[i] for i in range(len(names)) if i not in constexprs]

        classified: list[KernelArgDescriptor] = []
        for k, arg in enumerate(args):
            nm = runtime_names[k] if k < len(runtime_names) else ""
            if isinstance(arg, torch.Tensor):
                bid = sink.get_or_create_buffer(
                    arg.data_ptr(),
                    str(arg.dtype).replace("torch.", ""),
                    tuple(arg.shape),
                )
                entry: KernelArgDescriptor = {"kind": "tensor", "buffer_id": bid}
                if nm.startswith("out_ptr"):
                    entry["direction"] = "out"
                classified.append(entry)
            elif isinstance(arg, float):
                classified.append({"kind": "scalar", "dtype": "float32", "value": arg})
            else:
                classified.append(
                    {
                        "kind": "scalar",
                        "dtype": "int32",
                        "value": int(arg),
                    }
                )
        num_scratch = 0
        if getattr(self, "has_global_scratch", False):
            num_scratch += 1
        if getattr(self, "has_profile_scratch", False):
            num_scratch += 1

        sink.all_launches.append(
            LaunchTraceEntry(
                symbol=self.name,
                grid=(grid_x, grid_y, grid_z),
                shared_mem=int(self.shared),
                num_warps=getattr(self, "num_warps", 4),
                args=classified,
                num_scratch_args=num_scratch,
            )
        )
        return orig_run(self, grid_x, grid_y, grid_z, stream, *args)

    StaticallyLaunchedCudaKernel.run = spy_run
    try:
        yield sink
    finally:
        StaticallyLaunchedCudaKernel.run = orig_run
        sink.finalize()
