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

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
from onnxifier.logger import debug, warning

from . import inductor
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
    context exits. ``grid_constants`` holds the winning config's int
    constexprs (XBLOCK etc.) per kernel symbol, extracted from the
    listener's ASTSource; :func:`attach_grid_exprs` combines them with the
    wrapper graph's ``grid_type`` to fill ``launch.grid_expr``.
    """

    kernels: list[CompiledKernelInfo] = field(default_factory=list)
    grid_constants: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(
        self, compiled_kernel: Any, target: Any = None, src_hash: Any = None
    ) -> None:
        """Extract cubin + metadata from a triton ``CompiledKernel``.

        Args:
            compiled_kernel: a ``triton.compiler.CompiledKernel`` (or any
                duck-typed object exposing ``asm`` and ``metadata``).
            target: optional ``GPUTarget``-like; falls back to
                ``_DEFAULT_TARGET`` when omitted (e.g. listener path with
                no target dict).
            src_hash: optional triton cache-key hash identifying this exact
                compiled variant; the bundle keeps only variants whose hash
                actually launched (autotune losers are dropped).

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
        if src_hash:
            info["src_hash"] = str(src_hash)
        self.kernels.append(info)
        if os.environ.get("HYPERONNX_TTIR"):
            for key in ("ttir", "ttgir"):
                val = asm.get(key)
                if val:
                    info[key] = val.decode() if isinstance(val, bytes) else val

    def record_from_listener(
        self, src: Any, metadata: dict, metadata_group: dict
    ) -> None:
        """Build a CompiledKernel from listener data and record it."""
        name = metadata.get("name", f"kernel_{len(self.kernels)}")
        # ponytail: listener API doesn't surface the real hash; placeholder
        # is fine since the capture path never reads ck.hash.
        try:
            ck = inductor.compiled_kernel_cls()(
                src, metadata_group, hash="listener_captured"
            )
        except Exception as exc:  # pylint: disable=broad-except
            warning(f"failed to construct CompiledKernel for {name}: {exc}")
            return
        target = _extract_target_from_metadata(metadata)
        self.record(ck, target=target, src_hash=metadata.get("hash"))
        self.grid_constants[name] = inductor.extract_grid_constants(src)


def _binary_ext_for_target(target: Any) -> str:
    """Pick the ``asm`` key holding the kernel binary for a backend.

    CUDA → ``"cubin"``, HIP → ``"hsaco"``. Any other backend defaults to
    ``"cubin"``; add a branch when a third backend ships.
    """
    backend = getattr(target, "backend", None) if target else None
    return "hsaco" if backend == "hip" else "cubin"


def _extract_target_from_metadata(metadata: dict) -> Any:
    """Reconstruct a GPUTarget-like object from a listener metadata dict.

    The listener may pass ``target`` as a dict (JSON round-tripped) or as
    the original namedtuple; both are handled.
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
    # ponytail: v1 records minimal arg metadata; wrapper-codegen arg
    # parsing is handled by the pipeline merge in bundle.py.
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


def _json_safe(v: Any) -> Any:
    """Coerce a literal value to a JSON-serializable primitive.

    Codegen values can hold non-serializable objects (``Ellipsis``,
    ``torch.dtype``, ``torch.device``, sympy exprs); these break
    ``json.dumps`` at bundle-write time. Primitives pass through;
    containers recurse; anything else becomes its ``repr``.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return repr(v)


# ---------------------------------------------------------------------------
# Inductor wrapper-codegen structure capture.
#
# Inductor builds wrapper source from structured ``WrapperLine`` dataclasses
# (``PythonWrapperCodegen.lines``) and only stringifies them inside
# ``_generate``. Capturing the line list before stringification yields the
# wrapper's execution graph directly — no source re-parsing.
# ---------------------------------------------------------------------------


@contextmanager
def capture_wrapper_lines():
    """Capture inductor's structured wrapper lines after codegen.

    Patches ``PythonWrapperCodegen._generate`` to serialize ``self.lines``
    after codegen completes (memory planning + kernel codegen have both
    run, so ReuseLine/ReinterpretLine with layout changes are present).
    The IR nodes and ``V.graph`` are only alive inside codegen, so
    serialization must happen in the spy, not after the context exits.
    One entry per codegened graph (subgraphs/partitions included).

    Yields:
        ``list[dict]`` — JSON-safe graphs, see :func:`serialize_wrapper_graph`.
    """
    # pylint: disable=protected-access
    PythonWrapperCodegen = inductor.wrapper_codegen_cls()
    orig_generate = PythonWrapperCodegen._generate
    captured: list[dict] = []

    def _spy_generate(self, is_inference):
        result = orig_generate(self, is_inference)
        graph = inductor.codegen_graph()
        graph_inputs = dict(getattr(graph, "graph_inputs", {}) or {})
        # FX placeholder targets carry the original parameter/input names
        # ("conv1.weight", "bn1.running_mean", ...); placeholder order
        # matches graph_inputs order, so zip them to name each arg*.
        input_sources: dict[str, str] = {}
        try:
            import torch.fx as fx  # noqa: F401

            fx_graph: fx.Graph = graph.graph  # type: ignore[assignment]
            targets = [str(n.target) for n in fx_graph.nodes if n.op == "placeholder"]
            for name, target in zip(graph_inputs, targets):
                input_sources[name] = target
        except Exception:  # pylint: disable=broad-except
            input_sources = {}
        captured.append(
            serialize_wrapper_graph(
                getattr(graph, "name", ""),
                list(self.lines),
                dict(self.args_to_buffers),
                graph_inputs,
                input_sources,
            )
        )
        return result

    PythonWrapperCodegen._generate = _spy_generate
    try:
        yield captured
    finally:
        PythonWrapperCodegen._generate = orig_generate


def _line_type_name(line: Any) -> str:
    """``KernelCallLine`` → ``kernel_call``; ``AllocateLine`` → ``allocate``."""
    name = type(line).__name__
    name = name.removesuffix("Line")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _serialize_buffer(buf: Any) -> dict | None:
    """Best-effort shape/dtype/stride extraction from an inductor IR buffer."""
    if buf is None:
        return None
    out: dict[str, Any] = {}
    for key, getter, fmt in (
        ("shape", buf.get_size, str),
        ("stride", buf.get_stride, str),
        ("dtype", buf.get_dtype, lambda d: str(d).replace("torch.", "")),
    ):
        try:
            out[key] = [fmt(s) for s in getter()] if key != "dtype" else fmt(getter())
        except Exception:  # pylint: disable=broad-except
            pass
    return out or None


def _serialize_wrapper_line(line: Any) -> dict | None:
    """Convert one ``WrapperLine`` to a JSON-safe execution-step dict.

    Returns ``None`` for plain strings (already-text fragments carry no
    structure beyond what the neighbouring structured lines provide).
    """
    if isinstance(line, str):
        return None
    wc = inductor.wrapper_module()

    if isinstance(line, wc.KernelCallLine):
        return {
            "type": "triton_kernel" if line.triton else "kernel_call",
            "kernel": line.kernel_name,
            "args": _json_safe(list(line.call_args)),
            # Transient: consumed by attach_grid_exprs.
            "grid_type": (line.inductor_meta or {}).get("grid_type"),
        }
    if isinstance(line, wc.AllocateLine):
        step = {
            "type": "allocate",
            "buffer": line.node.get_name(),
            "comm_buffer": line.comm_buffer,
        }
        meta = _serialize_buffer(line.node)
        if meta:
            step.update(meta)
        return step
    if isinstance(line, wc.ReuseLine):
        # Storage alias. When the layout differs (channels_last vs
        # contiguous), inductor codegens reinterpret_tensor — record as
        # a view step with the reused buffer's shape/stride so the
        # pipeline merge preserves the layout transform.
        step = {
            "type": "reuse",
            "source": line.node.get_name(),
            "reused_as": line.reused_as.get_name(),
        }
        try:
            old_size = list(line.node.get_size())
            old_stride = list(line.node.get_stride())
            new_size = list(line.reused_as.get_size())
            new_stride = list(line.reused_as.get_stride())
            if old_size != new_size or old_stride != new_stride:
                meta = _serialize_buffer(line.reused_as)
                if meta:
                    step.update(meta)
        except Exception:  # pylint: disable=broad-except
            pass
        return step
    if isinstance(line, wc.ReinterpretLine):
        # View-as: same storage, different shape/stride. Recorded as a
        # view edge for buffer_id propagation at bundle time.
        step = {
            "type": "view",
            "source": line.node.get_name(),
            "view": line.reused_as.get_name(),
        }
        meta = _serialize_buffer(line.reused_as)
        if meta:
            step.update(meta)
        offset = getattr(line.layout, "offset", None)
        if offset is not None:
            step["offset"] = str(offset)
        return step
    if isinstance(
        line,
        (wc.ExternKernelAllocLine, wc.ExternKernelOutLine, wc.ExternKernelMultiOutLine),
    ):
        node = line.node
        step = {"type": "extern_kernel"}
        try:
            step["kernel"] = node.get_kernel_name()
            step["output"] = node.get_name()
            step["args"] = _json_safe(list(node.codegen_args()))
            step["kwargs"] = _json_safe(list(node.codegen_kwargs()))
        except Exception:  # pylint: disable=broad-except
            step["detail"] = repr(node)
        return step
    step = {"type": _line_type_name(line)}
    detail = None
    try:
        detail = str(line)
    except Exception:  # pylint: disable=broad-except
        pass
    if detail:
        step["detail"] = detail
    return step


def serialize_wrapper_graph(
    graph_name: str,
    lines: list[Any],
    args_to_buffers: dict[str, Any],
    graph_inputs: dict[str, Any],
    input_sources: dict[str, str] | None = None,
) -> dict:
    """Serialize one codegened graph's wrapper lines into a JSON-safe dict::

        {
          "graph": graph_name,
          "buffers": {name: {shape, stride, dtype, kind?, source?}},
          "steps": [execution-ordered step dicts],
        }

    ``buffers`` is the static definition table for every name the steps
    reference: graph inputs (``kind="input"``, with ``source`` holding the
    original FX placeholder target — e.g. ``"bn1.weight"`` — when known)
    plus any call-arg buffers inductor tracked. ``allocate``/``reuse``
    steps overlay it at bundle time. ``steps[i].kernel`` for
    ``triton_kernel`` entries joins to the launch trace and triton
    listener by symbol; ``extern_kernel`` entries are the vendor-library
    calls interleaved between them.

    Must be called while inductor codegen is live (the IR nodes referenced
    by the lines deref ``V.graph``); :func:`capture_wrapper_lines` calls it
    from inside the ``_generate`` spy.
    """
    buffers: dict[str, dict] = {}
    for name, node in graph_inputs.items():
        meta = _serialize_buffer(node)
        if meta:
            buffers[name] = {**meta, "kind": "input"}
            source = (input_sources or {}).get(name)
            if source:
                buffers[name]["source"] = source
    for name, buf in args_to_buffers.items():
        if name not in buffers:
            meta = _serialize_buffer(buf)
            if meta:
                buffers[name] = meta
    steps = []
    for line in lines:
        step = _serialize_wrapper_line(line)
        if step is not None:
            steps.append(step)
    return {"graph": graph_name, "buffers": buffers, "steps": steps}


@contextmanager
def capture_compiled_kernels():
    """Install a triton compilation listener to capture every compiled kernel.

    Uses triton's official `knobs.compilation.listener` hook. The listener is
    called for both cache hits and misses, with full metadata + metadata_group
    (which maps filenames like 'kernel.cubin' to local filesystem paths).

    Yields:
        CaptureSink populated as kernels compile.
    """
    kc = inductor.compilation_knobs()

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


def attach_grid_exprs(sink: CaptureSink, wrapper_graph: list[dict]) -> None:
    """Fill ``launch.grid_expr`` on captured kernels from the wrapper graph.

    Joins the wrapper's ``triton_kernel`` steps (which carry
    ``inductor_meta.grid_type``) with the listener-captured winning config
    (:attr:`CaptureSink.grid_constants`) by kernel symbol, builds inductor's
    :class:`GridExpr` to get the per-dim grid expressions, and translates
    them via :func:`translate_grid`. Kernels whose grid is unbuildable or
    untranslatable keep ``grid_expr=None`` and fall back to the ONNX
    function per the static-match contract.

    Must run after the compile (both the listener and the wrapper spy have
    fired) and before ``write_kernel_bundle``.
    """
    GridExpr = inductor.grid_expr_cls()

    grid_types: dict[str, str] = {}
    for graph in wrapper_graph:
        for step in graph["steps"]:
            if step.get("type") != "triton_kernel":
                continue
            grid_type = step.get("grid_type")
            if grid_type and step.get("kernel") not in grid_types:
                grid_types[step["kernel"]] = grid_type

    for k in sink.kernels:
        name = k["symbol"]
        grid_type = grid_types.get(name)
        cfg = sink.grid_constants.get(name)
        if not grid_type or not cfg:
            continue
        try:
            grid = GridExpr.from_meta({"grid_type": grid_type}, dict(cfg))
        except Exception as exc:  # pylint: disable=broad-except
            debug(f"grid build failed for {name}: {exc}")
            continue
        if grid.prefix:
            debug(f"grid for {name} has prefix assignments; leaving null")
            continue
        source = f"return ({grid.x_grid}, {grid.y_grid}, {grid.z_grid})"
        try:
            ast = translate_grid(source)
        except NotTranslatable as exc:
            debug(f"grid AST untranslatable for {name}: {exc}")
            continue
        except Exception as exc:  # pylint: disable=broad-except
            warning(f"grid AST failed for {name}: {exc}")
            continue
        k["launch"]["grid_expr"] = ast


@dataclass
class BufferInfo:
    """A device buffer seen during the launch trace.

    ``data_ptr`` is the GPU address at capture time; at replay the buffer
    is re-allocated and the mapping is by ``buffer_id``. ``stride`` is
    recorded because compiled kernels bake the capture-time memory layout
    (e.g. channels_last conv activations) into their indexing — replay
    must allocate with the same strides.
    """

    data_ptr: int
    kind: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...] = ()
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
    kernel_hash: str | None = None
    """Triton cache-key hash of the variant that actually launched — the
    winner fingerprint used to drop autotune loser cubins at bundle time."""


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
        self,
        data_ptr: int,
        dtype: str,
        shape: tuple[int, ...],
        stride: tuple[int, ...] = (),
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
                stride=stride,
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
        stride: tuple[int, ...] = (),
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
        ``out_ptr*`` / in-place ``in_out_ptr*``) to build the set of
        buffer_ids produced by triton.
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
    StaticallyLaunchedCudaKernel = inductor.static_launcher_cls()

    sink = LaunchTraceSink()

    # Parameters and persistent buffers (BN running_mean/var/
    # num_batches_tracked) — both are required replay inputs.
    for name, param in [
        *module.named_parameters(),
        *module.named_buffers(),
    ]:
        if param.is_cuda and param.data_ptr() != 0:
            sink.pre_register(
                param.data_ptr(),
                "parameter",
                str(param.dtype).replace("torch.", ""),
                tuple(param.shape),
                name,
                tuple(param.stride()),
            )

    for i, arg in enumerate(input_args):
        if torch.is_tensor(arg) and arg.is_cuda:
            sink.pre_register(
                arg.data_ptr(),
                "input",
                str(arg.dtype).replace("torch.", ""),
                tuple(arg.shape),
                f"input_{i}",
                tuple(arg.stride()),
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
                    tuple(arg.stride()),
                )

    orig_run = StaticallyLaunchedCudaKernel.run

    def spy_run(self, grid_x, grid_y, grid_z, stream, *args):
        # Inductor names write-back pointers "out_ptr*" (fresh outputs),
        # in-place "in_out_ptr*", read "in_ptr*". The static launcher's
        # runtime arg list excludes ``full_constexprs`` (a superset of
        # ``declared_constexprs``), so names must be aligned with the
        # same set.
        names = list(getattr(self, "arg_names", []) or [])
        constexprs = inductor.launcher_constexpr_indices(self)
        runtime_names = [names[i] for i in range(len(names)) if i not in constexprs]
        if len(runtime_names) != len(args):
            debug(
                f"{getattr(self, 'name', '?')}: arg_names/runtime args count "
                f"differs ({len(runtime_names)} vs {len(args)}); "
                "direction flags may be incomplete"
            )

        classified: list[KernelArgDescriptor] = []
        for k, arg in enumerate(args):
            nm = runtime_names[k] if k < len(runtime_names) else ""
            if isinstance(arg, torch.Tensor):
                bid = sink.get_or_create_buffer(
                    arg.data_ptr(),
                    str(arg.dtype).replace("torch.", ""),
                    tuple(arg.shape),
                    tuple(arg.stride()),
                )
                entry: KernelArgDescriptor = {
                    "kind": "tensor",
                    "buffer_id": bid,
                    # Per-arg layout: one pointer hosts different views over time.
                    "shape": [int(s) for s in arg.shape],
                    "stride": [int(s) for s in arg.stride()],
                }
                if nm.startswith(("out_ptr", "in_out_ptr")):
                    entry["direction"] = "out"
                classified.append(entry)
            elif isinstance(arg, float):
                classified.append({"kind": "scalar", "dtype": "float32", "value": arg})
            else:
                classified.append({
                    "kind": "scalar",
                    "dtype": "int32",
                    "value": int(arg),
                })
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
                kernel_hash=getattr(self, "hash", None),
            )
        )
        return orig_run(self, grid_x, grid_y, grid_z, stream, *args)

    StaticallyLaunchedCudaKernel.run = spy_run
    try:
        yield sink
    finally:
        StaticallyLaunchedCudaKernel.run = orig_run
        sink.finalize()
