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

Shared TypedDicts for the compile / kernel-bundle subsystem.

These structures are the in-memory mirror of the JSON manifest that ships next
to each compiled ONNX function (see ``bundle.write_kernel_bundle`` and the
``Kernel bundle manifest schema`` section of the compile-and-kernel-export
design doc). They are intentionally plain ``TypedDict``\\s rather than dataclasses
so that ``json.dumps``/``json.loads`` round-trips require no adapter layer.
"""

from typing import NotRequired, TypedDict

type GridAstNode = dict
"""A single translated grid-AST node (see ``grid_ast.translate_grid``).

The runtime-implemented op set is documented in the design doc's
"Grid AST" section: ``const``, ``meta``, ``shape_dim``, ``mul``,
``floordiv``, ``cdiv``. Typed loosely as ``dict`` because the field set
varies per ``op``.
"""

type GridLiteral = tuple[int, ...] | list[int] | None
"""A concrete grid value captured at export time.

``None`` means the grid was not captured (v1 default for ``captured_grid``).
"""


class GPUTarget(TypedDict):
    """Target device a cubin was compiled for.

    Mirrors ``triton.backends.compiler.GPUTarget``. The runtime uses this to
    reject bundles compiled for an incompatible architecture (e.g. an
    ``sm_90`` cubin on an ``sm_80`` device).
    """

    backend: str  # "cuda" | "hip" | "xpu" (v1 ships cuda only)
    arch: str  # CUDA compute capability, e.g. "sm_90"
    warp_size: int  # normally 32 on CUDA


class KernelArgDescriptor(TypedDict):
    """One cubin parameter slot, language-agnostic.

    A C/C++/Rust runtime reads ``args[i]`` to know how to push kernel
    arguments via the CUDA Driver API. The ``kind`` discriminates the union:
    ``"tensor"`` arguments carry ``buffer_id`` (v1 launch-trace path) or
    ``elem_offset`` (v1.1 inductor-wrapper path); ``"scalar`` arguments
    either carry a literal ``value`` (compile-time constant) or a ``from_``
    descriptor (derived from an input's shape or another arg).

    Only ``kind`` is always present; the rest are ``NotRequired`` because the
    v1 launch trace emits minimal dicts (e.g. ``{"kind": "tensor",
    "buffer_id": n}``) and enriches them in later versions.
    """

    kind: str  # "tensor" | "scalar" | "literal"
    name: NotRequired[str]
    """Static buffer symbol (e.g. ``"buf0"`` / ``"arg0_1"``) from the
    wrapper codegen — the name half of the symbol↔index correspondence;
    ``buffer_id`` is the index half."""
    dtype: NotRequired[str]
    buffer_id: NotRequired[int]  # v1 launch-trace: device buffer id
    direction: NotRequired[str]  # "out" for write-back pointers (out_ptr*)
    shape: NotRequired[list[int]]
    """This arg's logical shape at its own launch — one pointer can host
    different views over time, so per-arg layout beats the registry's."""
    stride: NotRequired[list[int]]
    elem_offset: NotRequired[int]  # v1.1: per-slot element offset
    value: NotRequired[int | float | list | str]
    expr: NotRequired[str]
    """Original wrapper expression when the arg is a compound expression;
    ``name`` then holds the base buffer symbol."""
    from_: NotRequired[dict]  # serialized with key "from"


class LaunchDescriptor(TypedDict):
    """Physical launch constraints + grid expression for a cubin.

    Everything here except ``grid_expr``/``captured_grid`` maps 1:1 onto
    ``CUlaunchAttribute`` / ``cuLaunchKernelEx`` arguments; the runtime must
    honour them or refuse to launch.
    """

    num_warps: int
    num_ctas: int
    shared_mem_bytes: int
    num_regs: int
    grid_expr: list[GridAstNode] | None
    captured_grid: list[int] | None


class CompiledKernelInfo(TypedDict):
    """In-memory representation of one captured kernel, pre-bundle.

    This is what ``CaptureSink.record`` populates from a triton
    ``CompiledKernel``; ``write_kernel_bundle`` then serialises each entry
    into a ``KernelEntry`` on disk.
    """

    cubin_bytes: bytes
    symbol: str
    device_target: GPUTarget
    launch: LaunchDescriptor
    args: list[KernelArgDescriptor]
    src_hash: NotRequired[str]
    """Triton's full cache-key hash (``metadata["hash"]``), identifying the
    exact compiled variant. Autotune may compile several variants of one
    symbol; the launch trace records the winner's hash and the bundle
    filters losers by it."""
    ttir: NotRequired[str]
    """Triton IR (TTIR) text — the high-level kernel representation before
    GPU-specific lowering. Absent when the backend provides no IR."""
    ttgir: NotRequired[str]
    """Triton GPU IR (TTGIR) text — the GPU-specific lowered representation
    (blocks, warps, shared memory). Absent when the backend provides no IR."""


class KernelEntry(TypedDict):
    """On-disk manifest entry for one kernel.

    Same shape as :class:`CompiledKernelInfo` except ``cubin_bytes`` is
    replaced by ``cubin`` (the filename inside the bundle directory) and
    ``variants`` is reserved for v2 autotune multi-version selection.
    """

    id: str
    cubin: str  # filename within bundle dir
    symbol: str
    device_target: GPUTarget
    launch: LaunchDescriptor
    args: list[KernelArgDescriptor]
    variants: list


class BufferEntry(TypedDict):
    """One device buffer used by the kernel sequence.

    ``kind`` discriminates: ``input`` (user-provided at replay),
    ``parameter`` (loaded from ``file``), ``intermediate`` (zeroed),
    ``output`` (produced by the kernel sequence).

    To rebuild a linear contiguous buffer from a strided entry: allocate
    ``span = sum((shape[i]-1) * stride[i]) + 1`` elements, view it with
    ``as_strided(shape, stride)``, then ``contiguous()`` the view.
    """

    id: int
    kind: str
    dtype: str
    shape: list[int]
    stride: NotRequired[list[int]]
    """Memory layout at capture time (element strides). Compiled kernels
    bake this layout into their indexing, so replay must allocate with
    the same strides (e.g. channels_last conv activations). Absent means
    row-major contiguous for ``shape``."""
    name: NotRequired[str]
    file: NotRequired[str]


class KernelBundleManifest(TypedDict):
    """Top-level manifest.json schema (v2).

    v2 merged the old top-level ``kernels`` list into ``pipeline``: each
    ``triton_kernel`` step inlines the kernel launch payload (``cubin``,
    ``device_target``, ``launch``, runtime ``args``). The old
    ``vendor_lib`` key was dropped — vendor calls are the pipeline's
    ``extern_kernel`` steps. ``module`` and ``io`` are kept loosely typed
    (``dict``) because their internal structure is small, stable, and
    documented in the design doc; tightening them to TypedDicts would buy
    little and force churn on every provenance field addition.
    """

    schema_version: int
    module: dict
    io: dict
    """``{"inputs": [entry, ...], "outputs": [entry, ...]}`` mirroring the
    ONNX function signature, in declaration order. Each entry is
    ``{"name", "dtype", "shape"}`` plus an optional ``buffer_id`` linking
    it unambiguously to a :class:`BufferEntry` in ``buffers[]``. The link
    is by data_ptr (captured at launch-trace time), not position:
    ``buffers[]`` is creation-order (params → inputs → intermediates →
    outputs) and need not match io order, so positional matching would be
    ambiguous. ``buffer_id`` is absent when the tensor was never sighted
    by the trace (e.g. a CPU-only or eager-only value)."""
    pipeline: list[dict]
    """The execution pipeline, one entry per codegened graph:
    ``{"graph": name, "buffers": {...}, "steps": [...]}``. Always present.

    ``buffers`` is the definition table for every buffer name the steps
    reference — graph inputs (``kind="input"``), allocations
    (``kind="allocate"``, with shape/stride/dtype), storage aliases
    (``alias_of``) and storage views (``view_of``, see below); entries are
    cross-validated against ``buffers[]`` and carry ``buffer_id`` on match.
    Allocator reuse can map one name to different runtime ids across steps;
    the per-step ``args`` ids stay authoritative.

    Extern-kernel outputs register as ``kind="extern_out"`` with the
    output's shape/stride/dtype (aten allocates them internally, so no
    ``allocate`` step exists) — the layout contract downstream kernels'
    indexing implies.

    View resolution rule: a ``view_of`` (or ``reinterpret_of``) entry adds
    no storage of its own — follow the chain to the base entry, summing
    each hop's ``offset`` (elements), then read/write through the base's
    ``buffer_id`` storage at that accumulated offset. Same rule for
    ``as_strided`` steps whose ``output.buffer_id`` is null.

    ``steps`` are execution-ordered and noise-free
    (asserts/guards/comments/frees stripped): ``allocate``,
    ``triton_kernel`` (kernel launch payload inlined),
    ``extern_kernel`` (vendor-library calls), and ``as_strided``
    (hoisted ``reinterpret_tensor`` layout transforms). Both kernel step
    kinds share one schema: ``args`` as :class:`KernelArgDescriptor`\\s
    — tensor args carry both ``name`` (static buffer symbol) and
    ``buffer_id`` (runtime index) — plus ``output`` of the same shape
    (also carrying the output's shape/stride when known, so a linear
    buffer can be rebuilt per the :class:`BufferEntry` stride note).
    Captured from ``PythonWrapperCodegen.lines`` before stringification
    by ``capture_wrapper_lines`` — no source parsing involved.
    """
    buffers: NotRequired[list[BufferEntry]]
