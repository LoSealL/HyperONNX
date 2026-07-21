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
design doc). They are intentionally plain ``TypedDict``\s rather than dataclasses
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
    ``"tensor"`` arguments carry ``elem_offset``; ``"scalar`` arguments
    either carry a literal ``value`` (compile-time constant) or a ``from_``
    descriptor (derived from an input's shape or another arg).
    """

    kind: str  # "tensor" | "scalar"
    name: str
    dtype: str
    elem_offset: NotRequired[int]
    value: NotRequired[int]
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


class KernelBundleManifest(TypedDict):
    """Top-level manifest.json schema.

    ``module`` and ``io`` are kept loosely typed (``dict``) because their
    internal structure is small, stable, and documented in the design doc;
    tightening them to TypedDicts would buy little and force churn on every
    provenance field addition.
    """

    schema_version: int
    module: dict
    io: dict
    kernels: list[KernelEntry]
