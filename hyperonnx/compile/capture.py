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

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from onnxifier.logger import debug, warning

from .grid_ast import NotTranslatable, translate_grid
from .typing import CompiledKernelInfo, GPUTarget, KernelArgDescriptor, LaunchDescriptor

_DEFAULT_TARGET: GPUTarget = {"backend": "cuda", "arch": "sm_70", "warp_size": 32}


@dataclass
class CaptureSink:
    kernels: list[CompiledKernelInfo] = field(default_factory=list)
    grid_sources: dict[str, str] = field(default_factory=dict)

    def record(self, compiled_kernel: Any, target: Any = None) -> None:
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
        self.kernels.append(
            CompiledKernelInfo(
                cubin_bytes=cubin_bytes,
                symbol=name,
                device_target=gpu_target,
                launch=launch,
                args=args,
            )
        )

    def attach_grid_source(self, kernel_name: str, source: str) -> None:
        self.grid_sources[kernel_name] = source

    def record_from_listener(
        self, src: Any, metadata: dict, metadata_group: dict
    ) -> None:
        """Build a CompiledKernel from listener data and record it.

        The listener passes raw metadata dict + metadata_group (filename->path).
        We construct a CompiledKernel the same way triton.compiler.compile does,
        so the existing record() method works unchanged.
        """
        from triton.compiler import (
            CompiledKernel,  # pyright: ignore[reportMissingImports]
        )

        name = metadata.get("name", f"kernel_{len(self.kernels)}")
        # ponytail: hash is not surfaced by the listener API; our capture path
        # never reads ck.hash, so a placeholder is fine. If a future consumer
        # needs the real hash, plumb it through the listener upstream.
        try:
            ck = CompiledKernel(src, metadata_group, hash="listener_captured")
        except Exception as exc:
            warning(f"failed to construct CompiledKernel for {name}: {exc}")
            return
        target = _extract_target_from_metadata(metadata)
        self.record(ck, target=target)


def _binary_ext_for_target(target: Any) -> str:
    backend = getattr(target, "backend", None) if target else None
    if backend == "cuda":
        return "cubin"
    if backend == "hip":
        return "hsaco"
    return "cubin"


def _extract_target_from_metadata(metadata: dict) -> Any:
    """Reconstruct a GPUTarget-like object from a listener metadata dict.

    The listener serializes metadata via `namedtuple._asdict()`, which only
    converts the top level — nested namedtuples stay as namedtuples. So
    `metadata['target']` may be a dict (JSON round-tripped) or a real
    GPUTarget. Both expose the attributes `record()` needs, so we just pass
    the value through when it isn't a dict.
    """
    from types import SimpleNamespace

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
    return {
        "backend": getattr(target, "backend", "cuda"),
        "arch": getattr(target, "arch", "sm_70"),
        "warp_size": int(getattr(target, "warp_size", 32)),
    }


def _infer_args(meta: Any) -> list[KernelArgDescriptor]:
    # ponytail: v1 records minimal arg metadata. A complete args list
    # requires parsing inductor's wrapper code, deferred to v1.1.
    return []


def extract_grid_value(lam: Any, meta: dict) -> tuple[int, ...] | None:
    try:
        out: Any = lam(meta) if callable(lam) else lam
        return tuple(int(x) for x in out)
    except Exception as exc:
        debug(f"grid extraction failed: {exc}")
        return None


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
    # ponytail: grid AST extraction (compile_static_grid=False path) is a no-op
    # in v1 because the inductor wrapper-codegen hook is not yet implemented.
    # All kernels get grid_expr=null. The grid_sources dict stays empty, so the
    # post-yield translate_grid loop never runs. The translate_grid/evaluate_grid
    # functions in grid_ast.py are tested and ready for v1.1 — see spec §"Grid AST".
    from triton.knobs import compilation as kc  # pyright: ignore[reportMissingImports]

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

    def _listener(*, src, metadata, metadata_group, times, cache_hit):
        try:
            sink.record_from_listener(src, metadata, metadata_group)
        except Exception as exc:
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
            except Exception as exc:
                warning(f"grid AST failed for {name}: {exc}")
                continue
            _attach_ast_to_kernel(sink, name, ast)


def _attach_ast_to_kernel(sink: CaptureSink, name: str, ast: list[dict] | None) -> None:
    for k in sink.kernels:
        if k["symbol"] == name:
            k["launch"]["grid_expr"] = ast
            return
