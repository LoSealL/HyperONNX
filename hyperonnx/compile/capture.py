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
            captured_grid=[0, 0, 0],
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


def _binary_ext_for_target(target: Any) -> str:
    backend = getattr(target, "backend", None) if target else None
    if backend == "cuda":
        return "cubin"
    if backend == "hip":
        return "hsaco"
    return "cubin"


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
        out = lam(meta) if callable(lam) else lam
        return tuple(int(x) for x in out)
    except Exception as exc:
        debug(f"grid extraction failed: {exc}")
        return None


@contextmanager
def capture_compiled_kernels(static_grid: bool = False):
    """Monkey-patch triton.compiler.compile to capture every compiled kernel.

    The patched function returns the original CompiledKernel unchanged (pure spy).
    Grid AST extraction is skipped entirely when static_grid=True.

    Args:
        static_grid: if True, leave grid_expr=None for every captured kernel.

    Yields:
        CaptureSink populated as kernels compile.
    """
    import triton.compiler as tc

    sink = CaptureSink()
    orig_compile = tc.compile

    def _spy(src, target=None, options=None, **kw):
        ck = orig_compile(src, target, options, **kw)
        try:
            sink.record(ck, target)
        except Exception as exc:
            warning(f"capture failed for kernel: {exc}")
        return ck

    tc.compile = _spy
    try:
        yield sink
    finally:
        tc.compile = orig_compile

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
