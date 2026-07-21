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
# triton is an optional runtime dependency (win32 in the cpu extra, linux via
# the cuda/xpu extras). Imports are lazy and feature-checked at runtime.

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from onnxifier.logger import warning

from .typing import CompiledKernelInfo, GPUTarget, LaunchDescriptor

_DEFAULT_TARGET: GPUTarget = {"backend": "cuda", "arch": "sm_70", "warp_size": 32}


@dataclass
class CaptureSink:
    kernels: list[CompiledKernelInfo] = field(default_factory=list)

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
        )
        self.kernels.append(
            CompiledKernelInfo(
                cubin_bytes=cubin_bytes,
                symbol=name,
                device_target=gpu_target,
                launch=launch,
            )
        )

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
        except Exception as exc:
            warning(f"failed to construct CompiledKernel for {name}: {exc}")
            return
        target = _extract_target_from_metadata(metadata)
        self.record(ck, target=target)


def _binary_ext_for_target(target: Any) -> str:
    backend = getattr(target, "backend", None) if target else None
    return "hsaco" if backend == "hip" else "cubin"


def _extract_target_from_metadata(metadata: dict) -> Any:
    """Return the raw target from listener metadata, or None if absent.

    The listener serializes metadata via `namedtuple._asdict()`, which only
    converts the top level — nested namedtuples stay as namedtuples. We pass
    the value through; `_target_to_dict` duck-types dict vs object.
    """
    return metadata.get("target")


def _target_to_dict(target: Any) -> GPUTarget:
    get = target.get if isinstance(target, dict) else lambda k, d: getattr(target, k, d)
    return {
        "backend": get("backend", "cuda"),
        "arch": get("arch", "sm_70"),
        "warp_size": int(get("warp_size", 32)),
    }


@contextmanager
def capture_compiled_kernels():
    """Install a triton compilation listener to capture every compiled kernel.

    Uses triton's official `knobs.compilation.listener` hook. The listener is
    called for both cache hits and misses, with full metadata + metadata_group
    (which maps filenames like 'kernel.cubin' to local filesystem paths).

    Yields:
        CaptureSink populated as kernels compile.
    """
    from triton.knobs import compilation as kc

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
