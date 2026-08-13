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

CUTLASS kernel config tuning registry.

Maps extern_kernel names to tuner functions that return CutlassConfig.
Each tuner takes (args, buffers, arch, configs) and returns the best config.
"""

import importlib
from collections.abc import Callable

_TUNER_MODULES: dict[str, tuple[str, str]] = {
    "extern_kernels.mm": (".mm", "tune_mm"),
    "extern_kernels.bmm": (".mm", "tune_mm"),
    "extern_kernels.addmm": (".mm", "tune_mm"),
    "extern_kernels.convolution": (".conv", "tune_conv"),
    "extern_kernels.cudnn_convolution": (".conv", "tune_conv"),
}


def _lazy_registry() -> dict[str, Callable]:
    """Build the registry with lazy imports."""

    reg: dict[str, Callable] = {}
    for key, (module_path, attr) in _TUNER_MODULES.items():
        try:
            mod = importlib.import_module(module_path, package=__package__)
            reg[key] = getattr(mod, attr)
        except (ImportError, AttributeError):
            pass
    return reg


_REGISTRY: dict[str, Callable] | None = None


def get_tuner(kernel_name: str) -> Callable | None:
    """Get the CUTLASS tuner for an extern kernel name.

    Returns None if the kernel is not supported.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _lazy_registry()
    return _REGISTRY.get(kernel_name)


def require_cutlass():
    """Import and return cutlass.cute, raising if unavailable."""
    try:
        import cutlass.cute as cute  # type: ignore[reportMissingImports]

        return cute
    except ImportError:
        raise RuntimeError(
            "CUTLASS CuTe DSL required. Install with: uv sync --group cutlass"
            " (Linux only)"
        ) from None
