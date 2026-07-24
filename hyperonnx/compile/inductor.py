"""Centralized access to triton and torch._inductor internals.

All triton / torch._inductor imports in the codebase go through this
module. Each accessor is cached: the first call does the import (lazily,
so CPU-only environments can still import this module without error) and
converts ``ImportError`` / ``ModuleNotFoundError`` into a readable
``RuntimeError`` that explains the missing dependency and its consequence.

Version-divergence points (``full_constexprs`` vs ``declared_constexprs``,
tuple vs str constexpr keys, listener-hook existence) are encapsulated
here so callers see one stable API across triton/torch versions.
"""

# pylint: disable=import-outside-toplevel, missing-function-docstring
# pylint: disable=raise-missing-from
# pyright: reportMissingImports=none

from functools import lru_cache
from types import ModuleType


def _missing_triton_error() -> RuntimeError:
    return RuntimeError(
        "triton is not available on this platform (CPU-only torch build "
        "or triton not installed). Kernel capture is unavailable — the "
        "model cannot be exported with compile_hier unless triton is "
        "installed. For GPU triton, install the 'triton' extra: "
        "pip install hyperonnx[triton]"
    )


def _missing_inductor_error(attr: str) -> RuntimeError:
    return RuntimeError(
        f"torch._inductor.{attr} is not available in this torch build "
        "(torch too old or CPU-only). Kernel capture is unavailable."
    )


def _attr_changed_error(path: str, hint: str = "") -> RuntimeError:
    msg = (
        f"{path} is missing — the installed torch/triton version changed "
        "this API. Kernel capture is unavailable."
    )
    if hint:
        msg += f" {hint}"
    return RuntimeError(msg)


@lru_cache(maxsize=1)
def triton_version() -> str:
    try:
        import triton
    except ImportError:
        raise _missing_triton_error()
    return triton.__version__


@lru_cache(maxsize=1)
def compiled_kernel_cls():
    try:
        from triton.compiler import CompiledKernel
    except ImportError:
        raise _missing_triton_error()
    return CompiledKernel


@lru_cache(maxsize=1)
def compilation_knobs():
    try:
        from triton.knobs import compilation as kc
    except ImportError:
        raise _missing_triton_error()
    return kc


@lru_cache(maxsize=1)
def wrapper_codegen_cls():
    try:
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    except ImportError:
        raise _missing_inductor_error("codegen.wrapper.PythonWrapperCodegen")
    if not hasattr(PythonWrapperCodegen, "run_wrapper_ir_passes"):
        raise _attr_changed_error(
            "PythonWrapperCodegen.run_wrapper_ir_passes",
            "Check torch._inductor.codegen.wrapper for the replacement method.",
        )
    return PythonWrapperCodegen


@lru_cache(maxsize=1)
def virtualized():
    try:
        from torch._inductor.virtualized import V
    except ImportError:
        raise _missing_inductor_error("virtualized.V")
    return V


def codegen_graph():
    """Return ``V.graph`` during a wrapper-codegen spy call.

    Must be called from inside ``PythonWrapperCodegen.run_wrapper_ir_passes``
    where ``V.graph`` is alive; outside codegen it is a ``NullHandler``.
    """
    return virtualized().graph


@lru_cache(maxsize=1)
def wrapper_module() -> ModuleType:
    try:
        from torch._inductor.codegen import wrapper as wc
    except ImportError:
        raise _missing_inductor_error("codegen.wrapper")
    return wc


@lru_cache(maxsize=1)
def grid_expr_cls():
    try:
        from torch._inductor.runtime.triton_heuristics import GridExpr
    except ImportError:
        raise _missing_inductor_error("runtime.triton_heuristics.GridExpr")
    if not hasattr(GridExpr, "from_meta"):
        raise _attr_changed_error(
            "GridExpr.from_meta",
            "Check torch._inductor.runtime.triton_heuristics for the API.",
        )
    return GridExpr


@lru_cache(maxsize=1)
def static_launcher_cls():
    try:
        from torch._inductor.runtime.static_triton_launcher import (
            StaticallyLaunchedCudaKernel,
        )
    except ImportError:
        raise _missing_inductor_error(
            "runtime.static_triton_launcher.StaticallyLaunchedCudaKernel"
        )
    if not hasattr(StaticallyLaunchedCudaKernel, "run"):
        raise _attr_changed_error(
            "StaticallyLaunchedCudaKernel.run",
            "Check torch._inductor.runtime.static_triton_launcher for the API.",
        )
    return StaticallyLaunchedCudaKernel


def has_listener_hook() -> bool:
    """Whether the installed triton version supports the compilation listener."""
    try:
        kc = compilation_knobs()
    except RuntimeError:
        return False
    return hasattr(kc, "listener")


def launcher_constexpr_indices(launcher) -> set[int]:
    """Return indices of constexpr args excluded from the runtime call.

    Handles both ``full_constexprs`` (newer triton) and
    ``declared_constexprs`` (older) by falling back via ``getattr``.
    """
    if full_constexprs := getattr(launcher, "full_constexprs", None):
        return set(full_constexprs)
    if declared_constexprs := getattr(launcher, "declared_constexprs", None):
        return set(declared_constexprs)
    return set()


def extract_grid_constants(src) -> dict[str, int]:
    """Extract integer constexpr values (XBLOCK etc.) from an ASTSource."""
    out: dict[str, int] = {}
    arg_names = list(getattr(getattr(src, "fn", None), "arg_names", []) or [])
    for key, val in (getattr(src, "constants", None) or {}).items():
        if isinstance(val, bool) or not isinstance(val, int):
            continue
        idx = key[0] if isinstance(key, tuple) else key
        if isinstance(idx, int) and idx < len(arg_names):
            out[arg_names[idx]] = val
        elif isinstance(idx, str):
            out[idx] = val
    return out
