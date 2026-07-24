# Selective torch.compile Module Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `compile` parameter to `export_hyper_onnx` that selectively `torch.compile`s marked modules and writes a cubin + manifest sidecar bundle next to each compiled module's ONNX function.

**Architecture:** Two new pipeline steps inserted after the existing `_export_hiera`: (1) `_collect_compiled_kernels` runs `torch.compile(module)(*args)` once per compiled module under a triton-hook context that captures `CompiledKernel` and grid values; (2) `_attach_kernel_bundle` writes the cubin files and `manifest.json` into `<external_directory>/<type_name>.kernels/`. The ONNX graph is never mutated — the bundle is a pure sidecar discovered by filename convention.

**Tech Stack:** Python 3.12+, PyTorch 2.5–2.10 (`<2.11`), Triton 3.x, onnxifier, onnxscript `<0.6.0`, stdlib `ast` / `json`.

## Global Constraints

- **Torch**: `<2.11` (2.5–2.10 verified). Do not introduce APIs that don't exist in 2.5.
- **onnxscript**: `<0.6.0`. Don't use post-0.6 IR APIs.
- **Python**: `>=3.12`. Use stdlib modules where possible (`ast`, `json`, `contextlib`).
- **Package style**: existing Apache-2.0 header on every new `.py` file (copy from `hyperonnx/utils.py`).
- **Logging**: use `from onnxifier.logger import debug, warning, error, nest` — same as `hyperonnx/hyper_export.py`.
- **Tests**: pytest, CUDA-gated via `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")`.
- **Commit title tag**: `[dev]` for feature work, per repo `AGENTS.md`. Append `Signed-off-by: AGENTS` with your agent name.
- **No emojis** in code or docs unless explicitly requested.
- **No comments** in code unless absolutely necessary for non-obvious logic.
- **Lint**: `uv run ruff check hyperonnx` and `uv run ruff format hyperonnx` before every commit.

## File Structure

```
hyperonnx/
├── compile/                      # NEW package
│   ├── __init__.py               # public re-exports
│   ├── capture.py                # triton + inductor hooks (step 3)
│   ├── grid_ast.py               # stdlib ast -> JSON AST translator (6 ops)
│   ├── bundle.py                 # manifest writer + cubin dumper (step 4)
│   └── typing.py                 # CompiledKernelInfo, GridLiteral TypedDicts
├── hyper_export.py               # MODIFY: add compile + compile_static_grid params,
│                                 #         call _collect_compiled_kernels and
│                                 #         _attach_kernel_bundle after _export_hiera
└── auto.py                       # MODIFY: pass compile + compile_static_grid through
tests/
├── expoter/
│   ├── test_compile_capture.py   # NEW (Tier 1 unit)
│   ├── test_compile_grid_ast.py  # NEW (Tier 1 unit)
│   └── test_compile_bundle.py    # NEW (Tier 1 unit)
└── test_export_compile_integration.py  # NEW (Tier 2 integration, CUDA-gated)
```

**Responsibilities:**
- `capture.py` — only hook installation and capture state. No I/O.
- `grid_ast.py` — pure translation `ast.AST -> dict | None`. No I/O, no torch imports.
- `bundle.py` — only filesystem writing and manifest serialization. Takes a typed `CompiledKernelInfo`, returns nothing.
- `typing.py` — TypedDicts shared across the three modules above.
- `hyper_export.py` orchestrates: imports from `compile/`, calls them in sequence, owns the `module_spec` mutation.

---

## Task 1: Skeleton package + shared TypedDicts

**Files:**
- Create: `hyperonnx/compile/__init__.py`
- Create: `hyperonnx/compile/typing.py`
- Test: `tests/expoter/test_compile_typing.py`

**Interfaces:**
- Produces: `CompiledKernelInfo`, `GridLiteral`, `KernelBundleManifest`, `LaunchDescriptor`, `KernelArgDescriptor` TypedDicts (see Step 3 for exact shapes).

- [ ] **Step 1: Write the failing test**

```python
# tests/expoter/test_compile_typing.py
"""Typing smoke tests for the compile package."""

from hyperonnx.compile.typing import (
    CompiledKernelInfo,
    GridLiteral,
    KernelBundleManifest,
    LaunchDescriptor,
    KernelArgDescriptor,
)


def test_typeddicts_are_importable():
    assert "cubin_bytes" in CompiledKernelInfo.__annotations__
    assert "grid_expr" in LaunchDescriptor.__annotations__
    assert "kernels" in KernelBundleManifest.__annotations__
    assert "args" in CompiledKernelInfo.__annotations__
    assert "kind" in KernelArgDescriptor.__annotations__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/expoter/test_compile_typing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperonnx.compile'`

- [ ] **Step 3: Create the package `__init__.py`**

```python
# hyperonnx/compile/__init__.py
"""Copyright (C) 2026 The HYPERONNX Authors.

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
```

- [ ] **Step 4: Write the TypedDicts**

```python
# hyperonnx/compile/typing.py
"""Copyright (C) 2026 The HYPERONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
... [same header as above]
"""

from typing import NotRequired, TypedDict


type GridAstNode = dict
type GridLiteral = tuple[int, ...] | list[int] | None


class GPUTarget(TypedDict):
    backend: str
    arch: str
    warp_size: int


class KernelArgDescriptor(TypedDict):
    kind: str  # "tensor" | "scalar"
    name: str
    dtype: str
    elem_offset: NotRequired[int]
    value: NotRequired[int]
    from_: NotRequired[dict]  # key is "from", written as from_ in python


class LaunchDescriptor(TypedDict):
    num_warps: int
    num_ctas: int
    shared_mem_bytes: int
    num_regs: int
    grid_expr: GridAstNode | None
    captured_grid: list[int]


class CompiledKernelInfo(TypedDict):
    cubin_bytes: bytes
    symbol: str
    device_target: GPUTarget
    launch: LaunchDescriptor
    args: list[KernelArgDescriptor]


class KernelEntry(TypedDict):
    id: str
    cubin: str  # filename within bundle dir
    symbol: str
    device_target: GPUTarget
    launch: LaunchDescriptor
    args: list[KernelArgDescriptor]
    variants: list


class KernelBundleManifest(TypedDict):
    schema_version: int
    module: dict
    io: dict
    kernels: list[KernelEntry]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/expoter/test_compile_typing.py -v`
Expected: PASS

- [ ] **Step 6: Lint + format**

Run: `uv run ruff format hyperonnx/compile && uv run ruff check hyperonnx/compile`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add hyperonnx/compile/__init__.py hyperonnx/compile/typing.py tests/expoter/test_compile_typing.py
git commit -m "[dev] add compile package skeleton with shared TypedDicts

Signed-off-by: AGENTS <claude>"
```

---

## Task 2: Grid AST translator

**Files:**
- Create: `hyperonnx/compile/grid_ast.py`
- Test: `tests/expoter/test_compile_grid_ast.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class NotTranslatable(Exception): pass`
  - `def translate_grid(source: str) -> list[dict] | None:` — takes the body source of `def grid_wrapper_for_<name>(meta): return (...)`, returns a list of AST nodes (one per grid dim) or `None` if anything is untranslatable.
  - `def evaluate_grid(ast_nodes: list[dict], io_shapes: dict[str, list[int | str]], meta: dict[str, int]) -> list[int]:` — runtime-side evaluator used at export for the §5 sanity check.

- [ ] **Step 1: Write the failing test (basic cdiv)**

```python
# tests/expoter/test_compile_grid_ast.py
"""Unit tests for grid AST translation and evaluation."""

import pytest

from hyperonnx.compile.grid_ast import (
    NotTranslatable,
    evaluate_grid,
    translate_grid,
)


def test_translate_cdiv_only():
    source = "return (cdiv(M, 128),)"
    ast = translate_grid(source)
    assert ast == [
        {
            "op": "cdiv",
            "a": {"op": "meta", "key": "M"},
            "b": {"op": "const", "value": 128},
        }
    ]


def test_translate_shape_subscript():
    source = "return (x.shape[0],)"
    ast = translate_grid(source)
    assert ast == [{"op": "shape_dim", "input": "x", "axis": 0}]


def test_translate_mul_chain():
    source = "return (x.shape[0] * x.shape[1],)"
    ast = translate_grid(source)
    assert ast == [
        {
            "op": "mul",
            "a": {"op": "shape_dim", "input": "x", "axis": 0},
            "b": {"op": "shape_dim", "input": "x", "axis": 1},
        }
    ]


def test_translate_unhandled_raises():
    source = "return (math.sin(x),)"
    with pytest.raises(NotTranslatable):
        translate_grid(source)


def test_translate_none_on_parse_error():
    assert translate_grid("not valid python ((") is None


def test_evaluate_cdiv():
    ast = [
        {
            "op": "cdiv",
            "a": {"op": "meta", "key": "M"},
            "b": {"op": "const", "value": 128},
        }
    ]
    assert evaluate_grid(ast, io_shapes={}, meta={"M": 1000}) == [8]


def test_evaluate_shape_dim():
    ast = [{"op": "shape_dim", "input": "x", "axis": 1}]
    assert evaluate_grid(ast, io_shapes={"x": [4, 7, 3]}, meta={}) == [7]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/expoter/test_compile_grid_ast.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperonnx.compile.grid_ast'`

- [ ] **Step 3: Implement the translator**

```python
# hyperonnx/compile/grid_ast.py
"""Copyright (C) 2026 The HYPERONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
... [same header]
"""

import ast as pyast
from collections.abc import Mapping


class NotTranslatable(Exception):
    """Raised when a grid expression cannot be translated to the v1 AST."""


_ALLOWED_BINOPS = {
    pyast.Add: None,  # not in v1 node set, but allowed in parsing
    pyast.Mult: "mul",
    pyast.FloorDiv: "floordiv",
}


def _translate_expr(node: pyast.AST) -> dict:
    if isinstance(node, pyast.Constant) and isinstance(node.value, int):
        return {"op": "const", "value": node.value}
    if isinstance(node, pyast.Name):
        return {"op": "meta", "key": node.id}
    if isinstance(node, pyast.Call):
        if not isinstance(node.func, pyast.Name):
            raise NotTranslatable(f"call target not a Name: {pyast.dump(node)}")
        fname = node.func.id
        if fname == "cdiv":
            args = [_translate_expr(a) for a in node.args]
            if len(args) != 2:
                raise NotTranslatable("cdiv needs 2 args")
            return {"op": "cdiv", "a": args[0], "b": args[1]}
        raise NotTranslatable(f"unknown call: {fname}")
    if isinstance(node, pyast.BinOp):
        op_kind = type(node.op)
        if op_kind not in _ALLOWED_BINOPS:
            raise NotTranslatable(f"unsupported BinOp: {op_kind.__name__}")
        op_name = _ALLOWED_BINOPS[op_kind]
        if op_name is None:
            raise NotTranslatable(f"binop {op_kind.__name__} not in v1 node set")
        return {
            "op": op_name,
            "a": _translate_expr(node.left),
            "b": _translate_expr(node.right),
        }
    if isinstance(node, pyast.Subscript):
        # Match x.shape[N] or x.size(N)
        if isinstance(node.value, pyast.Attribute) and node.value.attr in (
            "shape",
            "size",
        ):
            base = node.value.value
            if not isinstance(base, pyast.Name):
                raise NotTranslatable("shape base not Name")
            idx = node.slice
            if isinstance(idx, pyast.Constant) and isinstance(idx.value, int):
                return {"op": "shape_dim", "input": base.id, "axis": idx.value}
        raise NotTranslatable(f"unsupported subscript: {pyast.dump(node)}")
    raise NotTranslatable(f"unsupported expr: {type(node).__name__}")


def translate_grid(source: str) -> list[dict] | None:
    """Translate the body of a grid lambda into the v1 AST.

    Args:
        source: Python source containing a `return (expr, ...)` statement.

    Returns:
        A list of AST node dicts, one per grid dim, or None on parse error.
        Raises NotTranslatable if a node is outside the v1 set.
    """
    try:
        tree = pyast.parse(source)
    except SyntaxError:
        return None
    returns = [n for n in pyast.walk(tree) if isinstance(n, pyast.Return)]
    if len(returns) != 1:
        return None
    value = returns[0].value
    if value is None:
        return None
    if isinstance(value, pyast.Tuple):
        elems = value.elts
    else:
        elems = [value]
    return [_translate_expr(e) for e in elems]


def evaluate_grid(
    ast_nodes: list[dict],
    io_shapes: Mapping[str, list[int | str]],
    meta: Mapping[str, int],
) -> list[int]:
    """Evaluate a translated AST against concrete shapes and meta values."""

    def _ev(node: dict) -> int:
        op = node["op"]
        if op == "const":
            return int(node["value"])
        if op == "meta":
            return int(meta[node["key"]])
        if op == "shape_dim":
            return int(io_shapes[node["input"]][node["axis"]])
        if op == "mul":
            return _ev(node["a"]) * _ev(node["b"])
        if op == "floordiv":
            return _ev(node["a"]) // _ev(node["b"])
        if op == "cdiv":
            a = _ev(node["a"])
            b = _ev(node["b"])
            return (a + b - 1) // b
        raise ValueError(f"unknown op: {op}")

    return [_ev(n) for n in ast_nodes]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/expoter/test_compile_grid_ast.py -v`
Expected: all 7 PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff format hyperonnx/compile/grid_ast.py && uv run ruff check hyperonnx/compile/grid_ast.py`

- [ ] **Step 6: Commit**

```bash
git add hyperonnx/compile/grid_ast.py tests/expoter/test_compile_grid_ast.py
git commit -m "[dev] add grid AST translator with 6-op node set

Signed-off-by: AGENTS <claude>"
```

---

## Task 3: Triton + inductor hook capture context

**Files:**
- Create: `hyperonnx/compile/capture.py`
- Test: `tests/expoter/test_compile_capture.py`

**Interfaces:**
- Consumes: `translate_grid` from Task 2.
- Produces:
  - `@contextmanager def capture_compiled_kernels(static_grid: bool = False):` — yields a `CaptureSink` with attributes `.kernels: list[dict]` (each dict has keys `name`, `cubin_bytes`, `symbol`, `device_target`, `launch`, `args`) and `.grid_sources: dict[str, str]`.
  - `def extract_grid_value(lam: object, meta: dict) -> tuple[int, ...] | None:` — helper used internally by the spy; calls the grid lambda with `meta` and returns the tuple, or None on failure.

- [ ] **Step 1: Write the failing test (no GPU needed for hook shape)**

```python
# tests/expoter/test_compile_capture.py
"""Unit tests for the capture context manager (no CUDA required).

These tests monkey-patch triton.compiler.compile with a stub so the hook
logic can be tested without a real GPU. Integration with real triton
kernels is verified in the Tier 2 integration tests.
"""

from unittest.mock import MagicMock

from hyperonnx.compile.capture import CaptureSink, capture_compiled_kernels


class StubMetadata:
    def __init__(self):
        self.name = "stub_kernel_0"
        self.num_warps = 4
        self.num_ctas = 1
        self.shared = 1024
        self.num_regs = 32


class StubAsmDict(dict):
    pass


class StubCompiledKernel:
    def __init__(self):
        self.metadata = StubMetadata()
        self.asm = StubAsmDict({"cubin": b"\x00\x01\x02FAKE"})


def test_capture_sink_initially_empty():
    sink = CaptureSink()
    assert sink.kernels == []
    assert sink.grid_sources == {}


def test_capture_hook_spy_passes_through_and_records(monkeypatch):
    import triton.compiler as tc

    recorded = []
    fake_ck = StubCompiledKernel()

    def fake_compile(src, target=None, options=None, **kw):
        recorded.append(src)
        return fake_ck

    monkeypatch.setattr(tc, "compile", fake_compile)

    with capture_compiled_kernels(static_grid=True) as sink:
        result = tc.compile("kernel void foo() {}")
        assert result is fake_ck
        assert len(sink.kernels) == 1
        entry = sink.kernels[0]
        assert entry["cubin_bytes"] == b"\x00\x01\x02FAKE"
        assert entry["symbol"] == "stub_kernel_0"
        assert entry["launch"]["num_warps"] == 4
        assert entry["launch"]["grid_expr"] is None
        assert entry["device_target"]["backend"]  # filled from target or default

    assert recorded == ["kernel void foo() {}"]


def test_capture_hook_restores_original(monkeypatch):
    import triton.compiler as tc

    orig = tc.compile
    with capture_compiled_kernels(static_grid=True):
        pass
    assert tc.compile is orig


def test_capture_static_grid_skips_ast(monkeypatch):
    import triton.compiler as tc

    monkeypatch.setattr(tc, "compile", lambda *a, **k: StubCompiledKernel())
    with capture_compiled_kernels(static_grid=True) as sink:
        tc.compile("foo")
    assert all(k["launch"]["grid_expr"] is None for k in sink.kernels)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/expoter/test_compile_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperonnx.compile.capture'`

- [ ] **Step 3: Implement capture**

```python
# hyperonnx/compile/capture.py
"""Copyright (C) 2026 The HYPERONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
... [same header]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/expoter/test_compile_capture.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff format hyperonnx/compile/capture.py && uv run ruff check hyperonnx/compile/capture.py`

- [ ] **Step 6: Commit**

```bash
git add hyperonnx/compile/capture.py tests/expoter/test_compile_capture.py
git commit -m "[dev] add triton compile hook capture context manager

Signed-off-by: AGENTS <claude>"
```

---

## Task 4: Bundle writer (manifest + cubin files)

**Files:**
- Create: `hyperonnx/compile/bundle.py`
- Test: `tests/expoter/test_compile_bundle.py`

**Interfaces:**
- Consumes: `CaptureSink` from Task 3, `CompiledKernelInfo` from Task 1.
- Produces:
  - `def write_kernel_bundle(directory: Path, type_name: str, kernels: list[CompiledKernelInfo], io: dict, module_meta: dict) -> Path:` — writes `<directory>/<type_name>.kernels/` with `manifest.json` and one `kernel_NNNN.cubin` per entry. Returns the bundle dir path.

- [ ] **Step 1: Write the failing test**

```python
# tests/expoter/test_compile_bundle.py
"""Unit tests for the bundle writer."""

import json
from pathlib import Path

from hyperonnx.compile.bundle import write_kernel_bundle
from hyperonnx.compile.typing import CompiledKernelInfo


def _fake_kernel(symbol: str) -> CompiledKernelInfo:
    return CompiledKernelInfo(
        cubin_bytes=b"\x00FAKE",
        symbol=symbol,
        device_target={"backend": "cuda", "arch": "sm_80", "warp_size": 32},
        launch={
            "num_warps": 4,
            "num_ctas": 1,
            "shared_mem_bytes": 2048,
            "num_regs": 64,
            "grid_expr": None,
            "captured_grid": [1, 2, 1],
        },
        args=[],
    )


def test_write_bundle_creates_dir_and_files(tmp_path: Path):
    kernels = [_fake_kernel("k0"), _fake_kernel("k1")]
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="Attention:0",
        kernels=kernels,
        io={
            "inputs": [{"name": "x", "dtype": "float16", "shape": [1, 8]}],
            "outputs": [],
        },
        module_meta={
            "type_name": "Attention:0",
            "python_class": "M.Attention",
            "torch_version": "2.10.0",
            "triton_version": "3.5.0",
        },
    )
    assert out == tmp_path / "Attention:0.kernels"
    assert out.is_dir()
    assert (out / "manifest.json").is_file()
    assert (out / "kernel_0000.cubin").read_bytes() == b"\x00FAKE"
    assert (out / "kernel_0001.cubin").read_bytes() == b"\x00FAKE"


def test_manifest_is_well_formed(tmp_path: Path):
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="A",
        kernels=[_fake_kernel("k0")],
        io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
    )
    data = json.loads((out / "manifest.json").read_text())
    assert data["schema_version"] == 1
    assert data["module"]["type_name"] == "A"
    assert data["kernels"][0]["id"] == "kernel_0000"
    assert data["kernels"][0]["cubin"] == "kernel_0000.cubin"
    assert data["kernels"][0]["variants"] == []


def test_bundle_dir_is_legalized_for_unsafe_chars(tmp_path: Path):
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="A/B:0",
        kernels=[],
        io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A/B:0",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
    )
    assert out.exists()


def test_grid_expr_serializes_when_present(tmp_path: Path):
    k = _fake_kernel("k0")
    k["launch"]["grid_expr"] = [{"op": "const", "value": 1}]
    out = write_kernel_bundle(
        directory=tmp_path,
        type_name="A",
        kernels=[k],
        io={"inputs": [], "outputs": []},
        module_meta={
            "type_name": "A",
            "python_class": "X",
            "torch_version": "1",
            "triton_version": "1",
        },
    )
    data = json.loads((out / "manifest.json").read_text())
    assert data["kernels"][0]["launch"]["grid_expr"] == [{"op": "const", "value": 1}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/expoter/test_compile_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the writer**

```python
# hyperonnx/compile/bundle.py
"""Copyright (C) 2026 The HYPERONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
... [same header]
"""

import json
from pathlib import Path

from onnxifier.logger import info
from onnxifier.utils import legalize_path_name

from .typing import CompiledKernelInfo, KernelBundleManifest, KernelEntry


_SCHEMA_VERSION = 1


def write_kernel_bundle(
    directory: Path,
    type_name: str,
    kernels: list[CompiledKernelInfo],
    io: dict,
    module_meta: dict,
) -> Path:
    """Write the kernel bundle sidecar directory.

    Creates `<directory>/<legalized(type_name)>.kernels/` containing:
      - manifest.json
      - kernel_NNNN.cubin for each captured kernel

    The directory name is legalized via onnxifier.utils.legalize_path_name
    to be filesystem-safe while staying deterministic.

    Args:
        directory: parent directory (typically external_directory).
        type_name: module spec type_name (e.g. "Attention:0").
        kernels: list of captured kernels.
        io: {"inputs": [...], "outputs": [...]} mirroring the ONNX function.
        module_meta: provenance dict with python_class / torch_version / etc.

    Returns:
        The Path to the created bundle directory.
    """
    bundle_name = legalize_path_name(f"{type_name}.kernels")
    bundle_dir = Path(directory) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    entries: list[KernelEntry] = []
    for i, k in enumerate(kernels):
        cubin_filename = f"kernel_{i:04d}.cubin"
        (bundle_dir / cubin_filename).write_bytes(k["cubin_bytes"])
        entries.append(
            KernelEntry(
                id=f"kernel_{i:04d}",
                cubin=cubin_filename,
                symbol=k["symbol"],
                device_target=k["device_target"],
                launch=k["launch"],
                args=k["args"],
                variants=[],
            )
        )

    manifest = KernelBundleManifest(
        schema_version=_SCHEMA_VERSION,
        module=module_meta,
        io=io,
        kernels=entries,
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    info(f"wrote kernel bundle: {bundle_dir} ({len(entries)} kernels)")
    return bundle_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/expoter/test_compile_bundle.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff format hyperonnx/compile/bundle.py && uv run ruff check hyperonnx/compile/bundle.py`

- [ ] **Step 6: Commit**

```bash
git add hyperonnx/compile/bundle.py tests/expoter/test_compile_bundle.py
git commit -m "[dev] add kernel bundle writer (manifest.json + cubin files)

Signed-off-by: AGENTS <claude>"
```

---

## Task 5: Integrate compile path into `export_hyper_onnx`

**Files:**
- Modify: `hyperonnx/hyper_export.py` (function `export_hyper_onnx` and `_export_hiera`)
- Modify: `hyperonnx/auto.py` (function `AutoTraceMethod.export`)
- Test: `tests/test_export_compile_integration.py`

**Interfaces:**
- Consumes: `capture_compiled_kernels` from Task 3, `write_kernel_bundle` from Task 4.
- Produces: `export_hyper_onnx(..., compile=None, compile_static_grid=False, ...)`.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_export_compile_integration.py
"""Tier 2 integration tests for compile + kernel bundle export.

These require a real CUDA device + triton. Skipped otherwise.
"""

import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _Compiled(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


class _Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _Compiled()

    def forward(self, x):
        return self.child(x) * 2.0


def test_compile_produces_bundle(tmp_path: Path):
    from hyperonnx import export_hyper_onnx

    model = _Parent().cuda()
    args = (torch.randn(4, 8, device="cuda"),)
    export_hyper_onnx(
        model,
        args,
        str(tmp_path / "model.onnx"),
        hiera=[_Compiled],
        compile=[_Compiled],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    bundles = list(tmp_path.glob("*.kernels"))
    assert len(bundles) == 1
    manifest_path = bundles[0] / "manifest.json"
    assert manifest_path.is_file()
    data = json.loads(manifest_path.read_text())
    assert data["schema_version"] == 1
    assert len(data["kernels"]) >= 1
    cubin_path = bundles[0] / data["kernels"][0]["cubin"]
    assert cubin_path.stat().st_size > 0


def test_compile_subset_of_hiera(tmp_path: Path):
    from hyperonnx import export_hyper_onnx

    class _A(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class _B(torch.nn.Module):
        def forward(self, x):
            return x * 2

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _A()
            self.b = _B()

        def forward(self, x):
            return self.b(self.a(x))

    model = _M().cuda()
    args = (torch.randn(2, 4, device="cuda"),)
    export_hyper_onnx(
        model,
        args,
        str(tmp_path / "m.onnx"),
        hiera=[_A, _B],
        compile=[_A],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    a_bundles = list(tmp_path.glob("*A*.kernels"))
    b_bundles = list(tmp_path.glob("*B*.kernels"))
    assert len(a_bundles) == 1
    assert len(b_bundles) == 0


def test_compile_auto_promotes_into_hiera(tmp_path: Path):
    from hyperonnx import export_hyper_onnx

    model = _Parent().cuda()
    args = (torch.randn(2, 4, device="cuda"),)
    export_hyper_onnx(
        model,
        args,
        str(tmp_path / "m.onnx"),
        compile=[_Compiled],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    bundles = list(tmp_path.glob("*.kernels"))
    assert len(bundles) == 1


def test_compile_static_grid_skips_ast(tmp_path: Path):
    from hyperonnx import export_hyper_onnx

    model = _Parent().cuda()
    args = (torch.randn(2, 4, device="cuda"),)
    export_hyper_onnx(
        model,
        args,
        str(tmp_path / "m.onnx"),
        compile=[_Compiled],
        compile_static_grid=True,
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    bundle = next(tmp_path.glob("*.kernels"))
    data = json.loads((bundle / "manifest.json").read_text())
    assert all(k["launch"]["grid_expr"] is None for k in data["kernels"])


def test_bundle_deletion_leaves_valid_model(tmp_path: Path):
    from hyperonnx import export_hyper_onnx
    import onnx

    model = _Parent().cuda()
    args = (torch.randn(2, 4, device="cuda"),)
    out_onnx = tmp_path / "m.onnx"
    export_hyper_onnx(
        model,
        args,
        str(out_onnx),
        compile=[_Compiled],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    for bundle in tmp_path.glob("*.kernels"):
        for f in bundle.iterdir():
            f.unlink()
        bundle.rmdir()
    onnx.checker.check_model(str(out_onnx))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_export_compile_integration.py -v`
Expected: FAIL with `TypeError: export_hyper_onnx() got an unexpected keyword argument 'compile'`.

- [ ] **Step 3: Modify `export_hyper_onnx` signature**

Open `hyperonnx/hyper_export.py`. Add two new keyword params to the signature (after `hiera`):

```python
def export_hyper_onnx(  # noqa: C901
    model: Module,
    input_args: tuple,
    f: str | PathLike | BytesIO,
    *,
    kwargs: dict[str, AnyTensor] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    opset_version: int = ONNXIFIER_OPSET.version,
    dynamo: bool = False,
    external_data: bool = False,
    hiera: Collection[type[Module]] | None = None,
    compile: Collection[type[Module]] | None = None,          # NEW
    compile_static_grid: bool = False,                        # NEW
    module_spec: dict[Module, ModuleSpec] | None = None,
    do_optimization: bool = True,
    fold_nodes_to_functions: bool = True,
    fuse_constants_to_function: bool = True,
    external_directory: str | PathLike | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    **_: Any,
) -> Any | None:
```

Add the auto-promote logic near the top of the body (right after `logger = nest(model_typename)`):

```python
    if compile:
        hiera = set(hiera or []) | set(compile)
```

- [ ] **Step 4: Add the new pipeline steps after `_export_hiera`**

In the body of `export_hyper_onnx`, after the existing `_export_hiera(...)` call, add:

```python
    if compile:
        _collect_and_attach_kernels(
            model=model,
            compile=compile,
            module_spec=module_spec,
            external_directory=external_directory,
            compile_static_grid=compile_static_grid,
            logger=logger,
        )
```

Then add the helper function near `_export_hiera`:

```python
def _collect_and_attach_kernels(
    model: Module,
    compile: Collection[type[Module]],
    module_spec: dict[Module, ModuleSpec],
    external_directory: str | PathLike | None,
    compile_static_grid: bool,
    logger: Logger,
):
    """Run torch.compile per marked module and attach kernel bundles."""
    from pathlib import Path

    import torch

    from .compile.capture import capture_compiled_kernels
    from .compile.bundle import write_kernel_bundle

    if not external_directory:
        logger.warning("compile requested but external_directory is None; skipping.")
        return
    out_dir = Path(external_directory)

    for module in model.modules():
        if type(module) not in compile:
            continue
        spec = module_spec.get(module)
        if spec is None or spec.get("status") != ExportStatus.EXPORTED:
            continue
        with capture_compiled_kernels(static_grid=compile_static_grid) as sink:
            try:
                compiled = torch.compile(module)
                compiled(*spec["args"], **(spec.get("kwargs") or {}))
            except Exception as exc:
                logger.warning(
                    f"torch.compile failed for {type(module).__name__}: {exc}"
                )
                continue
        if not sink.kernels:
            logger.warning(
                f"no kernels captured for {type(module).__name__}; "
                "module may be fully eager."
            )
            continue
        type_name = spec["type_name"]
        module_meta = {
            "type_name": type_name,
            "python_class": f"{type(module).__module__}.{type(module).__qualname__}",
            "torch_version": torch.__version__,
            "triton_version": _safe_triton_version(),
        }
        io = {
            "inputs": _spec_io(spec["args"], spec.get("kwargs", {}), spec["signature"]),
            "outputs": _spec_io_from_output(spec.get("output")),
        }
        write_kernel_bundle(
            directory=out_dir,
            type_name=type_name,
            kernels=sink.kernels,
            io=io,
            module_meta=module_meta,
        )


def _safe_triton_version() -> str:
    try:
        import triton

        return triton.__version__
    except Exception:
        return "unknown"


def _spec_io(args, kwargs, signature) -> list[dict]:
    from .exporter.utils import plain_tensor_container

    out: list[dict] = []
    params = signature.parameters
    for arg, name in zip(args, params):
        for t in plain_tensor_container(arg):
            out.append({
                "name": name,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": list(t.shape),
            })
    return out


def _spec_io_from_output(output) -> list[dict]:
    from .exporter.utils import plain_tensor_container

    if output is None:
        return []
    out: list[dict] = []
    for t in plain_tensor_container(output):
        out.append({
            "name": "output",
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
        })
    return out
```

- [ ] **Step 5: Pass `compile` + `compile_static_grid` through `AutoTraceMethod.export`**

Open `hyperonnx/auto.py`. Extend the `export` method signature and forward the args:

```python
    def export(
        self,
        f: str | PathLike | BytesIO,
        *,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        opset_version: int = 19,
        dynamo: bool = False,
        external_data: bool = False,
        hiera: Collection[type[Module]] | None = None,
        compile: Collection[type[Module]] | None = None,          # NEW
        compile_static_grid: bool = False,                        # NEW
        module_spec: dict[Module, ModuleSpec] | None = None,
        do_optimization: bool = True,
        external_directory: str | PathLike | None = None,
    ):
```

And in the `export_hyper_onnx(...)` call inside `export`, add:

```python
compile = (compile,)
compile_static_grid = (compile_static_grid,)
```

- [ ] **Step 6: Run integration tests**

Run: `uv run pytest tests/test_export_compile_integration.py -v`
Expected: 5 PASS (requires CUDA + triton). If running on CPU, the suite is skipped at module level — Tier 1 tests still cover the unit logic.

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `uv run pytest -x`
Expected: PASS.

- [ ] **Step 8: Lint + format**

Run: `uv run ruff format hyperonnx && uv run ruff check hyperonnx`

- [ ] **Step 9: Commit**

```bash
git add hyperonnx/hyper_export.py hyperonnx/auto.py tests/test_export_compile_integration.py
git commit -m "[dev] integrate compile + kernel bundle export into export_hyper_onnx

Signed-off-by: AGENTS <claude>"
```

---

## Task 6: Update README + skill doc

**Files:**
- Modify: `README.md` (add compile example section)
- Modify: `README_CN.md` (mirror the English example)
- Modify: `.claude/skills/torch-dynamo-onnx-export/SKILL.md` (add a Pattern 4 for compile)

**Interfaces:**
- Consumes: the final `export_hyper_onnx(compile=..., compile_static_grid=...)` API from Task 5.

- [ ] **Step 1: Add README section**

Open `README.md`. After the existing example (2) (auto_trace_method), add a new section:

```markdown
### 3) Export compiled modules with CUDA kernel bundle

Mark specific modules with `compile=` to `torch.compile` them during export.
A `<TypeName>.kernels/` sidecar directory is written next to each compiled
module's ONNX function, containing the cubin files and a `manifest.json`.

```python
export_hyper_onnx(
    model,
    (torch.randn(8, 768),),
    "model.onnx",
    hiera=[DecoderLayer, Attention],
    compile=[Attention],            # Attention gets a kernel bundle
    compile_static_grid=False,      # set True to skip grid AST extraction
    dynamo=True,
    external_data=True,
    external_directory="out/",
)
```

The ONNX function body remains the portable fallback. The kernel bundle is
a pure sidecar — deleting it makes the model behave as if `compile=None`.
```

- [ ] **Step 2: Mirror in README_CN.md**

Add the equivalent Chinese section under the existing README_CN structure (translate the prose; keep the code identical).

- [ ] **Step 3: Update the dynamo skill doc**

Open `.claude/skills/torch-dynamo-onnx-export/SKILL.md`. Under "Pattern 2", add "Pattern 4: Compile + kernel bundle":

```markdown
### Pattern 4: Compile + kernel bundle export

For selective torch.compile with cubin sidecar:

```python
export_hyper_onnx(
    model,
    args,
    'model.onnx',
    compile=[Attention],      # auto-promoted into hiera
    compile_static_grid=True, # skip AST extraction for fixed shapes
    dynamo=True,
)
```

How it works:
1. HyperONNX runs the existing hiera flow to build the ONNX function body.
2. For each compiled module, torch.compile is invoked under a triton hook
   that captures every CompiledKernel.
3. cubin bytes + descriptor are written to `<TypeName>.kernels/` next to
   the function ONNX. The ONNX graph is never mutated.

See `docs/superpowers/specs/2026-07-20-compile-and-kernel-export-design.md`
for the full manifest schema.
```

- [ ] **Step 4: Verify docs render**

Run: `uv run python -c "from pathlib import Path; [print(p, len(p.read_text())) for p in [Path('README.md'), Path('README_CN.md'), Path('.claude/skills/torch-dynamo-onnx-export/SKILL.md')]]"`
Expected: all three paths exist with non-zero byte counts.

- [ ] **Step 5: Commit**

```bash
git add README.md README_CN.md .claude/skills/torch-dynamo-onnx-export/SKILL.md
git commit -m "[doc] document compile + kernel bundle feature

Signed-off-by: AGENTS <claude>"
```

---

## Self-review checklist (post-write)

**Spec coverage:**
- API (`compile`, `compile_static_grid`, auto-promote) → Task 5 Step 3-4 ✓
- Pipeline steps (3 + 4) → Task 3 + Task 4 + Task 5 Step 4 ✓
- Bundle layout (flat, sidecar, naming convention) → Task 4 ✓
- Manifest schema (all fields, schema_version, variants reserved) → Task 1 + Task 4 ✓
- Grid AST (6 ops, NotTranslatable, evaluate) → Task 2 ✓
- Static bypass mode → Task 3 (`static_grid=True` param) + Task 5 ✓
- Error handling (2 hard, 6 soft) → Task 5 Step 4 (try/except around torch.compile) + Task 3 (warning on capture failure) ✓
- Testing (Tier 1 unit, Tier 2 integration, CUDA-gated) → Tasks 1-4 (Tier 1) + Task 5 (Tier 2) ✓
- Version constraints → Global Constraints header ✓

**Placeholder scan:** none. All code blocks are complete and runnable.

**Type consistency:** `CompiledKernelInfo` shape is identical between Task 1 (definition), Task 3 (produces), and Task 4 (consumes). `KernelBundleManifest` schema_version is hardcoded to `1` in Task 4 and asserted equal to `1` in Task 4 tests and Task 5 integration tests.

**Scope:** single focused feature, one plan, no decomposition needed.
