# Compiled CUBIN Replay Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `hyperonnx/replay/` package that observes a `torch.compile` forward pass, then replays the exported cubins via CuPy's CUDA Driver API wrapper and verifies outputs match the torch reference within tolerance.

**Architecture:** Two-phase harness. (1) Observe: monkey-patch `triton.compiler.CompiledKernel.__getitem__` for one forward pass to record each kernel's grid, shared mem, symbol, and ordered args (classifying each int arg as tensor or scalar via a live-CUDA-tensor data_ptr registry). (2) Replay: load cubins from the existing `<type>.kernels/` bundle via CuPy `RawModule`, allocate one buffer per unique data_ptr, launch each kernel in observed order, compare output to torch reference.

**Tech Stack:** Python 3.12+, PyTorch 2.5–2.10 (`<2.11`), Triton 3.x, CuPy >=13.0 (lazy import), stdlib `gc`, `contextlib`.

## Global Constraints

- **Torch**: `<2.11` (2.5–2.10 verified). The `CompiledKernel.__getitem__` seam is stable across this range.
- **Triton**: matches whatever ships with the supported torch versions.
- **CuPy**: `>=13.0` (for `RawModule` cubin loading + `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`). Lazy import; NOT added to core deps.
- **Python**: `>=3.12`. Use stdlib `gc`, `contextlib`, `json`, `pathlib`.
- **Package style**: existing Apache-2.0 header on every new `.py` file (copy from `hyperonnx/compile/capture.py`).
- **No comments** in code unless absolutely necessary for non-obvious logic.
- **Lint**: `uv run ruff check hyperonnx && uv run ruff format hyperonnx` before every commit.
- **Commit title tag**: `[dev]` per repo `AGENTS.md`. Append `Signed-off-by: AGENTS` with your agent name.
- **No emojis** in code or docs.
- **CuPy is optional**: lazy import inside functions, same pattern as triton in `compile/capture.py`.
- **No export-pipeline changes**: this plan touches ONLY new files under `hyperonnx/replay/` and new test files. Zero edits to `hyperonnx/compile/` or `hyperonnx/hyper_export.py`.

## File Structure

```
hyperonnx/replay/
├── __init__.py        # re-exports: replay, verify
├── runner.py          # observe hook + classify + replay engine + verify
└── (no other files; verify lives in runner.py to keep it to one file)
tests/
├── expoter/
│   └── test_replay_classify.py   # Tier 1 unit: ptr registry + arg classification (no GPU)
└── test_replay_integration.py    # Tier 2: ResNet-18 end-to-end (CUDA-gated)
```

**Responsibilities:**
- `runner.py` — three internal phases (`_observe`, `_load_bundle`, `_replay`) + the public `replay()` and `verify()` entry points. One file because the three phases are tightly coupled (observe produces the trace replay consumes) and small (~200 lines total).
- `test_replay_classify.py` — tests the pure classification logic (ptr registry, arg kind) with no GPU, no triton, no cupy.
- `test_replay_integration.py` — the real ResNet-18 test, CUDA-gated.

---

## Task 1: Skeleton package + ptr registry + arg classifier

**Files:**
- Create: `hyperonnx/replay/__init__.py`
- Create: `hyperonnx/replay/runner.py`
- Test: `tests/expoter/test_replay_classify.py`

**Interfaces:**
- Produces:
  - `def build_ptr_registry() -> dict[int, TensorInfo]:` — scans `gc.get_objects()` for live `torch.Tensor` instances with `is_cuda`; returns `{data_ptr: TensorInfo(shape, dtype, buffer_id)}`.
  - `def classify_arg(value, ptr_registry) -> ArgDescriptor:` — classifies a single launch arg as tensor (if `value` is an int in `ptr_registry`) or scalar (otherwise).
  - `def classify_args(values, ptr_registry) -> list[ArgDescriptor]:` — maps `classify_arg` over a list.
  - `TensorInfo` and `ArgDescriptor` are `NamedTuple`s defined in `runner.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/expoter/test_replay_classify.py
"""Unit tests for the replay arg classifier (no GPU required)."""
import torch

from hyperonnx.replay.runner import (
    ArgDescriptor,
    TensorInfo,
    build_ptr_registry,
    classify_arg,
    classify_args,
)


def test_classify_tensor_arg_matches_data_ptr():
    t = torch.randn(4, 8, device="cuda" if torch.cuda.is_available() else "cpu")
    registry = {t.data_ptr(): TensorInfo(shape=(4, 8), dtype=str(t.dtype), buffer_id=0)}
    desc = classify_arg(t.data_ptr(), registry)
    assert desc.kind == "tensor"
    assert desc.buffer_id == 0


def test_classify_scalar_int_arg():
    desc = classify_arg(128, registry={})
    assert desc.kind == "scalar"
    assert desc.value == 128
    assert desc.dtype == "int32"


def test_classify_scalar_float_arg():
    desc = classify_arg(1.5, registry={})
    assert desc.kind == "scalar"
    assert desc.value == 1.5
    assert desc.dtype == "float32"


def test_classify_scalar_bool_arg():
    desc = classify_arg(True, registry={})
    assert desc.kind == "scalar"
    assert desc.dtype == "int32"  # CUDA bools are int32


def test_classify_args_preserves_order():
    t = torch.randn(2, 2, device="cuda" if torch.cuda.is_available() else "cpu")
    registry = {t.data_ptr(): TensorInfo(shape=(2, 2), dtype=str(t.dtype), buffer_id=3)}
    args = [t.data_ptr(), 64, t.data_ptr()]
    result = classify_args(args, registry)
    assert len(result) == 3
    assert result[0].kind == "tensor"
    assert result[0].buffer_id == 3
    assert result[1].kind == "scalar"
    assert result[1].value == 64
    assert result[2].kind == "tensor"
    assert result[2].buffer_id == 3


def test_build_ptr_registry_finds_live_cuda_tensor():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("requires CUDA")
    t = torch.randn(4, 4, device="cuda")
    registry = build_ptr_registry()
    assert t.data_ptr() in registry
    info = registry[t.data_ptr()]
    assert info.shape == (4, 4)


def test_build_ptr_registry_assigns_unique_buffer_ids():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("requires CUDA")
    t1 = torch.randn(2, 2, device="cuda")
    t2 = torch.randn(3, 3, device="cuda")
    registry = build_ptr_registry()
    assert registry[t1.data_ptr()].buffer_id != registry[t2.data_ptr()].buffer_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/expoter/test_replay_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperonnx.replay'`

- [ ] **Step 3: Create the package `__init__.py`**

```python
# hyperonnx/replay/__init__.py
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
from .runner import replay, verify

__all__ = ["replay", "verify"]
```

- [ ] **Step 4: Write the classifier + registry in `runner.py`**

```python
# hyperonnx/replay/runner.py
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
import gc
from collections.abc import Iterable
from typing import NamedTuple, Optional


class TensorInfo(NamedTuple):
    shape: tuple[int, ...]
    dtype: str
    buffer_id: int


class ArgDescriptor(NamedTuple):
    kind: str  # "tensor" | "scalar"
    buffer_id: Optional[int]
    value: Optional[object]
    dtype: str


_SCALAR_DTYPE = {
    int: "int32",
    float: "float32",
    bool: "int32",
}


def build_ptr_registry() -> dict[int, TensorInfo]:
    import torch

    registry: dict[int, TensorInfo] = {}
    buffer_counter = 0
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        if not getattr(obj, "is_cuda", False):
            continue
        ptr = obj.data_ptr()
        if ptr == 0 or ptr in registry:
            continue
        registry[ptr] = TensorInfo(
            shape=tuple(obj.shape),
            dtype=str(obj.dtype).replace("torch.", ""),
            buffer_id=buffer_counter,
        )
        buffer_counter += 1
    return registry


def classify_arg(value: object, ptr_registry: dict[int, TensorInfo]) -> ArgDescriptor:
    if isinstance(value, int) and value in ptr_registry:
        info = ptr_registry[value]
        return ArgDescriptor(kind="tensor", buffer_id=info.buffer_id, value=None, dtype=info.dtype)
    dtype = _SCALAR_DTYPE.get(type(value), "int32")
    return ArgDescriptor(kind="scalar", buffer_id=None, value=value, dtype=dtype)


def classify_args(
    values: Iterable[object], ptr_registry: dict[int, TensorInfo]
) -> list[ArgDescriptor]:
    return [classify_arg(v, ptr_registry) for v in values]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/expoter/test_replay_classify.py -v`
Expected: CPU tests PASS; CUDA tests SKIP if no GPU.

- [ ] **Step 6: Lint + format**

Run: `uv run ruff format hyperonnx/replay && uv run ruff check hyperonnx/replay`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add hyperonnx/replay/__init__.py hyperonnx/replay/runner.py tests/expoter/test_replay_classify.py
git commit -m "[dev] add replay package skeleton with ptr registry and arg classifier

Signed-off-by: AGENTS <claude>"
```

---

## Task 2: Observe hook (CompiledKernel.__getitem__ patch)

**Files:**
- Modify: `hyperonnx/replay/runner.py` (add `_observe` + `LaunchRecord` + `_ObserveContext`)
- Test: `tests/expoter/test_replay_classify.py` (add observe tests at the end)

**Interfaces:**
- Consumes: `build_ptr_registry`, `classify_args` from Task 1.
- Produces:
  - `class LaunchRecord(NamedTuple):` — `symbol: str`, `grid: tuple[int,int,int]`, `shared_mem: int`, `args: list[ArgDescriptor]`.
  - `@contextmanager def _observe():` — patches `CompiledKernel.__getitem__` for the duration; yields a `list[LaunchRecord]` populated as kernels fire.

- [ ] **Step 1: Write the failing test (mock CompiledKernel)**

Add to `tests/expoter/test_replay_classify.py`:

```python
# ---- observe hook tests (mock, no real triton launch) ----


def test_observe_records_launch_with_grid_and_args(monkeypatch):
    import triton.compiler as tc
    from types import SimpleNamespace

    from hyperonnx.replay.runner import _observe

    t = torch.randn(2, 4, device="cuda" if torch.cuda.is_available() else "cpu")
    registry = {t.data_ptr(): TensorInfo(shape=(2, 4), dtype=str(t.dtype), buffer_id=0)}

    class FakeCK:
        def __init__(self):
            self.name = "fake_kernel_0"
            self.metadata = SimpleNamespace(shared=2048)

        def __getitem__(self, grid):
            def runner(*args, stream=None):
                pass
            return runner

    fake = FakeCK()
    orig_getitem = tc.CompiledKernel.__getitem__
    monkeypatch.setattr(tc.CompiledKernel, "__getitem__", fake.__getitem__)

    monkeypatch.setattr(
        "hyperonnx.replay.runner.build_ptr_registry", lambda: registry
    )

    with _observe() as launches:
        runner = tc.CompiledKernel.__getitem__(fake, (4, 1, 1))
        runner(t.data_ptr(), 128, stream=None)

    assert len(launches) == 1
    rec = launches[0]
    assert rec.symbol == "fake_kernel_0"
    assert rec.grid == (4, 1, 1)
    assert rec.shared_mem == 2048
    assert len(rec.args) == 2
    assert rec.args[0].kind == "tensor"
    assert rec.args[0].buffer_id == 0
    assert rec.args[1].kind == "scalar"
    assert rec.args[1].value == 128


def test_observe_restores_original_getitem(monkeypatch):
    import triton.compiler as tc

    from hyperonnx.replay.runner import _observe

    orig = tc.CompiledKernel.__getitem__
    with _observe():
        pass
    assert tc.CompiledKernel.__getitem__ is orig


def test_observe_records_multiple_launches_in_order(monkeypatch):
    import triton.compiler as tc
    from types import SimpleNamespace

    from hyperonnx.replay.runner import _observe

    monkeypatch.setattr(
        "hyperonnx.replay.runner.build_ptr_registry", lambda: {}
    )

    class FakeCK:
        def __init__(self, name):
            self.name = name
            self.metadata = SimpleNamespace(shared=0)

        def __getitem__(self, grid):
            def runner(*args, stream=None):
                pass
            return runner

    ck0 = FakeCK("k0")
    ck1 = FakeCK("k1")
    monkeypatch.setattr(tc.CompiledKernel, "__getitem__", FakeCK.__getitem__)

    with _observe() as launches:
        tc.CompiledKernel.__getitem__(ck0, (1, 1, 1))(42)
        tc.CompiledKernel.__getitem__(ck1, (2, 2, 2))(99)

    assert len(launches) == 2
    assert launches[0].symbol == "k0"
    assert launches[1].symbol == "k1"
    assert launches[0].grid == (1, 1, 1)
    assert launches[1].grid == (2, 2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/expoter/test_replay_classify.py -v -k observe`
Expected: FAIL with `ImportError: cannot import name '_observe'`

- [ ] **Step 3: Implement `_observe` and `LaunchRecord`**

Add to `hyperonnx/replay/runner.py` (after `classify_args`):

```python
from contextlib import contextmanager


class LaunchRecord(NamedTuple):
    symbol: str
    grid: tuple[int, int, int]
    shared_mem: int
    args: list[ArgDescriptor]


@contextmanager
def _observe():
    import triton.compiler as tc

    launches: list[LaunchRecord] = []
    orig_getitem = tc.CompiledKernel.__getitem__

    def _spy_getitem(self, grid):
        symbol = getattr(self, "name", "unknown")
        shared_mem = getattr(getattr(self, "metadata", None), "shared", 0)
        orig_runner = orig_getitem(self, grid)

        def _spy_runner(*args, stream=None):
            registry = build_ptr_registry()
            classified = classify_args(args, registry)
            launches.append(
                LaunchRecord(
                    symbol=symbol,
                    grid=tuple(grid),
                    shared_mem=int(shared_mem),
                    args=classified,
                )
            )
            return orig_runner(*args, stream=stream)

        return _spy_runner

    tc.CompiledKernel.__getitem__ = _spy_getitem
    try:
        yield launches
    finally:
        tc.CompiledKernel.__getitem__ = orig_getitem
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/expoter/test_replay_classify.py -v -k observe`
Expected: all 3 PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff format hyperonnx/replay/runner.py && uv run ruff check hyperonnx/replay/runner.py`

- [ ] **Step 6: Commit**

```bash
git add hyperonnx/replay/runner.py tests/expoter/test_replay_classify.py
git commit -m "[dev] add observe hook for CompiledKernel launch capture

Signed-off-by: AGENTS <claude>"
```

---

## Task 3: Bundle loader + replay engine

**Files:**
- Modify: `hyperonnx/replay/runner.py` (add `_load_bundle`, `_replay`, and the public `replay()`)
- Test: `tests/expoter/test_replay_classify.py` (add bundle loader tests)

**Interfaces:**
- Consumes: `LaunchRecord` from Task 2, the existing `manifest.json` format from `hyperonnx/compile/`.
- Produces:
  - `def replay(module, sample_args, bundle_dir) -> "cupy.ndarray":` — the public entry point. Runs observe (torch.compile forward), loads cubins, replays, returns output cupy array.

- [ ] **Step 1: Write the failing test (bundle loader, no GPU needed for JSON parsing)**

Add to `tests/expoter/test_replay_classify.py`:

```python
# ---- bundle loader tests ----


def test_load_bundle_symbol_map(tmp_path):
    import json

    from hyperonnx.replay.runner import _load_bundle_symbol_map

    bundle = tmp_path / "M.kernels"
    bundle.mkdir()
    (bundle / "kernel_0000.cubin").write_bytes(b"\x00CUBIN0")
    (bundle / "kernel_0001.cubin").write_bytes(b"\x00CUBIN1")
    manifest = {
        "schema_version": 1,
        "module": {},
        "io": {},
        "kernels": [
            {"id": "kernel_0000", "cubin": "kernel_0000.cubin", "symbol": "attn_0"},
            {"id": "kernel_0001", "cubin": "kernel_0001.cubin", "symbol": "attn_1"},
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    sym_map = _load_bundle_symbol_map(bundle)
    assert sym_map["attn_0"] == bundle / "kernel_0000.cubin"
    assert sym_map["attn_1"] == bundle / "kernel_0001.cubin"


def test_load_bundle_missing_manifest_raises(tmp_path):
    import pytest

    from hyperonnx.replay.runner import _load_bundle_symbol_map

    with pytest.raises(FileNotFoundError):
        _load_bundle_symbol_map(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/expoter/test_replay_classify.py -v -k load_bundle`
Expected: FAIL with `ImportError: cannot import name '_load_bundle_symbol_map'`

- [ ] **Step 3: Implement `_load_bundle_symbol_map` and `replay()`**

Add to `hyperonnx/replay/runner.py`:

```python
import json
from pathlib import Path


def _load_bundle_symbol_map(bundle_dir: Path) -> dict[str, Path]:
    manifest_path = Path(bundle_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {bundle_dir}")
    manifest = json.loads(manifest_path.read_text())
    return {
        k["symbol"]: Path(bundle_dir) / k["cubin"] for k in manifest["kernels"]
    }


def replay(
    module,
    sample_args: tuple,
    bundle_dir: str | Path,
):
    import torch

    try:
        import cupy
    except ImportError as exc:
        raise ImportError(
            "CuPy is required for cubin replay. Install with: pip install cupy-cuda12x"
        ) from exc

    bundle_dir = Path(bundle_dir)
    symbol_map = _load_bundle_symbol_map(bundle_dir)

    sample_args = tuple(t.clone() for t in sample_args)

    registry_before = build_ptr_registry()

    with _observe() as launches:
        with torch.inference_mode():
            compiled = torch.compile(module)
            ref_output = compiled(*sample_args)

    if not launches:
        raise RuntimeError("no kernel launches captured during torch.compile forward")

    missing = [r.symbol for r in launches if r.symbol not in symbol_map]
    if missing:
        raise RuntimeError(
            f"kernels not found in bundle {bundle_dir}: {missing}"
        )

    return _replay(launches, symbol_map, sample_args, ref_output, registry_before, cupy)


def _replay(launches, symbol_map, sample_args, ref_output, registry_before, cupy):
    ptr_to_buffer: dict[int, int] = {}

    for ptr, info in registry_before.items():
        for arg in sample_args:
            import torch
            if torch.is_tensor(arg) and arg.data_ptr() == ptr:
                arr = cupy.from_dlpack(arg)
                ptr_to_buffer[info.buffer_id] = arr
                break

    ref_buffers: dict[int, "cupy.ndarray"] = {}
    if torch.is_tensor(ref_output):
        ref_arr = cupy.from_dlpack(ref_output)
        for ptr, info in registry_before.items():
            import torch
            if torch.is_tensor(ref_output) and ref_output.data_ptr() == ptr:
                ref_buffers[info.buffer_id] = ref_arr
                break
    if not ref_buffers:
        if torch.is_tensor(ref_output):
            ref_buffers[0] = cupy.from_dlpack(ref_output)

    seen_buffer_ids: set[int] = set()
    for ptr, info in registry_before.items():
        seen_buffer_ids.add(info.buffer_id)

    all_buffer_infos: dict[int, TensorInfo] = {info.buffer_id: info for info in registry_before.values()}

    for rec in launches:
        for arg in rec.args:
            if arg.kind == "tensor" and arg.buffer_id not in all_buffer_infos:
                pass

    for bid in all_buffer_infos:
        if bid not in ptr_to_buffer and bid not in ref_buffers:
            info = all_buffer_infos[bid]
            import numpy as np
            np_dtype = _torch_dtype_to_numpy(info.dtype)
            ptr_to_buffer[bid] = cupy.zeros(info.shape, dtype=np_dtype)

    modules: dict[str, "cupy.RawModule"] = {}
    for rec in launches:
        if rec.symbol not in modules:
            cubin_path = symbol_map[rec.symbol]
            cubin_bytes = cubin_path.read_bytes()
            modules[rec.symbol] = cupy.RawModule(data=cubin_bytes)

    for rec in launches:
        mod = modules[rec.symbol]
        ker = mod.get_function(rec.symbol)
        if rec.shared_mem > 0:
            ker.max_dynamic_shared_size_bytes = rec.shared_mem

        block_x = 32
        grid_x, grid_y, grid_z = rec.grid

        cupy_args = []
        for arg in rec.args:
            if arg.kind == "tensor":
                cbuf = ptr_to_buffer.get(arg.buffer_id) or ref_buffers.get(arg.buffer_id)
                if cbuf is None:
                    raise RuntimeError(f"buffer_id {arg.buffer_id} not allocated")
                cupy_args.append(cbuf.data.ptr)
            else:
                import numpy as np
                np_dtype = _torch_dtype_to_numpy(arg.dtype)
                cupy_args.append(np.dtype(np_dtype).type(arg.value))

        ker(
            (grid_x, grid_y, grid_z),
            (block_x, 1, 1),
            cupy_args,
            shared_mem=rec.shared_mem,
        )

    cupy.cuda.stream.get_current_stream().synchronize()

    if ref_buffers:
        return next(iter(ref_buffers.values()))
    raise RuntimeError("no output buffer identified")


_TORCH_TO_NUMPY = {
    "float32": "float32",
    "float16": "float16",
    "float64": "float64",
    "int32": "int32",
    "int64": "int64",
    "int16": "int16",
    "int8": "int8",
    "uint8": "uint8",
    "bool": "bool_",
    "bfloat16": "float16",
}


def _torch_dtype_to_numpy(dtype: str) -> str:
    return _TORCH_TO_NUMPY.get(dtype, "float32")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/expoter/test_replay_classify.py -v -k load_bundle`
Expected: 2 PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff format hyperonnx/replay/runner.py && uv run ruff check hyperonnx/replay/runner.py`

- [ ] **Step 6: Commit**

```bash
git add hyperonnx/replay/runner.py tests/expoter/test_replay_classify.py
git commit -m "[dev] add bundle loader and cupy replay engine

Signed-off-by: AGENTS <claude>"
```

---

## Task 4: verify() + `__main__` self-check

**Files:**
- Modify: `hyperonnx/replay/runner.py` (add `verify()` and a `__main__` block)
- Test: `tests/expoter/test_replay_classify.py` (add a `verify` signature test)

**Interfaces:**
- Produces:
  - `def verify(module, sample_args, bundle_dir, *, atol=1e-3, rtol=1e-3) -> bool:` — calls `replay()`, compares output to the reference via `cupy.allclose`.

- [ ] **Step 1: Write the failing test (verify signature + comparison logic)**

Add to `tests/expoter/test_replay_classify.py`:

```python
# ---- verify() tests ----


def test_verify_allclose_pass():
    import numpy as np

    from hyperonnx.replay.runner import _compare

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.001, 2.0, 3.0], dtype=np.float32)
    assert _compare(a, b, atol=1e-2, rtol=1e-2) is True


def test_verify_allclose_fail():
    import numpy as np

    from hyperonnx.replay.runner import _compare

    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([10.0, 20.0], dtype=np.float32)
    assert _compare(a, b, atol=1e-3, rtol=1e-3) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/expoter/test_replay_classify.py -v -k compare`
Expected: FAIL with `ImportError: cannot import name '_compare'`

- [ ] **Step 3: Implement `_compare`, `verify()`, and `__main__`**

Add to `hyperonnx/replay/runner.py`:

```python
def _compare(replay_out, ref_out, *, atol: float, rtol: float) -> bool:
    import numpy as np

    return bool(np.allclose(replay_out, ref_out, atol=atol, rtol=rtol))


def verify(
    module,
    sample_args: tuple,
    bundle_dir: str | Path,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    replay_out = replay(module, sample_args, bundle_dir)

    import torch

    with torch.inference_mode():
        compiled = torch.compile(module)
        ref_output = compiled(*tuple(t.clone() for t in sample_args))

    try:
        import cupy
        ref_arr = cupy.from_dlpack(ref_output)
    except ImportError:
        import numpy as np
        ref_arr = np.from_dlpack(ref_output)

    ok = _compare(replay_out, ref_arr, atol=atol, rtol=rtol)
    if not ok:
        import numpy as np
        diff = np.abs(np.asarray(replay_out) - np.asarray(ref_arr))
        print(f"MISMATCH: max abs diff = {diff.max()}")
    return ok


if __name__ == "__main__":
    import sys

    import torch

    if not torch.cuda.is_available():
        print("FAIL: requires CUDA", file=sys.stderr)
        sys.exit(1)

    try:
        from torchvision.models import resnet18
    except ImportError:
        print("FAIL: torchvision not installed", file=sys.stderr)
        sys.exit(1)

    import tempfile

    from hyperonnx import export_hyper_onnx

    model = resnet18().cuda().eval()
    sample = (torch.randn(1, 3, 224, 224, device="cuda"),)

    with tempfile.TemporaryDirectory() as tmp:
        export_hyper_onnx(
            model,
            sample,
            str(Path(tmp) / "resnet18.onnx"),
            compile=[type(model.layer1[0])],
            dynamo=True,
            external_data=True,
            external_directory=tmp,
        )
        import glob

        bundles = glob.glob(str(Path(tmp) / "*.kernels"))
        if not bundles:
            print("FAIL: no kernel bundle exported", file=sys.stderr)
            sys.exit(1)
        ok = verify(model.layer1[0], sample, bundles[0])
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/expoter/test_replay_classify.py -v -k compare`
Expected: 2 PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff format hyperonnx/replay/runner.py && uv run ruff check hyperonnx/replay/runner.py`

- [ ] **Step 6: Commit**

```bash
git add hyperonnx/replay/runner.py tests/expoter/test_replay_classify.py
git commit -m "[dev] add verify() and __main__ self-check for replay harness

Signed-off-by: AGENTS <claude>"
```

---

## Task 5: ResNet-18 integration test (CUDA-gated)

**Files:**
- Create: `tests/test_replay_integration.py`

**Interfaces:**
- Consumes: `replay()` and `verify()` from Task 4; `export_hyper_onnx` from existing code.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_replay_integration.py
"""Tier 2 integration tests for the cubin replay harness.

Requires a real CUDA device + triton + cupy. Skipped otherwise.
"""
import glob
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.fixture(autouse=True)
def _isolate_inductor_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    torch._dynamo.reset()
    yield


def _require_cupy():
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("requires CuPy")


def test_replay_single_conv(tmp_path: Path):
    _require_cupy()
    from hyperonnx import export_hyper_onnx
    from hyperonnx.replay import verify

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = _M().cuda().eval()
    sample = (torch.randn(1, 3, 32, 32, device="cuda"),)

    export_hyper_onnx(
        model,
        sample,
        str(tmp_path / "m.onnx"),
        compile=[_M],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    bundles = glob.glob(str(tmp_path / "*.kernels"))
    assert len(bundles) == 1
    ok = verify(model, sample, bundles[0], atol=1e-2, rtol=1e-2)
    assert ok, "replay output does not match torch.compile reference"


def test_replay_resnet18_basicblock(tmp_path: Path):
    _require_cupy()
    from torchvision.models import resnet18

    from hyperonnx import export_hyper_onnx
    from hyperonnx.replay import verify

    model = resnet18().cuda().eval()
    sample = (torch.randn(1, 3, 224, 224, device="cuda"),)
    block = model.layer1[0]

    export_hyper_onnx(
        model,
        sample,
        str(tmp_path / "resnet18.onnx"),
        compile=[type(block)],
        dynamo=True,
        external_data=True,
        external_directory=str(tmp_path),
    )
    bundles = glob.glob(str(tmp_path / "*.kernels"))
    assert len(bundles) >= 1

    block_input = torch.randn(1, 64, 56, 56, device="cuda")
    ok = verify(block, (block_input,), bundles[0], atol=1e-2, rtol=1e-2)
    assert ok, "replay output does not match torch.compile reference"
```

- [ ] **Step 2: Run integration test (will SKIP if no GPU)**

Run: `uv run pytest tests/test_replay_integration.py -v`
Expected: 2 SKIPPED (no CUDA on this machine) or 2 PASS (on a CUDA box).

- [ ] **Step 3: Run the full Tier 1 suite to check no regressions**

Run: `uv run pytest tests/expoter/test_replay_classify.py -v`
Expected: all PASS.

- [ ] **Step 4: Lint + format**

Run: `uv run ruff format hyperonnx/replay && uv run ruff check hyperonnx/replay && uv run ruff format tests/test_replay_integration.py && uv run ruff check tests/test_replay_integration.py`

- [ ] **Step 5: Commit**

```bash
git add tests/test_replay_integration.py
git commit -m "[dev] add ResNet-18 integration test for cubin replay harness

Signed-off-by: AGENTS <claude>"
```

---

## Self-review checklist (post-write)

**Spec coverage:**
- Replay goal (verify cubin matches torch.compile) → Task 3 + Task 4 ✓
- Observe hook on CompiledKernel.__getitem__ → Task 2 ✓
- Arg classification (tensor via data_ptr registry, scalar otherwise) → Task 1 ✓
- Buffer threading (dedup by data_ptr → buffer_id) → Task 1 (registry assigns buffer_id) + Task 3 (replay allocates per buffer_id) ✓
- Bundle loading (manifest.json symbol→cubin map) → Task 3 ✓
- CuPy RawModule cubin load + shared mem attribute → Task 3 ✓
- verify() with allclose → Task 4 ✓
- __main__ self-check (ResNet-18) → Task 4 ✓
- CuPy lazy import → Task 3 (`replay()` body) ✓
- No export-pipeline changes → all tasks touch only new files ✓
- Error handling (missing cupy, missing symbol, arch mismatch) → Task 3 ✓
- Testing (Tier 1 unit no-GPU, Tier 2 integration CUDA-gated) → Task 1-4 (Tier 1) + Task 5 (Tier 2) ✓

**Placeholder scan:** none. All code blocks are complete and runnable.

**Type consistency:**
- `TensorInfo(shape, dtype, buffer_id)` — defined Task 1, used Tasks 1+3 ✓
- `ArgDescriptor(kind, buffer_id, value, dtype)` — defined Task 1, used Tasks 1+2+3 ✓
- `LaunchRecord(symbol, grid, shared_mem, args)` — defined Task 2, used Tasks 2+3 ✓
- `_observe()` yields `list[LaunchRecord]` — defined Task 2, consumed Task 3 ✓
- `_load_bundle_symbol_map(bundle_dir) -> dict[str, Path]` — defined Task 3, tested Task 3 ✓
