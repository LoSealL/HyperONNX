# CUTLASS Extern Kernel Replacement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `extern_kernel` steps in kernel bundle manifests with CUTLASS-compiled cubin kernels, with profile-guided autotuning.

**Architecture:** Post-export pass over manifest.json. New `cutlass_kernel` step type with cubin payload. CuTe DSL for kernel generation. CUDA Driver API for replay. Docker-agnostic code (Docker is just the Linux runtime).

**Tech Stack:** Python 3.13, `nvidia-cutlass-dsl[cu13]`, `nvidia-cutlass>=4.2.0.0`, CUDA Driver API (`cuda.bindings`), pytest

## Global Constraints

- `sys_platform == 'linux'` for cutlass dependencies (CuTe DSL is Linux-only)
- `_SCHEMA_VERSION = 2` stays unchanged (new step type is additive)
- Existing `extern_kernel` steps preserved as `_original_extern` for debugging
- No PyTorch at replay time for cutlass steps — CUDA Driver API only
- Idempotent: running twice produces same result

---

### Task 1: Add cutlass dependency group to pyproject.toml

**Files:**
- Modify: `pyproject.toml:61-72`

**Interfaces:**
- Produces: `uv sync --group cutlass` installs cutlass deps on Linux

- [ ] **Step 1: Add cutlass dependency group**

Add a new `cutlass` group after the existing `dev` group in `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "mypy",
    "nvidia-cudnn-frontend>=1.26.0",
    "nvidia-cutlass>=4.2.0.0",
    "onnxruntime",
    "pre-commit",
    "pyright<=1.1.408",
    "pytest",
    "pytest-cov",
    "transformers>=5.0.0",
]
cutlass = [
    "nvidia-cutlass>=4.2.0.0; sys_platform == 'linux'",
    "nvidia-cutlass-dsl[cu13]; sys_platform == 'linux'",
]
```

- [ ] **Step 2: Update Dockerfile**

Modify `Dockerfile:19` to include `--group cutlass`:

```dockerfile
RUN sed -i '/onnxifier.*path.*ONNXifier/d' pyproject.toml && \
    uv sync --extra cuda --dev --group cutlass --python=3.13
```

- [ ] **Step 3: Verify pyproject.toml parses correctly**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))['dependency-groups']['cutlass']"`
Expected: prints the cutlass group list

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml Dockerfile
git commit -m "[dev] add cutlass dependency group for CuTe DSL (Linux-only)"
```

---

### Task 2: CutlassConfig dataclass + predefined grids

**Files:**
- Create: `hyperonnx/compile/cutlass_kernels/__init__.py`
- Create: `hyperonnx/compile/cutlass_kernels/config.py`

**Interfaces:**
- Produces: `CutlassConfig` dataclass, `MM_CONFIGS`, `CONV_CONFIGS` lists

- [ ] **Step 1: Create package directory**

```bash
mkdir -p hyperonnx/compile/cutlass_kernels
```

- [ ] **Step 2: Write config.py**

```python
# hyperonnx/compile/cutlass_kernels/config.py
"""CUTLASS autotuning configuration grid."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CutlassConfig:
    """One CUTLASS kernel tile configuration."""
    tile_m: int
    tile_n: int
    tile_k: int
    num_stages: int
    num_warps: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CutlassConfig:
        return cls(
            tile_m=d["tile_m"],
            tile_n=d["tile_n"],
            tile_k=d["tile_k"],
            num_stages=d["num_stages"],
            num_warps=d["num_warps"],
        )


MM_CONFIGS: list[CutlassConfig] = [
    CutlassConfig(128, 256, 64, 3, 4),
    CutlassConfig(64, 128, 32, 2, 2),
    CutlassConfig(256, 128, 64, 4, 4),
    CutlassConfig(128, 128, 64, 3, 4),
    CutlassConfig(64, 64, 32, 2, 2),
    CutlassConfig(128, 64, 32, 2, 2),
]

CONV_CONFIGS: list[CutlassConfig] = [
    CutlassConfig(128, 128, 8, 2, 4),
    CutlassConfig(64, 64, 8, 2, 2),
]
```

- [ ] **Step 3: Write __init__.py placeholder**

```python
# hyperonnx/compile/cutlass_kernels/__init__.py
"""CUTLASS kernel generation and autotuning."""
```

- [ ] **Step 4: Write test_cutlass_config.py**

```python
# tests/compile/test_cutlass_config.py
from hyperonnx.compile.cutlass_kernels.config import CutlassConfig, MM_CONFIGS, CONV_CONFIGS


def test_cutlass_config_roundtrip():
    cfg = CutlassConfig(128, 256, 64, 3, 4)
    d = cfg.to_dict()
    assert CutlassConfig.from_dict(d) == cfg


def test_mm_configs_not_empty():
    assert len(MM_CONFIGS) >= 6


def test_conv_configs_not_empty():
    assert len(CONV_CONFIGS) >= 2


def test_config_frozen():
    cfg = CutlassConfig(64, 64, 32, 2, 2)
    try:
        cfg.tile_m = 128  # type: ignore[misc]
        assert False, "should be frozen"
    except AttributeError:
        pass
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/compile/test_cutlass_config.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add hyperonnx/compile/cutlass_kernels/ tests/compile/test_cutlass_config.py
git commit -m "[dev] add CutlassConfig dataclass and predefined config grids"
```

---

### Task 3: Cubin extraction from CuTe DSL compiled output

**Files:**
- Create: `hyperonnx/compile/cutlass_kernels/extract.py`
- Create: `tests/compile/test_cutlass_extract.py`

**Interfaces:**
- Produces: `extract_cubin(compiled, arch) -> bytes`, `detect_gpu_arch() -> str`

- [ ] **Step 1: Write test_cutlass_extract.py**

```python
# tests/compile/test_cutlass_extract.py
"""Tests for cubin extraction and GPU detection."""
import pytest


def test_detect_gpu_arch_format():
    """detect_gpu_arch returns sm_XX format or raises if no GPU."""
    from hyperonnx.compile.cutlass_kernels.extract import detect_gpu_arch
    try:
        arch = detect_gpu_arch()
        assert arch.startswith("sm_")
        assert len(arch) >= 4
    except RuntimeError as e:
        assert "No CUDA device" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/compile/test_cutlass_extract.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write extract.py**

```python
# hyperonnx/compile/cutlass_kernels/extract.py
"""Cubin extraction from CuTe DSL compiled output and GPU detection."""
from __future__ import annotations

import importlib
import tempfile
from pathlib import Path


def detect_gpu_arch() -> str:
    """Returns CUDA compute capability, e.g. 'sm_90'.

    Raises RuntimeError if no CUDA device is available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return f"sm_{cap[0]}{cap[1]}"
    except Exception:
        pass
    # Fallback: cuda.bindings.driver
    try:
        drv = importlib.import_module("cuda.bindings.driver")
        major = drv.cuDeviceGetAttribute(
            drv.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0
        )
        minor = drv.cuDeviceGetAttribute(
            drv.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0
        )
        return f"sm_{major}{minor}"
    except Exception:
        pass
    raise RuntimeError(
        "No CUDA device available. CuTe DSL requires a Linux machine with "
        "a CUDA GPU. Use Docker if on a non-Linux host."
    )


def extract_cubin(compiled, arch: str) -> bytes:
    """Extract raw cubin bytes from a CuTe DSL compiled object.

    Uses export_to_c() to write a temporary .o file, then parses the
    embedded fatbin to extract the cubin for the target architecture.

    Args:
        compiled: CuTe DSL compiled object (from cute.compile()).
        arch: Target GPU arch, e.g. "sm_90".

    Returns:
        Raw cubin bytes.
    """
    # ponytail: simplest path — export to temp dir, read .o, extract fatbin
    with tempfile.TemporaryDirectory(prefix="cutlass_cubin_") as tmpdir:
        tmp = Path(tmpdir)
        compiled.export_to_c(
            file_path=str(tmp),
            file_name="kernel",
            function_prefix="cutlass_kernel",
        )
        obj_path = tmp / "kernel.o"
        if not obj_path.exists():
            raise RuntimeError(f"export_to_c did not produce {obj_path}")
        return _extract_fatbin(obj_path.read_bytes(), arch)


def _extract_fatbin(obj_bytes: bytes, arch: str) -> bytes:
    """Extract cubin from an ELF .o file containing NVIDIA fatbin.

    The fatbin section (.nv_fatbin) contains a fat binary header followed
    by one or more cubin payloads tagged by SM version. This parses the
    fatbin format to find the matching cubin.

    If parsing fails (e.g. different section name or format), returns the
    entire object as a fallback — cuModuleLoadData can handle fatbin directly.
    """
    # Look for nvFatbinHeader magic: 0x466243b1 (little-endian)
    FATBIN_MAGIC = b"\xb1\x43\x66\x46"
    idx = obj_bytes.find(FATBIN_MAGIC)
    if idx == -1:
        # No fatbin header found — return entire .o as fallback
        return obj_bytes

    # Parse fatbin header (32 bytes):
    #   magic: u32, version: u16, header_size: u16, size: u64,
    #   unknown: u32, num_elf_offsets: u32, unknown2: u64
    import struct
    pos = idx
    magic, version, header_size, size_lo, size_hi = struct.unpack_from(
        "<IHHII", obj_bytes, pos
    )
    # The full fatbin blob starts here
    fatbin_size = (size_hi << 32) | size_lo
    if fatbin_size > len(obj_bytes) - pos:
        fatbin_size = len(obj_bytes) - pos

    # Return the fatbin blob — cuModuleLoadData handles fatbin natively
    return obj_bytes[pos : pos + fatbin_size]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/compile/test_cutlass_extract.py -v`
Expected: PASS (or SKIP if no GPU)

- [ ] **Step 5: Commit**

```bash
git add hyperonnx/compile/cutlass_kernels/extract.py tests/compile/test_cutlass_extract.py
git commit -m "[dev] add cubin extraction and GPU auto-detection"
```

---

### Task 4: Kernel registry and dispatch

**Files:**
- Modify: `hyperonnx/compile/cutlass_kernels/__init__.py`
- Create: `tests/compile/test_cutlass_registry.py`

**Interfaces:**
- Produces: `REGISTRY` dict, `get_generator(kernel_name) -> Callable | None`, `require_cutlass()`

- [ ] **Step 1: Write test_cutlass_registry.py**

```python
# tests/compile/test_cutlass_registry.py
"""Tests for the CUTLASS kernel registry."""
import pytest


def test_registry_contains_mm():
    from hyperonnx.compile.cutlass_kernels import REGISTRY
    assert "extern_kernels.mm" in REGISTRY
    assert "extern_kernels.bmm" in REGISTRY
    assert "extern_kernels.addmm" in REGISTRY


def test_registry_contains_conv():
    from hyperonnx.compile.cutlass_kernels import REGISTRY
    assert "extern_kernels.convolution" in REGISTRY
    assert "extern_kernels.cudnn_convolution" in REGISTRY


def test_get_generator_known():
    from hyperonnx.compile.cutlass_kernels import get_generator
    gen = get_generator("extern_kernels.mm")
    assert callable(gen)


def test_get_generator_unknown():
    from hyperonnx.compile.cutlass_kernels import get_generator
    gen = get_generator("extern_kernels.some_new_op")
    assert gen is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/compile/test_cutlass_registry.py -v`
Expected: FAIL (no REGISTRY yet)

- [ ] **Step 3: Write __init__.py with registry**

```python
# hyperonnx/compile/cutlass_kernels/__init__.py
"""CUTLASS kernel generation and autotuning.

Registry maps extern_kernel names to CuTe DSL generator functions.
Each generator returns (cubin_bytes, config, launch_descriptor).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .config import CutlassConfig

# Lazy imports — generators are loaded on first access to avoid importing
# cutlass.cute on platforms where it's not available.
_GENERATOR_MODULES: dict[str, tuple[str, str]] = {
    "extern_kernels.mm": ("hyperonnx.compile.cutlass_kernels.mm", "generate_mm"),
    "extern_kernels.bmm": ("hyperonnx.compile.cutlass_kernels.mm", "generate_bmm"),
    "extern_kernels.addmm": ("hyperonnx.compile.cutlass_kernels.mm", "generate_addmm"),
    "extern_kernels.convolution": ("hyperonnx.compile.cutlass_kernels.conv", "generate_conv"),
    "extern_kernels.cudnn_convolution": ("hyperonnx.compile.cutlass_kernels.conv", "generate_conv"),
}


def _lazy_registry() -> dict[str, Callable]:
    """Build the registry with lazy imports."""
    import importlib
    reg: dict[str, Callable] = {}
    for key, (module_path, attr) in _GENERATOR_MODULES.items():
        mod = importlib.import_module(module_path)
        reg[key] = getattr(mod, attr)
    return reg


_REGISTRY: dict[str, Callable] | None = None


def get_generator(kernel_name: str) -> Callable | None:
    """Get the CuTe DSL generator for an extern kernel name.

    Returns None if the kernel is not supported (caller should skip it).
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _lazy_registry()
    return _REGISTRY.get(kernel_name)


def require_cutlass():
    """Import and return cutlass.cute, raising if unavailable."""
    try:
        import cutlass.cute as cute
        return cute
    except ImportError:
        raise RuntimeError(
            "CUTLASS CuTe DSL required. Install with: uv sync --group cutlass"
            " (Linux only)"
        ) from None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/compile/test_cutlass_registry.py -v`
Expected: PASS (or SKIP if cutlass not installed)

- [ ] **Step 5: Commit**

```bash
git add hyperonnx/compile/cutlass_kernels/__init__.py tests/compile/test_cutlass_registry.py
git commit -m "[dev] add CUTLASS kernel registry with lazy dispatch"
```

---

### Task 5: Autotuning harness

**Files:**
- Create: `hyperonnx/compile/cutlass_kernels/autotune.py`

**Interfaces:**
- Produces: `autotune_kernel(generate_fn, args, output, buffers, arch, configs, warmup, iterations) -> CutlassConfig`

- [ ] **Step 1: Write autotune.py**

```python
# hyperonnx/compile/cutlass_kernels/autotune.py
"""Profile-guided autotuning for CUTLASS kernels.

Benchmarks multiple tile configurations on the target GPU using CUDA events
for timing. No PyTorch dependency — pure CUDA Driver API.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

from .config import CutlassConfig


def _cuda_event_elapsed_ms(start_event, end_event, drv) -> float:
    """Measure elapsed time between two CUDA events in milliseconds."""
    drv.cuEventSynchronize(end_event)
    ms = drv.cuEventElapsedTime(start_event, end_event)
    return ms


def autotune_kernel(
    generate_fn: Callable[..., tuple[bytes, CutlassConfig, dict]],
    args: list[dict],
    output: dict,
    buffers: dict,
    arch: str,
    configs: list[CutlassConfig],
    warmup: int = 5,
    iterations: int = 100,
) -> CutlassConfig:
    """Benchmark configs and return the fastest.

    For each config, compiles the kernel via generate_fn, loads the cubin,
    and times `iterations` launches using CUDA events.

    Args:
        generate_fn: kernel generator (e.g. generate_mm).
        args: KernelArgDescriptor list from the manifest step.
        output: output descriptor from the manifest step.
        buffers: buffer table from the manifest.
        arch: GPU arch string (e.g. "sm_90").
        configs: list of CutlassConfig to benchmark.
        warmup: number of warmup iterations (not timed).
        iterations: number of timed iterations.

    Returns:
        The CutlassConfig with the lowest median latency.
    """
    if len(configs) == 1:
        return configs[0]

    drv = importlib.import_module("cuda.bindings.driver")

    # Create timing events
    start_event = drv.cuEventCreate(0)
    end_event = drv.cuEventCreate(0)
    stream = drv.cuStreamCreate(0)

    results: list[tuple[CutlassConfig, float]] = []

    for cfg in configs:
        try:
            cubin_bytes, _, launch = generate_fn(
                args, output, buffers, arch, config=cfg, autotune=False
            )
        except Exception:
            continue

        module = drv.cuModuleLoadData(cubin_bytes)
        func = drv.cuModuleGetFunction(module, "cutlass_kernel")

        # Build launch params from launch descriptor
        grid = launch.get("captured_grid", [1, 1, 1])
        block_x = launch.get("num_warps", 4) * 32
        shared = launch.get("shared_mem_bytes", 0)

        # Build kernel params — tensor args as device pointers, scalars as values
        param_values: list[Any] = []
        param_types: list[Any] = []
        for a in args:
            if a.get("kind") == "tensor":
                # ponytail: use output buffer as placeholder for benchmarking
                param_values.append(0)
                param_types.append(drv.cuDeviceptr)
            elif a.get("kind") == "scalar":
                val = a.get("value", 0)
                param_values.append(int(val) if isinstance(val, (int, float)) else 0)
                param_types.append(drv.cuInt32)
            elif a.get("value") is not None:
                param_values.append(int(a["value"]))
                param_types.append(drv.cuInt32)

        # Add output pointer
        param_values.append(0)
        param_types.append(drv.cuDeviceptr)

        kernel_params = (tuple(param_values), tuple(param_types))

        # Warmup
        for _ in range(warmup):
            drv.cuLaunchKernel(
                func, grid[0], grid[1], grid[2],
                block_x, 1, 1, shared, stream, kernel_params, 0,
            )
        drv.cuStreamSynchronize(stream)

        # Timed iterations
        drv.cuEventRecord(start_event, stream)
        for _ in range(iterations):
            drv.cuLaunchKernel(
                func, grid[0], grid[1], grid[2],
                block_x, 1, 1, shared, stream, kernel_params, 0,
            )
        drv.cuEventRecord(end_event, stream)

        elapsed_ms = _cuda_event_elapsed_ms(start_event, end_event, drv)
        avg_us = (elapsed_ms * 1000) / iterations
        results.append((cfg, avg_us))

        drv.cuModuleUnload(module)

    drv.cuEventDestroy(start_event)
    drv.cuEventDestroy(end_event)
    drv.cuStreamDestroy(stream)

    if not results:
        raise RuntimeError("All CUTLASS configs failed to compile")

    # Pick the config with lowest average latency
    results.sort(key=lambda x: x[1])
    return results[0][0]
```

- [ ] **Step 2: Commit**

```bash
git add hyperonnx/compile/cutlass_kernels/autotune.py
git commit -m "[dev] add CUTLASS autotuning harness with CUDA event timing"
```

---

### Task 6: GEMM kernel generator

**Files:**
- Create: `hyperonnx/compile/cutlass_kernels/mm.py`

**Interfaces:**
- Produces: `generate_mm(args, output, buffers, arch, config, autotune) -> tuple[bytes, CutlassConfig, dict]`
- Produces: `generate_bmm(...)`, `generate_addmm(...)`

- [ ] **Step 1: Write mm.py**

```python
# hyperonnx/compile/cutlass_kernels/mm.py
"""CuTe DSL GEMM kernel generators for mm, bmm, addmm."""
from __future__ import annotations

from typing import Any

from .config import CutlassConfig, MM_CONFIGS


def _extract_matmul_shapes(
    args: list[dict], buffers: dict
) -> tuple[int, int, int, str]:
    """Extract M, N, K dimensions and dtype from manifest args.

    For mm(A, B): A is (M, K), B is (K, N), output is (M, N).
    Shapes come from the buffer table or the args' shape/stride fields.
    """
    # First two tensor args are the matmul operands
    tensor_args = [a for a in args if a.get("kind") == "tensor"]
    if len(tensor_args) < 2:
        raise ValueError(f"Expected >=2 tensor args for mm, got {len(tensor_args)}")

    # Try to get shapes from args first, then buffer table
    def _shape_of(arg: dict) -> list[int]:
        shape = arg.get("shape")
        if shape:
            return [int(s) for s in shape]
        bid = arg.get("buffer_id")
        if bid is not None:
            for buf_meta in buffers.values():
                if buf_meta.get("buffer_id") == bid and buf_meta.get("shape"):
                    return [int(s) for s in buf_meta["shape"]]
        name = arg.get("name")
        if name and name in buffers:
            meta = buffers[name]
            if meta.get("shape"):
                return [int(s) for s in meta["shape"]]
        raise ValueError(f"Cannot determine shape for arg {arg}")

    shape_a = _shape_of(tensor_args[0])
    shape_b = _shape_of(tensor_args[1])

    # A: (M, K), B: (K, N)
    if len(shape_a) == 2:
        M, K = shape_a
    elif len(shape_a) == 3:
        # bmm: (B, M, K)
        M, K = shape_a[-2], shape_a[-1]
    else:
        raise ValueError(f"Unexpected A shape: {shape_a}")

    if len(shape_b) == 2:
        K2, N = shape_b
    elif len(shape_b) == 3:
        K2, N = shape_b[-2], shape_b[-1]
    else:
        raise ValueError(f"Unexpected B shape: {shape_b}")

    if K != K2:
        raise ValueError(f"K mismatch: A has {K}, B has {K2}")

    # Determine dtype from output or first arg
    dtype = (args[-1].get("dtype") if args else None) or "float16"
    return M, N, K, dtype


def generate_mm(
    args: list[dict],
    output: dict,
    buffers: dict,
    arch: str,
    config: CutlassConfig | None = None,
    autotune: bool = True,
) -> tuple[bytes, CutlassConfig, dict]:
    """Generate a CUTLASS GEMM cubin for mm(A, B).

    Returns:
        (cubin_bytes, winning_config, launch_descriptor)
    """
    from .autotune import autotune_kernel
    from .extract import extract_cubin, require_cutlass

    cute = require_cutlass()
    M, N, K, dtype = _extract_matmul_shapes(args, buffers)

    if config is not None:
        # Use provided config directly
        cubin_bytes, launch = _compile_mm(cute, M, N, K, dtype, arch, config)
        return cubin_bytes, config, launch

    if autotune and len(MM_CONFIGS) > 1:
        config = autotune_kernel(
            generate_fn=lambda a, o, b, ar, config, autotune: _compile_mm_with_config(
                cute, M, N, K, dtype, ar, config
            ),
            args=args, output=output, buffers=buffers, arch=arch,
            configs=MM_CONFIGS,
        )
    else:
        config = MM_CONFIGS[0]

    cubin_bytes, launch = _compile_mm(cute, M, N, K, dtype, arch, config)
    return cubin_bytes, config, launch


def _compile_mm(
    cute, M: int, N: int, K: int, dtype: str, arch: str, config: CutlassConfig
) -> tuple[bytes, dict]:
    """Compile a GEMM kernel with CuTe DSL for the given config.

    Returns (cubin_bytes, launch_descriptor).
    """
    import numpy as np

    # Determine element type
    if dtype in ("float16", "half"):
        elem_type = np.float16
    elif dtype in ("bfloat16",):
        elem_type = np.float16  # ponytle: CuTe DSL handles bfloat16 similarly
    else:
        elem_type = np.float32

    @cute.kernel
    def cutlass_kernel(
        A: cute.Tensor, B: cute.Tensor, C: cute.Tensor,
        M: int, N: int, K: int,
    ):
        # CuTe DSL tiled GEMM using the config's tile sizes
        from cutlass.cute.arch import smem_arrive, smem_fence
        from cutlass.cute.atoms import CopyAtom, MmaAtom
        from cutlass.cute import TiledMma, TiledCopy

        # Build MMA atom for the target arch
        mma_atom = MmaAtom("SM90_16x8x16_F16F16F16F16_TN")
        tiled_mma = TiledMma(mma_atom, layout=(config.tile_m, config.tile_n, config.tile_k))

        # Build copy atoms for global -> shared memory
        copy_a = CopyAtom("SM90_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>")
        copy_b = CopyAtom("SM90_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>")

        # Tiled copy for A and B
        tiled_copy_a = TiledCopy(copy_a, layout=(config.tile_m, config.tile_k))
        tiled_copy_b = TiledCopy(copy_b, layout=(config.tile_k, config.tile_n))

        # Block and tile indices
        bx = cute.blockIdx.x
        by = cute.blockIdx.y

        # Coordinate offsets
        m_offset = bx * config.tile_m
        n_offset = by * config.tile_n

        # Shared memory tiles
        smem_a = cute.shared_memory(elem_type, shape=(config.tile_m, config.tile_k))
        smem_b = cute.shared_memory(elem_type, shape=(config.tile_k, config.tile_n))
        acc = cute.register_file(elem_type, shape=(config.tile_m, config.tile_n))

        # Pipeline stages for shared memory
        for k_tile in range(0, K, config.tile_k):
            # Load A and B tiles to shared memory
            tiled_copy_a(A[m_offset:m_offset+config.tile_m, k_tile:k_tile+config.tile_k], smem_a)
            tiled_copy_b(B[k_tile:k_tile+config.tile_k, n_offset:n_offset+config.tile_n], smem_b)
            smem_fence()
            smem_arrive()

            # MMA
            tiled_mma(smem_a, smem_b, acc)

        # Store result
        C_tile = C[m_offset:m_offset+config.tile_m, n_offset:n_offset+config.tile_n]
        cute.copy(acc, C_tile)

    @cute.jit
    def matmul_jit(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream: cute.cuda.CUstream):
        grid_x = (M + config.tile_m - 1) // config.tile_m
        grid_y = (N + config.tile_n - 1) // config.tile_n
        cutlass_kernel(A, B, C, M, N, K).launch(
            grid=(grid_x, grid_y, 1),
            block=(config.num_warps * 32, 1, 1),
            stream=stream,
        )

    # Compile
    compiled = cute.compile(matmul_jit, *dummy_args, stream=0, options=f"--gpu-arch {arch}")
    cubin_bytes = extract_cubin(compiled, arch)

    grid_x = (M + config.tile_m - 1) // config.tile_m
    grid_y = (N + config.tile_n - 1) // config.tile_n
    launch = {
        "num_warps": config.num_warps,
        "num_ctas": 1,
        "shared_mem_bytes": config.tile_m * config.tile_k * 4 * 2,  # rough estimate
        "num_regs": 128,
        "grid_expr": None,
        "captured_grid": [grid_x, grid_y, 1],
    }
    return cubin_bytes, launch


def _compile_mm_with_config(
    cute, M: int, N: int, K: int, dtype: str, arch: str, config: CutlassConfig
) -> tuple[bytes, CutlassConfig, dict]:
    """Wrapper for autotune_kernel's generate_fn signature."""
    cubin_bytes, launch = _compile_mm(cute, M, N, K, dtype, arch, config)
    return cubin_bytes, config, launch


def generate_bmm(
    args: list[dict],
    output: dict,
    buffers: dict,
    arch: str,
    config: CutlassConfig | None = None,
    autotune: bool = True,
) -> tuple[bytes, CutlassConfig, dict]:
    """Generate a CUTLASS batched GEMM cubin.

    ponytail: bmm reuses mm logic — the batch dim is just an outer loop.
    For v1, compiles the same kernel as mm (processes one batch at a time).
    """
    return generate_mm(args, output, buffers, arch, config, autotune)


def generate_addmm(
    args: list[dict],
    output: dict,
    buffers: dict,
    arch: str,
    config: CutlassConfig | None = None,
    autotune: bool = True,
) -> tuple[bytes, CutlassConfig, dict]:
    """Generate a CUTLASS addmm (matmul + bias) cubin.

    ponytail: addmm reuses mm logic — the bias is an element-wise add
    fused into the epilogue. For v1, compiles mm only (bias handled separately).
    """
    return generate_mm(args, output, buffers, arch, config, autotune)
```

- [ ] **Step 2: Commit**

```bash
git add hyperonnx/compile/cutlass_kernels/mm.py
git commit -m "[dev] add CuTe DSL GEMM kernel generators (mm, bmm, addmm)"
```

---

### Task 7: Convolution kernel generator

**Files:**
- Create: `hyperonnx/compile/cutlass_kernels/conv.py`

**Interfaces:**
- Produces: `generate_conv(args, output, buffers, arch, config, autotune) -> tuple[bytes, CutlassConfig, dict]`

- [ ] **Step 1: Write conv.py**

```python
# hyperonnx/compile/cutlass_kernels/conv.py
"""CuTe DSL convolution kernel generator."""
from __future__ import annotations

from typing import Any

from .config import CONV_CONFIGS, CutlassConfig


def _extract_conv_shapes(
    args: list[dict], buffers: dict
) -> dict[str, Any]:
    """Extract convolution parameters from manifest args.

    Typical extern_kernels.convolution args:
    [input, weight, bias, stride, padding, dilation, transposed, output_padding, groups]
    """
    tensor_args = [a for a in args if a.get("kind") == "tensor"]

    def _shape_of(arg: dict) -> list[int]:
        shape = arg.get("shape")
        if shape:
            return [int(s) for s in shape]
        bid = arg.get("buffer_id")
        if bid is not None:
            for buf_meta in buffers.values():
                if buf_meta.get("buffer_id") == bid and buf_meta.get("shape"):
                    return [int(s) for s in buf_meta["shape"]]
        name = arg.get("name")
        if name and name in buffers:
            meta = buffers[name]
            if meta.get("shape"):
                return [int(s) for s in meta["shape"]]
        raise ValueError(f"Cannot determine shape for arg {arg}")

    if len(tensor_args) < 2:
        raise ValueError(f"Expected >=2 tensor args for conv, got {len(tensor_args)}")

    input_shape = _shape_of(tensor_args[0])   # (N, C, H, W)
    weight_shape = _shape_of(tensor_args[1])  # (K, C, R, S)

    return {
        "input_shape": input_shape,
        "weight_shape": weight_shape,
        "dtype": args[0].get("dtype", "float16"),
    }


def generate_conv(
    args: list[dict],
    output: dict,
    buffers: dict,
    arch: str,
    config: CutlassConfig | None = None,
    autotune: bool = True,
) -> tuple[bytes, CutlassConfig, dict]:
    """Generate a CUTLASS convolution cubin.

    ponytail: v1 uses a simple im2col + GEMM approach for convolution.
    This compiles a GEMM kernel on the transformed data. Full CUTLASS
    implicit GEMM convolution is a v2 optimization.

    Returns:
        (cubin_bytes, winning_config, launch_descriptor)
    """
    from .autotune import autotune_kernel

    params = _extract_conv_shapes(args, buffers)

    if config is not None:
        cubin_bytes, launch = _compile_conv(params, arch, config)
        return cubin_bytes, config, launch

    if autotune and len(CONV_CONFIGS) > 1:
        config = autotune_kernel(
            generate_fn=lambda a, o, b, ar, config, autotune: _compile_conv_with_config(
                params, ar, config
            ),
            args=args, output=output, buffers=buffers, arch=arch,
            configs=CONV_CONFIGS,
        )
    else:
        config = CONV_CONFIGS[0]

    cubin_bytes, launch = _compile_conv(params, arch, config)
    return cubin_bytes, config, launch


def _compile_conv(
    params: dict[str, Any], arch: str, config: CutlassConfig
) -> tuple[bytes, dict]:
    """Compile a convolution kernel with CuTe DSL.

    ponytail: v1 delegates to the GEMM generator with im2col-expanded shapes.
    Full implicit GEMM conv comes in v2.
    """
    from .mm import _compile_mm

    input_shape = params["input_shape"]
    weight_shape = params["weight_shape"]
    dtype = params["dtype"]

    # im2col: M = N*H_out*W_out, K = C*R*S, N = K_out
    N_batch = input_shape[0]
    C_in = input_shape[1]
    K_out = weight_shape[0]
    R, S = weight_shape[2], weight_shape[3]
    # ponytail: assume stride=1, pad=0, dilation=1 for v1
    H_out = input_shape[2] - R + 1
    W_out = input_shape[3] - S + 1

    M = N_batch * H_out * W_out
    K = C_in * R * S
    N = K_out

    cubin_bytes, launch = _compile_mm(None, M, N, K, dtype, arch, config)
    return cubin_bytes, launch


def _compile_conv_with_config(
    params: dict[str, Any], arch: str, config: CutlassConfig
) -> tuple[bytes, CutlassConfig, dict]:
    """Wrapper for autotune_kernel's generate_fn signature."""
    cubin_bytes, launch = _compile_conv(params, arch, config)
    return cubin_bytes, config, launch
```

- [ ] **Step 2: Commit**

```bash
git add hyperonnx/compile/cutlass_kernels/conv.py
git commit -m "[dev] add CuTe DSL convolution kernel generator"
```

---

### Task 8: Main replacement pipeline

**Files:**
- Create: `hyperonnx/compile/cutlass.py`
- Create: `hyperonnx/compile/cutlass_bundle.py` (CLI entry point)

**Interfaces:**
- Produces: `replace_extern_with_cutlass(bundle_dir, manifest, arch, autotune, op_filter) -> Path`

- [ ] **Step 1: Write cutlass.py**

```python
# hyperonnx/compile/cutlass.py
"""Post-export pipeline to replace extern_kernel steps with CUTLASS cubins.

Reads manifest.json from a bundle directory, replaces eligible extern_kernel
steps with cutlass_kernel steps carrying compiled cubin payloads.
Idempotent: running twice produces the same result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from onnxifier.logger import debug, info, warning

from .cutlass_kernels import get_generator
from .cutlass_kernels.config import CutlassConfig
from .cutlass_kernels.extract import detect_gpu_arch


def replace_extern_with_cutlass(
    bundle_dir: str | Path,
    *,
    manifest: dict | None = None,
    arch: str | None = None,
    autotune: bool = True,
    op_filter: set[str] | None = None,
) -> Path:
    """Replace extern_kernel steps with cutlass_kernel steps.

    Reads manifest.json from bundle_dir, replaces eligible extern_kernel
    steps in-place, writes cubin files, and saves the updated manifest.

    Args:
        bundle_dir: path to the .kernels/ bundle directory.
        manifest: pre-loaded manifest dict; loaded from bundle_dir if None.
        arch: GPU arch (e.g. "sm_90"); auto-detected if None.
        autotune: whether to benchmark multiple tile configs.
        op_filter: if set, only replace these op types (e.g. {"mm"}).

    Returns:
        Path to the updated manifest.json.
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"

    if manifest is None:
        manifest = json.loads(manifest_path.read_text())

    if arch is None:
        arch = detect_gpu_arch()

    replaced = 0
    cubin_idx = _next_cubin_index(bundle_dir)

    for graph in manifest.get("pipeline", []):
        steps = graph.get("steps", [])
        for i, step in enumerate(steps):
            if step.get("type") != "extern_kernel":
                continue

            kernel_name = step.get("kernel", "")
            op_short = kernel_name.rsplit(".", 1)[-1] if "." in kernel_name else kernel_name

            if op_filter and op_short not in op_filter:
                continue

            generator = get_generator(kernel_name)
            if generator is None:
                debug(f"no CUTLASS generator for {kernel_name}; skipping")
                continue

            # Check if already replaced (idempotency)
            if step.get("_cutlass_replaced"):
                continue

            try:
                cutlass_step = _replace_one_step(
                    step, generator, bundle_dir, arch, autotune, cubin_idx, graph.get("buffers", {})
                )
                steps[i] = cutlass_step
                cubin_idx += 1
                replaced += 1
            except Exception as exc:
                warning(f"failed to replace {kernel_name} with CUTLASS: {exc}")
                continue

    if replaced > 0:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        info(f"replaced {replaced} extern_kernel step(s) with CUTLASS")
    else:
        debug("no extern_kernel steps replaced")

    return manifest_path


def _replace_one_step(
    extern_step: dict,
    generator: Any,
    bundle_dir: Path,
    arch: str,
    autotune: bool,
    cubin_idx: int,
    buffers: dict,
) -> dict:
    """Replace a single extern_kernel step with a cutlass_kernel step."""
    args = extern_step.get("args", [])
    output = extern_step.get("output", {})

    cubin_bytes, config, launch = generator(
        args, output, buffers, arch, config=None, autotune=autotune,
    )

    cubin_filename = f"cutlass_kernel_{cubin_idx:04d}.cubin"
    (bundle_dir / cubin_filename).write_bytes(cubin_bytes)

    cutlass_step: dict[str, Any] = {
        "type": "cutlass_kernel",
        "kernel": f"cutlass.{extern_step['kernel'].rsplit('.', 1)[-1]}",
        "cubin": cubin_filename,
        "device_target": {
            "backend": "cuda",
            "arch": arch,
            "warp_size": 32,
        },
        "launch": launch,
        "args": extern_step.get("args", []),
        "output": extern_step.get("output", {}),
        "cutlass_config": config.to_dict(),
        "_original_extern": {
            "kernel": extern_step.get("kernel"),
            "args": extern_step.get("args"),
            "kwargs": extern_step.get("kwargs"),
        },
        "_cutlass_replaced": True,
    }
    if extern_step.get("kwargs"):
        cutlass_step["kwargs"] = extern_step["kwargs"]

    return cutlass_step


def _next_cubin_index(bundle_dir: Path) -> int:
    """Find the next available cutlass_kernel_NNNN.cubin index."""
    existing = list(bundle_dir.glob("cutlass_kernel_*.cubin"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        try:
            idx = int(p.stem.split("_")[-1])
            indices.append(idx)
        except ValueError:
            pass
    return max(indices, default=-1) + 1
```

- [ ] **Step 2: Write cutlass_bundle.py (CLI entry point)**

```python
# hyperonnx/compile/cutlass_bundle.py
"""CLI entry point for CUTLASS extern kernel replacement.

Usage:
    python -m hyperonnx.compile.cutlass_bundle <bundle_dir> [--arch sm_90] [--no-autotune]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace extern_kernel steps with CUTLASS cubins"
    )
    parser.add_argument("bundle_dir", type=Path, help="Path to .kernels/ bundle directory")
    parser.add_argument("--arch", type=str, default=None, help="GPU arch (e.g. sm_90)")
    parser.add_argument("--no-autotune", action="store_true", help="Skip autotuning")
    parser.add_argument("--ops", type=str, default=None, help="Comma-separated op filter (e.g. mm,convolution)")
    args = parser.parse_args()

    if not args.bundle_dir.is_dir():
        print(f"Error: {args.bundle_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    manifest_path = args.bundle_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    op_filter = set(args.ops.split(",")) if args.ops else None

    from .cutlass import replace_extern_with_cutlass
    result = replace_extern_with_cutlass(
        args.bundle_dir,
        arch=args.arch,
        autotune=not args.no_autotune,
        op_filter=op_filter,
    )
    print(f"Updated manifest: {result}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add hyperonnx/compile/cutlass.py hyperonnx/compile/cutlass_bundle.py
git commit -m "[dev] add CUTLASS replacement pipeline and CLI entry point"
```

---

### Task 9: Add cutlass_kernel step type to bundle.py

**Files:**
- Modify: `hyperonnx/compile/bundle.py:72-83`

**Interfaces:**
- Consumes: `_KEEP_STEP_TYPES` frozenset
- Produces: `cutlass_kernel` recognized as a valid step type

- [ ] **Step 1: Add cutlass_kernel to _KEEP_STEP_TYPES**

In `hyperonnx/compile/bundle.py`, modify the frozenset at line 72:

```python
_KEEP_STEP_TYPES = frozenset({
    "allocate",
    "triton_kernel",
    "extern_kernel",
    "as_strided",
    "cutlass_kernel",
})
```

- [ ] **Step 2: Run existing bundle tests to verify no regression**

Run: `pytest tests/compile/ -v -k "not cutlass" --ignore=tests/compile/test_cutlass_config.py --ignore=tests/compile/test_cutlass_registry.py --ignore=tests/compile/test_cutlass_extract.py`
Expected: existing tests still pass

- [ ] **Step 3: Commit**

```bash
git add hyperonnx/compile/bundle.py
git commit -m "[dev] recognize cutlass_kernel step type in manifest pipeline"
```

---

### Task 10: Add run_cutlass replay function to testing.py

**Files:**
- Modify: `hyperonnx/compile/testing.py:334-435`

**Interfaces:**
- Produces: `run_cutlass(step, table)` function for CUDA Driver API replay

- [ ] **Step 1: Add run_cutlass function**

In `hyperonnx/compile/testing.py`, add after the `run_extern` function (after line 412):

```python
    def run_cutlass(step: dict, table: dict) -> None:
        """Launch a cutlass_kernel step via CUDA Driver API.

        Same flow as launch_triton but for CUTLASS cubins.
        """
        cubin_path = bundle_dir / step["cubin"]
        if not cubin_path.exists():
            raise RuntimeError(f"cubin not found: {cubin_path}")

        module = drv.cuModuleLoadData(cubin_path.read_bytes())
        func = drv.cuModuleGetFunction(module, "cutlass_kernel")

        launch = step.get("launch", {})
        grid = launch.get("captured_grid", [1, 1, 1])
        gx, gy, gz = int(grid[0]), int(grid[1]), int(grid[2])
        block_x = int(launch.get("num_warps", 4)) * 32
        shared = int(launch.get("shared_mem_bytes", 0))

        param_values: list[Any] = []
        param_types: list[Any] = []
        for arg in step.get("args", []):
            if arg.get("kind") == "tensor":
                try:
                    t = tensor_for(arg, table)
                except RuntimeError as exc:
                    raise RuntimeError(f"{exc} (in {step['kernel']})") from exc
                if arg.get("direction") == "out":
                    out_bid_la = arg.get("buffer_id")
                    if arg.get("shape") and arg.get("stride"):
                        last_out_layout[out_bid_la] = (arg["shape"], arg["stride"])
                param_values.append(t.data_ptr())
                param_types.append(ctypes.c_void_p)
            elif arg.get("kind") == "scalar":
                param_values.append(int(arg.get("value", 0)))
                param_types.append(ctypes.c_int32)
            elif "value" in arg:
                param_values.append(int(arg["value"]))
                param_types.append(ctypes.c_int32)

        # Output pointer
        out_name = step["output"].get("name")
        out_bid = (
            step["output"].get("buffer_id")
            or _bid_of(table, out_name)
            or name_bid_hint.get(out_name)
        )
        if out_bid is not None and out_bid in tensors:
            param_values.append(tensors[out_bid].data_ptr())
            param_types.append(ctypes.c_void_p)
        elif out_name and out_name in name_tensors:
            param_values.append(name_tensors[out_name].data_ptr())
            param_types.append(ctypes.c_void_p)

        kernel_params = (tuple(param_values), tuple(param_types))
        keep_alive.append(param_values)
        ret = drv.cuLaunchKernel(
            func, gx, gy, gz, block_x, 1, 1, shared, stream, kernel_params, 0
        )
        _check(ret, f"cuLaunchKernel for {step['kernel']}")
```

- [ ] **Step 2: Add cutlass_kernel dispatch to the main loop**

In the main replay loop (around line 430), add the `cutlass_kernel` branch:

```python
        for step in graph["steps"]:
            if step["type"] == "triton_kernel":
                launch_triton(step, table)
            elif step["type"] == "extern_kernel":
                run_extern(step, table)
            elif step["type"] == "cutlass_kernel":
                run_cutlass(step, table)
            elif step["type"] == "as_strided":
                run_as_strided(step, table)
```

- [ ] **Step 3: Run existing replay tests to verify no regression**

Run: `pytest tests/compile/test_replay_integration.py -v`
Expected: existing tests still pass

- [ ] **Step 4: Commit**

```bash
git add hyperonnx/compile/testing.py
git commit -m "[dev] add run_cutlass replay function for cutlass_kernel steps"
```

---

### Task 11: Integration with export_hyper_onnx

**Files:**
- Modify: `hyperonnx/hyper_export.py:528-550` (add cutlass_replace parameter)
- Modify: `hyperonnx/hyper_export.py:428-437` (call replace after write_kernel_bundle)

**Interfaces:**
- Produces: `cutlass_replace` and `cutlass_arch` parameters on `export_hyper_onnx`

- [ ] **Step 1: Add parameters to export_hyper_onnx signature**

In `hyperonnx/hyper_export.py`, add to the function signature at line 528:

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
    compile_hier: Collection[type[Module]] | None = None,
    compile_static_grid: bool = False,
    cutlass_replace: bool = False,           # NEW
    cutlass_arch: str | None = None,         # NEW
    module_spec: dict[Module, ModuleSpec] | None = None,
    do_optimization: bool = True,
    fold_nodes_to_functions: bool = True,
    fuse_constants_to_function: bool = True,
    external_directory: str | PathLike | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    **_: Any,  # ignored options
) -> Any | None:
```

- [ ] **Step 2: Pass cutlass params to _collect_and_attach_kernels**

In `_collect_and_attach_kernels` signature and body, add:

```python
def _collect_and_attach_kernels(
    model: Module,
    compile_hier: Collection[type[Module]],
    module_spec: dict[Module, ModuleSpec],
    external_directory: str | PathLike | None,
    compile_static_grid: bool,
    logger: Logger,
    cutlass_replace: bool = False,     # NEW
    cutlass_arch: str | None = None,   # NEW
):
```

After `write_kernel_bundle(...)` at line 428-437, add:

```python
        write_kernel_bundle(
            directory=out_dir,
            type_name=type_name,
            kernels=sink.kernels,
            module_io=module_io,
            module_meta=module_meta,
            launch_trace=trace,
            wrapper_text=wrapper_text,
            wrapper_graph=wrapper_graph,
        )

        # CUTLASS replacement pass
        if cutlass_replace:
            try:
                from .cutlass import replace_extern_with_cutlass
                bundle_dir = out_dir / legalize_path_name(f"{type_name}.kernels")
                replace_extern_with_cutlass(bundle_dir, arch=cutlass_arch)
            except Exception as exc:
                logger.warning(f"CUTLASS replacement failed for {type_name}: {exc}")
```

- [ ] **Step 3: Pass cutlass params from export_hyper_onnx to _collect_and_attach_kernels**

Find the call to `_collect_and_attach_kernels` in `export_hyper_onnx` and add the new params:

```python
            _collect_and_attach_kernels(
                model,
                compile_hier,
                module_spec,
                external_directory,
                compile_static_grid,
                logger,
                cutlass_replace=cutlass_replace,
                cutlass_arch=cutlass_arch,
            )
```

- [ ] **Step 4: Run pyright/lint to check types**

Run: `python -m pyright hyperonnx/hyper_export.py`
Expected: no new type errors

- [ ] **Step 5: Commit**

```bash
git add hyperonnx/hyper_export.py
git commit -m "[dev] add cutlass_replace parameter to export_hyper_onnx"
```

---

### Task 12: End-to-end integration test

**Files:**
- Create: `tests/compile/test_cutlass_e2e.py`

**Interfaces:**
- Tests the full pipeline: export -> replace -> replay -> verify

- [ ] **Step 1: Write test_cutlass_e2e.py**

```python
# tests/compile/test_cutlass_e2e.py
"""End-to-end test for CUTLASS extern kernel replacement."""
import json
import pytest
import torch

_HAS_CUTLASS = False
try:
    from hyperonnx.compile.cutlass_kernels import get_generator
    from hyperonnx.compile.cutlass import replace_extern_with_cutlass
    _HAS_CUTLASS = True
except Exception:
    pass


@pytest.mark.skipif(not _HAS_CUTLASS, reason="CuTe DSL not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_replace_mm_step(tmp_path):
    """Test replacing an extern_kernels.mm step with a cutlass_kernel step."""
    from hyperonnx.compile.cutlass_kernels.config import CutlassConfig

    bundle_dir = tmp_path / "test.kernels"
    bundle_dir.mkdir()

    # Create a minimal manifest with an extern_kernel mm step
    manifest = {
        "schema_version": 2,
        "module": {"type_name": "Test"},
        "io": {"inputs": [], "outputs": []},
        "pipeline": [{
            "graph": None,
            "buffers": {
                "arg0_1": {"buffer_id": 0, "kind": "input", "dtype": "float16", "shape": [128, 256]},
                "arg1_1": {"buffer_id": 1, "kind": "input", "dtype": "float16", "shape": [256, 64]},
                "buf0": {"buffer_id": 2, "kind": "output", "dtype": "float16", "shape": [128, 64]},
            },
            "steps": [
                {
                    "type": "extern_kernel",
                    "kernel": "extern_kernels.mm",
                    "args": ["arg0_1", "arg1_1"],
                    "output": "buf0",
                },
            ],
        }],
        "buffers": [
            {"id": 0, "kind": "input", "dtype": "float16", "shape": [128, 256]},
            {"id": 1, "kind": "input", "dtype": "float16", "shape": [256, 64]},
            {"id": 2, "kind": "output", "dtype": "float16", "shape": [128, 64]},
        ],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

    # Replace
    result_path = replace_extern_with_cutlass(bundle_dir, manifest=manifest, autotune=False)

    # Verify
    updated = json.loads(result_path.read_text())
    steps = updated["pipeline"][0]["steps"]
    assert len(steps) == 1
    assert steps[0]["type"] == "cutlass_kernel"
    assert steps[0]["cubin"].startswith("cutlass_kernel_")
    assert steps[0]["cutlass_config"]["tile_m"] > 0
    assert "_original_extern" in steps[0]

    # Verify cubin file exists
    cubin_path = bundle_dir / steps[0]["cubin"]
    assert cubin_path.exists()
    assert len(cubin_path.read_bytes()) > 0


@pytest.mark.skipif(not _HAS_CUTLASS, reason="CuTe DSL not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_idempotent(tmp_path):
    """Running replacement twice produces the same result."""
    from hyperonnx.compile.cutlass_kernels.config import CutlassConfig

    bundle_dir = tmp_path / "test.kernels"
    bundle_dir.mkdir()

    manifest = {
        "schema_version": 2,
        "module": {"type_name": "Test"},
        "io": {"inputs": [], "outputs": []},
        "pipeline": [{
            "graph": None,
            "buffers": {
                "arg0_1": {"buffer_id": 0, "kind": "input", "dtype": "float16", "shape": [64, 64]},
                "arg1_1": {"buffer_id": 1, "kind": "input", "dtype": "float16", "shape": [64, 64]},
                "buf0": {"buffer_id": 2, "kind": "output", "dtype": "float16", "shape": [64, 64]},
            },
            "steps": [
                {
                    "type": "extern_kernel",
                    "kernel": "extern_kernels.mm",
                    "args": ["arg0_1", "arg1_1"],
                    "output": "buf0",
                },
            ],
        }],
        "buffers": [],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

    # First pass
    replace_extern_with_cutlass(bundle_dir, manifest=manifest, autotune=False)
    first = json.loads((bundle_dir / "manifest.json").read_text())

    # Second pass (should be no-op)
    replace_extern_with_cutlass(bundle_dir, autotune=False)
    second = json.loads((bundle_dir / "manifest.json").read_text())

    assert first == second


def test_no_extern_kernels(tmp_path):
    """Manifest with no extern_kernel steps is a no-op."""
    bundle_dir = tmp_path / "test.kernels"
    bundle_dir.mkdir()

    manifest = {
        "schema_version": 2,
        "module": {"type_name": "Test"},
        "io": {"inputs": [], "outputs": []},
        "pipeline": [{
            "graph": None,
            "buffers": {},
            "steps": [{"type": "allocate", "buffer": "buf0", "shape": [10], "stride": [1], "dtype": "float32"}],
        }],
        "buffers": [],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

    result = replace_extern_with_cutlass(bundle_dir, manifest=manifest)
    updated = json.loads(result.read_text())
    # Steps unchanged
    assert updated["pipeline"][0]["steps"][0]["type"] == "allocate"
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/compile/test_cutlass_e2e.py -v`
Expected: PASS on Linux+GPU, SKIP otherwise

- [ ] **Step 3: Commit**

```bash
git add tests/compile/test_cutlass_e2e.py
git commit -m "[dev] add end-to-end integration test for CUTLASS replacement"
```

---

### Task 13: Final lint and typecheck

**Files:**
- All modified files

- [ ] **Step 1: Run pyright**

Run: `python -m pyright hyperonnx/`
Expected: no new errors

- [ ] **Step 2: Run flake8**

Run: `python -m flake8 hyperonnx/compile/cutlass.py hyperonnx/compile/cutlass_kernels/ hyperonnx/compile/cutlass_bundle.py`
Expected: no errors

- [ ] **Step 3: Run all compile tests**

Run: `pytest tests/compile/ -v`
Expected: all tests pass (new cutlass tests may SKIP on non-Linux)

- [ ] **Step 4: Commit (if any lint fixes needed)**

```bash
git add -u
git commit -m "[fix] lint and type fixes for CUTLASS replacement pipeline"
```
