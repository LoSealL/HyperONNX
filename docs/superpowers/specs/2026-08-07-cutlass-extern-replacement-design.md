# CUTLASS Extern Kernel Replacement Pipeline

- **Status**: Draft
- **Date**: 2026-08-07
- **Owner**: HyperONNX Authors
- **Branch**: TBD (new feature branch)

## Summary

Add a post-export pipeline that replaces `extern_kernel` steps in a kernel bundle
manifest with CUTLASS-compiled cubin kernels. The replacement uses CuTe DSL
(`nvidia-cutlass-dsl`) to generate and compile kernels, and profile-guided
autotuning to select the best tile configuration for each operation+shape+GPU
combination.

**Design principle**: the replacement is a pure post-export pass over the manifest.
No changes to the existing triton capture or export flow. Running it on a manifest
with no extern kernels is a no-op. Running it twice doesn't double-replace.

**Docker principle**: all code is docker-agnostic Python. Docker is simply the
Linux runtime environment required by CuTe DSL (Linux-only). No Docker-specific
imports, APIs, or assumptions in the codebase.

## Goals

1. Replace `extern_kernel` steps (cuDNN/cuBLAS calls) with `cutlass_kernel`
   steps that carry compiled cubin payloads.
2. Profile-guided autotuning: benchmark multiple tile configurations on the
   target GPU and pick the fastest.
3. GPU auto-detect: determine `sm_*` arch from the current CUDA device.
4. Graceful fallback: unknown op types are skipped (logged as warning),
   keeping the original `extern_kernel` step.
5. Idempotent: running replacement twice on the same manifest produces the
   same result.
6. No PyTorch dependency at replay time for replaced kernels — launch via
   CUDA Driver API (same as triton steps).

## Non-goals (v1)

- **No exhaustive autotuning.** v1 uses a predefined grid of 6-8 tile
  configurations per op type. Full CUTLASS profiler sweep is deferred.
- **No fused multi-op kernels.** Each extern step is replaced independently.
  Fusion (e.g. matmul+bias+relu) is a v2 optimization.
- **No dynamic shape compilation.** v1 compiles cubins for the exact shapes
  captured in the manifest. Grid expression AST for variable shapes is
  deferred to v2.
- **No Windows support.** CuTe DSL is Linux-only. The code is
  docker-agnostic but must run in a Linux environment.
- **No ONNX graph mutation.** Same principle as the kernel bundle: the ONNX
  function body is untouched. The manifest's pipeline steps carry the cubin.

## API

### New function

```python
def replace_extern_with_cutlass(
    bundle_dir: str | Path,
    *,
    manifest: dict | None = None,  # load from bundle_dir if None
    arch: str | None = None,  # auto-detect if None
    autotune: bool = True,  # benchmark configs
    op_filter: set[str] | None = None,  # e.g. {"mm", "convolution"}
) -> Path:
    """Replace extern_kernel steps with cutlass_kernel steps.

    Reads manifest.json from bundle_dir, replaces eligible extern_kernel
    steps in-place, writes cubin files, and saves the updated manifest.

    Returns the path to the updated manifest.json.
    """
```

### CLI entry point

```bash
python -m hyperonnx.compile.cutlass_bundle <bundle_dir> [--arch sm_90] [--no-autotune]
```

### Integration with export_hyper_onnx

```python
export_hyper_onnx(
    model,
    args,
    "model.onnx",
    compile=[Attention],
    cutlass_replace=True,  # NEW: run replacement after export
    cutlass_arch=None,  # NEW: auto-detect if None
    dynamo=True,
    external_data=True,
    external_directory="out/",
)
```

When `cutlass_replace=True`, `_collect_and_attach_kernels()` calls
`replace_extern_with_cutlass()` on each bundle after writing.

## New Step Type: `cutlass_kernel`

Extends `_KEEP_STEP_TYPES` in `bundle.py`:

```python
_KEEP_STEP_TYPES = frozenset({
    "allocate",
    "triton_kernel",
    "extern_kernel",
    "as_strided",
    "cutlass_kernel",  # NEW
})
```

### Step schema

```json
{
    "type": "cutlass_kernel",
    "kernel": "cutlass.mm",
    "cubin": "cutlass_kernel_0000.cubin",
    "device_target": {
        "backend": "cuda",
        "arch": "sm_90",
        "warp_size": 32
    },
    "launch": {
        "num_warps": 4,
        "num_ctas": 1,
        "shared_mem_bytes": 49152,
        "num_regs": 128,
        "grid_expr": null,
        "captured_grid": [32, 16, 1]
    },
    "args": [
        {"kind": "tensor", "name": "arg0_1", "buffer_id": 2},
        {"kind": "tensor", "name": "vbuf1", "buffer_id": null},
        {"kind": "scalar", "value": 768}
    ],
    "output": {"name": "buf0", "buffer_id": 3, "direction": "out"},
    "cutlass_config": {
        "tile_m": 128,
        "tile_n": 256,
        "tile_k": 64,
        "num_stages": 3,
        "num_warps": 4
    }
}
```

### `cutlass_config` field

Stores the winning autotune configuration for reproducibility. Present only
when autotuning was performed. Omitted when using a fixed config.

## Module Structure

```
hyperonnx/compile/
    cutlass.py              # Main pipeline: replace_extern_with_cutlass()
    cutlass_kernels/
        __init__.py         # Registry + dispatch
        mm.py               # GEMM: mm, bmm, addmm
        conv.py             # Convolution: convolution, cudnn_convolution
        config.py           # CutlassConfig dataclass + predefined grids
        autotune.py         # Benchmarking harness
        extract.py          # Cubin extraction from CuTe DSL compiled output
```

### Registry (`cutlass_kernels/__init__.py`)

```python
_REGISTRY: dict[str, Callable] = {
    "extern_kernels.mm": generate_mm,
    "extern_kernels.bmm": generate_bmm,
    "extern_kernels.addmm": generate_addmm,
    "extern_kernels.convolution": generate_conv,
    "extern_kernels.cudnn_convolution": generate_conv,
}
```

Unknown ops produce a warning and are skipped.

### Generator signature

Each generator follows:

```python
def generate_mm(
    args: list[dict],  # KernelArgDescriptor list from step
    output: dict,  # output descriptor
    buffers: dict,  # buffer table from manifest
    arch: str,  # e.g. "sm_90"
    config: CutlassConfig | None = None,
    autotune: bool = True,
) -> tuple[bytes, CutlassConfig, dict]:
    """Returns (cubin_bytes, winning_config, launch_descriptor)"""
```

### Cubin extraction (`extract.py`)

CuTe DSL's `cute.compile()` produces a compiled object. The cubin bytes
are extracted from the embedded fatbin:

```python
def extract_cubin(compiled, arch: str) -> bytes:
    """Extract raw cubin bytes from a CuTe DSL compiled object."""
    # Method 1: export_to_c() -> .o file -> parse fatbin
    # Method 2: CuTe DSL API if available for direct cubin access
```

## Autotuning

### Configuration grid (`config.py`)

```python
@dataclass
class CutlassConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    num_stages: int
    num_warps: int


MM_CONFIGS = [
    CutlassConfig(128, 256, 64, 3, 4),
    CutlassConfig(64, 128, 32, 2, 2),
    CutlassConfig(256, 128, 64, 4, 4),
    CutlassConfig(128, 128, 64, 3, 4),
    CutlassConfig(64, 64, 32, 2, 2),
    CutlassConfig(128, 64, 32, 2, 2),
]

CONV_CONFIGS = [
    CutlassConfig(128, 128, 8, 2, 4),
    CutlassConfig(64, 64, 8, 2, 2),
]
```

### Benchmarking (`autotune.py`)

```python
def autotune_kernel(
    generate_fn: Callable,
    args: list[dict],
    output: dict,
    buffers: dict,
    arch: str,
    configs: list[CutlassConfig],
    warmup: int = 5,
    iterations: int = 100,
) -> CutlassConfig:
    """Benchmark configs and return the fastest."""
```

Uses CUDA events for timing (`cuEventCreate`, `cuEventRecord`,
`cuEventElapsedTime`). No PyTorch timing — pure CUDA Driver API.

## Replay Integration

### `testing.py` changes

Add `run_cutlass` alongside `run_extern`:

```python
def run_cutlass(step: dict, table: dict) -> None:
    """Launch a cutlass_kernel step via CUDA Driver API."""
    cubin_path = bundle_dir / step["cubin"]
    module = drv.cuModuleLoadData(cubin_path.read_bytes())
    func = drv.cuModuleGetFunction(module, "cutlass_kernel")
    # Build kernel_params from step["args"] (same as triton steps)
    # Launch with step["launch"]["captured_grid"]
```

The dispatch in the main loop becomes:

```python
for step in graph["steps"]:
    if step["type"] == "triton_kernel":
        launch_triton(step, table)
    elif step["type"] == "extern_kernel":
        run_extern(step, table)
    elif step["type"] == "cutlass_kernel":
        run_cutlass(step, table)  # NEW
    elif step["type"] == "as_strided":
        run_as_strided(step, table)
```

## GPU Auto-Detection

```python
def detect_gpu_arch() -> str:
    """Returns CUDA compute capability, e.g. 'sm_90'."""
    try:
        import torch

        cap = torch.cuda.get_device_capability()
        return f"sm_{cap[0]}{cap[1]}"
    except Exception:
        from cuda.bindings.driver import (
            cuDeviceGetAttribute,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )

        major = cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0)
        minor = cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0)
        return f"sm_{major}{minor}"
```

## Dependency Management

### pyproject.toml

```toml
[dependency-groups]
cutlass = [
    "nvidia-cutlass>=4.2.0.0; sys_platform == 'linux'",
    "nvidia-cutlass-dsl[cu13]; sys_platform == 'linux'",
]
```

- Separate `cutlass` dependency group (not in `dev`)
- `uv sync --group cutlass` in Docker/Linux
- Not installed on Windows/Mac
- Existing `nvidia-cutlass>=4.2.0.0` in `dev` can be moved to this group

### Lazy imports

```python
def _require_cutlass():
    try:
        import cutlass.cute as cute

        return cute
    except ImportError:
        raise RuntimeError(
            "CUTLASS CuTe DSL required. Install with: uv sync --group cutlass"
            " (Linux only)"
        )
```

## Docker Usage

The existing Dockerfile gets a `--group cutlass` addition:

```dockerfile
RUN sed -i '/onnxifier.*path.*ONNXifier/d' pyproject.toml && \
    uv sync --extra cuda --dev --group cutlass --python=3.13
```

No other Docker changes. The replacement pipeline runs as normal Python
inside the container.

## Manifest Schema Changes

### Schema version

`_SCHEMA_VERSION` stays at 2 — `cutlass_kernel` is a new step type within
the existing schema, not a breaking change. Runtimes that don't understand
`cutlass_kernel` can fall back to the `extern_kernel` step (which remains
in the manifest as a comment field, see below).

### Extern step preservation

After replacement, the original `extern_kernel` step is kept as a
`_original_extern` field on the `cutlass_kernel` step for debugging:

```json
{
    "type": "cutlass_kernel",
    "kernel": "cutlass.mm",
    "cubin": "cutlass_kernel_0000.cubin",
    "_original_extern": {
        "kernel": "extern_kernels.mm",
        "args": [...],
        "kwargs": [...]
    }
}
```

This field is stripped before replay (not consumed by runtime).

## Testing

### Unit tests

- `tests/compile/test_cutlass_config.py`: config grid, serialization
- `tests/compile/test_cutlass_registry.py`: registry dispatch, unknown ops
- `tests/compile/test_cutlass_extract.py`: cubin extraction mock

### Integration tests (require GPU + CuTe DSL)

- `tests/compile/test_cutlass_mm.py`: replace mm step, verify cubin
- `tests/compile/test_cutlass_autotune.py`: benchmark configs, pick winner
- `tests/compile/test_cutlass_replay.py`: replay manifest with cutlass steps

### Test markers

```python
@pytest.mark.skipif(not _HAS_CUTLASS, reason="CuTe DSL not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
```

## Implementation Phases

### Phase 1: Foundation
- `config.py`: CutlassConfig dataclass + predefined grids
- `extract.py`: cubin extraction from CuTe DSL output
- `__init__.py`: registry skeleton + dispatch
- `cutlass.py`: `replace_extern_with_cutlass()` main loop

### Phase 2: GEMM kernels
- `mm.py`: `generate_mm()`, `generate_bmm()`, `generate_addmm()`
- Autotuning for GEMM configs
- Integration test: replace mm in a real manifest

### Phase 3: Convolution kernels
- `conv.py`: `generate_conv()`
- Autotuning for conv configs

### Phase 4: Replay + integration
- `testing.py`: `run_cutlass()` replay function
- `bundle.py`: `cutlass_kernel` step type support
- `hyper_export.py`: `cutlass_replace=True` parameter
- End-to-end test: export -> replace -> replay -> verify

## Open Questions

1. **CuTe DSL cubin extraction**: `export_to_c()` produces ELF `.o` files, not
   raw cubins. `cuModuleLoadData` cannot load these directly. The verification
   uses `compiled()` directly (CuTe DSL runtime). A future approach could use
   `cuLibraryLoadData` (CUDA 12+) or `cuobjdump --dump-cubin` to extract raw
   cubins from the ELF.
2. **Grid/block dims**: CuTe DSL requires explicit grid/block in
   `kernel.launch()`. These are config-dependent and must be computed
   from the tile sizes and problem shape.
3. **Shared memory**: CUTLASS kernels often use shared memory. The
   `launch.shared_mem_bytes` must be set correctly from the compiled
   kernel's metadata.

## Verification Results (2026-08-07)

ResNet18 BasicBlock_7 kernel bundle with CUTLASS replacement verified on
Blackwell (sm_120) in Docker:

| Metric | Value |
|--------|-------|
| GEMM max diff (fp16) | 0.003906 |
| Conv1 max diff | 0.001949 |
| Full block max diff | 0.000790 |
| NaN/Inf | None |

CUTLASS tiled GEMM kernel produces correct results compared to PyTorch
reference. The kernel uses a simple tiled approach (16x16 tiles) —
production performance would benefit from MMA atoms and shared memory.
