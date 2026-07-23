# Compiled CUBIN Replay Harness via CuPy

- **Status**: Draft
- **Date**: 2026-07-22
- **Owner**: HyperONNX Authors
- **Branch**: TBD (new feature branch)

## Summary

A verification harness that loads the cubin files written by the existing
`compile=` export path, replays the full kernel sequence via the CuPy CUDA
Driver API wrapper (no PyTorch on the launch path), and compares the output
against the original `torch.compile` reference. This closes the
"runtime-side correctness verification" gap that the compile-and-kernel-export
design explicitly deferred as a non-goal — not as a permanent exclusion, but
as a separable concern. This spec owns that concern.

**Design principle**: the harness observes a live forward pass to learn each
kernel's exact launch signature (grid, shared mem, ordered arg list with
tensor/scalar classification), then replays the cubins from the existing
bundle using CuPy alone. The export pipeline (`hyperonnx/compile/`,
`hyperonnx/hyper_export.py`) is **not modified** — the manifest's deferred
`args=[]` / `grid_expr=null` fields stay deferred; the harness fills that
gap at runtime instead of on disk.

## Goal

1. Prove a compiled module's cubins produce bit-close outputs to the
   `torch.compile` reference, for a real model (ResNet-18).
2. Zero export-pipeline changes. The harness reads cubins from the existing
   `<type_name>.kernels/` bundle and re-observes launches live.
3. Pure-CuPy launch path: no PyTorch in the replay loop (torch is used only
   to produce the reference output and to drive the one observed forward).

## Non-goals

- **No on-disk launch trace.** The manifest's `args`/`grid_expr` completion
  is a v1.1 roadmap item owned by the compile-export spec; this harness does
  not write a `launch_trace.json` or modify the manifest. The observed trace
  is ephemeral (in-process, discarded after replay).
- **No dynamic shapes.** Replay runs on the same shape that was observed.
  Grid is read verbatim from the observed launch; `grid_expr` evaluation is
  out of scope.
- **No general runtime.** This is a verification tool, not a production
  dispatcher. It always runs every kernel in the captured order; it does
  not fall back to the ONNX function or make dispatch decisions.
- **No per-kernel intermediate verification.** Intermediates are threaded
  through by buffer identity (see Buffer threading); only the module's final
  output is compared against the reference.

## Architecture

Two components under a new `hyperonnx/replay/` package. No modifications to
existing files.

```
hyperonnx/replay/
├── __init__.py        # re-exports replay and verify
├── runner.py          # observe launches + replay cubins via CuPy
└── verify.py          # end-to-end check: export -> replay -> allclose
```

### Data flow

```
 ┌──────────────────────────────────────────────────────────────┐
 │  replay(module, sample_args, bundle_dir)                     │
 └────────────────────────────┬─────────────────────────────────┘
                              ▼
   (1) OBSERVE (one torch.compile forward, CompiledKernel hook)
       for each kernel launch, record:
         {symbol, grid (gx,gy,gz), shared_mem,
          args: [(kind, ...)] ordered}
       classify each arg:
         tensor  -> buffer_id (by data_ptr), shape, dtype
         scalar  -> value, dtype
       save reference output tensor
                              ▼
   (2) LOAD BUNDLE
       read manifest.json -> {symbol: cubin_filename} map
       load each cubin once via cupy.RawModule(path=...)
       resolve each symbol via mod.get_function(symbol)
                              ▼
   (3) ALLOCATE BUFFERS
       one cupy array per unique buffer_id:
         input buffers  <- copied from sample_args
         intermediate   <- cupy.zeros(shape, dtype)
         output buffer  <- identified by matching ref output's data_ptr
                              ▼
   (4) REPLAY (pure cupy, kernel-by-kernel in observed order)
       for each recorded launch:
         build args list:
           tensor arg -> cupy_array[buffer_id].data.ptr
           scalar arg -> numpy scalar
         set CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
         launch: ker((gx,gy,gz), (block_x,1,1), args, shared_mem=...)
       sync stream
                              ▼
   (5) RETURN output cupy array (the buffer matching ref output)
```

### The observe hook

The seam is `triton.compiler.CompiledKernel.__call__` (the method inductor's
generated wrapper invokes to launch a compiled kernel on-device). One
monkey-patch, installed for exactly one `torch.compile(module)(*args)`
forward, captures every launch with:

- **symbol** — from the `CompiledKernel.metadata.name` (matches
  `manifest.kernels[i].symbol`, so the cubin can be looked up in the bundle).
- **grid** — the `(grid_x, grid_y, grid_z)` integer args.
- **shared_mem** — the `shared` kwarg (or 0 if absent).
- **args** — the remaining positional args, classified at observe time.
  The classifier uses `cupy.cuda.runtime.pointerGetAttributes(ptr)` as the
  primary discriminator: a device pointer → **tensor** arg; anything else
  → **scalar** arg. For tensor args, shape and dtype are recovered by
  looking up the `data_ptr` in a registry built from `gc.get_objects()`
  (scanning for live `torch.Tensor` instances with `is_cuda`). The registry
  is built once per observe pass; it covers inputs and all intermediates.
  - **tensor** → `(buffer_id, shape, dtype)` where `buffer_id` is the
    stable int assigned per unique `data_ptr`.
  - **scalar** → `(value, dtype)` recorded as a numpy scalar.

**Exact triton `__call__` signature discovery** is implementation work
(it varies slightly across triton 3.x minors); the spec pins only the seam
(the `CompiledKernel` launch entry point) and the data captured. If the
`__call__` surface shifts across the supported torch range (2.5–2.10), the
hook falls back to patching the nearest stable launch entry (e.g.
`CompiledKernel.run`) — both routes expose the same data.

### Vendor-library ops are invisible (known limitation)

The observe hook targets inductor's triton launch path
(`StaticallyLaunchedCudaKernel.run`). Inductor does **not** lower every op
to triton: it delegates convolutions to **cuDNN** (`aten::cudnn_convolution`)
and many matmuls to **cuBLAS** (`aten::mm`/`addmm`/`_scaled_mm`). Those calls
run through the vendor library's own kernel-launch path, never pass through
the triton launcher, and produce **no capturable cubin**.

Consequence: when a compiled module's graph contains a vendor-library op,
the buffer it produces is never written by any captured triton kernel, and
every downstream consumer computes on garbage at replay.

This is confirmed empirically: a pure pointwise module (`y = x*2 + x`)
replays bit-close to the reference, while a single `nn.Conv2d` does not —
profiling the compiled forward shows `aten::cudnn_convolution` producing the
conv-result buffer that the captured triton kernels only consume.

### Capture-time coverage detection (write-direction)

Rather than maintaining a brittle op-name whitelist, the capture pipeline
detects vendor-library delegation **generically** via buffer-write coverage.
Inductor's launcher exposes `arg_names` for each compiled kernel, using a
stable naming convention: write-back pointers are named `out_ptr*`, reads
are `in_ptr*`. The launch-trace spy records `"direction": "out"` on every
tensor arg bound to an `out_ptr*` name.

After the forward, `LaunchTraceSink.vendor_lib_gaps()` computes the set of
buffer_ids written by triton (every tensor arg flagged `direction="out"`
across all launches) and returns any intermediate/output buffer not in that
set. Those gap buffers were produced by a vendor-library call — no cubin,
no triton launch, invisible to the spy.

This approach is:
- **Generic** — detects cuDNN, cuBLAS, or any future vendor lib by coverage
  gap, with zero op-name coupling.
- **Version-robust** — relies only on inductor's `in_ptr`/`out_ptr` naming
  convention (stable since torch 2.4+).
- **Cheap** — arg-name inspection at capture, no memory snapshotting.

### Partial bundles with a ``vendor_lib`` marker

When gaps are detected, the bundle is **still written** — all triton
kernels dump normally (they are valid cubins that replay correctly for the
subgraphs they cover). The manifest gains a top-level ``vendor_lib`` key:

```json
"vendor_lib": {"unwritten_buffers": [5, 7]}
```

listing the buffer_ids no triton kernel produced. A downstream runtime reads
this key to know it must fill those buffers itself (e.g. via raw cuDNN/cuBLAS
calls — see Open questions) or fall back to the ONNX function for the
affected subgraph. Absent ``vendor_lib`` ⇒ full triton coverage.

Dead ends confirmed during investigation:
- `torch.backends.cudnn.enabled = False` triggers an inductor stride/layout
  assertion failure on the supported torch range.
- `TorchDispatchMode` cannot see the calls because inductor runs its
  compiled graph under its own dispatch, bypassing Python dispatch modes.
- Monkey-patching `cuLaunchKernel` in Python does not help because inductor
  launches from C++ via the CUDA driver directly.

### Buffer threading

A compiled module is a pipeline: kernel #1's output tensor is kernel #2's
input. The harness threads these correctly by deduping tensors via
`data_ptr()`:

- During observe, each unique `data_ptr` is assigned a stable `buffer_id`
  (an incrementing int).
- Multiple kernels referencing the same `data_ptr` share the `buffer_id`.
- At replay, **one** cupy array is allocated per `buffer_id`. Inputs are
  populated from `sample_args`; intermediates are zeroed. Every kernel
  referencing that `buffer_id` receives the same `cupy_array.data.ptr`.

This is what makes a multi-kernel module verify without per-kernel
intermediate reference data — the intermediate buffer is produced by the
upstream cubin and consumed by the downstream cubin, exactly as torch did.

**Input identification**: `sample_args` tensors are matched to buffer_ids
by their `data_ptr()` at observe time. The output buffer_id is the one
whose `data_ptr()` matches the reference output tensor.

### Block dimensions

Triton kernels are 1-D within a block: `block = (num_warps * warp_size, 1, 1)`.
`num_warps` comes from `manifest.kernels[i].launch.num_warps`; `warp_size`
from `manifest.kernels[i].device_target.warp_size`. The harness reads both
from the manifest rather than re-observing.

### Shared memory

`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` is set on each loaded
`cupy.cuda.Function` before launch when `shared_mem > 0`. The value comes
from the observed launch (the `shared` kwarg to `CompiledKernel.__call__`),
cross-checked against `manifest.kernels[i].launch.shared_mem_bytes`.

### CuPy as optional dependency

CuPy is imported lazily inside `runner.py` and `verify.py` (same pattern as
triton in `compile/capture.py`). A missing CuPy produces a clear
`ImportError` with install instructions, never a silent skip.

## API

### `replay(module, sample_args, bundle_dir)`

```python
def replay(
    module: Module,
    sample_args: tuple[Tensor, ...],
    bundle_dir: str | Path,
) -> cupy.ndarray:
    """Observe one torch.compile forward, then replay cubins via CuPy.

    Args:
        module: the nn.Module whose cubins were exported. Must be on CUDA.
        sample_args: concrete input tensors (CUDA). Must match the shapes
            used during export.
        bundle_dir: path to the ``<type_name>.kernels/`` directory written
            by the compile export path.

    Returns:
        The output cupy.ndarray (the buffer matching the reference output).

    Raises:
        ImportError: if CuPy is not installed.
        RuntimeError: if a cubin symbol in the trace is not found in the
            bundle, or if the GPU arch is incompatible.
    """
```

### `verify(module, sample_args, bundle_dir, atol, rtol)`

```python
def verify(
    module: Module,
    sample_args: tuple[Tensor, ...],
    bundle_dir: str | Path,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Replay and compare against torch.compile reference.

    Returns True if ``cupy.allclose(replay_out, ref_out)``; False otherwise.
    Prints max abs diff on failure.
    """
```

### `__main__` self-check

`python -m hyperonnx.replay.verify` runs the ResNet-18 smoke test (see
Testing), prints `PASS` / `FAIL` + max diff, exits 0 / 1.

## Example usage

```python
import torch
from torchvision.models import resnet18
from hyperonnx import export_hyper_onnx
from hyperonnx.replay import verify

model = resnet18().cuda().eval()
sample = (torch.randn(1, 3, 224, 224, device="cuda"),)

export_hyper_onnx(
    model, sample, "resnet18.onnx",
    compile=[type(model.layer1[0])],   # BasicBlock
    dynamo=True,
    external_data=True,
    external_directory="out/",
)

ok = verify(model, sample, "out/BasicBlock:0.kernels/")
print("verified" if ok else "MISMATCH")
```

## Error handling

| # | Failure | Severity | Action |
|---|---------|----------|--------|
| 1 | CuPy not installed | Hard | `ImportError` with install hint. |
| 2 | Triton not available / no GPU | Hard | `RuntimeError` — the harness is meaningless without a CUDA device that can compile. |
| 3 | A symbol in the trace has no matching cubin in the bundle | Hard | `RuntimeError` listing the missing symbol. |
| 4 | `device_target.arch` mismatch with runtime GPU | Hard | `RuntimeError` naming both archs. |
| 5 | `allclose` fails | Soft (verify) / raised (assert in self-check) | Print max abs diff + failing buffer shape. |

No silent degradation. The harness is a verification tool; a failure must
be loud.

## Testing

### File locations

```
tests/
├── expoter/
│   └── test_replay_runner.py        # observe hook + arg classification (mock)
└── test_replay_integration.py       # CUDA: ResNet-18 end-to-end
```

### Tier 1 — Unit (no GPU)

| Test | Asserts |
|------|---------|
| `test_classify_tensor_arg` | An int matching a live tensor's `data_ptr` is classified as `tensor` with correct `buffer_id`, shape, dtype. |
| `test_classify_scalar_arg` | A bare int/float/bool is classified as `scalar` with correct value. |
| `test_buffer_dedup` | Two args sharing the same `data_ptr` get the same `buffer_id`. |
| `test_symbol_to_cubin_map` | Given a manifest with 2 kernels, the symbol→cubin map resolves correctly. |

These test the classification + mapping logic in isolation by calling the
internal helpers directly (no real triton launch).

### Tier 2 — Integration (CUDA + triton + cupy required)

| Test | Asserts |
|------|---------|
| `test_replay_resnet18_basicblock` | ResNet-18 `BasicBlock` cubins replay to within `atol=1e-3` of `torch.compile` reference. |
| `test_replay_single_conv` | A single `nn.Conv2d` (1 cubin) replays correctly — minimal multi-buffer sanity check. |

Gated by:

```python
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
```

CuPy availability checked at import inside the test body (skip with clear
message if absent).

### `__main__` self-check

`python -m hyperonnx.replay.verify` constructs a ResNet-18, exports a
compiled `BasicBlock`, runs `verify(...)`, asserts the result, prints a
one-line PASS/FAIL. This is the ponytail-compliant runnable check that
fails if the replay logic breaks.

## Version compatibility

- **CuPy**: `>=13.0` (for `RawModule(path=...)` cubin loading + dynamic
  shared mem attribute setting). Lazy import; not added to core deps.
- **Torch**: `<2.11` (same as the rest of HyperONNX). The observe hook
  targets the `CompiledKernel` surface stable across this range.
- **Triton**: whatever ships with the supported torch versions.

## Open questions

None blocking. If the `CompiledKernel.__call__` signature proves unstable
across the supported triton range, the implementation falls back to a
nearby stable launch entry point; the spec pins the seam, not the exact
method name.
