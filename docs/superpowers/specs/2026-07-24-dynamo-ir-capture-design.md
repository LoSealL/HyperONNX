# Dynamo IR Capture (per-module FX graph text dump)

- **Status**: Draft
- **Date**: 2026-07-24
- **Owner**: HyperONNX Authors

## Summary

For every module exported through the `compile=` path, capture the FX graph
that inductor receives (the dynamo IR) and write it as a human-readable text
dump, one file per compiled module, into `external_directory`. The dump is
written **unconditionally** — even when no kernel bundle is produced (fully
eager module) — so users can always inspect what inductor was asked to
compile.

## Goal

1. Given a compiled module, produce `<legalized(type_name)>.dynamo_ir.txt`
   in `external_directory` containing the FX graph as printed by
   `gm.print_readable()`.
2. Works for every compiled module, including those that capture zero triton
   kernels (no `.kernels/` bundle written).
3. No new CLI flags; rides the existing `compile=` path.

## Non-goals

- **No structured/JSON form.** The dump is text for human inspection, not a
  machine-parseable format. (A JSON form could be a follow-up if needed.)
- **No re-execution guarantees.** The dump is documentation, not a runnable
  artifact.
- **No IR for non-compiled modules.** Only modules in `compile=[...]`.

## Architecture

The `compile_fx` seam already used by `capture_vendor_ops` sees every FX
graph `gm` handed to inductor. Extend that same wrapper — no new seam.

```
capture_vendor_ops()                    (hyperonnx/compile/capture.py)
  └─ _record(gm, ...)                   wraps torch._inductor.compile_fx
        ├─ records compute-node descriptors   (existing)
        └─ appends gm.print_readable() → sink.ir_dumps   (NEW)

write_dynamo_ir(directory, type_name, dumps)   (NEW, capture.py)
  └─ legalizes name, writes UTF-8 text file(s)

hyper_export.py _collect_and_attach_kernels()
  └─ after the forward: write_dynamo_ir(out_dir, type_name, vendor_ops.ir_dumps)
     (before / independent of the `not sink.kernels → continue` check)
```

### Components

- `VendorOpSink.ir_dumps: list[str]` — one entry per FX graph seen at the
  compile_fx boundary. Dynamo may split a module into multiple graphs (graph
  breaks); each gets an entry.
- `_record` wrapper — appends `gm.print_readable()`; falls back to
  `str(gm.graph)` when `print_readable()` returns `None` (version-dependent).
- `write_dynamo_ir(directory, type_name, dumps)` — file writer, uses
  `legalize_path_name(f"{type_name}")` for naming consistency with bundles.
  Empty `dumps` → writes nothing, debug-log only.
- `hyper_export.py` — invokes the writer after the compiled forward,
  regardless of whether a kernel bundle is written.

### Naming

- **1 graph** (common case): `<legalized(type_name)>.dynamo_ir.txt` —
  name-matched to the module and its `.kernels/` bundle (if any).
- **N graphs** (dynamo graph breaks): `<legalized(type_name)>_<i>.dynamo_ir.txt`,
  `i = 0..N-1`.

### Data flow

1. `torch.compile(module)` triggers inductor; each FX graph passes through
   the wrapped `compile_fx`.
2. The wrapper appends the readable text to `sink.ir_dumps`.
3. After the forward, `hyper_export` writes the dump(s) to
   `external_directory`.
4. Kernel-bundle capture proceeds independently (bundle written only when
   `sink.kernels` is non-empty).

## Error handling

| Failure | Behavior |
|---------|----------|
| `gm.print_readable()` returns `None` | Fall back to `str(gm.graph)`. |
| `ir_dumps` empty (compile cached/failed before compile_fx) | Write nothing; debug log. |
| Write I/O error | Propagate (same as `write_kernel_bundle`). |

No silent data loss: an empty dump set is logged, never silently skipped
without a trace.

## Testing

Extend `tests/compile/test_replay_integration.py`:

| Test | Asserts |
|------|---------|
| pointwise model | `*_P_0.dynamo_ir.txt` exists; contains the fused mul/add triton kernel name. |
| conv model | `*_M_0.dynamo_ir.txt` exists; contains `conv2d`. |

Both ride the existing CUDA-gated integration file; no new fixtures.

## Version compatibility

- `gm.print_readable()` exists across the supported torch range
  (`torch.fx.GraphModule.print_readable`); the `str(gm.graph)` fallback
  covers versions where it returns `None` instead of the string.
- The `compile_fx` wrap is the same seam already used for vendor-op capture.

## Open questions

None blocking. A JSON IR form, if ever needed, would be a separate feature
with its own spec.
