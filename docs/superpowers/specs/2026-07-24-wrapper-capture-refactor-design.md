# Refactor: Triton-First Bundle — Wrapper Capture, TTIR Behind Flag

- **Status**: Draft
- **Date**: 2026-07-24
- **Owner**: HyperONNX Authors

## Summary

Refocus the kernel bundle on what a torch-free replay runtime actually needs:
cubins (always), the inductor `def call()` execution procedure (when
vendor-lib ops interleave), and TTIR/TTGIR (debug-only, behind an env var).

Removes TTIR/TTGIR from the default bundle output — they are developer
diagnostics, not replay inputs. Adds the inductor wrapper's `def call()`
body as `wrapper.py` inside the bundle when buffer-coverage detects
vendor-library gaps — this is the authoritative execution order showing
how triton kernels and vendor-lib calls (`extern_kernels.convolution`,
etc.) interleave.

## Goal

1. **Cubins always** — unchanged, the core artifact.
2. **Wrapper `def call()` body when vendor gaps exist** — saved as
   `wrapper.py` in the bundle; shows the exact triton↔vendor call order.
3. **TTIR/TTGIR behind `HYPERONNX_TTIR=1`** — off by default; on only for
   debugging.

## Non-goals

- No structured/JSON wrapper form (text only).
- No wrapper capture for pure-triton modules (no vendor gaps → no wrapper).
- No parsing of the wrapper into a replay dispatcher — it is documentation
  of the execution procedure for a human (or future runtime) to read.

## Design

### Env-var gating for TTIR/TTGIR

`capture.py` `CaptureSink.record()` checks `os.environ.get("HYPERONNX_TTIR")`
before extracting `asm["ttir"]`/`asm["ttgir"]`. Off → fields absent →
`bundle.py` writes no `.ttir`/`.ttgir` files. On → current behavior.

### Wrapper extraction

`hyper_export.py` `_collect_and_attach_kernels`: after the compiled forward
(while `cache_dir` still exists), when `trace.vendor_lib_gaps()` is
non-empty:

1. Scan `cache_dir` for the inductor wrapper `.py` (the file containing
   `def call(`).
2. Parse via `ast.parse(text)`, find `FunctionDef` named `call`, extract
   via `ast.get_source_segment`.
3. Pass the extracted text to `write_kernel_bundle(..., wrapper_text=text)`.

`write_kernel_bundle` writes `wrapper.py` (UTF-8) into the bundle when
`wrapper_text` is a non-empty string.

### Error handling

| Failure | Behavior |
|---------|----------|
| No wrapper `.py` found in cache dir | Warning; bundle ships without `wrapper.py`. |
| `ast.parse` fails | Warning; continue without wrapper. |
| No `def call` function in the file | Warning; continue without wrapper. |
| TTIR env var unset | No `.ttir`/`.ttgir` files; cubins unaffected. |

## Bundle layouts

**Pure-triton module (default)**
```
_P_0.kernels/
├── manifest.json
├── kernel_0000.cubin
└── ...
```

**Vendor-lib module**
```
MetaFormer_1.kernels/
├── manifest.json          # vendor_lib.ops[] + unwritten_buffers
├── kernel_0000.cubin
└── wrapper.py             # def call() body
```

**Debug (`HYPERONNX_TTIR=1`)**
```
_P_0.kernels/
├── manifest.json
├── kernel_0000.cubin
├── kernel_0000.ttir
├── kernel_0000.ttgir
└── ...
```

## Testing

| Test | Asserts |
|------|---------|
| pointwise (existing) | No `.ttir`/`.ttgir` by default; no `wrapper.py` (pure triton). |
| conv (existing) | `wrapper.py` exists; contains `extern_kernels.convolution`. |
| conv with `HYPERONNX_TTIR=1` (monkeypatch) | `.ttir`/`.ttgir` files present. |
