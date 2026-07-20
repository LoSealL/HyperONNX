# Selective `torch.compile` Module Export with CUDA Kernel Bundle

- **Status**: Draft
- **Date**: 2026-07-20
- **Owner**: HyperONNX Authors
- **Branch**: TBD (new feature branch)

## Summary

Add the ability to selectively `torch.compile` a subset of `nn.Module`s in a model
during HyperONNX export, and emit — alongside the existing ONNX function body for
each such module — a **kernel bundle**: the compiled CUDA cubin(s) plus a
self-describing JSON manifest. The ONNX function body remains the portable,
language-agnostic fallback that any ONNX runtime can execute; the kernel bundle
is a strictly optional sidecar that a third-party runtime may dispatch for
performance.

**Design principle**: the kernel bundle is a *pure sidecar*. Deleting the
`.kernels/` directory must leave a valid ONNX model indistinguishable from one
exported without `compile`. No ONNX graph mutation, no new custom op, no
runtime coupling to Python.

## Goals

1. Let the user mark specific module types for `torch.compile` during export.
2. Capture each compiled kernel's cubin + launch metadata at export time.
3. Produce a language-agnostic manifest (JSON) describing kernel args, launch
   constraints, and (when extractable) a grid expression AST so the runtime
   can launch the cubin on shapes other than the captured one.
4. Provide a static-grid bypass mode for fixed-shape deployments where AST
   extraction is unnecessary.
5. Zero coupling to PyTorch at runtime — the bundle must be consumable from
   C / C++ / Rust via the CUDA Driver API alone.

## Non-goals (v1)

- **No AOTI `.so` escape hatch.** v1 ships cubin-only. The AOTI path (Tier 2
  from brainstorming) is deferred to v2; the manifest schema reserves no field
  for it to avoid premature abstraction.
- **No autotune multi-version selection.** v1 captures the single kernel
  inductor actually ran. The manifest has a `variants: []` list reserved so v2
  can extend without a schema break.
- **No runtime-side correctness verification.** HyperONNX produces a
  well-formed bundle; it does not load or execute the cubin. Dispatch
  correctness is the runtime's responsibility.
- **No PyTorch roundtrip.** The user's stated target is a third-party
  language-agnostic runtime; loading the bundle back through PyTorch is not
  in scope.

## API

### New parameters

`export_hyper_onnx` gains two keyword-only parameters:

```python
def export_hyper_onnx(
    model, input_args, f, *,
    hiera: Collection[type[Module]] | None = None,
    compile: Collection[type[Module]] | None = None,        # NEW
    compile_static_grid: bool = False,                      # NEW
    ...
)
```

`auto_trace_method(...).export(...)` passes both through verbatim to
`export_hyper_onnx`.

### Parameter semantics

- **`compile`**: a container of `nn.Module` subclasses, same matching rule as
  `hiera` (`type(child) in compile`).
- **`compile_static_grid`**: when `True`, skip grid AST extraction entirely;
  treat every captured grid as a static 3-tuple. Default `False`.

### Relationship to `hiera`

`compile ⊆ hiera` — every compiled module must also be an ONNX function
(the function body *is* the fallback the runtime executes when it declines
to dispatch the kernel). Concretely:

- If a type is in `compile` but not `hiera`, HyperONNX **auto-promotes** it:
  internally `hiera = set(hiera or []) | set(compile or [])`. Silent.
- If a type is in `hiera` but not `compile`, unchanged behaviour: ONNX
  function only, no kernel sidecar.
- A type in both: function exported as today, **plus** a kernel bundle is
  written next to it.

This keeps the existing `hiera` path 100% backward-compatible; `compile` is
purely additive.

### Example

```python
export_hyper_onnx(
    model,
    (torch.randn(8, 768),),
    "model.onnx",
    hiera=[DecoderLayer, Attention],
    compile=[Attention],            # Attention gets a kernel bundle
    dynamo=True,
    external_data=True,
    external_directory="out/",
)
```

## Pipeline / data flow

The export pipeline extends the existing `hiera` flow with two new steps
(`_collect_compiled_kernels` and `_attach_kernel_bundle`), inserted *after*
`_export_hiera` so the ONNX function body is built before any compile work.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  export_hyper_onnx(model, ..., hiera=[A], compile=[A])          │
 └────────────────────────────┬────────────────────────────────────┘
                              ▼
   (1) trace_module_spec (UNCHANGED)
       forward hooks grab spec per module:
         spec[A] = {args, kwargs, signature, output, status:FORWARDED}
                              ▼
   (2) _export_hiera (UNCHANGED)
       builds ONNX function body:
         spec[A].onnx = <ModelProto>, spec[A].status = EXPORTED
                              ▼
   (3) _collect_compiled_kernels (NEW, only if compile is non-empty)
       with _capture_triton_kernels() as cap:
           for module where type in compile and status == EXPORTED:
               torch.compile(module)(*spec.args, **spec.kwargs)
       cap.kernels  = [CompiledKernel, ...]          # from triton compile hook
       cap.grids    = {kernel_name: (x,y,z), ...}    # from inductor wrapper hook
                              ▼
   (4) _attach_kernel_bundle (NEW)
       for each compiled module:
           for each captured kernel k:
               dump cubin bytes
               extract descriptor from k.metadata
               if not compile_static_grid:
                   parse grid expression -> AST (see "Grid AST")
                   on failure: grid_expr = null
               else:
                   grid_expr = null
               write bundle files under external_directory/<type>.kernels/
           attach manifest reference into spec[A]["kernel_bundle"]
                              ▼
   (5) rest of export_hyper_onnx (UNCHANGED)
       combine top-level model, run rewriters, save ONNX.
       The ONNX graph has NO reference to the kernel bundle; the manifest
       is discovered by filename convention (see "Bundle layout").
```

### Why a separate compile pass (not fused into step 1)

`trace_module_spec` (step 1) must stay unchanged — it is the contract with
the existing `hiera` path and runs the **uncompiled** model to capture I/O
shapes. `torch.compile` mutates module state (wraps `forward`, installs
guards); that pollution must not leak into tracing. Compile therefore runs
in a **separate** pass, after the ONNX function is already built. Both
paths stay independently testable.

### Triton hooks (step 3 capture points)

Two separate captures happen during step 3, each via a monkey-patch installed
by the `_capture_triton_kernels` context manager:

**(a) Cubin capture** — patch `triton.compiler.compile`:

```python
@contextmanager
def _capture_triton_kernels():
    import triton.compiler as tc
    captured = {"kernels": [], "grids": {}}
    orig_compile = tc.compile
    def _spy(src, target=None, options=None, **kw):
        ck = orig_compile(src, target, options, **kw)
        captured["kernels"].append({"kernel": ck, "src": src})
        return ck
    tc.compile = _spy
    # (b) Grid capture — patch the inductor wrapper's grid-lambda emitter.
    _install_grid_spy(captured["grids"])
    try:
        yield captured
    finally:
        tc.compile = orig_compile
        _uninstall_grid_spy()
```

`CompiledKernel` exposes everything we need:
- `ck.asm[binary_ext]` — the cubin bytes (`binary_ext = "cubin"` for CUDA).
- `ck.metadata` — a namedtuple with `name`, `num_warps`, `num_ctas`,
  `shared`, `num_regs`, and the `GPUTarget` (`backend`, `arch`, `warp_size`).

**(b) Grid capture** — patch `PythonWrapperCodegen` (specifically the
function that emits `def grid_wrapper_for_<name>(meta):`) to record, per
kernel name, both:
- the **source** of the grid lambda body (for AST extraction in non-bypass mode);
- the **concrete integer 3-tuple** actually returned when the lambda is
  invoked once at export time with the captured inputs (this becomes
  `captured_grid`).

The two captures are correlated by kernel name: `ck.metadata.name` from (a)
matches the `<name>` suffix in `grid_wrapper_for_<name>` from (b).

**Why two hooks, not one**: triton's `compile` runs *before* inductor emits
the wrapper code that calls the compiled kernel with a concrete grid. The
grid depends on runtime shapes and inductor's blocking heuristics, so it is
not knowable at triton-compile time. Two hooks are necessary; combining
them would require inductor internals that aren't part of triton's API.

### Why per-module compile (not whole-model)

Step 3 runs `torch.compile(module)` **per compiled module**, not on the
whole model. This is cheaper and isolates failures: one module failing to
compile does not abort the export of the others (each module's failure is
categorised per the error table below).

## Bundle layout

```
<external_directory>/
├── <ModelTypename>_combined.onnx      # top-level ONNX (unchanged)
├── <ModelTypename>.onnx               # function body for the top-level
├── Attention:0.onnx                   # ONNX function body (fallback path)
├── Attention:0.kernels/               # NEW: bundle for "Attention:0"
│   ├── manifest.json
│   ├── kernel_0000.cubin
│   ├── kernel_0001.cubin
│   └── ...
└── DecoderLayer:0.onnx                # hiera-only module (no .kernels/)
```

### Naming convention (no ONNX graph reference)

- The bundle directory name **must** equal `<type_name>` where
  `type_name = spec["type_name"]` — the same identifier used by
  `ComposeOnnxAsFunctionRewriter` to label the ONNX function and by
  `module_spec` generally. Naming is deterministic and round-trippable.
- **Discovery contract**: a runtime loads `Attention:0.onnx` (the ONNX
  function), then probes for a sibling `Attention:0.kernels/manifest.json`.
  If present, the kernel path is available. If absent, run the ONNX
  function. **No new ONNX op, no graph mutation, no runtime coupling.**

### Why flat kernel files

Inductor sometimes produces 50+ kernels for one module. A flat layout
(`kernel_0000.cubin`, `kernel_0001.cubin`, ...) is easier for a C runtime
to `opendir` / scan than a nested per-kernel directory tree. There is
exactly one structured metadata file per bundle: `manifest.json`. If a
kernel needs more metadata in a future version, it is added inside
`kernels[i]` of the manifest — never as a side file.

### Safety property

Deleting the `.kernels/` directory (or never writing it) makes the model
behave exactly as today (function-only). The bundle is strictly optional
and never corrupts the ONNX.

## Manifest schema

```jsonc
{
  "schema_version": 1,
  "module": {
    "type_name": "Attention:0",
    "python_class": "transformers.models.qwen2.modeling_qwen2.Qwen2Attention",
    "torch_version": "2.13.0+cu130",
    "triton_version": "3.7.1"
  },
  "io": {
    "inputs":  [{"name": "x", "dtype": "float16", "shape": ["batch", "seq", 768]}],
    "outputs": [{"name": "y", "dtype": "float16", "shape": ["batch", "seq", 768]}]
  },
  "kernels": [
    {
      "id": "kernel_0000",
      "cubin": "kernel_0000.cubin",
      "symbol": "Attention_0_pooled_0",
      "device_target": { "backend": "cuda", "arch": "sm_90", "warp_size": 32 },
      "launch": {
        "num_warps": 4,
        "num_ctas": 1,
        "shared_mem_bytes": 24576,
        "num_regs": 128,
        "grid_expr": { /* AST or null */ },
        "captured_grid": null
      },
      "args": [
        {"kind": "tensor", "name": "x",   "dtype": "float16", "elem_offset": 0},
        {"kind": "tensor", "name": "y",   "dtype": "float16", "elem_offset": 0},
        {"kind": "scalar", "name": "M",   "dtype": "int32",
         "from": {"io": "x", "dim": 0}},
        {"kind": "scalar", "name": "BLOCK_M", "dtype": "int32", "value": 128}
      ],
      "variants": []
    }
  ]
}
```

### Field semantics

- **`schema_version`** at top level — the manifest can evolve; runtime gates
  on this.
- **`module.python_class`** is informational only; runtime must not rely on
  it for dispatch (the class lives in Python, which the runtime doesn't
  have).
- **`module.torch_version` / `triton_version`** — provenance for debugging;
  a runtime may warn if they don't match the toolchain used to build it.
- **`io`** mirrors the ONNX function signature exactly (same names, same
  order). This lets the runtime confirm the bundle matches the function
  before dispatching.
  - `shape` entries are either integers (static dim) or strings (symbolic
    dim name, e.g. `"batch"`); the same symbolic name appearing in two
    places means they must be equal at runtime.
- **`device_target`** — critical for runtime to skip incompatible bundles
  (e.g. `sm_90` cubin on `sm_80`). `arch` is the CUDA compute capability
  string; `backend` is reserved for future xpu/rocm.
- **`launch.shared_mem_bytes`** comes from `metadata.shared`;
  **`num_regs`** from `metadata.num_regs`. Both are physical constraints
  the runtime must respect (or reject launch).
- **`args` ordering** matches the cubin's parameter order. This is what
  makes the bundle language-agnostic: a C runtime reads `args[i]` and
  knows how to set up `CUlaunchAttribute`s / push parameters via
  `cuLaunchKernelEx`. The descriptor tells you *what* each slot is (a
  tensor pointer, a scalar derived from an input's shape, or a
  compile-time constant), not just how many.
- **`launch.grid_expr`** — see "Grid AST".
- **`launch.captured_grid`** — the concrete grid value at capture time,
  or `null` in v1 (the inductor wrapper hook that records the runtime
  grid is not yet wired in — see §"Grid AST" extraction pipeline).
  Runtime uses it as a debug reference and as the launch grid when
  `grid_expr` is null, the runtime shape matches `io` exactly, AND
  `captured_grid` is non-null. If `captured_grid` is null the runtime
  must fall back to the ONNX function.
- **`variants: []`** is deliberately empty in v1 — the upgrade seam for
  v2 autotune (multiple cubins for the same kernel slot, picked by shape
  / tile heuristics).

## Grid AST

### Why a small AST, not raw Python

The grid lambda is Python source — fine for PyTorch, useless for a C
runtime. We parse it into a tiny JSON AST that any language can evaluate.
The node set is minimal (6 nodes cover ~95% of inductor grids); we fall
back to `grid_expr: null` on anything we cannot translate.

### Node set (v1)

Every node is a JSON object `{"op": "...", ...}`. The runtime implements
exactly these 6 ops:

| op            | fields                       | semantics                                                                                      |
|---------------|------------------------------|------------------------------------------------------------------------------------------------|
| `"const"`     | `value: int`                 | literal integer                                                                                |
| `"shape_dim"` | `input: str, axis: int`      | value of `spec.io.inputs[input].shape[axis]` at runtime (symbolic name or static int)          |
| `"cdiv"`      | `a: node, b: node`           | ceil division `(a + b - 1) // b`                                                               |
| `"mul"`       | `a: node, b: node`           | `a * b`                                                                                        |
| `"floordiv"`  | `a: node, b: node`           | `a // b`                                                                                       |
| `"meta"`      | `key: str`                   | value from autotuner meta dict (e.g. `BLOCK_M`); resolved against `args` constants in manifest |

### Worked examples

Typical attention grid `cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), 1` where
`M = x.shape[0] * x.shape[1]`:

```jsonc
"grid_expr": [
  {"op": "cdiv",
   "a": {"op": "mul",
         "a": {"op": "shape_dim", "input": "x", "axis": 0},
         "b": {"op": "shape_dim", "input": "x", "axis": 1}},
   "b": {"op": "meta", "key": "BLOCK_M"}},
  {"op": "cdiv",
   "a": {"op": "shape_dim", "input": "x", "axis": 2},
   "b": {"op": "meta", "key": "BLOCK_N"}},
  {"op": "const", "value": 1}
]
```

Simple elementwise grid `(numel // 1024,)`:

```jsonc
"grid_expr": [
  {"op": "floordiv",
   "a": {"op": "shape_dim", "input": "x", "axis": 0},
   "b": {"op": "const", "value": 1024}}
]
```

### Extraction pipeline (in `_collect_compiled_kernels`)

1. **At triton `compile` hook** — record `(src_hash, CompiledKernel)`.
2. **At inductor wrapper codegen** — hook `PythonWrapperCodegen` (or scan
   the generated source) for `def grid_wrapper_for_<name>(meta):` bodies.
   Match by `name` to `src_hash`. Parse the body with Python's stdlib
   `ast` module, walking only `Call` / `BinOp` / `Name` / `Constant` /
   `Subscript` nodes. Anything outside the 6-op set raises
   `NotTranslatable`; we catch it, set `grid_expr: null`, log a warning,
   and move on.
3. **At `kernel.run` call** — capture the actual grid value (always a
   3-tuple of ints). Stored as `captured_grid`.

### Runtime verification contract

- `grid_expr` is authoritative. If non-null, the runtime evaluates it
  against the new shape's dims and trusts the result.
- If `grid_expr` is `null`, the runtime may launch only if the runtime
  shape matches `io.shapes` exactly (static dims equal; symbolic dims
  may take any value but `captured_grid` is used verbatim). Otherwise
  the runtime must fall back to the ONNX function.
- `captured_grid` is the launch grid for the static-match case **when
  non-null**. In v1 it is `null` for every kernel because the inductor
  wrapper hook that records the concrete grid is not yet wired in (see
  §"Grid AST" — extraction pipeline). When `captured_grid is null`, the
  runtime MUST fall back to the ONNX function regardless of shape match;
  launching a 0-block grid off a null captured value would be silent
  garbage. v1.1 populates `captured_grid` and re-enables the static-match
  fast path.

### Static grid bypass

When `compile_static_grid=True`:

- **Skip AST extraction entirely** — no `ast.parse`, no visitor, no
  `grid_wrapper_for_*` scan.
- Each kernel's manifest entry gets `"grid_expr": null` and the
  captured grid value.
- Runtime contract: launch the cubin only if runtime shape matches
  `io.shapes` exactly; otherwise fall back to the ONNX function.

The flag is explicit (not auto-detected) because AST extraction has a
cost and the user knows whether their model runs at a fixed shape.
Both modes produce the same manifest schema (`grid_expr` + `captured_grid`
fields), so runtime code is unchanged — it always checks
`grid_expr != null` first.

## Error handling & fallback

| # | Failure                                                    | Severity | Action                                                                              |
|---|------------------------------------------------------------|----------|-------------------------------------------------------------------------------------|
| 1 | `compile` type not in `hiera`                              | —        | Auto-promote (silent)                                                               |
| 2 | Module not CUDA / triton unavailable                       | Soft     | Skip kernel bundle for that module; ONNX function still exported. `logger.warning`  |
| 3 | `torch.compile(module)` raises                             | Hard     | Export aborts with `RuntimeError`. User explicitly asked to compile; silent skip is a footgun. |
| 4 | Triton hook captures 0 kernels for a compiled module       | Soft     | Skip bundle, log warning. Module may be fully eager.                                |
| 5 | Grid AST extraction fails on a kernel (non-bypass mode)    | Soft     | Set `grid_expr: null` for that kernel; keep cubin + `captured_grid`. `logger.debug` |
| 6 | Grid AST evaluates but mismatches `captured_grid` (export sanity check) | Soft | Demote to `grid_expr: null`. Keep cubin. `logger.warning` |
| 7 | Bundle write fails (disk full, permission)                 | Hard     | `OSError` propagates; user asked for external kernel data, can't silently ship broken bundle. |
| 8 | Torch version missing `torch.compile` (<2.0)               | Hard     | Raise at API entry.                                                                 |

### Principles

- **Two hard exits, six soft degradations.** Only #3 (compile explicitly
  requested but failed) and #7 (bundle write failed) abort. Everything
  else degrades to "ONNX function only" — the existing behaviour, never
  a regression.
- **Never partial state.** If a module has 5 kernels and kernel #3's AST
  fails, the bundle still contains all 5 cubins. Kernel #3 just has
  `grid_expr: null`; the runtime falls back to the ONNX function for
  that kernel on non-captured shapes. The other 4 stay fully dynamic.
- **Logging, not exceptions, for soft failures.** Match the existing
  `logger = nest(model_typename)` convention from `hyper_export.py`. All
  kernel warnings go through the same nested logger.

### Static-bypass mode

Failures #5 and #6 are structurally impossible when
`compile_static_grid=True` (no AST work happens). Everything else is
identical.

### Runtime contract (documented, not enforced by HyperONNX)

HyperONNX's responsibility ends at producing a well-formed bundle. The
runtime — which is third-party — must:

1. Load `manifest.json`, check `schema_version`.
2. Check `device_target.arch` against the runtime device.
3. For each kernel: if `grid_expr` is non-null, evaluate it; otherwise
   require exact shape match with `io.shapes` (static dims must match;
   symbolic dims may take any value but `captured_grid` is used literally).
4. Launch cubin with `args` ordering; or skip and execute the sibling
   ONNX function.

HyperONNX does **not** verify runtime dispatch correctness. We only
guarantee the bundle is well-formed.

## Testing

### File locations

```
tests/expoter/
├── test_compile_capture.py           # NEW — triton hook + grid extraction
├── test_compile_bundle.py            # NEW — manifest writer + schema
└── test_compile_grid_ast.py          # NEW — AST translator + evaluator
tests/
└── test_export_hyper_onnx.py         # extended with compile integration test
```

### Tier 1 — Unit (fast, no GPU required)

| Test                                        | Asserts                                                                                          |
|---------------------------------------------|--------------------------------------------------------------------------------------------------|
| `test_capture_triton_compile_hook`          | Patching `triton.compiler.compile` records every call; the spy returns the original `CompiledKernel` unchanged. |
| `test_static_grid_capture`                  | With `compile_static_grid=True`, the grid value tuple is captured from `kernel.run` and stored verbatim. |
| `test_grid_ast_cdiv_only`                   | Translator turns `cdiv(M, 128)` into the 2-node AST correctly.                                   |
| `test_grid_ast_unhandled_raises_nottranslatable` | A grid expression using `sin(...)` raises `NotTranslatable`, caught by caller.              |
| `test_manifest_well_formed`                 | Written `manifest.json` parses, has `schema_version: 1`, `kernels[i].cubin` files exist.        |
| `test_grid_ast_eval_matches_captured`       | AST evaluated against captured shapes equals `captured_grid` (the §5 #6 sanity check).          |
| `test_bundle_optional_deletion`             | Deleting `.kernels/` dir leaves a valid model indistinguishable from a `compile=None` export.    |

### Tier 2 — Integration (requires CUDA + triton)

| Test                                        | Asserts                                                                                          |
|---------------------------------------------|--------------------------------------------------------------------------------------------------|
| `test_compile_end_to_end_cubin`             | A model with `compile=[Hiera1]` produces `Hiera1:0.kernels/kernel_0000.cubin` with non-zero bytes and a valid manifest. |
| `test_compile_subset_of_hiera`              | `hiera=[A, B], compile=[A]` → only `A` has a `.kernels/` dir; B unchanged.                       |
| `test_compile_auto_promote`                 | `compile=[A]` with `hiera=None` → A still gets an ONNX function (auto-promoted).                  |
| `test_compile_static_grid_skips_ast`        | With `compile_static_grid=True`, all `grid_expr` are `null`.                                     |
| `test_compile_dynamic_grid_ast_present`     | With default flag, at least one kernel has non-null `grid_expr` (on a model with dynamic-ish shape). |
| `test_grid_mismatch_demotes_to_null`        | Inject a corrupt grid expression; export still succeeds, `grid_expr` is null in manifest.        |

### Tier 3 — Smoke (optional, gated by env var)

- `test_compile_qwen_attention_kernel` — runs the README example against a
  small transformer; verifies bundle layout matches convention. Skipped
  if `HYPERONNX_TEST_LLM=0` or transformers not installed.

### Skip markers

CUDA-only tests use the existing project convention (no custom plugin).
Tests that need CUDA get:

```python
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
```

Triton availability is implied by CUDA (per `pyproject.toml` the triton
dep rides alongside torch).

### Non-tests (YAGNI)

- **No cubin correctness test.** We do not load and execute the cubin —
  that's the runtime's job. We only verify the bytes are non-empty and
  match `CompiledKernel.asm[binary_ext]`.
- **No roundtrip reload-and-run test.** Per the stated target
  (third-party runtime, language-agnostic), a PyTorch-side reload is not
  in scope.
- **No autotune variant test.** `variants: []` is v2.
- **No multi-kernel ordering invariant test.** Capture order is preserved
  but not contracted as a guarantee.

## Version compatibility

This feature follows the existing HyperONNX hard constraints:

- **Torch**: `<2.11` (2.5–2.10 verified). `torch.compile` and the triton
  hook surface used here are stable across this range.
- **Triton**: matches whatever ships with the supported torch versions.
  The `CompiledKernel.asm[binary_ext]` and `metadata` namedtuple shape
  are stable for triton 3.x.
- **onnxscript**: `<0.6.0` (existing constraint, unchanged).

If the triton `compile` hook surface changes in a future torch/triton
version, failure mode #2 (Soft) kicks in: the bundle is skipped and the
ONNX function still exports.

## Open questions

None blocking. The following are explicitly deferred and have reserved
slots in the manifest schema:

1. **Autotune variants** — `kernels[i].variants: []` reserved for v2.
2. **AOTI `.so` escape hatch** — no schema field reserved; would land as
   a new top-level `kernel_format: "cubin" | "aoti"` selector in v2.
3. **XPU / ROCm backends** — `device_target.backend` is a string so
   non-CUDA values are speculatively permitted, but the v1 implementation
   only exercises the CUDA path.
