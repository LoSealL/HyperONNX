---
name: torch-dynamo-onnx-export
description: Use this skill when exporting PyTorch models to ONNX using the modern torch.export (dynamo) path in HyperONNX. Best for models with dynamic control flow, complex tensor operations, and hierarchical custom operators defined via onnxscript.
argument-hint: Describe the model, custom operations (if any), target opset, and whether you need hierarchical export or simple custom operator definitions.
---

# Torch Dynamo ONNX Export

Use this skill when the task is about the modern Torch ONNX export path in HyperONNX, meaning the `torch.export`-based path with `dynamo=True`.

This skill is specific to this repository. It captures the actual workflow used by HyperONNX code and tests, rather than generic PyTorch ONNX advice.

## Torch Behaviour

- PyTorch 2.6+ introduced `torch.export` for AOT (Ahead-of-Time) tracing, enabling more dynamic models than TorchScript.
- PyTorch 2.9+ made `dynamo=True` the default in `torch.onnx.export()`.
- The dynamo path is based on FX graphs and bytecode analysis, not static tracing.
- On PyTorch 2.5 and below, `dynamo` parameter may not exist; always check `torch.onnx.export` signature and use the wrapper `torch_export_handle_lower_version` from HyperONNX.
- In this repository, prefer being explicit: always pass `dynamo=True` to clarify intent when using the modern exporter.

⚠️ **Version Constraint**: This skill is tested and supported on **torch < 2.11** with **onnxscript < 0.6.0**. These versions have breaking changes in ONNX/module structure that require future work to resolve.

## What This Skill Produces

Produce one of these outcomes:

- A correct export snippet using the modern `torch.export` path with `dynamo=True`.
- A HyperONNX code change that leverages `torch.export` for better dynamic model support.
- A custom operator definition using onnxscript and custom_translation_table for the dynamo path.
- A debugging checklist for why a dynamo export is failing or behaving unexpectedly.
- A migration guide from legacy (dynamo=False) to modern (dynamo=True) export.

## When To Use It

Use this skill when the user asks for any of the following:

- Use `torch.onnx.export` with the modern or dynamo path.
- Force torch.export instead of TorchScript tracing.
- Export a model with `dynamo=True`.
- Create custom ONNX operators using onnxscript for the dynamo path.
- Build a custom_translation_table for hierarchical export.
- Understand why dynamo export behaves differently from legacy export.
- Export models with dynamic shapes, loops, or data-dependent control flow.

Do not use this skill for generic ONNX export or torch.export advice unrelated to HyperONNX.

## Repository Facts To Ground Decisions

- HyperONNX exposes `torch_export_handle_lower_version` in [hyperonnx/torch_export.py](../../../hyperonnx/torch_export.py) to bridge version differences.
- The main hierarchical export entrypoint is [hyperonnx/hyper_export.py](../../../hyperonnx/hyper_export.py).
- The dynamo-specific branch for hierarchical replacement lives in [hyperonnx/exporter/dynamo.py](../../../hyperonnx/exporter/dynamo.py).
- Custom operator definition using onnxscript is built dynamically in [hyperonnx/exporter/dynamo.py](../../../hyperonnx/exporter/dynamo.py) via `build_onnxscript` and IRBuilder.
- Transformer attention for the dynamo path is handled in [hyperonnx/transformers/attention.py](../../../hyperonnx/transformers/attention.py) via `attention_translation_table()`.
- Reference tests: [tests/expoter/test_dynamo_replace_custom_op.py](../../../tests/expoter/test_dynamo_replace_custom_op.py), [tests/expoter/test_dynamo_build_onnxscript.py](../../../tests/expoter/test_dynamo_build_onnxscript.py), [tests/expoter/test_dynamo_make_custom_op.py](../../../tests/expoter/test_dynamo_make_custom_op.py).

## Dynamo Export Fundamentals

### Core Concepts

**torch.export**: Produces an ExportedProgram (AOT traced graph) that preserves more semantics than TorchScript, including data-dependent control flow and dynamic shapes.

**custom_translation_table**: A dictionary mapping PyTorch callables (e.g., custom ops, modules) to their ONNX counterparts (via onnxscript). The dynamo exporter uses this table to translate FX nodes to ONNX operators.

**onnxscript**: A framework for defining ONNX functions programmatically. HyperONNX uses `onnxscript.OnnxFunction` and `irbuilder.IRFunction` to build custom operators dynamically.

**replace_with_custom_op**: Context manager in [hyperonnx/exporter/dynamo.py](../../../hyperonnx/exporter/dynamo.py) that builds a custom_translation_table from module_spec, yielding it for use in export.

**build_onnxscript**: Dynamically constructs an ONNX function from a module's spec (inputs, outputs, dtype info). Used to create custom operator definitions without writing onnxscript manually.

### Key Differences from Legacy Export (TorchScript)

| Aspect | Legacy (dynamo=False) | Dynamo (dynamo=True) |
|--------|----------------------|---------------------|
| **Tracer** | TorchScript (static tracing) | torch.export (FX bytecode) |
| **Dynamic Shapes** | Limited | Better supported |
| **Control Flow** | Loops unrolled, if/else lost | Can be preserved (with torch.cond) |
| **Custom Op Registration** | torch.onnx.register_custom_op_symbolic | custom_translation_table + onnxscript |
| **Module Replacement** | DuckForward (torch.autograd.Function) | Direct FX node replacement |
| **Symbolic Gen** | Automatic from DuckForward.symbolic | Explicit via onnxscript |
| **Attention Registration** | register_attention_opsets() | attention_translation_table() |
| **Performance** | Often faster export | Slower export (bytecode analysis) |

## Defining Custom Operators for Dynamo Export

### Pattern 1: Direct onnxscript function (simple custom op)

For single custom operations without hierarchical wrapping:

```python
import onnxscript
import torch

# Step 1: Define an onnxscript function
_MY_OPSET = onnxscript.values.Opset("custom_domain", 1)

@onnxscript.script(_MY_OPSET)
def my_custom_op(x: onnxscript.FLOAT) -> onnxscript.FLOAT:
    return x * 2.0 + 1.0

# Step 2: Build custom_translation_table
custom_translation_table = {
    torch.ops.custom_lib.my_op: my_custom_op,
}

# Step 3: Export with dynamo=True and the table
torch.onnx.export(
    model,
    example_inputs,
    'model.onnx',
    dynamo=True,
    custom_translation_table=custom_translation_table,
    opset_version=18,
)
```

Key rules:
- The onnxscript function signature must match the PyTorch operation signature.
- The dictionary key must be a callable (e.g., torch.ops, a custom op).
- Always pass opset_version to align with onnxscript opset expectations.

### Pattern 2: HyperONNX hierarchical export with dynamic onnxscript

For complex modules that need hierarchical export, use `replace_with_custom_op` and `build_onnxscript`:

```python
import torch
from hyperonnx import export_hyper_onnx

class MyCustomModule(torch.nn.Module):
    """A complex module with loops or dynamic shapes."""
    def forward(self, x):
        for i in range(x.shape[0]):
            x = x * 2
        return x

model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    MyCustomModule(),
    torch.nn.Linear(20, 5),
)

export_hyper_onnx(
    model,
    (torch.randn(1, 10),),
    'model.onnx',
    hiera=[MyCustomModule],  # Export as hierarchical function
    dynamo=True,             # Use torch.export, not TorchScript
    opset_version=18,
)
```

How it works:
1. HyperONNX traces the model and caches module specs (inputs, outputs, dtypes).
2. `build_onnxscript` dynamically creates an ONNX function from the module spec.
3. `replace_with_custom_op` builds the custom_translation_table.
4. torch.onnx.export (with dynamo=True) translates the FX graph using the table.

Reference: [tests/expoter/test_dynamo_replace_custom_op.py](../../../tests/expoter/test_dynamo_replace_custom_op.py).

### Pattern 3: Attention-specific translation for dynamo

For transformer attention layers:

```python
import torch
from hyperonnx.transformers.attention import attention_translation_table

# Step 1: Get the translation table for dynamo path
translation_table = attention_translation_table()

# Step 2: Export with dynamo=True and the table
torch.onnx.export(
    model,
    example_inputs,
    'model.onnx',
    opset_version=24,
    dynamo=True,
    custom_translation_table=translation_table,
)
```

This uses the modern onnxscript-based Attention definition. See [hyperonnx/transformers/attention.py](../../../hyperonnx/transformers/attention.py) for implementation.

## Building Custom Operators with onnxscript

### Overview

HyperONNX's `build_onnxscript` function constructs ONNX operators automatically from module specs, eliminating the need to write onnxscript manually. However, you can also write custom onnxscript functions for fine-grained control.

### Auto-generation via build_onnxscript

```python
from hyperonnx.exporter.dynamo import build_onnxscript
from hyperonnx.typing import default_module_spec, ModuleSpec

# Assume module_spec is populated via trace_module_spec
spec: ModuleSpec = {...}  # Contains: args, output, signature, etc.

# Auto-build onnxscript function
onnx_func = build_onnxscript(spec)

# Use in custom_translation_table
custom_translation_table = {
    MyModule: onnx_func,
}
```

### Manual onnxscript Definition

For more control, write onnxscript directly:

```python
import onnxscript

# Define a custom opset
CUSTOM_OPSET = onnxscript.values.Opset(domain="my_domain", version=1)

@onnxscript.script(CUSTOM_OPSET)
def my_complex_op(x: onnxscript.FLOAT, mask: onnxscript.BOOL) -> onnxscript.FLOAT:
    """Custom operator that applies masking."""
    return x * mask.cast(onnxscript.FLOAT)

# Use in translation table
custom_translation_table = {
    my_pytorch_op: my_complex_op,
}
```

Key considerations:
- onnxscript functions use ONNX IR types (FLOAT, INT32, BOOL, etc.), not PyTorch types.
- The function signature must match the number of inputs/outputs of the PyTorch operation.
- Use `.cast()` for type conversions when needed.

## Workflow

### 0. Decide between direct onnxscript and hierarchical export

Before writing code, ask:
- **Is this a single atomic custom operation?** → Use direct onnxscript (Pattern 1).
- **Is this a module hierarchy that needs reusable ONNX functions?** → Use hierarchical export (Pattern 2).
- **Is this transformer attention?** → Use the built-in pattern (Pattern 3).

If unsure, check:
- [tests/expoter/test_dynamo_make_custom_op.py](../../../tests/expoter/test_dynamo_make_custom_op.py) for simple custom ops.
- [tests/expoter/test_dynamo_replace_custom_op.py](../../../tests/expoter/test_dynamo_replace_custom_op.py) for hierarchical examples.
- [tests/expoter/test_dynamo_build_onnxscript.py](../../../tests/expoter/test_dynamo_build_onnxscript.py) for build_onnxscript usage.

### 1. Check torch version and dynamo availability

```python
import torch
from inspect import signature

# Verify dynamo parameter exists
sig = signature(torch.onnx.export)
if 'dynamo' not in sig.parameters:
    # Use torch_export_handle_lower_version from HyperONNX
    from hyperonnx.torch_export import torch_export_handle_lower_version
    torch_export_handle_lower_version(model, args, f, dynamo=True, ...)
else:
    torch.onnx.export(model, args, f, dynamo=True, ...)
```

### 2. Identify and define custom operators

For each custom operator:
- Determine its input/output signatures.
- Write an onnxscript function or use build_onnxscript.
- Add it to custom_translation_table.

Example:
```python
custom_translation_table = {}

# Add manually-written onnxscript functions
custom_translation_table[torch.ops.custom_lib.op1] = onnxscript_op1

# Or add auto-generated ones from module specs
custom_translation_table[MyModule] = build_onnxscript(my_module_spec)
```

### 3. Build custom_translation_table

Use the context manager `replace_with_custom_op` for hierarchical export:

```python
from hyperonnx.exporter.dynamo import replace_with_custom_op

with replace_with_custom_op(model, module_spec) as custom_translation_table:
    torch.onnx.export(
        model,
        example_inputs,
        'model.onnx',
        dynamo=True,
        custom_translation_table=custom_translation_table,
        opset_version=18,
    )
```

### 4. Call torch.onnx.export with dynamo=True

```python
torch.onnx.export(
    model,
    example_inputs,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=18,
    dynamo=True,
    custom_translation_table=custom_translation_table,
    dynamic_shapes={'input': {0: 'batch_size'}},  # Optional: specify dynamic shapes
)
```

Key parameters:
- `dynamo=True`: Use torch.export instead of TorchScript.
- `custom_translation_table`: Dict mapping callables to onnxscript functions.
- `dynamic_shapes`: Preferred over dynamic_axes for dynamo path.
- `opset_version`: Must align with onnxscript opset.

### 5. Handle export failures and fallback

The dynamo path may fail on complex models. HyperONNX provides fallback:

```python
from hyperonnx.hyper_export import export_hyper_onnx

# export_hyper_onnx tries fallback automatically
export_hyper_onnx(
    model,
    args,
    'model.onnx',
    dynamo=True,
    # If dynamo export fails, HyperONNX may retry with dynamo=False
)
```

To disable fallback and fail fast:
```python
# Call torch_export_handle_lower_version directly without fallback
torch.onnx.export(
    model,
    args,
    f,
    dynamo=True,
    # No fallback logic here
)
```

### 6. Validate the exported ONNX model

```python
import onnx

# Load and check
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model, full_check=True)

# Verify custom operators appear
for node in onnx_model.graph.node:
    print(f"{node.op_type} in domain {node.domain}")
```

## Common Errors & Recovery

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError` in custom_translation_table | torch.ops key not in table | Verify the key exactly matches the PyTorch op reference |
| `TypeError: ... got an unexpected keyword argument 'dynamo'` | torch.onnx.export too old | Use `torch_export_handle_lower_version` wrapper |
| onnxscript function signature mismatch | ONNX function has wrong number of inputs/outputs | Verify spec['args'] and spec['output'] match onnxscript function |
| `UnflatteningError` in build_onnxscript | Output structure not flattened | Use `plain_tensor_container` to extract only tensors |
| Export succeeds but ONNX model is wrong | dynamic_shapes not specified | Add `dynamic_shapes` parameter for dynamic-shape models |
| torch.export fails on data-dependent branches | Model has unhandled control flow | Wrap loops/conditionals with `torch.cond` or fall back to dynamo=False |

## Branching Guidance

If the request is about library implementation:
- Prefer HyperONNX abstractions (export_hyper_onnx, replace_with_custom_op).
- Preserve backward compatibility with older torch versions.
- Use torch_export_handle_lower_version wrapper for version bridging.

If the request is about custom operators:
- Start with simple onnxscript functions (Pattern 1) for atomic ops.
- Use build_onnxscript for auto-generation from module specs.
- Write manual onnxscript only when fine-grained control is needed.

If the request is about a failing export:
- Check if the model uses data-dependent control flow; consider torch.cond.
- Verify custom_translation_table keys match torch.ops exactly.
- Check onnxscript function signatures match specs.
- Try lowering torch version requirement (dynamo may not exist in older PyTorch).
- Consider falling back to dynamo=False if dynamo export is unreliable for this model.

## Quality Bar

A good answer or code change using this skill should:
- Clearly distinguish between dynamo=True (modern) and dynamo=False (legacy).
- Use repository-native abstractions (export_hyper_onnx, replace_with_custom_op, build_onnxscript).
- Specify dynamic_shapes when exporting models with variable batch size or sequence length.
- Include custom_translation_table definitions with clear onnxscript signatures.
- Provide validation (onnx.checker.check_model) and explain expected ONNX structure.
- Mention torch version requirements and fallback behavior.

## Example Prompts

- Export this model using dynamo=True and custom attention operators.
- Add dynamic shape support to this ONNX export.
- Create a custom_translation_table for my hierarchical model.
- Why does dynamo export fail on this model with torch.cond?
- Migrate this export from dynamo=False to dynamo=True.
- Build an onnxscript function for this custom module.

## Weak Spots To Clarify Before Extending This Skill

If you evolve this skill later, clarify these points first:

- Which torch versions (2.5, 2.6, 2.8, 2.9+) should be officially supported for dynamo export examples?
- Should we recommend torch.cond for all data-dependent branches, or only certain patterns?
- Is build_onnxscript reliable enough to recommend for user-defined modules, or should we prefer manual onnxscript?
- How to handle dtype inference in build_onnxscript for models with mixed precision (float16, bfloat16)?
- Should fallback behavior be exposed as a user-facing option, or remain internal to HyperONNX?
