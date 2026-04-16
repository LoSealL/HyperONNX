---
name: torch-legacy-onnx-export
description: Use this skill when working in HyperONNX and you need the legacy Torch ONNX export path with dynamo=False, especially for TorchScript-based export behavior, lower-version compatibility, custom attention registration, or post-export ONNX fixes.
argument-hint: Describe the model, export entrypoint, torch version, and whether custom attention or custom input structures are involved.
---

# Torch Legacy ONNX Export

Use this skill when the task is about the legacy Torch ONNX export path in HyperONNX, meaning the TorchScript-style path with dynamo=False.

This skill is specific to this repository. It captures the actual workflow used by HyperONNX code and tests, rather than generic PyTorch advice.

## Torch Behaviour

- PyTorch 2.9 changed `torch.onnx.export(..., dynamo=...)` so that `dynamo=True` is the default.
- On PyTorch 2.8 and earlier, do not assume the default selects the modern exporter. If legacy behavior matters, set `dynamo=False` explicitly. If modern exporter behavior matters, set `dynamo=True` explicitly.
- In this repository, prefer being explicit either way. Do not rely on the library default because the default changed across PyTorch versions.

⚠️ **Version Constraint**: This skill is tested and supported on **torch < 2.11** with **onnxscript < 0.6.0**. These versions have breaking changes in ONNX/module structure that require future work to resolve.

## What This Skill Produces

Produce one of these outcomes:

- A correct export snippet that uses the legacy path with dynamo=False.
- A HyperONNX code change that preserves legacy export compatibility.
- A debugging checklist for why a legacy export is failing in this repository.
- A validation plan that proves the exported ONNX model is structurally correct.

## When To Use It

Use this skill when the user asks for any of the following:

- Use torch.onnx.export with the old or legacy path.
- Force TorchScript export instead of dynamo export.
- Keep HyperONNX compatible with older torch.onnx.export behavior.
- Export a model with dynamo=False.
- Make custom attention export work on the TorchScript path.
- Understand why legacy export behaves differently from dynamo export.

Do not use this skill for generic ONNX export advice that is unrelated to HyperONNX.

## Repository Facts To Ground Decisions

- HyperONNX exposes a version-bridging wrapper in [hyperonnx/torch_export.py](../../../hyperonnx/torch_export.py) named torch_export_handle_lower_version.
- The main hierarchical export entrypoint is [hyperonnx/hyper_export.py](../../../hyperonnx/hyper_export.py), which accepts dynamo and may fall back between export modes.
- The TorchScript branch for hierarchical replacement lives in [hyperonnx/exporter/__init__.py](../../../hyperonnx/exporter/__init__.py).
- Transformer attention registration for the legacy path is implemented in [hyperonnx/transformers/attention.py](../../../hyperonnx/transformers/attention.py).
- Repository examples already show dynamo=False in [README.md](../../../README.md) and [README_CN.md](../../../README_CN.md).
- The duck-typing and symbolic operation system for TorchScript export is in [hyperonnx/exporter/torchscript.py](../../../hyperonnx/exporter/torchscript.py).

## Registering Custom Operators for Legacy Export

The legacy TorchScript export path requires explicit symbolic function registration for custom operators. Use this section when the task is to export custom PyTorch operations to ONNX while using dynamo=False.

### Pattern 1: Direct symbolic registration (torch-native style)

For simple custom operations that do not need hierarchical module wrapping:

```python
import torch
from torch.onnx import register_custom_op_symbolic

# Step 1: Define a symbolic function for ONNX
def symbolic_my_op(g, input, param):
    """ONNX symbolic for my_op.

    Args:
        g: The ONNX graph context.
        input: The input tensor from PyTorch.
        param: A parameter.

    Returns:
        ONNX operator node(s).
    """
    return g.op('custom_domain::MyOp', input, param_f=param)

# Step 2: Register the symbolic before export
register_custom_op_symbolic('custom_domain::my_op', symbolic_my_op, opset_version=18)

# Step 3: Export with dynamo=False
torch.onnx.export(
    model,
    example_inputs,
    'model.onnx',
    opset_version=18,
    dynamo=False,
)
```

Key rules:
- The symbolic function name must match the registered name in `"<domain>::<op>"` format.
- Always use opset_version consistently between registration and export.
- Register before calling export; registration persists in the current process.

### Pattern 2: HyperONNX hierarchical module with duck-typed custom op

For complex models where the custom operation is itself a module that needs hierarchical export, use the duck-typing system from [hyperonnx/exporter/torchscript.py](../../../hyperonnx/exporter/torchscript.py):

```python
import torch
from hyperonnx import export_hyper_onnx

class MyCustomModule(torch.nn.Module):
    """A module that implements custom logic."""
    def forward(self, x):
        # Custom computation
        return x * 2 + 1

model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    MyCustomModule(),
    torch.nn.Linear(20, 5),
)

export_hyper_onnx(
    model,
    (torch.randn(1, 10),),
    'model.onnx',
    hiera=[MyCustomModule],  # Export MyCustomModule as a function
    dynamo=False,
    opset_version=18,
)
```

How it works:
1. When you pass `hiera=[MyCustomModule]`, HyperONNX will trace and export the module separately.
2. The module is then replaced with a duck-type that acts like a custom operator node.
3. The DuckForward class (from torchscript.py) handles the ONNX symbolic generation automatically.

Reference: [tests/expoter/test_dynamo_replace_custom_op.py](../../../tests/expoter/test_dynamo_replace_custom_op.py) for hierarchical examples.

### Pattern 3: Attention-specific registration

For transformer attention layers, HyperONNX provides a pre-built pattern:

```python
import torch
from hyperonnx.transformers.attention import register_attention_opsets

# Register custom attention ONNX symbolic for the legacy path
register_attention_opsets()

# Then proceed with normal export
torch.onnx.export(
    model,
    example_inputs,
    'model.onnx',
    opset_version=24,
    dynamo=False,
)
```

This registration only affects the TorchScript (legacy) path. See "Handle transformer attention correctly" in Workflow step 4 for post-export adjustments.

## Understanding torchscript.py: Duck-Typing and Symbolic Export

The [hyperonnx/exporter/torchscript.py](../../../hyperonnx/exporter/torchscript.py) module implements a clever pattern for exporting PyTorch modules as ONNX custom operators without requiring explicit symbolic functions. Here is how it works:

### Core Concepts

**DuckForward**: A torch.autograd.Function subclass that:
- Stores the precomputed output in module_spec at forward time.
- Implements a `symbolic()` static method that generates an ONNX node instead of computing.
- Acts as a "duck type" – it quacks like a normal forward pass during tracing, but produces ONNX during export.

**make_duck_forward**: Creates a DuckForward instance with closure over a specific module_spec.

**replace_duck_forward**: Context manager that temporarily replaces module.forward with duck-typed versions.

### Workflow (Three Phases)

**Phase 1: Tracing and Caching**

Before export, HyperONNX calls `trace_module_spec()` which:
1. Registers forward hooks on all modules in the hiera list.
2. Runs the model with example inputs to capture: input signatures, actual output values, loop count.
3. Stores all this in module_spec.

**Phase 2: Duck-Typing (During torch.onnx.export)**

When `torch.onnx.export` traces the model with dynamo=False:
1. `replace_duck_forward()` context manager replaces each hiera module's forward with a duck-typed version.
2. Each duck forward is a DuckForward.apply() call that:
   - During normal tracing: calls DuckForward.forward(), returns the precomputed cached output.
   - During ONNX symbolic tracing: calls DuckForward.symbolic(), returns an ONNX op node.
3. The exporter builds the ONNX graph, inserting custom op nodes where the duck forwards were.

**Phase 3: Function Embedding**

After export, HyperONNX uses other rewriters to:
1. Embed the actual ONNX subgraph for each module into a function definition.
2. Replace the custom op nodes with function calls.

### Key Implementation Details

**Forward method**: Stores precomputed output in module_spec, indexed by loop counter. Handles multiple calls to the same module and flattens nested output structures.

**Symbolic method**: Uses the module's type_name (set during tracing) as the ONNX op name. Filters out None args and specifies the number of outputs.

**Forward wrapper (_forward function)**:
- Runs in inference mode to avoid backprop.
- Flattens input structures the same way as the real forward.
- Reorders kwargs to match the original function signature.
- Calls DuckForward.apply which triggers either forward or symbolic depending on context.

### Why This Pattern Matters for Legacy Export

1. **No need for explicit symbolic functions**: The duck-typing pattern generates ONNX nodes automatically.
2. **Handles complex module interactions**: Multiple calls, nested outputs, custom input types are all cached.
3. **Integrates with TorchScript tracer**: The exporter sees normal forward calls; duck-typing is transparent.
4. **Preserves correctness**: By caching actual outputs during tracing, the ONNX graph matches PyTorch semantics.

### When to Use vs. When Not To Use

**Use duck-typing (torchscript.py) when:**
- You are using HyperONNX hierarchical export (passing hiera list).
- The module is complex enough that writing a correct symbolic function is error-prone.
- You want the output to be an embedded ONNX function, not just a custom op node.

**Use direct symbolic registration when:**
- You only have a single custom operator, not a module hierarchy.
- You already have a working symbolic function.
- You want the ONNX output to reference an external custom op domain.

### Common Gotchas

1. **Mismatched signatures**: If the cached output from tracing does not match the actual forward call structure, duck forward will return wrong data. Always verify with trace_module_spec.
2. **Module called multiple times**: Duck forward increments loops and indexes into loop_outputs. Ensure tracing and export see the same number of calls.
3. **Non-tensor outputs**: The symbolic method must return an op or tuple of ops. Use plain_tensor_container to extract just tensors.
4. **Backward incompatibility**: Do not call backward on a DuckForward result. The backward method raises RuntimeError by design.

## Workflow

### 0. Decide between direct symbolic registration and duck-typing

Before writing code, ask:
- **Is this a single custom op that will appear in the final ONNX?** → Use direct symbolic registration (Pattern 1).
- **Is this a PyTorch module that should become a reusable ONNX function?** → Use HyperONNX hierarchical export with duck-typing (Pattern 2).
- **Is this transformer attention?** → Use the built-in pattern (Pattern 3) and refer to step 4 of Workflow.

If unsure, check the test files:
- [tests/expoter/test_dynamo_replace_custom_op.py](../../../tests/expoter/test_dynamo_replace_custom_op.py) for hierarchical examples.
- [hyperonnx/transformers/attention.py](../../../hyperonnx/transformers/attention.py) for attention-specific patterns.

### 1. Identify the export entrypoint

First determine which layer of the repository the task belongs to:

- If the task is about HyperONNX library internals, prefer torch_export_handle_lower_version instead of calling raw torch.onnx.export directly.
- If the task is about exporting a full hierarchical model, work through export_hyper_onnx in [hyperonnx/hyper_export.py](../../../hyperonnx/hyper_export.py).
- If the task is a direct usage example or a focused test, raw torch.onnx.export is acceptable when it matches the repository’s current test style.

### 2. Choose the export mode deliberately

Use this decision logic:

- If the requirement is legacy or TorchScript export, set dynamo=False explicitly.
- If the user only says export must work and does not require legacy behavior, note that HyperONNX may try fallback modes internally in some paths.
- If the code is in trace_module_spec, remember the repository intentionally uses dynamo=False there to keep tracing behavior aligned with TorchScript export semantics.

### 3. Apply the correct calling pattern

For repository code changes:

- Preserve the wrapper call shape used in [hyperonnx/torch_export.py](../../../hyperonnx/torch_export.py).
- Keep kwargs handling compatible with torch versions below 2.5.0, where kwargs are merged into args.
- Do not remove the OperatorExportTypes.ONNX_ATEN_FALLBACK behavior for older torch versions unless the task explicitly requires it and the change is validated.

For direct usage examples:

- Pass dynamo=False explicitly.
- Provide input_names and output_names when the model interface is non-trivial.
- Keep opset_version explicit when export semantics matter.

Minimal example:

```python
import torch

model = MyModule().eval()
example_inputs = (x, y)

torch.onnx.export(
    model,
    example_inputs,
    "model.onnx",
    input_names=["x", "y"],
    output_names=["out"],
    opset_version=19,
    dynamo=False,
)
```

### 4. Handle transformer attention correctly

If the task involves HyperONNX transformer attention, the legacy path has a required branch:

- Before export on the TorchScript path, call register_attention_opsets() from [hyperonnx/transformers/attention.py](../../../hyperonnx/transformers/attention.py).
- This registration exists specifically for the torchscript export path with dynamo=False.
- If the export also relies on the custom Attention op becoming default-domain ONNX Attention opset 24, promote the exported model with promote_onnx_model_to_opset24() after export.

Pattern:

```python
import onnx
import torch

from hyperonnx.transformers.attention import (
    promote_onnx_model_to_opset24,
    register_attention_opsets,
)

register_attention_opsets()

torch.onnx.export(
    model,
    example_inputs,
    output_path,
    opset_version=24,
    dynamo=False,
)

onnx_model = onnx.load_model(str(output_path))
onnx_model = promote_onnx_model_to_opset24(onnx_model)
onnx.checker.check_model(onnx_model, full_check=True)
```

Reference implementation: [tests/transformers/test_attention.py](../../../tests/transformers/test_attention.py)

### 5. Watch for input-structure limitations

Legacy export does not make arbitrary Python objects exportable.

Use these rules from the repository tests:

- Plain custom objects fail export.
- Tuple or list subclasses may pass flattening but still lose class behavior.
- A torch.Tensor subclass can sometimes preserve the needed behavior well enough for export.

Reference: [tests/expoter/test_non_pod_export.py](../../../tests/expoter/test_non_pod_export.py)

If the user is trying to export non-POD inputs, explain that the fix is usually to flatten or tensorize inputs, not to keep arbitrary Python objects in the ONNX boundary.

### 6. Validate the exported model

Minimum completion checks:

- The export artifact exists or the in-memory export object is non-null.
- onnx.checker.check_model passes.
- The chosen export path really used dynamo=False when legacy behavior was required.
- If custom attention is involved, verify the expected Attention node and domain/opset normalization.
- If changing library code, verify at least the smallest relevant test for the touched path.

Good repository-aligned validation targets:

- [tests/transformers/test_attention.py](../../../tests/transformers/test_attention.py)
- [tests/transformers/test_static_cache.py](../../../tests/transformers/test_static_cache.py)
- [tests/expoter/test_dynamo_replace_custom_op.py](../../../tests/expoter/test_dynamo_replace_custom_op.py) for contrast with the dynamo path

## Branching Guidance

If the request is about library implementation:

- Prefer HyperONNX wrappers and existing abstractions.
- Preserve backward-compatibility branches.
- Avoid replacing the wrapper with direct raw torch.onnx.export unless there is a strong reason.

If the request is about examples or docs:

- Show the shortest correct snippet with dynamo=False.
- Mention any required registration step up front.
- Link the example back to the repository’s existing test or README pattern.

If the request is about a failing export:

- Check whether the failure is caused by unsupported input structures.
- Check whether the user forgot dynamo=False.
- Check whether custom attention registration was skipped.
- Check whether post-export opset promotion is missing.
- Check whether the code path should have used torch_export_handle_lower_version.

## Quality Bar

A good answer or code change using this skill should:

- Distinguish clearly between legacy TorchScript export and dynamo export.
- Use repository-native entrypoints and helpers.
- Mention version-compatibility handling when touching core export code.
- Include concrete validation, not just export code.
- Avoid generic ONNX advice that ignores HyperONNX’s actual implementation.

## Example Prompts

- Use the legacy torch.onnx.export path for this HyperONNX test.
- Add a sample that exports with dynamo=False and validates the ONNX.
- Debug why this custom attention export fails on the TorchScript path.
- Update HyperONNX export code without breaking lower torch versions.
- Show me the correct legacy export flow for a transformer module in this repo.

## Weak Spots To Clarify Before Extending This Skill

If you evolve this skill later, clarify these points first:

- Whether the team wants to prefer legacy export in docs, or only keep it as a compatibility path.
- Which torch version range should be treated as officially supported for legacy export examples.
- Whether future transformer examples should standardize on post-export opset promotion or hide that behind a helper.
