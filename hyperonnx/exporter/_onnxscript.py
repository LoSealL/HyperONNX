"""
Copyright (C) 2026 The HYPERONNX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

onnxscript ``build_onnxscript`` dispatcher and modern implementation.

* When ``onnxscript<0.6`` is detected:
* ``build_onnxscript`` — dispatches to ``_onnxscript_legacy`` or the modern
  path below.
"""

from typing import Any

import onnx_ir as ir
import onnxscript
import torch
from onnx_ir import DataType
from packaging.version import Version

from ..typing import ModuleSpec
from .utils import NAMESPACE, plain_tensor_container


def torch_dtype_to_onnxscript_type(dtype: torch.dtype):
    """Map a torch dtype to the onnxscript type class (works on both APIs)."""
    _map = {
        torch.float32: onnxscript.FLOAT,
        torch.float16: onnxscript.FLOAT16,
        torch.bfloat16: onnxscript.BFLOAT16,
        torch.float64: onnxscript.DOUBLE,
        torch.float8_e5m2: onnxscript.FLOAT8E5M2,
        torch.float8_e5m2fnuz: onnxscript.FLOAT8E5M2FNUZ,
        torch.float8_e4m3fn: onnxscript.FLOAT8E4M3FN,
        torch.float8_e4m3fnuz: onnxscript.FLOAT8E4M3FNUZ,
        torch.int8: onnxscript.INT8,
        torch.int16: onnxscript.INT16,
        torch.int32: onnxscript.INT32,
        torch.int64: onnxscript.INT64,
        torch.uint8: onnxscript.UINT8,
        torch.uint16: onnxscript.UINT16,
        torch.uint32: onnxscript.UINT32,
        torch.uint64: onnxscript.UINT64,
        torch.bool: onnxscript.BOOL,
    }
    if dtype not in _map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _map[dtype]


def _build_onnxscript_modern(spec: ModuleSpec) -> onnxscript.OnnxFunction:
    # pylint: disable=import-outside-toplevel
    from onnxscript._internal.irbuilder import IRFunction

    func_name = (spec["name"] or "main").replace(".", "_") + "_func"
    # Note: op_name must not be the same as function name, or it would cause
    # onnx infinite recursion (function referencing itself).
    op_name = spec["type_name"]
    result = IRFunction(func_name, NAMESPACE)

    _os_to_dtype: dict[Any, DataType] = {
        onnxscript.FLOAT: DataType.FLOAT,
        onnxscript.FLOAT16: DataType.FLOAT16,
        onnxscript.BFLOAT16: DataType.BFLOAT16,
        onnxscript.DOUBLE: DataType.DOUBLE,
        onnxscript.FLOAT8E5M2: DataType.FLOAT8E5M2,
        onnxscript.FLOAT8E5M2FNUZ: DataType.FLOAT8E5M2FNUZ,
        onnxscript.FLOAT8E4M3FN: DataType.FLOAT8E4M3FN,
        onnxscript.FLOAT8E4M3FNUZ: DataType.FLOAT8E4M3FNUZ,
        onnxscript.INT8: DataType.INT8,
        onnxscript.INT16: DataType.INT16,
        onnxscript.INT32: DataType.INT32,
        onnxscript.INT64: DataType.INT64,
        onnxscript.UINT8: DataType.UINT8,
        onnxscript.UINT16: DataType.UINT16,
        onnxscript.UINT32: DataType.UINT32,
        onnxscript.UINT64: DataType.UINT64,
        onnxscript.BOOL: DataType.BOOL,
        onnxscript.STRING: DataType.STRING,
    }

    def _to_ir_type(os_type):
        return ir.TensorType(_os_to_dtype[os_type])

    input_params: list[tuple[str, type]] = []
    return_types: list[type] = []
    for args, name in zip(spec["args"], spec["signature"].parameters):
        for i, arg in enumerate(plain_tensor_container(args)):
            if arg is None:
                continue
            param_name = f"{name}_{i}"
            if isinstance(arg, str):
                os_type = onnxscript.STRING
            else:
                os_type = torch_dtype_to_onnxscript_type(arg.dtype)
            value = ir.Value(name=param_name, type=_to_ir_type(os_type))
            result.append_parameter(value)
            input_params.append((param_name, os_type))
    if kwargs := spec.get("kwargs"):
        sig = spec["signature"]
        ordered_kwargs = {k: kwargs[k] for k in sig.parameters if k in kwargs}
        for name, args in ordered_kwargs.items():
            for i, arg in enumerate(plain_tensor_container(args)):
                if not isinstance(arg, torch.Tensor):
                    continue
                param_name = f"{name}_{i}"
                os_type = torch_dtype_to_onnxscript_type(arg.dtype)
                value = ir.Value(name=param_name, type=_to_ir_type(os_type))
                result.append_parameter(value)
                input_params.append((param_name, os_type))
    num_outputs = 0
    if "output" in spec:
        outputs = [o for o in plain_tensor_container(spec["output"]) if o is not None]
        num_outputs = len(outputs)
        return_types = [torch_dtype_to_onnxscript_type(o.dtype) for o in outputs]
    input_values = list(result.inputs)
    node = ir.Node(
        NAMESPACE, op_name, inputs=input_values, num_outputs=max(num_outputs, 1)
    )
    result.append_node(node)
    if num_outputs > 0:
        result.graph.outputs.extend(node.outputs)

    # Build a python function with proper type annotations so the torch.onnx
    # exporter can infer the correct number of inputs and outputs.
    param_decls = ", ".join(
        f"{n}: _os_types[{i!r}]" for i, (n, _) in enumerate(input_params)
    )
    if num_outputs == 0:
        ret_decl = "None"
    elif num_outputs == 1:
        ret_decl = "_ret_types[0]"
    else:
        ret_decl = (
            "tuple[" + ", ".join(f"_ret_types[{i}]" for i in range(num_outputs)) + "]"
        )
    ns: dict[str, Any] = {
        "onnxscript": onnxscript,
        "_os_types": [t for _, t in input_params],
        "_ret_types": return_types,
    }
    code = f"def {func_name}({param_decls}) -> {ret_decl}:\n    pass\n"
    exec(code, ns)  # noqa: S102, pylint: disable=exec-used
    pyfun = ns[func_name]

    return onnxscript.OnnxFunction(None, pyfun, result, "", {})


def build_onnxscript(spec: ModuleSpec) -> onnxscript.OnnxFunction:
    """Dynamically build an onnx script for custom translation table."""

    # pylint: disable=import-outside-toplevel,import-error,ungrouped-imports
    # Detect API generation: onnxscript<0.6 exposes a public ``irbuilder`` module
    # with ``IRStmt``/``IRVar``; newer versions moved it to ``_internal`` and use
    # the standalone ``onnx_ir`` package instead.
    if Version(onnxscript.__version__) < Version("0.6.0"):
        from ._onnxscript_legacy import build_onnxscript as _build

        return _build(spec)
    else:
        return _build_onnxscript_modern(spec)
