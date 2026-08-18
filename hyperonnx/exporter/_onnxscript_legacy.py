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
"""

# pylint: disable=import-error

from ast import AST
from collections import OrderedDict
from inspect import Parameter, Signature
from typing import Any

import onnxscript
import onnxscript.ir._schemas as schemas
import onnxscript.irbuilder as irbuilder  # type: ignore
import onnxscript.sourceinfo  # type: ignore
import torch

from ..typing import ModuleSpec
from .utils import NAMESPACE, plain_tensor_container

DOMAIN = onnxscript.values.Opset(domain=NAMESPACE, version=1)  # type: ignore


def _torch_dtype_to_onnxscript_type(dtype: torch.dtype):
    _map = {
        torch.float32: onnxscript.onnx_types.FLOAT,
        torch.float16: onnxscript.onnx_types.FLOAT16,
        torch.bfloat16: onnxscript.onnx_types.BFLOAT16,
        torch.float64: onnxscript.onnx_types.DOUBLE,
        torch.float8_e5m2: onnxscript.onnx_types.FLOAT8E5M2,
        torch.float8_e5m2fnuz: onnxscript.onnx_types.FLOAT8E5M2FNUZ,
        torch.float8_e4m3fn: onnxscript.onnx_types.FLOAT8E4M3FN,
        torch.float8_e4m3fnuz: onnxscript.onnx_types.FLOAT8E4M3FNUZ,
        torch.int8: onnxscript.onnx_types.INT8,
        torch.int16: onnxscript.onnx_types.INT16,
        torch.int32: onnxscript.onnx_types.INT32,
        torch.int64: onnxscript.onnx_types.INT64,
        torch.uint8: onnxscript.onnx_types.UINT8,
        torch.uint16: onnxscript.onnx_types.UINT16,
        torch.uint32: onnxscript.onnx_types.UINT32,
        torch.uint64: onnxscript.onnx_types.UINT64,
        torch.bool: onnxscript.onnx_types.BOOL,
    }
    if dtype not in _map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _map[dtype]


def build_onnxscript(spec: ModuleSpec) -> onnxscript.OnnxFunction:
    """Dynamically build an onnx script for custom translation table."""

    func_name = spec["name"] + "_func"
    # Note: op_name must not be the same as function name, or it would cause
    # onnx infinite recursion (function referencing itself).
    op_name = spec["type_name"]
    result = irbuilder.IRFunction(func_name, NAMESPACE)
    stmt = irbuilder.IRStmt([], onnxscript.values.Op(DOMAIN, op_name), [], [])  # type: ignore
    annotations: dict[str, Any] = OrderedDict()
    sig_parameters: list[Parameter] = []
    return_types: list[Any] = []
    for args, name in zip(spec["args"], spec["signature"].parameters):
        for i, arg in enumerate(plain_tensor_container(args)):
            if arg is None:
                continue
            elif isinstance(arg, str):
                irtype = onnxscript.onnx_types.STRING
            else:
                irtype = _torch_dtype_to_onnxscript_type(arg.dtype)
            sourceinfo = onnxscript.sourceinfo.SourceInfo(AST())
            result.append_input(irbuilder.IRVar(f"{name}:{i}", irtype, sourceinfo))
            annotations[f"{name}_{i}"] = irtype
            sig_parameters.append(
                Parameter(
                    name=f"{name}_{i}",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
    if kwargs := spec.get("kwargs"):
        sig = spec["signature"]
        ordered_kwargs = {k: kwargs[k] for k in sig.parameters if k in kwargs}
        for name, args in ordered_kwargs.items():
            for i, arg in enumerate(plain_tensor_container(args)):
                if not isinstance(arg, torch.Tensor):
                    continue
                irtype = _torch_dtype_to_onnxscript_type(arg.dtype)
                sourceinfo = onnxscript.sourceinfo.SourceInfo(AST())
                result.append_input(irbuilder.IRVar(f"{name}:{i}", irtype, sourceinfo))
                annotations[f"{name}_{i}"] = irtype
                sig_parameters.append(
                    Parameter(
                        name=f"{name}_{i}",
                        kind=Parameter.POSITIONAL_OR_KEYWORD,
                    )
                )
    if "output" in spec:
        for i, output in enumerate(plain_tensor_container(spec["output"])):
            irtype = _torch_dtype_to_onnxscript_type(output.dtype)
            sourceinfo = onnxscript.sourceinfo.SourceInfo(AST())
            result.append_output(irbuilder.IRVar(f"output:{i}", irtype, sourceinfo))
            return_types.append(_torch_dtype_to_onnxscript_type(output.dtype))
    stmt.args = [i.name for i in result.inputs]
    stmt.result = [i.name for i in result.outputs]
    result.append_stmt(stmt)

    def _f(*args, **kwargs):  # this function does nothing
        return getattr(DOMAIN, op_name)(*args, **kwargs)

    onnx_fn = onnxscript.OnnxFunction(DOMAIN, _f, result, "", {})
    if onnx_fn.op_schema is not None:
        # FIXME: this hack will cause infinite loop during translate the graph in ONNX
        op_signature = schemas.OpSignature.from_op_schema(onnx_fn.op_schema)  # type: ignore
        onnx_fn.op_signature = op_signature
        if len(return_types) == 1:
            annotations["return"] = return_types[0]
        else:
            annotations["return"] = tuple[tuple(return_types)]  # type: ignore
        setattr(
            onnx_fn,
            "__signature__",
            Signature(sig_parameters, return_annotation=annotations["return"]),
        )
        setattr(onnx_fn, "__annotations__", annotations)

    return onnx_fn
