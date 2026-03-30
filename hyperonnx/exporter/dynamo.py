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

import inspect
from ast import AST
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager

import onnxscript
import onnxscript.ir._schemas as schemas
import onnxscript.irbuilder as irbuilder
import onnxscript.sourceinfo
import torch
from onnxifier.logger import warning
from torch.library import custom_op
from torch.nn import Module

from ..typing import AnyTensor, ExportStatus, ModuleSpec
from .utils import plain_tensor_container

NAMESPACE = "hyper"
DOMAIN = onnxscript.values.Opset(domain=NAMESPACE, version=1)


def _tensor_dtype_to_onnx_dtype(dtype: torch.dtype):
    if dtype == torch.float32:
        return onnxscript.onnx_types.FLOAT
    elif dtype == torch.float16:
        return onnxscript.onnx_types.FLOAT16
    elif dtype == torch.bfloat16:
        return onnxscript.onnx_types.BFLOAT16
    elif dtype == torch.float64:
        return onnxscript.onnx_types.DOUBLE
    elif dtype == torch.float8_e5m2:
        return onnxscript.onnx_types.FLOAT8E5M2
    elif dtype == torch.float8_e5m2fnuz:
        return onnxscript.onnx_types.FLOAT8E5M2FNUZ
    elif dtype == torch.float8_e4m3fn:
        return onnxscript.onnx_types.FLOAT8E4M3FN
    elif dtype == torch.float8_e4m3fnuz:
        return onnxscript.onnx_types.FLOAT8E4M3FNUZ
    elif dtype == torch.int8:
        return onnxscript.onnx_types.INT8
    elif dtype == torch.int16:
        return onnxscript.onnx_types.INT16
    elif dtype == torch.int32:
        return onnxscript.onnx_types.INT32
    elif dtype == torch.int64:
        return onnxscript.onnx_types.INT64
    elif dtype == torch.uint8:
        return onnxscript.onnx_types.UINT8
    elif dtype == torch.uint16:
        return onnxscript.onnx_types.UINT16
    elif dtype == torch.uint32:
        return onnxscript.onnx_types.UINT32
    elif dtype == torch.uint64:
        return onnxscript.onnx_types.UINT64
    elif dtype == torch.bool:
        return onnxscript.onnx_types.BOOL
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_onnxscript(spec: ModuleSpec) -> onnxscript.OnnxFunction:
    """Dynamically build an onnx script for custom translation table."""

    func_name = spec["name"] + "_func"
    result = irbuilder.IRFunction(func_name, NAMESPACE)
    # Note: op_name must not be the same as function name, or it would cause
    # onnx infinite recursion (function referencing itself).
    op_name = spec["type_name"]
    stmt = irbuilder.IRStmt([], onnxscript.values.Op(DOMAIN, op_name), [], [])
    annotations: dict[str, type] = OrderedDict()
    sig_parameters: list[inspect.Parameter] = []
    return_types = []
    for args, name in zip(spec["args"], spec["signature"].parameters):
        for i, arg in enumerate(plain_tensor_container(args)):
            if arg is None:
                continue
            elif isinstance(arg, str):
                irtype = onnxscript.onnx_types.STRING
            else:
                irtype = _tensor_dtype_to_onnx_dtype(arg.dtype)
            sourceinfo = onnxscript.sourceinfo.SourceInfo(AST())
            result.append_input(irbuilder.IRVar(f"{name}:{i}", irtype, sourceinfo))
            annotations[f"{name}_{i}"] = irtype
            sig_parameters.append(
                inspect.Parameter(
                    name=f"{name}_{i}",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
    if kwargs := spec.get("kwargs"):
        sig = spec["signature"]
        ordered_kwargs = {k: kwargs[k] for k in sig.parameters if k in kwargs}
        for name, args in ordered_kwargs.items():
            for i, arg in enumerate(plain_tensor_container(args)):
                if not isinstance(arg, torch.Tensor):
                    continue
                irtype = _tensor_dtype_to_onnx_dtype(arg.dtype)
                sourceinfo = onnxscript.sourceinfo.SourceInfo(AST())
                result.append_input(irbuilder.IRVar(f"{name}:{i}", irtype, sourceinfo))
                annotations[f"{name}_{i}"] = irtype
                sig_parameters.append(
                    inspect.Parameter(
                        name=f"{name}_{i}",
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                )
    if "output" in spec:
        for i, output in enumerate(plain_tensor_container(spec["output"])):
            irtype = _tensor_dtype_to_onnx_dtype(output.dtype)
            sourceinfo = onnxscript.sourceinfo.SourceInfo(AST())
            result.append_output(irbuilder.IRVar(f"output:{i}", irtype, sourceinfo))
            return_types.append(_tensor_dtype_to_onnx_dtype(output.dtype))
    stmt.args = [i.name for i in result.inputs]
    stmt.result = [i.name for i in result.outputs]
    result.append_stmt(stmt)

    def _f(*args, **kwargs):  # this function does nothing
        return getattr(DOMAIN, op_name)(*args, **kwargs)

    onnx_fn = onnxscript.OnnxFunction(DOMAIN, _f, result, "", {})
    if onnx_fn.op_schema is not None:
        # FIXME: this hack will cause infinite loop during translate the graph in ONNX
        op_signature = schemas.OpSignature.from_op_schema(onnx_fn.op_schema)
        onnx_fn.op_signature = op_signature
        if len(return_types) == 1:
            annotations["return"] = return_types[0]
        else:
            annotations["return"] = tuple[*return_types]  # type: ignore
        setattr(
            onnx_fn,
            "__signature__",
            inspect.Signature(sig_parameters, return_annotation=annotations["return"]),
        )
        setattr(onnx_fn, "__annotations__", annotations)

    return onnx_fn


def _assign_plain_tensors(container: dict, name: str, value: AnyTensor):
    plain_values = plain_tensor_container(value)
    for i, arg in enumerate(plain_values):
        if len(plain_values) == 1:
            container[name] = arg
        else:
            container[f"{name}_{i}"] = arg


def _plain_args_and_kwargs(args: tuple, kwargs: dict, signature: inspect.Signature):
    new_args: dict[str, AnyTensor] = OrderedDict()
    params = signature.parameters
    for args, name in zip(args, params):
        _assign_plain_tensors(new_args, name, args)
    signature_has_var_kw = None
    for k in params:
        v = kwargs.pop(k, None)
        if v is not None:
            _assign_plain_tensors(new_args, k, v)
        elif params[k].kind == inspect.Parameter.VAR_KEYWORD:
            # forward like forward(self, x, y, **kwargs) and invoked with
            # forward(x, y, a=1, b=2)
            signature_has_var_kw = True
    if kwargs and signature_has_var_kw:
        # extend the signature from **kwargs to specific names
        for k, v in kwargs.items():
            _assign_plain_tensors(new_args, k, v)
    return new_args


def make_custom_op(module: Module, spec: ModuleSpec):
    """Create a custom op and registered into `torch.library`.
    To replace the module with created custom op during dynamo export.
    """
    spec_name = spec["name"] or "main"  # can't be empty
    spec_name = spec_name.replace(".", "_")  # no dot allowed in op name
    name = f"{NAMESPACE}::{spec_name}"
    new_args = _plain_args_and_kwargs(
        spec["args"], spec.get("kwargs", {}).copy(), spec["signature"]
    )
    # simple schema inference, refer to torch._library.infer_schema.infer_schema
    # for complete logic.
    schemas_str = []
    for k, v in new_args.items():
        if isinstance(v, str):
            # WA: in onnx_ir._convenience._constructors, there is a bug for
            # str encoding. So we filter out str arguments.
            pass
        # elif v is None:
        #     # Treat None as boolean
        #     schemas_str.append(f"bool {k}")
        else:
            schemas_str.append(f"{type(v).__name__} {k}")
    schema = f"({','.join(schemas_str)})"
    if "output" in spec:
        outputs = plain_tensor_container(spec["output"])
        if len(outputs) == 1:
            schema += f" -> {type(outputs[0]).__name__}"
        elif len(outputs) > 1:
            return_vals = [type(o).__name__ for o in outputs if o is not None]
            schema += f" -> ({','.join(return_vals)})"

    def _duck_forward(*args, **kwargs):
        output = spec.get("output", None)
        if fw := getattr(module, "__ori_forward", None):
            if output is None:
                output = fw(*args, **kwargs)
        if isinstance(output, dict):
            warning(
                "dynamo custom op doesn't support dict output, "
                f"while {type(module).__qualname__} returns a dict."
            )
            output = plain_tensor_container(output)
        assert output is not None
        return output

    custom_fn = custom_op(
        name,
        _duck_forward,
        mutates_args=(),
        schema=schema,
    )
    custom_fn.register_fake(_duck_forward)
    onnx_fn = build_onnxscript(spec)

    class _CustomWrapper(torch.nn.Module):
        def __init__(self, fn: Callable, signature: inspect.Signature):
            super().__init__()
            self._fn = fn
            self._sig = signature

        def forward(self, *args, **kwargs):
            new_args = _plain_args_and_kwargs(args, kwargs.copy(), self._sig)
            # WA: in onnx_ir._convenience._constructors, there is a bug for
            # str encoding. So we filter out str arguments here. Note this
            # requires that the string argument must have a default value.
            for k in list(new_args.keys()):
                if isinstance(new_args[k], str):
                    new_args.pop(k)
            return self._fn(**new_args)

    return _CustomWrapper(custom_fn, spec["signature"]), {name: onnx_fn}


@contextmanager
def replace_with_custom_op(model: Module, module_spec: dict[Module, ModuleSpec]):
    """Replace the forward function of modules in `module_spec` with a duck type.

    It's used to laterly replace the duck type with the embedded onnx functions.

    Args:
        model (Module): The torch module which is the top level of the model.
        module_spec (Dict[Module, ModuleSpec]): The dictionary to store the spec of
            modules. See :func:`make_hierarchical_hook` for more details.
    """

    try:
        custom_translation_table = {}
        for child in filter(lambda c: c in module_spec, model.modules()):
            spec = module_spec[child]
            setattr(child, "__ori_forward", child.forward)
            if spec["status"] == ExportStatus.EXPORTED:
                setattr(child, "__ori_forward", child.forward)
                custom_mod, translation_table = make_custom_op(child, spec)
                custom_translation_table.update(translation_table)
                child.forward = custom_mod.forward
        yield custom_translation_table
    finally:
        for child in model.modules():
            if getattr(child, "__ori_forward", None):
                child.forward = getattr(child, "__ori_forward")
                delattr(child, "__ori_forward")
