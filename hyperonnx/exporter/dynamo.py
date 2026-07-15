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
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager

import torch
from onnxifier.logger import warning
from torch.library import custom_op
from torch.nn import Module

from ..typing import AnyTensor, ExportStatus, ModuleSpec
from ._onnxscript import build_onnxscript
from .utils import NAMESPACE, plain_tensor_container


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
