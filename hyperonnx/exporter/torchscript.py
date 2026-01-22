"""
Copyright (C) 2025 The HYPERONNX Authors.

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

from contextlib import contextmanager
from functools import update_wrapper
from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from ..typing import AnyTensor, ExportStatus, ModuleSpec
from .utils import plain_tensor_container


def make_duck_forward(module_spec: ModuleSpec):
    """Make a duck forward function to replace this module with a duck type."""

    class DuckForward(torch.autograd.Function):  # pylint: disable=abstract-method
        """A duck forward function to export a custom op in onnx."""

        @staticmethod
        def forward(*args, **kwargs):
            loop = module_spec["loops"]
            if loop == 0:
                output = module_spec.get("output")
            else:
                output = module_spec.get("loop_outputs")[loop - 1]
            if module_spec.get("loop_outputs"):
                module_spec["loops"] += 1
            if isinstance(output, torch.Tensor) or output is None:
                return output
            plained_outputs = plain_tensor_container(output)
            return tuple(i for i in plained_outputs if i is not None)

        @staticmethod
        def backward(ctx, *grad_outputs):
            raise RuntimeError("A duck forward function can't be backwarded.")

        @staticmethod
        def symbolic(g, *args):
            """Make onnx symbolic.

            TODO:
                This method is high likely to be changed in the future.
                Refer to this link to use dynamo to export custom op:

            """
            output = module_spec.get("output")
            type_name = module_spec["type_name"]
            assert output is not None
            if isinstance(output, Tensor):
                outputs = 1
            else:
                outputs = len(plain_tensor_container(output))
            # TODO: g.op doesn't accept None as one of inputs
            # but we don't know the side effect of this WA.
            vargs = filter(lambda x: x is not None, args)
            op = g.op(type_name, *vargs, outputs=outputs)
            return op

    @torch.inference_mode()
    def _forward(*args, **kwargs):
        new_args = list(plain_tensor_container(args))
        if kwargs:
            # ISSUE (#158): caller kwargs order may not follow the signature order,
            # to replace forward function with a duck type, we need to reverse the
            # caller order to match the signature order.
            sig = module_spec["signature"]
            ordered_kwargs = {k: kwargs[k] for k in sig.parameters if k in kwargs}
            new_args.extend(plain_tensor_container(ordered_kwargs))
        # NOTE: apply doesn't take **kwargs, so we need to plain input container
        # and convert objects to Tensor
        return DuckForward.apply(*new_args)

    return _forward


@contextmanager
def replace_duck_forward(model: Module, module_spec: Dict[Module, ModuleSpec]):
    """Replace the forward function of modules in `module_spec` with a duck type.

    It's used to laterly replace the duck type with the embedded onnx functions.

    Args:
        model (Module): The torch module which is the top level of the model.
        module_spec (Dict[Module, ModuleSpec]): The dictionary to store the spec of
            modules. See :func:`make_hierarchical_hook` for more details.
    """

    def _hook_output_restore(
        module: Module,
        args: tuple,  # pylint: disable=unused-argument
        result: AnyTensor,  # pylint: disable=unused-argument
    ):
        spec = module_spec[module]
        output = spec.get("output")
        return output

    handles: List[RemovableHandle] = []
    try:
        for child in filter(lambda c: c in module_spec, model.modules()):
            if module_spec[child]["status"] == ExportStatus.EXPORTED:
                setattr(child, "__ori_forward", child.forward)
                if module_spec[child]["output_need_to_restore"]:
                    handle = child.register_forward_hook(_hook_output_restore)
                    handles.append(handle)
                # makes child.forward keep the same signature as original forward
                child.forward = update_wrapper(
                    make_duck_forward(module_spec[child]), child.forward
                )
                module_spec[child]["loops"] = 0
        yield
    finally:
        for child in model.modules():
            if getattr(child, "__ori_forward", None):
                child.forward = getattr(child, "__ori_forward")
                delattr(child, "__ori_forward")
        for handle in handles:
            handle.remove()
