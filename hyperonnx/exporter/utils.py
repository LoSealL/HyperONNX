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

from itertools import chain
from typing import Sequence, Tuple

import torch
from onnxifier.logger import warning
from torch import Tensor

from ..typing import AnyTensor, ModuleSpec


def plain_tensor_container(obj: AnyTensor) -> Tuple[Tensor, ...]:
    """Iteratively flatten an arbitrary output of Tensors into a tuple of Tensors.

    Args:
        obj (AnyTensor): Acceptable object could be a tuple, list, dict, or a nested
            tuple, list, or dict of Tensors.

    Examples::

        obj = dict(a=[torch.rand(1), torch.rand(2)], b=dict(x=torch.ones(1)))
        y = plain_tensor_container(obj)
        # y => (torch.rand(1), torch.rand(2), torch.ones(1))

    Raises:
        ValueError: If any object can't be recursively flattened.

    Returns:
        Tuple[Tensor, ...]: a tuple of tensors.
    """
    if isinstance(obj, dict):
        return plain_tensor_container(tuple(obj.values()))
    elif isinstance(obj, Sequence):
        if isinstance(obj, str):
            # WA: for string type, return directly
            return (obj,)  # type: ignore
        obj_seq = tuple(chain(*[plain_tensor_container(v) for v in obj]))
        cls = type(obj)
        try:
            # restore tuple or list
            return cls(obj_seq)  # type: ignore
        except Exception:  # pylint: disable=broad-except
            warning(
                "During the flattening of module output, "
                f"Can't construct {cls.__name__} from a tuple, we turn this output "
                "into a pure tuple. However, this may lead to unexpected behavior in "
                "forward function."
            )
            return obj_seq
    elif isinstance(obj, Tensor):
        return (obj,)
    else:  # may be constant POD
        try:
            return (torch.as_tensor(obj),)
        except (RuntimeError, ValueError):
            # I.e. None, Cache
            return (obj,)
        except Exception as ex:
            raise ValueError(f"Unsupported output type: {type(obj)}") from ex


def detach_module_outputs(output: AnyTensor, spec: ModuleSpec) -> AnyTensor:
    """Detach tensors in module outputs and record structural information.

    This function recursively processes module outputs to detach tensors and
    record the structural information needed for reconstruction. It handles
    various output types including single tensors, dictionaries, sequences,
    and None values.

    Note:
        If one of the output is None, it will be replaced with a tensor of False
        to avoid breaking the torchscript forward function.

    Args:
        output (AnyTensor): The module output to process. Can be a single tensor,
            a dictionary of tensors, a sequence of tensors, or None.
        spec (ModuleSpec): A specification dictionary to record structural
            information. Will be modified in-place to store metadata about
            the output structure.

    Returns:
        AnyTensor: The processed output with tensors detached. The structure
            is preserved when possible, but may be converted to simpler types
            (like tuple) if reconstruction fails.

    Raises:
        TypeError: If the output type is not supported (not Tensor, dict,
            Sequence, or None).
    """

    if isinstance(output, Tensor):
        return output.detach()
    elif isinstance(output, dict):
        spec["output_need_to_restore"] = True
        dict_value = {k: detach_module_outputs(v, spec) for k, v in output.items()}
        try:
            cls = type(output)
            return cls(**dict_value)
        except Exception:  # pylint: disable=broad-except
            warning(
                f"The original output type was {type(output).__name__}, "
                "but failed to construct it from dict, so return as a dict directly."
            )
            return dict_value
    elif isinstance(output, Sequence):
        spec["output_need_to_restore"] = True
        out_seq = tuple(detach_module_outputs(v, spec) for v in output)
        cls = type(output)
        try:
            # restore tuple or list
            return cls(out_seq)  # type: ignore
        except Exception:  # pylint: disable=broad-except
            warning(
                "During the analysis of module output, "
                f"Can't construct {cls.__name__} from a tuple, we turn this output "
                "into a pure tuple. However, this may lead to unexpected behavior in "
                "forward function."
            )
            return out_seq
    elif output is None:
        # return None in duck module is a bad case, because it will be treated as
        # an output of the module, while None is not a valid torch IR, leads to
        # an export failure in torch.
        # However, this WA has a risk that codes like `if tensor is None: ...` will
        # fail.
        return torch.tensor(False)
    else:
        raise TypeError(f"Unsupported output type: {type(output)}")
