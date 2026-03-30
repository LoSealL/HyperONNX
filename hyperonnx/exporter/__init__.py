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

from torch.nn import Module

from ..typing import ModuleSpec
from .dynamo import replace_with_custom_op
from .torchscript import replace_duck_forward


@contextmanager
def replace_with_duck_module(
    model: Module, dynamo: bool, module_spec: dict[Module, ModuleSpec]
):
    """Replace the forward function of modules in `module_spec` with a duck type.

    It's used to laterly replace the duck type with the embedded onnx functions.

    Args:
        model (Module): The torch module which is the top level of the model.
        dynamo (bool): Whether to export the model with ``torch.export`` ExportedProgram
            instead of TorchScript.
        module_spec (Dict[Module, ModuleSpec]): The dictionary to store the spec of
            modules. See :func:`make_hierarchical_hook` for more details.
    """

    try:
        if dynamo:
            with replace_with_custom_op(model, module_spec) as tb:
                yield tb
        else:
            with replace_duck_forward(model, module_spec) as _:
                yield _
    finally:
        pass
