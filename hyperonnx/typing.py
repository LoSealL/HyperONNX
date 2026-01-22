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

from collections import defaultdict
from enum import Enum
from inspect import Signature
from pathlib import Path
from typing import Callable, Dict, List, NotRequired, Tuple, TypeAlias, TypedDict

from onnx import ModelProto
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

AnyTensor: TypeAlias = Tensor | Tuple["AnyTensor", ...] | Dict[str, "AnyTensor"]
HookCallback: TypeAlias = Callable[
    [Module, Tuple[Tensor], Dict[str, AnyTensor], AnyTensor], None
]


class ExportStatus(Enum):
    """The status of exporting a module."""

    INITED = 0
    FORWARDED = 1
    IN_EXPORTING = 2
    EXPORTED = 3


class ModuleSpec(TypedDict):
    """The specification of a module to be exported into onnx."""

    name: str  # name of the module
    type_name: str  # onnx type name of the module
    signature: Signature  # signature of the module's forward method
    args: Tuple[Tensor, ...]  # input args of the module
    kwargs: NotRequired[Dict[str, AnyTensor]]  # input kwargs of the module
    output: NotRequired[AnyTensor]  # output of the module
    handle: NotRequired[RemovableHandle]  # registered hook handle
    onnx: NotRequired[ModelProto | Path]  # exported onnx of the module
    status: ExportStatus  # keep each module only export once
    output_need_to_restore: bool  # whether the output is tuplized
    input_names: List[str]  # input names of the module
    output_names: List[str]  # output names of the module
    unused_inputs: Tuple[str, ...]  # input that been optimized out
    unused_outputs: Tuple[str, ...]  # output that been optimized out
    loops: int  # current loop count, i.e. an RNN call sequence
    loop_outputs: List[AnyTensor]  # the output of each loop iter


def _module_spec_defaultdict_factory():
    return ModuleSpec(
        name="",
        type_name="",
        signature=Signature(),
        args=(),
        status=ExportStatus.INITED,
        output_need_to_restore=False,
        input_names=[],
        output_names=[],
        unused_inputs=(),
        unused_outputs=(),
        loops=0,
        loop_outputs=[],
    )


def default_module_spec() -> defaultdict[Module, ModuleSpec]:
    """Create a defaultdict with type Dict[Module, ModuleSpec]"""

    return defaultdict(_module_spec_defaultdict_factory)
