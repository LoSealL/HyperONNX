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

import warnings
from collections.abc import Callable, Mapping, Sequence
from inspect import signature
from os import PathLike
from typing import Any

import torch
from torch.onnx import OperatorExportTypes
from torch.onnx import export as torch_export

from .utils import capture_torch_jit_warnings

if torch_export.__module__ != "torch.onnx":
    warnings.warn(
        "torch.onnx.export has been replaced by "
        f"'{torch_export}' of {torch_export.__module__}"
    )


@capture_torch_jit_warnings
def torch_export_handle_lower_version(
    model: torch.nn.Module,
    args: tuple[Any, ...] = (),
    f: str | PathLike | None = None,
    *,
    kwargs: dict[str, Any] | None = None,
    export_params: bool = True,
    verbose: bool | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamic_axes: (
        Mapping[str, Mapping[int, str]] | Mapping[str, Sequence[int]] | None
    ) = None,
    keep_initializers_as_inputs: bool = False,
    dynamo: bool = False,
    # Dynamo only options
    external_data: bool = True,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    custom_translation_table: (
        dict[Callable, Callable | Sequence[Callable]] | None
    ) = None,
    report: bool = False,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | PathLike = ".",
    fallback: bool = False,
    # Deprecated options
    **_: Any,  # ignored options
) -> Any | None:
    r"""Exports a model into ONNX format.

    This function handles different versions of PyTorch and aligns API with the latest
    version.

    See `torch.onnx.export`_ for more details.

    .. _torch.onnx.export: https://pytorch.org/docs/stable/onnx.html
    """

    if torch.__version__ < "2":
        raise RuntimeError(
            f"PyTorch version < 2 is not supported. Your version is {torch.__version__}"
        )

    if torch.__version__ < "2.5.0":
        if kwargs is not None:
            if args and isinstance(args[-1], dict):
                args[-1].update(kwargs)
            else:
                args = args + (kwargs,)
        return torch_export(
            model,
            args,
            f,
            export_params=export_params,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            do_constant_folding=_.get("do_constant_folding", True),
            custom_opsets=_.get("custom_opsets"),
            export_modules_as_functions=_.get("export_modules_as_functions", False),
            autograd_inlining=_.get("autograd_inlining", True),
            # avoid onnx checker in old torch versions
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
    else:
        sign = signature(torch_export)
        if any(
            i not in sign.parameters
            for i in (
                "kwargs",
                "export_params",
                "verbose",
                "input_names",
                "output_names",
                "opset_version",
                "dynamic_axes",
                "keep_initializers_as_inputs",
                "dynamo",
                "external_data",
                "dynamic_shapes",
                # "custom_translation_table",
                "report",
                "verify",
                "profile",
                "dump_exported_program",
                "artifacts_dir",
                "fallback",
            )
        ):
            raise RuntimeError(
                f"Current PyTorch version is too high: {torch.__version__}, "
                "Try downgrade to torch>=2.5.0,<2.9.0"
            )
        return torch_export(
            model,
            args,
            f,
            kwargs=kwargs,
            export_params=export_params,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            dynamo=dynamo,
            external_data=external_data,
            dynamic_shapes=dynamic_shapes,
            custom_translation_table=custom_translation_table,
            report=report,
            verify=verify,
            profile=profile,
            dump_exported_program=dump_exported_program,
            artifacts_dir=artifacts_dir,
            fallback=fallback,
        )
