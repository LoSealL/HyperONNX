"""
Copyright (C) 2025-2026 The HYPERONNX Authors.

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

from collections.abc import Collection, Container
from contextlib import suppress
from inspect import signature
from io import BytesIO
from logging import Logger
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import onnx
from onnxifier import ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.logger import nest
from onnxifier.utils import chdir, legalize_path_name
from torch import Tensor
from torch.nn import Module

from .exporter import replace_with_duck_module
from .exporter.utils import detach_module_outputs, plain_tensor_container
from .function_rewriter import (
    ComposeNodesToFunctionsRewriter,
    ComposeOnnxAsFunctionRewriter,
    FuseConstantsToFunctionRewriter,
)
from .torch_export import torch_export_handle_lower_version
from .typing import (
    AnyTensor,
    ExportStatus,
    HookCallback,
    ModuleSpec,
    default_module_spec,
)
from .utils import HYPER_DOMAIN, OPTIMIZER_PASSES


def _get_input_names(spec: ModuleSpec) -> list[str]:
    names: list[str] = []
    params = spec["signature"].parameters

    def _arg_to_name(name: str, args):
        # WA: filter out None
        plain_arg = [x for x in args if x is not None]
        if len(plain_arg) > 1:
            names.extend([f"{name.lower()}_{i}" for i in range(len(plain_arg))])
        elif len(plain_arg) > 0:
            names.append(name.lower())

    for arg, param in zip(spec["args"], params):
        plain_arg = plain_tensor_container(arg)
        _arg_to_name(param, plain_arg)
    if kwargs := spec.get("kwargs"):
        for param in [p for p in params if p in kwargs]:
            plain_arg = plain_tensor_container(kwargs[param])
            _arg_to_name(param, plain_arg)
    return names


def _get_output_names(spec: ModuleSpec) -> None | list[str]:
    output = spec.get("output")
    if output is None:
        return None

    def _extract_names_from_dict(out_dict: dict) -> list[str]:
        names: list[str] = []
        for key, value in out_dict.items():
            if isinstance(value, dict):
                names.extend(_extract_names_from_dict(value))
                continue
            plain_value = plain_tensor_container(value)
            # WA: filter out None and False
            plain_value = [
                x for x in plain_value if x.ndim != 0 or x.item() is not False
            ]
            if len(plain_value) > 1:
                names.extend([f"{key}_{i:02d}" for i, _ in enumerate(plain_value)])
            elif len(plain_value) > 0:
                names.append(key)
        return names

    if isinstance(output, dict):
        return _extract_names_from_dict(output)
    else:
        return _extract_names_from_dict({spec["name"]: output})


def make_hierarchical_hook(
    hiera: Container[type[Module]], module_spec: dict[Module, ModuleSpec], index: int
) -> HookCallback:
    """Make a forward hook to record spec of modules in `hiera`.

    Args:
        hiera (Container[Module]): The container of modules to be recorded.
        module_spec (Dict[Module, ModuleSpec]): The dictionary to store the spec of
            modules.
        index (int): An unique index of enumerated modules

    Returns:
        HookCallback: The forward hook function.
    """

    def _hook(
        module: Module,
        args: tuple[Tensor],
        kwargs: dict[str, AnyTensor],
        output: AnyTensor,
    ) -> None:
        spec = module_spec[module]
        if type(module) in hiera and spec["status"] == ExportStatus.INITED:
            spec["type_name"] = f"{type(module).__name__}:{index}"
            spec["signature"] = signature(module.forward)
            spec["args"] = args
            spec["kwargs"] = kwargs
            spec["output"] = detach_module_outputs(output, spec)
            spec["status"] = ExportStatus.FORWARDED
            spec["input_names"] = _get_input_names(spec)
            spec["output_names"] = _get_output_names(spec) or []
        elif spec["status"] == ExportStatus.FORWARDED:
            spec["loops"] += 1
            spec["loop_outputs"].append(detach_module_outputs(output, spec))
        return

    return _hook


def trace_module_spec(
    model: Module,
    input_args: tuple,
    kwargs: dict[str, AnyTensor] | None,
    opset_version: int,
    hiera: Container[type[Module]],
    module_spec: dict[Module, ModuleSpec],
    dynamo: bool = False,
) -> dict[Module, ModuleSpec]:
    """Register forward hooks to modules in the `hiera` and record forward
    information to `module_spec`.

    Args:
        model (Module): the model to be traced (exported).
        input_args (tuple): a tuple of input args to the model.
        kwargs (Optional[Dict[str, AnyTensor]]): a dictionary of input kwargs to the
            model.
        hiera (Container[Module]): a container of modules to be recorded.
        module_spec (Dict[Module, ModuleSpec]): a dictionary to store the spec of
            modules.
        dynamo: Use dynamo to export the model. Defaults to False.
    """
    for i, (name, child) in enumerate(model.named_modules()):
        hook = make_hierarchical_hook(hiera, module_spec, i)
        if type(child) in hiera:
            handle = child.register_forward_hook(hook, with_kwargs=True)
            module_spec[child]["name"] = name or "main"  # avoid empty name
            module_spec[child]["handle"] = handle
    if kwargs is None:
        kwargs = {}
    kwargs = kwargs.copy()
    if input_args and isinstance(input_args[-1], dict):
        kwargs.update(input_args[-1])
        input_args = input_args[:-1]
    # To treat some special codes that it differs logic of the original forward
    # from the one in exporting. For example:
    # ```python
    # if torch.jit.is_tracing():
    #    forward1(...)
    # else:
    #    forward2(...)
    # ```
    # We have to fill module spec with the same logic in tracing.
    export_ok = None
    with TemporaryDirectory() as tmpdir, suppress(Exception):
        torch_export_handle_lower_version(
            model,
            input_args,
            tmpdir + "/_trace.onnx",
            kwargs=kwargs,
            opset_version=opset_version,
            # currently dynamo exporting can be run more than once
            dynamo=False,
            external_data=False,
        )
        export_ok = Path(tmpdir + "/_trace.onnx").exists()
    if not export_ok:
        # fallback to call model directly, leave `_` to debug
        _ = model(*input_args, **kwargs)
    for spec in module_spec.values():
        if "handle" in spec:
            spec["handle"].remove()
            spec.pop("handle")
    return module_spec


def _export_hiera(
    opset_version: int,
    dynamo: bool,
    external_data: bool,
    do_optimization: bool,
    fold_nodes_to_functions: bool,
    fuse_constants_to_function: bool,
    external_directory: str | PathLike | None,
    module_spec: dict[Module, ModuleSpec],
    hiera: Collection[type[Module]],
    logger: Logger,
):
    def _get_sub_spec(module: Module, spec: dict[Module, ModuleSpec]):
        child_spec: dict[Module, ModuleSpec] = default_module_spec()
        for child in module.modules():
            if child in spec:
                child_spec[child] = spec[child]
        return child_spec

    def _job(module: Module, spec: ModuleSpec):
        with BytesIO() as modelbytes:
            spec["status"] = ExportStatus.IN_EXPORTING
            child_spec = _get_sub_spec(module, module_spec)
            module_input_names = spec["input_names"]
            module_output_names = spec["output_names"]
            model_path: BytesIO | Path = modelbytes
            if external_data and external_directory:
                model_name = legalize_path_name(f"{spec['type_name']}.onnx")
                model_path = Path(external_directory) / model_name
            export_hyper_onnx(
                module,
                spec["args"],
                model_path,  # type: ignore
                kwargs=spec.get("kwargs"),
                input_names=module_input_names,
                output_names=module_output_names,
                opset_version=opset_version,
                dynamo=dynamo,
                external_data=external_data,
                hiera=hiera,
                module_spec=child_spec,
                do_optimization=do_optimization,
                fold_nodes_to_functions=fold_nodes_to_functions,
                fuse_constants_to_function=fuse_constants_to_function,
                external_directory=external_directory,
            )
            # update since child_spec may be updated
            module_spec.update(child_spec)
            if isinstance(model_path, Path):
                onnx_model = onnx.load_model(model_path, load_external_data=False)
            else:
                onnx_model = onnx.load_model_from_string(model_path.getvalue())
        # WA[torch ~2.5]: jit trace for some of modules may offer wrong inputs, hence
        # they won't follow input_names perfectly. We should replace this strict logic
        # by comparing the number of inputs.
        unused_inputs = set(module_input_names).difference(
            [i.name for i in onnx_model.graph.input]
        )
        if len(unused_inputs) != len(module_input_names) - len(onnx_model.graph.input):
            # Do not assign it, let fallback logic in `ComposeOnnxAsFunctionRewriter`
            # to determine unused inputs automatically (but unsafe).
            unused_inputs.clear()
        unused_outputs = ()
        if module_output_names:
            unused_outputs = set(module_output_names).difference(
                [i.name for i in onnx_model.graph.output]
            )
        base_dir = None
        if isinstance(model_path, Path):
            base_dir = model_path.parent.as_posix()
        graph = OnnxGraph(onnx_model, base_dir=base_dir)
        passes = ("initializer_to_constant",)
        if do_optimization:
            passes = OPTIMIZER_PASSES + passes
        with chdir(external_directory):
            graph = PassManager(passes).optimize(graph, strict=False)

        if external_directory:
            external_path = Path(external_directory) / f"{spec['type_name']}.onnx"
            external_path = legalize_path_name(external_path)
            logger.info(f"Saving {type(module)} to {external_path}")
            graph.save(
                external_path,
                save_as_external_data=external_data,
                check=False,
            )
            onnx_model = external_path.resolve()
        else:
            onnx_model = graph.model

        spec["unused_inputs"] = tuple(unused_inputs)
        spec["unused_outputs"] = tuple(unused_outputs)
        spec["onnx"] = onnx_model
        spec["status"] = ExportStatus.EXPORTED

    for i, (module, spec) in enumerate(module_spec.items()):
        if spec["status"] != ExportStatus.FORWARDED:
            # This function could be called recursively, no need to export same
            # function twice.
            continue
        # TODO: map to a threadpool
        _job(module, spec)


def export_hyper_onnx(  # noqa: C901
    model: Module,
    input_args: tuple,
    f: str | PathLike | BytesIO,
    *,
    kwargs: dict[str, AnyTensor] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    opset_version: int = ONNXIFIER_OPSET.version,
    dynamo: bool = False,
    external_data: bool = False,
    hiera: Collection[type[Module]] | None = None,
    module_spec: dict[Module, ModuleSpec] | None = None,
    do_optimization: bool = True,
    fold_nodes_to_functions: bool = True,
    fuse_constants_to_function: bool = True,
    external_directory: str | PathLike | None = None,
    **_: Any,  # ignored options
) -> Any | None:
    r"""Export a Pytorch module to ONNX format with hierarchical structure,
    so called hyper-onnx.

    Args:
        model (Module): The model to be exported. Must be a subclass of
            :class:`torch.nn.Module`.
        input_args (tuple): Example positional inputs. Any non-Tensor arguments will be
            hard-coded into the exported model; any Tensor arguments will become inputs
            of the exported model, in the order they occur in the tuple.
        f (str | PathLike | BytesIO): The filename or a BytesIO to save the model.
        kwargs (Optional[Dict[str, AnyTensor]]): Optional example keyword inputs.
        input_names: names to assign to the input nodes of the graph, in order.
        output_names: names to assign to the output nodes of the graph, in order.
        opset_version: The version of the `default opset`_ to target.
            Must be >= 7.
        dynamo: Whether to export the model with ``torch.export`` ExportedProgram
            instead of TorchScript.
        external_data: Whether to save the model weights as an external data file.
            This is required for models with large weights that exceed the ONNX file
            size limit (2GB). When False, the weights are saved in the ONNX file with
            the model architecture.
        hiera (Optional[Collection[type[Module]]]): A container of types of module to be
            composed as a onnx function.
        module_spec (Optional[Dict[Module, ModuleSpec]]): A dictionary to store the
            detail spec of modules.
        do_optimization: Whether to optimize the exported ONNX model.
        fold_nodes_to_functions: Whether to compose individual nodes into a function.
            Defaults to True.
        fuse_constants_to_function: Whether to fuse constants into function.
            Defaults to True.
        external_directory (Optional[str | PathLike]): The directory to save the onnx
            model exported to be composed. If not specified, the model will be saved
            in the memory. Set to True if functions to be composed are too large.

    .. _default opset: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    """

    model_typename = type(model).__name__
    logger = nest(model_typename)
    if _:
        ignored_params = "\n  ".join(_.keys())
        logger.warning(f"These arguments are ignored:\n  {ignored_params}")
    if external_data and external_directory is None:
        logger.warning("external_data is True but external_directory is not specified.")
        external_directory = Path.cwd()
        logger.info(f"using external_directory={external_directory}.")
    if external_directory:
        external_directory = Path(external_directory)
        external_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting {model_typename}")

    if module_spec is None:
        module_spec = default_module_spec()
    if not hiera or (model in module_spec and len(module_spec) == 1):
        logger.info(f"  Exporting hiera {model_typename}")
        for _dyn in (dynamo, not dynamo):  # fallback
            try:
                return torch_export_handle_lower_version(
                    model,
                    input_args,
                    f,  # type: ignore
                    kwargs=kwargs,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=opset_version,
                    dynamo=_dyn,
                    external_data=external_data,
                )
            except RuntimeError as ex:
                if _dyn == dynamo:
                    logger.warning(
                        f"  Failed to export {model_typename}, "
                        f"try use dynamo={not _dyn}, "
                        "set ONNXIFIER_LOG_LEVEL=DEBUG to see error logs."
                    )
                logger.debug(f"  <<<\n{ex}")
        raise RuntimeError(f"  Failed to export {model_typename}.")

    # forward to record the modules' spec, and only forward once
    if model not in module_spec or module_spec[model]["status"] == ExportStatus.INITED:
        trace_module_spec(
            model=model,
            input_args=input_args,
            kwargs=kwargs,
            opset_version=opset_version,
            hiera=hiera,
            module_spec=module_spec,
            dynamo=dynamo,
        )

    _export_hiera(
        opset_version=opset_version,
        dynamo=dynamo,
        external_data=external_data,
        do_optimization=do_optimization,
        fold_nodes_to_functions=fold_nodes_to_functions,
        fuse_constants_to_function=fuse_constants_to_function,
        external_directory=external_directory,
        module_spec=module_spec,
        hiera=hiera,
        logger=logger,
    )

    if model in module_spec:
        spec = module_spec[model]
        model_typename = spec["type_name"]
        if not input_names:
            input_names = spec["input_names"]
        if not output_names:
            output_names = spec["output_names"]

    with replace_with_duck_module(model, dynamo, module_spec) as tb, BytesIO() as mb:
        model_path: BytesIO | Path = mb
        if external_directory:
            modelname = legalize_path_name(f"{model_typename}_combined.onnx")
            model_path = Path(external_directory) / modelname
        logger.info(f"Combining {model_typename} to {model_path}")
        try:
            torch_export_handle_lower_version(
                model,
                input_args,
                model_path,  # type: ignore
                kwargs=kwargs,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                dynamo=dynamo,
                external_data=external_data,
                custom_translation_table=tb,
            )
        except RuntimeError as e:
            raise RuntimeError(f"Failed to export {model_typename}") from e
        if isinstance(model_path, Path):
            onnx_model = onnx.load_model(model_path, load_external_data=False)
        else:
            onnx_model = onnx.load_model_from_string(model_path.getvalue())

    typenames: list[str] = [module_spec[i]["type_name"] for i in module_spec]
    for node in onnx_model.graph.node:
        if node.op_type in typenames:
            node.domain = HYPER_DOMAIN  # add domain tag to run rewriter
    graph = OnnxGraph(onnx_model)
    passes: list[Any]
    passes = [ComposeOnnxAsFunctionRewriter(HYPER_DOMAIN, tuple(module_spec.values()))]
    if do_optimization:
        passes.extend(OPTIMIZER_PASSES)
    if fold_nodes_to_functions:
        passes.append(ComposeNodesToFunctionsRewriter(model_typename))
    if fuse_constants_to_function:
        passes.append(FuseConstantsToFunctionRewriter())
    with chdir(external_directory):
        graph = PassManager(
            # onnxsim always fail to process large models, so we bypass it.
            passes,
            exclude=["onnx_simplifier"] if external_data else [],
        ).optimize(graph, strict=False)
    if graph.external_base is None and external_data:
        graph.external_base = external_directory
    logger.info(f"Saving hyper-onnx {model_typename}...")
    graph.save(f, check=False, save_as_external_data=external_data)
    logger.info(f"{model_typename} saved to {f}")
