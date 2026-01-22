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

import inspect
from contextlib import contextmanager
from functools import wraps
from io import BytesIO
from os import PathLike
from pathlib import Path
from types import MethodType
from typing import Collection, Dict, List, Optional, Tuple

from torch import inference_mode
from torch.nn import Module

from .hyper_export import export_hyper_onnx
from .typing import AnyTensor, ModuleSpec


class AutoTraceMethod(Module):
    """Automatically trace export all functionalities of a class method.

    Example::

        class MyClass:

            def _some_real_work(self, *args):
                ...

            def sample(self, input_tokens):
                # some code here
                self._some_real_work(features)


        my_class = MyClass()
        with AutoTraceMethod(my_class._some_real_work) as tracer:
            # normal inference
            my_class.sample(input_tokens=tokens)
            tracer.export("some_real_work.onnx", hiera=[MySubClass])

    Note:

        * The method should be a method of any object, can't be a pure function.
        * The method should be "tensor-to-tensor", any non-tensor and non-POD data
          or cached members can't be traced.
    """

    def __init__(self, clsmethod: MethodType, expected_stages: int = 1):
        super().__init__()
        self.pos_args: List[Tuple[AnyTensor, ...]] = []
        self.kwargs: List[Dict[str, AnyTensor]] = []
        model = clsmethod.__self__
        if not isinstance(model, Module):
            for k, v in model.__dict__.items():
                setattr(self, k, v)
            self._forward = None
        else:
            self._forward = model.forward
        self._method = clsmethod
        self._cnt = 0
        self._expected_stages = expected_stages

    def forward(self, *args, **kwargs):
        """Redirect to the original method"""
        if self._cnt >= self._expected_stages:
            raise StopIteration(
                f"Method call exceeds the expected stages {self._expected_stages}."
            )
        self.pos_args.append(args)
        self.kwargs.append(kwargs)
        self._cnt += 1
        return self._method(*args, **kwargs)

    def export(
        self,
        f: str | PathLike | BytesIO,
        *,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 19,
        dynamo: bool = False,
        external_data: bool = False,
        hiera: Optional[Collection[type[Module]]] = None,
        module_spec: Optional[Dict[Module, ModuleSpec]] = None,
        do_optimization: bool = True,
        external_directory: Optional[str | PathLike] = None,
    ):
        """Export onnx model according to the traced data."""
        if not self.pos_args and not self.kwargs:
            raise RuntimeError(
                "No input data traced, did you call the method under context "
                "HYPERONNX.export.auto_trace_method?"
            )
        for i in range(self._expected_stages):
            args = self.pos_args[i]
            kwargs = self.kwargs[i]
            if isinstance(self._method.__self__, Module):
                model: Module = self._method.__self__
                model.forward = self._method
            else:
                model = self
                args = tuple(args) + tuple(kwargs.values())
                kwargs = {}
            try:
                if isinstance(f, BytesIO) and self._expected_stages > 1:
                    raise RuntimeError(
                        "Can't export multiple stages into a single BytesIO."
                    )
                elif isinstance(f, (str, PathLike)) and self._expected_stages > 1:
                    f = Path(f)
                    f = f.with_suffix(f".{i}{f.suffix}")
                if external_directory and self._expected_stages > 1:
                    external_directory = Path(external_directory)
                    external_directory = external_directory / f"{i}"
                    external_directory.mkdir(parents=True, exist_ok=True)
                export_hyper_onnx(
                    model,
                    args,
                    f,
                    kwargs=kwargs,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=opset_version,
                    dynamo=dynamo,
                    external_data=external_data,
                    hiera=hiera,
                    module_spec=module_spec,
                    do_optimization=do_optimization,
                    external_directory=external_directory,
                )
            finally:
                if (
                    isinstance(self._method.__self__, Module)
                    and self._forward is not None
                ):
                    self._method.__self__.forward = self._forward


@contextmanager
def auto_trace_method(clsmethod: MethodType, expected_stages: int = 1):
    """Enter a context to trace the given method, and try to export this method
    directly by wrapping it with AutoTraceMethod.

    Args:
        clsmethod (MethodType): a class method to export
        expected_stages (int): number of expected `clsmethod` calls, will raise
            a StopIteration exception if call of `clsmethod` exceeds this number.
            Defaults to 1.

    Yields:
        AutoTraceMethod: a tracer handle to help export with traced data.

    Raises:
        StopIteration: if the call of `clsmethod` exceeds `expected_stages`.

    Note:
        The method be traced must be a tensor-to-tensor method. The array of
        numpy is not supported.

    Warning:
        This function is still in experimental stage, and may change in the future.

    Example::

        with auto_trace_method(my_model.function1) as tracer:
            # inference normally
            my_model(input_data)
            # call tracer.export within the context
            tracer.export("function1.onnx")
    """

    if not inspect.ismethod(clsmethod):
        raise ValueError(
            "clsmethod should be a method of a nn.Module object, "
            "call auto_trace_method(Foo().bar) rather than auto_trace_method(Foo.bar)."
        )

    self = clsmethod.__self__
    method_ori = clsmethod
    context = inference_mode()
    try:
        # pylint: disable=unnecessary-dunder-call
        context.__enter__()
        tracer = AutoTraceMethod(clsmethod, expected_stages)

        @wraps(clsmethod)
        def _func(*args, **kwargs):
            return tracer(*args, **kwargs)

        setattr(self, clsmethod.__name__, _func)
        yield tracer
    finally:
        setattr(self, clsmethod.__name__, method_ori)
        context.__exit__(None, None, None)
