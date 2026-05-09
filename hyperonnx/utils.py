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

import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any

from onnxifier.logger import debug

HYPER_DOMAIN = "hyper"

OPTIMIZER_PASSES = (
    "constantofshape_to_constant",
    "fold_constant",
    "eliminate_identity",
    "eliminate_dead_nodes",
    "fold_constant",
    "erase_output_types",
    "onnx_simplifier",
)


def capture_torch_jit_warnings(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator for a function to capture warnings,
    emitted during legacy torch onnx export.

    Args:
        fn (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: The decorated function.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = fn(*args, **kwargs)
            debug("Captured %s warnings during torch export.", len(w))
            for msg in w:
                debug("- %s", msg)
            return res

    return wrapper
