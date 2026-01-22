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

import torch

from hyperonnx.exporter.utils import detach_module_outputs
from hyperonnx.typing import _module_spec_defaultdict_factory


def test_detach_normal_outputs():
    x = torch.randn(1, 2, 3)
    x.requires_grad_(True)
    output = detach_module_outputs(x, _module_spec_defaultdict_factory())
    assert isinstance(output, torch.Tensor)
    assert not output.requires_grad


def test_detach_nested_outputs():
    x = torch.randn(1, 2, 3).requires_grad_(True)
    y = torch.randn(1).requires_grad_(True)
    outputs = detach_module_outputs((x, [y]), _module_spec_defaultdict_factory())
    assert isinstance(outputs, tuple)
    assert isinstance(outputs[0], torch.Tensor)
    assert not outputs[0].requires_grad
    assert isinstance(outputs[1], list)
    assert isinstance(outputs[1][0], torch.Tensor)
    assert not outputs[1][0].requires_grad
