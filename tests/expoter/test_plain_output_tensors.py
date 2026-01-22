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

from hyperonnx.exporter.utils import plain_tensor_container


def test_plain_single_tensor():
    t = torch.empty([1])
    p = plain_tensor_container(t)
    assert isinstance(p, tuple)
    assert len(p) == 1
    assert p[0] is t


def test_plain_nested_list():
    t = [torch.empty([1]), [torch.empty([1]), torch.empty([1])]]
    p = plain_tensor_container(t)  # type: ignore
    assert isinstance(p, list)
    assert len(p) == 3
    assert p[0] is t[0]
    assert p[1] is t[1][0]
    assert p[2] is t[1][1]


def test_plain_nested_tuple():
    t = (torch.empty([1]), [torch.empty([1]), torch.empty([1])])
    p = plain_tensor_container(t)  # type: ignore
    assert isinstance(p, tuple)
    assert len(p) == 3
    assert p[0] is t[0]
    assert p[1] is t[1][0]
    assert p[2] is t[1][1]


def test_plain_nested_dict():
    t = dict(a=torch.empty([1]), b=[torch.empty([1]), torch.empty([1])])
    p = plain_tensor_container(t)  # type: ignore
    assert isinstance(p, tuple)
    assert len(p) == 3
    assert p[0] is t["a"]
    assert p[1] is t["b"][0]
    assert p[2] is t["b"][1]


def test_plain_nested_tuple_with_dict():
    t = (
        torch.empty([1]),
        dict(a=torch.empty([1]), b=[torch.empty([1]), torch.empty([1])]),
    )
    p = plain_tensor_container(t)  # type: ignore
    assert isinstance(p, tuple)
    assert len(p) == 4
    assert p[0] is t[0]
    assert p[1] is t[1]["a"]
    assert p[2] is t[1]["b"][0]
    assert p[3] is t[1]["b"][1]
