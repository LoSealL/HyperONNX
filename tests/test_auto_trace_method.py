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

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import pytest
import torch
from torch import nn

from hyperonnx import auto_trace_method


class AutoTraceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)

    def _foo(self, x, y, z):
        res = self.conv1(x)
        mask = y + torch.sigmoid(z).unsqueeze(1)
        res = self.pool(res * torch.cat([mask, mask], 1))
        return res

    def forward(self, x):
        y = torch.softmax(x, dim=1)
        z = x.sum(1)
        return self._foo(x, y, z)


@pytest.mark.parametrize("dynamo", [True, False])
def test_auto_trace_method(dynamo):
    model = AutoTraceModel()
    x = torch.randn(1, 3, 224, 224)
    model.eval()

    with auto_trace_method(model._foo) as tracer, BytesIO() as f:
        model(x)
        tracer.export(f, hiera=[nn.Conv2d], input_names=["x", "y", "z"], dynamo=dynamo)
        model = onnx.load_model_from_string(f.getvalue())
    onnx.checker.check_model(model, True)


class AutoTraceRnnCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.entry = nn.Linear(input_size, hidden_size)
        self.tail = nn.Linear(hidden_size, 1)

    def _rnn_cell(self, input, hidden):
        hy = torch.tanh(self.i2h(input) + self.h2h(hidden))
        return hy

    def forward(self, input):
        hidden = self.entry(input)
        for i in range(3):
            hidden = self._rnn_cell(input, hidden)
        return self.tail(hidden)


@pytest.mark.parametrize("dynamo", [True, False])
def test_auto_trace_rnn_cell(dynamo):
    model = AutoTraceRnnCell(10, 20)
    x = torch.randn(1, 10)
    model.eval()

    with (
        auto_trace_method(model._rnn_cell, 2) as tracer,
        TemporaryDirectory() as tmpdir,
    ):
        with pytest.raises(StopIteration):
            model(x)
        tracer.export(
            Path(tmpdir) / "rnn.onnx",
            input_names=["input", "hidden"],
            output_names=["hy"],
            dynamo=dynamo,
        )
        assert len(list(Path(tmpdir).glob("*.onnx"))) == 2
        for onnx_file in Path(tmpdir).glob("*.onnx"):
            model = onnx.load_model(onnx_file)
            onnx.checker.check_model(model, True)
