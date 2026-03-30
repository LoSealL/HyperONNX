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

# pylint: disable=missing-class-docstring,missing-function-docstring

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import onnx
import pytest
import torch
import torchvision as tv
from onnx.external_data_helper import load_external_data_for_model
from onnxifier import OnnxGraph
from onnxifier.evaluator import Evaluator
from onnxifier.utils import chdir
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models.vision_transformer import Encoder, EncoderBlock, vit_b_16

from hyperonnx import export_hyper_onnx
from hyperonnx.transformers import StaticCache, patch_transformers


class ModuleLvl2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, attr1, attr2=2):
        return x * attr1 + y * attr2, x + y


class ModuleLvl1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(1, 1, 1)
        self.body = torch.nn.ModuleList([ModuleLvl2(), ModuleLvl2()])

    def forward(self, x):
        y = self.head(x)
        z1, z2 = x, y
        for i, net in enumerate(self.body):
            z1, z2 = net(z1, z2, attr1=i, attr2=i + 1)
        return {"z1": z1, "z2": z2}


class ModuleTop(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = ModuleLvl1()

    def forward(self, x):
        x = self.body(x)
        return x


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_module_with_2_levels(dynamo):
    model = ModuleTop()
    with BytesIO() as f:
        export_hyper_onnx(
            model,
            (torch.randn(1, 1, 1, 1),),
            f,
            hiera=[ModuleLvl1, ModuleLvl2],
            input_names=["x"],
            output_names=["z1", "z2"],
            dynamo=dynamo,
        )
        onnx_model = onnx.load_from_string(f.getvalue())
    try:
        onnx.checker.check_model(onnx_model, True)
        types = set(
            ":".join(k.split(":")[:-1]) for k in OnnxGraph(onnx_model).functions
        )
        assert len(types) == 3
        assert "ModuleLvl1" in types
        assert "ModuleLvl2" in types
    except Exception:
        onnx.save(onnx_model, "module_2_levels_bad.onnx")
        raise

    data = torch.randn(1, 1, 1, 1)
    ref_output = model(data)
    runner = Evaluator(onnx_model, backend="onnxruntime")
    z1, z2 = runner([], {"x": data.numpy()})
    torch.testing.assert_close(torch.from_numpy(z1), ref_output["z1"])
    torch.testing.assert_close(torch.from_numpy(z2), ref_output["z2"])


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_module_with_input_dict(dynamo):
    model = ModuleTop()
    with BytesIO() as f:
        input_dict = {"x": torch.randn(1, 1, 1, 1)}
        export_hyper_onnx(
            model,
            (input_dict,),
            f,
            hiera=[ModuleLvl1],
            input_names=["x"],
            output_names=["z1", "z2"],
            dynamo=dynamo,
        )
        onnx_model = onnx.load_from_string(f.getvalue())
    onnx.checker.check_model(onnx_model, True)
    types = set(":".join(k.split(":")[:-1]) for k in OnnxGraph(onnx_model).functions)
    assert len(types) == 1
    assert "ModuleLvl1" in types

    data = torch.randn(1, 1, 1, 1)
    ref_output = model(data)
    runner = Evaluator(onnx_model, backend="onnxruntime")
    z1, z2 = runner([], {"x": data.numpy()})
    torch.testing.assert_close(torch.from_numpy(z1), ref_output["z1"])
    torch.testing.assert_close(torch.from_numpy(z2), ref_output["z2"])


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_resnet_with_basicblock_and_bottleneck(dynamo):
    resnet = tv.models.resnet18().eval()
    with BytesIO() as f:
        export_hyper_onnx(
            resnet,
            (torch.randn(1, 3, 224, 224),),
            f,
            input_names=["input.1"],
            hiera=[ResNet, BasicBlock, Bottleneck],
            do_optimization=False,
            dynamo=dynamo,
        )
        model = onnx.load_from_string(f.getvalue())
    try:
        onnx.checker.check_model(model, True)
        types = set(":".join(k.split(":")[:-1]) for k in OnnxGraph(model).functions)
        assert len(types) == 2
        assert "ResNet" in types
        assert "BasicBlock" in types
        assert "Bottleneck" not in types  # there is no Bottleneck in ResNet18

        data = torch.randn(1, 3, 224, 224)
        ref_output = resnet(data)
        runner = Evaluator(model, backend="onnxruntime")
        actual_output = runner([], {"input.1": data.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual_output), ref_output)
    except Exception:
        onnx.save(model, "resnet_bad.onnx")
        raise


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_vit_with_encoder_and_encoderblock(dynamo, tmp_path):
    vit = vit_b_16().eval()
    with BytesIO() as f:
        with chdir(tmp_path):
            export_hyper_onnx(
                vit,
                (torch.randn(1, 3, 224, 224),),
                f,
                hiera=[Encoder, EncoderBlock],
                external_directory=tmp_path,
                external_data=True,
                dynamo=dynamo,
            )
            model = onnx.load_from_string(f.getvalue())
        load_external_data_for_model(model, str(tmp_path))
        external_files = sorted(Path(tmp_path).glob("*.onnx"))
        assert len(external_files) == 15
    try:
        onnx.checker.check_model(model, True)
        types = set(":".join(k.split(":")[:-1]) for k in OnnxGraph(model).functions)
        assert len(types) == 4
        assert "Encoder" in types
        assert "EncoderBlock" in types

        data = torch.randn(1, 3, 224, 224)
        ref_output = vit(data)
        runner = Evaluator(model, backend="onnxruntime")
        actual_output = runner([], {"x": data.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual_output), ref_output)
    except Exception:
        onnx.save(model, "vit_bad.onnx")
        raise


@pytest.mark.parametrize("dynamo", [True, False])
@pytest.mark.parametrize("cache", [True, False])
def test_export_llama_transformers(dynamo, cache, tmp_path):
    if dynamo:
        pytest.skip("dynamo export not work for llama model yet")
    try:
        # pylint: disable=import-outside-toplevel
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            LlamaDecoderLayer,
            LlamaModel,
        )
    except ImportError:
        pytest.skip("transformers not installed")

    config = LlamaConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        use_cache=cache,
        attn_implementation="eager",
    )
    assert config.vocab_size is not None
    assert config.num_hidden_layers is not None
    assert config.num_key_value_heads is not None
    assert config.hidden_size is not None
    assert config.num_attention_heads is not None
    llama = LlamaModel(config).eval()
    example_inputs: dict = dict(
        input_ids=torch.randint(config.vocab_size, [1, 16]).long(),
        attention_mask=torch.ones([1, 16]),
        position_ids=torch.arange(16).long()[None],
        cache_position=torch.arange(16).long(),
    )
    if cache:
        example_inputs["use_cache"] = True
        example_inputs["past_key_values"] = StaticCache(
            size=[config.num_hidden_layers]
        ).early_initialization(
            1,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
            dtype=torch.float32,
        )
    with torch.no_grad():
        ref_output = llama(**example_inputs)
    with BytesIO() as f, patch_transformers(), torch.no_grad():
        input_names = ["input_ids", "attention_mask", "position_ids", "cache_position"]
        if cache:
            input_names.insert(-1, "past_key_values")
        output_names = ["last_hidden_state"]
        if cache:
            output_names.append("present_key_values")
        export_hyper_onnx(
            llama,
            (example_inputs,),
            f,
            input_names=input_names,
            output_names=output_names,
            hiera=[LlamaAttention, LlamaDecoderLayer],
            external_directory=tmp_path,
            dynamo=dynamo,
        )
        model = onnx.load_from_string(f.getvalue())
        load_external_data_for_model(model, "")
    try:
        onnx.checker.check_model(model, True)
    except Exception:
        onnx.save(model, "llama_bad.onnx")
        patch.stopall()
        raise

    onnx.save(model, "llama.onnx")
    runner = Evaluator(model, backend="onnxruntime")
    numpy_inputs = {
        k: v.numpy() for k, v in example_inputs.items() if isinstance(v, torch.Tensor)
    }
    if cache:
        numpy_inputs["past_key_values"].fill(0)
    actual_output = runner([], numpy_inputs)[0]
    torch.testing.assert_close(torch.from_numpy(actual_output), ref_output[0])


class RNNCell(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2


class RNNLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = RNNCell()

    def forward(self, x):
        for _ in range(3):
            x = self.cell(x)
        return x


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_rnnlike(dynamo):
    model = RNNLike()
    with BytesIO() as f:
        # Do not do optimization, especially do not fuse constants into functions
        # So keep RNNCell as one function
        export_hyper_onnx(
            model,
            (torch.randn(1, 1, 1, 1),),
            f,
            hiera=[RNNCell],
            input_names=["x"],
            do_optimization=False,
            dynamo=dynamo,
        )
        onnx_model = onnx.load_from_string(f.getvalue())
    try:
        onnx.checker.check_model(onnx_model, True)
        types = set(
            ":".join(k.split(":")[:-1]) for k in OnnxGraph(onnx_model).functions
        )
        assert len(types) == 1
        assert "RNNCell" in types
    except Exception:
        onnx.save(onnx_model, "rnnlike_bad.onnx")
        raise

    data = torch.randn(1, 1, 1, 1)
    ref_output = model(data)
    runner = Evaluator(onnx_model, backend="onnxruntime")
    x = runner([], {"x": data.numpy()})
    torch.testing.assert_close(torch.from_numpy(x[0]), ref_output)


class ModuleReturnNone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.relu(x)
        return y, None


class ModuleReturnNoneTop(torch.nn.Module):
    def __init__(self, accept_none=False):
        super().__init__()
        self.module1 = ModuleReturnNone()
        self.module2 = ModuleReturnNone()
        self.accept_none = accept_none

    def forward(self, x):
        x, _ = self.module1(x)
        ret = self.module2(x)
        if self.accept_none:
            return ret
        return ret[0]


@pytest.mark.parametrize("accept_none", [True, False])
@pytest.mark.parametrize("dynamo", [True, False])
def test_export_module_return_none(accept_none, dynamo):
    model = ModuleReturnNoneTop(accept_none)
    with BytesIO() as f:
        export_hyper_onnx(
            model,
            (torch.randn(1, 1, 1, 1),),
            f,
            hiera=[ModuleReturnNone, ModuleReturnNoneTop],
            input_names=["x"],
            output_names=["y"],
            dynamo=dynamo,
        )
        onnx_model = onnx.load_from_string(f.getvalue())
    onnx.checker.check_model(onnx_model, True)

    data = torch.randn(1, 1, 1, 1)
    ref_output = model(data)
    if accept_none:
        ref_output = ref_output[0]
    runner = Evaluator(onnx_model, backend="onnxruntime")
    x = runner(["y"], {"x": data.numpy()})
    torch.testing.assert_close(torch.from_numpy(x[0]), ref_output)


class ModuleReturnList(torch.nn.Module):
    def __init__(self, return_as_tuple=False):
        super().__init__()
        self.mlist = torch.nn.ModuleList()
        self.mlist.extend(
            [
                torch.nn.Linear(8, 8),
                torch.nn.Linear(8, 8),
                torch.nn.Linear(8, 8),
            ]
        )
        self.return_as_tuple = return_as_tuple

    def forward(self, x):
        outs = []
        for m in self.mlist:
            outs.append(m(x))
        if self.return_as_tuple:
            return tuple(outs)
        return outs


@pytest.mark.parametrize("return_as_tuple", [True, False])
@pytest.mark.parametrize("dynamo", [True, False])
def test_export_module_return_list(return_as_tuple, dynamo):
    model = ModuleReturnList(return_as_tuple)
    with BytesIO() as f:
        export_hyper_onnx(
            model,
            (torch.randn(1, 8),),
            f,
            hiera=[ModuleReturnList],
            input_names=["x"],
            output_names=["y"],
            dynamo=dynamo,
        )
        onnx_model = onnx.load_from_string(f.getvalue())
    onnx.checker.check_model(onnx_model, True)

    data = torch.randn(1, 8)
    ref_output = model(data)
    runner = Evaluator(onnx_model, backend="onnxruntime")
    x = runner(["y"], {"x": data.numpy()})
    for i, j in zip(ref_output, x):
        torch.testing.assert_close(torch.from_numpy(j), i)
