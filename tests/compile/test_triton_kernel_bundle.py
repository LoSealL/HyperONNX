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

import json

import pytest
import torch
import torchvision as tv

from hyperonnx import export_hyper_onnx
from hyperonnx.compile.testing import replay

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)
triton = pytest.importorskip("triton", reason="Triton is not available")


class TestCompileModuleA(torch.nn.Module):
    """Test A for layer normalization with affine parameters"""

    def __init__(self, affine_shape):
        super().__init__()
        self.normalized_dim = (-1,)
        self.use_scale = True
        self.use_bias = False
        self.weight = torch.nn.Parameter(torch.ones(affine_shape))
        self.eps = 1e-6

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        x = x * self.weight
        return x


class TestCompileModuleB(torch.nn.Module):
    """Test B for linear softmax"""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 256, 16, 2))

    def forward(self, x):
        B, N, H, _ = x.shape
        x_2d = x.view(B, N, H, 2 * 4 + 2)
        left = x_2d[..., : 2 * 4 + 1]
        right = x_2d[..., 1 : 2 * 4 + 2]
        x = torch.stack([left, right], dim=3)
        x_weighted = x.softmax(dim=-1) * self.weight.unsqueeze(-1)
        merged = x_weighted.new_zeros(B, N, H, 2 * 4 + 2)
        merged[..., : 2 * 4 + 1] += x_weighted[:, :, :, 0, :]
        merged[..., 1 : 2 * 4 + 2] += x_weighted[:, :, :, 1, :]
        return merged


class TestCompileModuleC(torch.nn.Module):
    """Test C for self-attention with causal masking"""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(256, 256)
        self.k_proj = torch.nn.Linear(256, 256)
        self.v_proj = torch.nn.Linear(256, 256)
        self.out_proj = torch.nn.Linear(256, 256)

    def forward(self, x):
        B, N, H = x.shape
        q = self.q_proj(x).view(B, N, 4, H // 4).transpose(1, 2)
        k = self.k_proj(x).view(B, N, 4, H // 4).transpose(1, 2)
        v = self.v_proj(x).view(B, N, 4, H // 4).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (H // 4) ** 0.5
        causal_mask = torch.tril(torch.ones(N, N)).to(attn_weights.device)
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float("-inf"))
        attn_probs = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, H)
        output = self.out_proj(attn_output)
        return output


@pytest.mark.parametrize(
    "module,inputs",
    [
        (
            TestCompileModuleA(affine_shape=(1, 3, 1, 1)),
            [torch.randn(1, 3, 224, 224).cuda()],
        ),
        (
            TestCompileModuleB(),
            [torch.randn(1, 256, 16, 10).cuda()],
        ),
        (
            TestCompileModuleC(),
            [torch.randn(1, 16, 256).cuda()],
        ),
        (
            tv.models.resnet.BasicBlock(64, 64),
            [torch.randn(1, 64, 56, 56).cuda()],
        ),
    ],
    ids=["LayerNorm", "LinearSoftmax", "SelfAttentionCausal", "ResNetBasicBlock"],
)
def test_export_triton_kernel_bundle(tmp_path, module, inputs):
    """Test export of a Triton kernel bundle"""
    module.cuda().eval()
    export_hyper_onnx(
        module,
        tuple(inputs),
        tmp_path / "module_dut.onnx",
        dynamo=True,
        compile_hier=[type(module)],
        compile_static_grid=True,
        external_directory=tmp_path,
        cutlass_tune=True,
    )
    with open(tmp_path / f"{type(module).__name__}_0.kernels/manifest.json") as f:
        manifest = json.load(f)
    assert manifest.get("schema_version") == 2
    manifest_module = manifest.get("module")
    manifest_io = manifest.get("io")
    manifest_pipeline = manifest.get("pipeline")
    manifest_buffers = manifest.get("buffers")

    assert manifest_module["type_name"] == f"{type(module).__name__}:0"
    assert (
        manifest_module["python_class"]
        == f"{type(module).__module__}.{type(module).__name__}"
    )
    assert manifest_module["torch_version"] == torch.__version__
    assert manifest_module["triton_version"] == triton.__version__
    assert manifest_io["inputs"][0]["name"] == "x"
    assert manifest_io["inputs"][0]["dtype"] == "float32"
    assert manifest_io["inputs"][0]["shape"] == list(inputs[0].shape)
    assert manifest_io["inputs"][0]["buffer_id"] > 0
    assert manifest_io["outputs"][0]["dtype"] == "float32"
    assert manifest_io["outputs"][0]["buffer_id"] > 0
    assert len(manifest_pipeline[0]["steps"]) > 0
    assert len(manifest_buffers) > 0

    output = replay(
        tmp_path / f"{type(module).__name__}_0.kernels",
        inputs,
    )
    ref_output = module(*inputs)
    assert torch.allclose(output, ref_output, atol=1e-4)
    inputs[0].detach().cpu().numpy().tofile(tmp_path / "x.bin")
    output.detach().cpu().numpy().tofile(tmp_path / "output.bin")
