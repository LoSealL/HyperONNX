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

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL is not available")

from hyperonnx.compile.cutlass_kernels.extract import detect_gpu_arch  # noqa: E402
from hyperonnx.compile.cutlass_kernels.run import run_cutlass_extern  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

CONFIG = {
    "tile_m": 128,
    "tile_n": 128,
    "tile_k": 8,
    "num_stages": 2,
    "num_warps": 4,
}


@pytest.mark.parametrize(
    "N,C,H,W,K,R,groups,stride,pad,bias",
    [
        (2, 512, 12, 20, 512, 3, 512, (1, 1), (1, 1), True),  # matchstereo dwconv
        (2, 64, 24, 40, 64, 3, 64, (1, 1), (1, 1), False),
        (1, 32, 16, 16, 64, 3, 4, (2, 2), (1, 1), True),  # general grouped
    ],
    ids=["dwconv-512", "dwconv-64", "grouped-4-stride2"],
)
def test_run_conv_grouped(N, C, H, W, K, R, groups, stride, pad, bias):
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, device="cuda")
    weight = torch.randn(K, C // groups, R, R, device="cuda")
    b = torch.randn(K, device="cuda") if bias else None

    ref = torch.nn.functional.conv2d(x, weight, b, stride, pad, groups=1 * groups)

    kwargs = {
        "stride": list(stride),
        "padding": list(pad),
        "dilation": [1, 1],
        "groups": groups,
        "bias": b,
    }
    got = run_cutlass_extern(
        "convolution", [x, weight], CONFIG, detect_gpu_arch(), kwargs=kwargs
    )

    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=2e-3), (got - ref).abs().max()
