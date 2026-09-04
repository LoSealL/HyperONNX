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

Tests for the cuBLAS-aware tuning decision (no GPU needed).
"""

import json

import pytest

try:
    import cutlass.cute  # noqa: F401

    _HAS_CUTLASS = True
except ImportError:
    _HAS_CUTLASS = False

_requires_cutlass = pytest.mark.skipif(
    not _HAS_CUTLASS, reason="CUTLASS CuTe DSL not available"
)


@_requires_cutlass
def test_is_tiled_gemm_eligible():
    from hyperonnx.compile.cutlass_kernels.config import CutlassConfig
    from hyperonnx.compile.cutlass_kernels.mm import is_tiled_gemm_eligible

    cfg = CutlassConfig(128, 128, 64, 3, 4)
    assert is_tiled_gemm_eligible(256, 128, 128, cfg)
    assert not is_tiled_gemm_eligible(256, 130, 128, cfg)  # N unaligned
    assert not is_tiled_gemm_eligible(256, 128, 124, cfg)  # K unaligned
    # smem overflow: (128*64 + 128*64) * 8 stages * 2B > 99KB
    big = CutlassConfig(128, 128, 64, 8, 4)
    assert not is_tiled_gemm_eligible(256, 128, 128, big)
    assert not is_tiled_gemm_eligible(256, 128, 128, CutlassConfig(64, 64, 8, 3, 4))


def _mm_args(M=128, N=128, K=128, dtype="float16"):
    return [
        {"kind": "tensor", "shape": [M, K], "dtype": dtype},
        {"kind": "tensor", "shape": [K, N], "dtype": dtype},
    ]


@_requires_cutlass
def test_tune_mm_keeps_config_when_cublas_wins(monkeypatch):
    from hyperonnx.compile.cutlass_kernels import mm

    monkeypatch.setattr(mm, "_bench_tiled_mm", lambda *a, **k: 1.0)
    monkeypatch.setattr(mm, "_bench_cublas_mm", lambda *a: 0.5)

    config, bench = mm.tune_mm(_mm_args(), {}, "sm_120")

    # config is never null — it's the best CUTLASS config tried; the bench
    # record decides actual usage at replay.
    assert config is not None
    assert config.naive is False
    assert bench["winner"] == "cublas"
    assert bench["cutlass_ms"] == 1.0
    assert bench["cublas_ms"] == 0.5


@_requires_cutlass
def test_tune_mm_returns_config_when_cutlass_wins(monkeypatch):
    from hyperonnx.compile.cutlass_kernels import mm

    monkeypatch.setattr(mm, "_bench_tiled_mm", lambda *a, **k: 0.4)
    monkeypatch.setattr(mm, "_bench_cublas_mm", lambda *a: 0.5)

    config, bench = mm.tune_mm(_mm_args(), {}, "sm_120")

    assert config is not None
    assert config.naive is False
    assert bench["winner"] == "cutlass"


@_requires_cutlass
def test_tune_mm_margin_rejects_ties(monkeypatch):
    """Within the 2% margin, cuBLAS stays (noise must not swap kernels)."""
    from hyperonnx.compile.cutlass_kernels import mm

    monkeypatch.setattr(mm, "_bench_tiled_mm", lambda *a, **k: 0.5)
    monkeypatch.setattr(mm, "_bench_cublas_mm", lambda *a: 0.5)

    config, bench = mm.tune_mm(_mm_args(), {}, "sm_120")

    assert config is not None
    assert config.naive is False
    assert bench["winner"] == "cublas"


@_requires_cutlass
def test_tune_mm_fp32_keeps_naive_config():
    from hyperonnx.compile.cutlass_kernels import mm

    config, bench = mm.tune_mm(_mm_args(dtype="float32"), {}, "sm_120")

    assert config is not None
    assert config.naive is True
    assert bench["winner"] == "cublas"
    assert "unsupported" in bench["reason"]


@_requires_cutlass
def test_annotate_keeps_config_when_cublas_wins(tmp_path, monkeypatch):
    """Losing steps keep a (naive) CUTLASS config; cuBLAS is the recorded winner."""
    from hyperonnx.compile import cutlass as cutlass_mod
    from hyperonnx.compile.cutlass_kernels.config import CutlassConfig

    monkeypatch.setattr(
        cutlass_mod,
        "get_tuner",
        lambda name: (
            lambda *a, **k: (
                CutlassConfig(128, 128, 32, 3, 4),
                {
                    "winner": "cublas",
                    "cutlass_ms": 2.0,
                    "cublas_ms": 1.0,
                },
            )
        ),
    )

    bundle_dir = tmp_path / "test.kernels"
    bundle_dir.mkdir()
    manifest = {
        "schema_version": 2,
        "module": {"type_name": "Test"},
        "io": {"inputs": [], "outputs": []},
        "pipeline": [
            {
                "graph": None,
                "buffers": {},
                "steps": [
                    {
                        "type": "extern_kernel",
                        "kernel": "extern_kernels.mm",
                        "args": [],
                        "output": "buf0",
                    },
                ],
            }
        ],
        "buffers": [],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

    cutlass_mod.annotate_cutlass_config(bundle_dir)
    updated = json.loads((bundle_dir / "manifest.json").read_text())
    step = updated["pipeline"][0]["steps"][0]
    assert step["cutlass_config"] is not None
    assert step["cutlass_config"]["naive"] is False
    assert step["cutlass_bench"]["winner"] == "cublas"

    # Idempotent: configured steps are skipped on re-run.
    cutlass_mod.annotate_cutlass_config(bundle_dir)
    again = json.loads((bundle_dir / "manifest.json").read_text())
    assert again == updated


def _conv_args():
    # input (N, C, H, W), weight (K_out, C_in/groups, R, S)
    return [
        {"kind": "tensor", "shape": [2, 128, 48, 80], "dtype": "float16"},
        {"kind": "tensor", "shape": [128, 1, 3, 3], "dtype": "float16"},
    ]


@_requires_cutlass
def test_conv_to_gemm_args_matches_replay_shape():
    """Tuning must see the same per-group im2col GEMM that replay runs."""
    from hyperonnx.compile.cutlass_kernels import conv

    kwargs = [
        "stride=(1, 1)",
        "padding=(1, 1)",
        "dilation=(1, 1)",
        "transposed=False",
        "output_padding=(0, 0)",
        "groups=128",
        "bias=None",
    ]
    gemm_args = conv._conv_to_gemm_args(_conv_args(), {}, kwargs)
    # depthwise 3x3: M=K_out/groups=1, K=1*3*3, N=H_out*W_out=48*80
    assert gemm_args[0]["shape"] == [1, 9]
    assert gemm_args[1]["shape"] == [9, 3840]

    # groups==1 honors stride: H_out=(48-2-1)//2+1=23, W_out=(80-2-1)//2+1=39
    plain = [
        {"kind": "tensor", "shape": [2, 16, 48, 80], "dtype": "float16"},
        {"kind": "tensor", "shape": [32, 16, 3, 3], "dtype": "float16"},
    ]
    gemm_args = conv._conv_to_gemm_args(plain, {}, ["stride=(2, 2)", "groups=1"])
    assert gemm_args[0]["shape"] == [32, 16 * 3 * 3]
    assert gemm_args[1]["shape"] == [16 * 3 * 3, 23 * 39]


@_requires_cutlass
def test_tune_conv_depthwise_picks_cublas(monkeypatch):
    """Per-group depthwise shapes are tiled-ineligible: cuBLAS stays winner."""
    from hyperonnx.compile.cutlass_kernels import conv, mm

    monkeypatch.setattr(mm, "_bench_tiled_mm", lambda *a, **k: 0.1)
    monkeypatch.setattr(mm, "_bench_cublas_mm", lambda *a: 1.0)

    kwargs = ["stride=(1, 1)", "padding=(1, 1)", "groups=128"]
    config, bench = conv.tune_conv(_conv_args(), {}, "sm_120", kwargs=kwargs)

    assert config is not None
    assert config.naive is True
    assert bench["winner"] == "cublas"
    assert bench["cutlass_ms"] is None
