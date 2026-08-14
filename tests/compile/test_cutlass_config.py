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

from hyperonnx.compile.cutlass_kernels.config import (
    CONV_CONFIGS,
    MM_CONFIGS,
    CutlassConfig,
)


def test_cutlass_config_roundtrip():
    cfg = CutlassConfig(128, 256, 64, 3, 4)
    d = cfg.to_dict()
    assert CutlassConfig.from_dict(d) == cfg


def test_mm_configs_not_empty():
    assert len(MM_CONFIGS) >= 6


def test_conv_configs_not_empty():
    assert len(CONV_CONFIGS) >= 2


def test_config_frozen():
    cfg = CutlassConfig(64, 64, 32, 2, 2)
    try:
        cfg.tile_m = 128  # type: ignore[misc]
        assert False, "should be frozen"
    except AttributeError:
        pass
