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

CUTLASS autotuning configuration grid.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CutlassConfig:
    """One CUTLASS kernel tile configuration."""

    tile_m: int
    tile_n: int
    tile_k: int
    num_stages: int
    num_warps: int
    naive: bool = False

    @property
    def m_warps(self) -> int:
        return max(1, self.num_warps // 2)

    @property
    def n_warps(self) -> int:
        return max(1, self.num_warps // self.m_warps)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CutlassConfig":
        return cls(
            tile_m=d["tile_m"],
            tile_n=d["tile_n"],
            tile_k=d["tile_k"],
            num_stages=d["num_stages"],
            num_warps=d["num_warps"],
            naive=d.get("naive", False),
        )


MM_CONFIGS: list[CutlassConfig] = [
    CutlassConfig(128, 128, 32, 4, 4),
    CutlassConfig(128, 128, 64, 3, 4),
    CutlassConfig(128, 256, 32, 4, 8),
    CutlassConfig(128, 256, 32, 3, 8),
    CutlassConfig(128, 128, 32, 3, 4),
    CutlassConfig(128, 128, 32, 4, 8),
]

CONV_CONFIGS: list[CutlassConfig] = [
    CutlassConfig(128, 128, 32, 3, 4),
    CutlassConfig(128, 256, 32, 3, 8),
]

# Fallback config kept in the manifest when no tiled CUTLASS kernel can run
# (fp32, unaligned shapes). The bench record still documents cuBLAS winning.
NAIVE_CONFIG = CutlassConfig(32, 32, 8, 1, 1, naive=True)
