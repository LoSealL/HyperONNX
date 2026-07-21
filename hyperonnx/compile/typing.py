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

from typing import TypedDict


class GPUTarget(TypedDict):
    backend: str
    arch: str
    warp_size: int


class LaunchDescriptor(TypedDict):
    num_warps: int
    num_ctas: int
    shared_mem_bytes: int
    num_regs: int


class CompiledKernelInfo(TypedDict):
    cubin_bytes: bytes
    symbol: str
    device_target: GPUTarget
    launch: LaunchDescriptor


class KernelEntry(TypedDict):
    id: str
    cubin: str  # filename within bundle dir
    symbol: str
    device_target: GPUTarget
    launch: LaunchDescriptor


class KernelBundleManifest(TypedDict):
    schema_version: int
    module: dict
    io: dict
    kernels: list[KernelEntry]
