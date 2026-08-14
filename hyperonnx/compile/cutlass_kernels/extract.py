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

GPU auto-detection for CUTLASS tuning.
"""

import importlib


def detect_gpu_arch() -> str:
    """Returns CUDA compute capability, e.g. 'sm_90'.

    Raises RuntimeError if no CUDA device is available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return f"sm_{cap[0]}{cap[1]}"
    except Exception:
        pass
    try:
        drv = importlib.import_module("cuda.bindings.driver")
        major = drv.cuDeviceGetAttribute(
            drv.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0
        )
        minor = drv.cuDeviceGetAttribute(
            drv.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0
        )
        return f"sm_{major}{minor}"
    except Exception:
        pass
    raise RuntimeError(
        "No CUDA device available. CUTLASS tuning requires a Linux machine "
        "with a CUDA GPU. Use Docker if on a non-Linux host."
    )
