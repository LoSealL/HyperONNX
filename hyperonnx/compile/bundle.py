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
from pathlib import Path

from onnxifier.logger import info
from onnxifier.utils import legalize_path_name

from .typing import CompiledKernelInfo, KernelBundleManifest, KernelEntry

_SCHEMA_VERSION = 1


def write_kernel_bundle(
    directory: Path,
    type_name: str,
    kernels: list[CompiledKernelInfo],
    module_io: dict,
    module_meta: dict,
) -> Path:
    """Write the kernel bundle sidecar directory.

    Creates `<directory>/<legalized(type_name)>.kernels/` containing:
      - manifest.json
      - kernel_NNNN.cubin for each captured kernel

    The directory name is legalized via onnxifier.utils.legalize_path_name
    to be filesystem-safe while staying deterministic.

    Args:
        directory: parent directory (typically external_directory).
        type_name: module spec type_name (e.g. "Attention:0").
        kernels: list of captured kernels.
        module_io: {"inputs": [...], "outputs": [...]} mirroring the ONNX
            function. Serialized as the `io` key in the manifest JSON.
        module_meta: provenance dict with python_class / torch_version / etc.

    Returns:
        The Path to the created bundle directory.
    """
    bundle_name = legalize_path_name(f"{type_name}.kernels")
    bundle_dir = Path(directory) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    entries: list[KernelEntry] = []
    for i, k in enumerate(kernels):
        cubin_filename = f"kernel_{i:04d}.cubin"
        (bundle_dir / cubin_filename).write_bytes(k["cubin_bytes"])
        entries.append(
            KernelEntry(
                id=cubin_filename.removesuffix(".cubin"),
                cubin=cubin_filename,
                symbol=k["symbol"],
                device_target=k["device_target"],
                launch=k["launch"],
                args=k["args"],
                variants=[],
            )
        )

    manifest = KernelBundleManifest(
        schema_version=_SCHEMA_VERSION,
        module=module_meta,
        io=module_io,
        kernels=entries,
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    info(f"wrote kernel bundle: {bundle_dir} ({len(entries)} kernels)")
    return bundle_dir
