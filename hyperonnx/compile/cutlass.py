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

Post-export pipeline to annotate extern_kernel steps with CUTLASS tuning config.

Reads manifest.json from a bundle directory, runs CuTe DSL autotuning for
each extern_kernel step, and writes the best config as `cutlass_config`
on the step. The C++ runtime can use this config to instantiate the same
CUTLASS template.

Idempotent: running twice on the same manifest produces the same result.
"""

import json
import sys
from pathlib import Path

from onnxifier.logger import debug, info, warning

from .cutlass_kernels import get_tuner
from .cutlass_kernels.extract import detect_gpu_arch


def _is_cutlass_supported() -> bool:
    """Check if CUTLASS CuTe DSL is likely available."""
    if sys.platform != "linux":
        return False
    try:
        import cutlass.cute  # noqa: F401  # type: ignore[reportMissingImports]

        return True
    except ImportError:
        return False


def annotate_cutlass_config(
    bundle_dir: str | Path,
    *,
    manifest: dict | None = None,
    arch: str | None = None,
    op_filter: set[str] | None = None,
) -> Path:
    """Annotate extern_kernel steps with CUTLASS tuning config.

    Reads manifest.json from bundle_dir, runs CuTe DSL autotuning for
    each eligible extern_kernel step, and writes the best config as
    ``cutlass_config`` on the step. Does NOT replace the step type.

    Args:
        bundle_dir: path to the .kernels/ bundle directory.
        manifest: pre-loaded manifest dict; loaded from bundle_dir if None.
        arch: GPU arch (e.g. "sm_90"); auto-detected if None.
        op_filter: if set, only tune these op types (e.g. {"mm"}).

    Returns:
        Path to the updated manifest.json.
    """
    if not _is_cutlass_supported():
        warning(
            "CUTLASS tuning unavailable: requires Linux with CuTe DSL "
            "(nvidia-cutlass-dsl). Extern kernel steps will not be annotated."
        )
        return Path(bundle_dir) / "manifest.json"

    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"

    if manifest is None:
        manifest = json.loads(manifest_path.read_text())

    if arch is None:
        arch = detect_gpu_arch()

    tuned = 0

    for graph in manifest.get("pipeline", []):  # type: ignore[union-attr]
        buffers = graph.get("buffers", {})
        for step in graph.get("steps", []):
            if step.get("type") != "extern_kernel":
                continue

            kernel_name = step.get("kernel", "")
            op_short = (
                kernel_name.rsplit(".", 1)[-1] if "." in kernel_name else kernel_name
            )

            if op_filter and op_short not in op_filter:
                continue

            # Idempotency: skip already-tuned steps (null counts as tuned)
            if "cutlass_config" in step:
                continue

            tuner = get_tuner(kernel_name)
            if tuner is None:
                debug(f"no CUTLASS tuner for {kernel_name}; skipping")
                continue

            try:
                args = step.get("args", [])
                config, bench = tuner(args, buffers, arch)
                # ``cutlass_config`` always carries the best CUTLASS config
                # (a naive fallback when none is eligible). ``cutlass_bench``
                # records cuBLAS vs CUTLASS timings and ``winner``; the runner
                # picks the kernel from ``winner`` at replay time.
                step["cutlass_config"] = config.to_dict()
                step["cutlass_bench"] = bench
                if not config.naive:
                    tuned += 1
            except Exception as exc:
                warning(f"CUTLASS tuning failed for {kernel_name}: {exc}")
                continue

    if tuned > 0:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        info(
            f"annotated {tuned} extern_kernel step(s) with CUTLASS config "
            f"(loser steps keep their cuBLAS/cuDNN call, see cutlass_bench)"
        )
    else:
        debug("no extern_kernel steps annotated")

    return manifest_path
