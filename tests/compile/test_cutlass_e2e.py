"""End-to-end test for CUTLASS config annotation."""

import json

_HAS_CUTLASS = False
try:
    _HAS_CUTLASS = True
except Exception:
    pass


def test_no_extern_kernels(tmp_path):
    """Manifest with no extern_kernel steps is a no-op."""
    from hyperonnx.compile.cutlass import annotate_cutlass_config

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
                        "type": "allocate",
                        "buffer": "buf0",
                        "shape": [10],
                        "stride": [1],
                        "dtype": "float32",
                    },
                ],
            }
        ],
        "buffers": [],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

    result = annotate_cutlass_config(bundle_dir, manifest=manifest)
    updated = json.loads(result.read_text())
    assert updated["pipeline"][0]["steps"][0]["type"] == "allocate"
    assert "cutlass_config" not in updated["pipeline"][0]["steps"][0]


def test_unknown_kernel_skipped(tmp_path):
    """Unknown extern kernel types are skipped."""
    from hyperonnx.compile.cutlass import annotate_cutlass_config

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
                        "kernel": "extern_kernels.some_unknown_op",
                        "args": [],
                        "output": "buf0",
                    },
                ],
            }
        ],
        "buffers": [],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

    result = annotate_cutlass_config(bundle_dir, manifest=manifest)
    updated = json.loads(result.read_text())
    step = updated["pipeline"][0]["steps"][0]
    assert step["type"] == "extern_kernel"
    assert "cutlass_config" not in step


def test_op_filter(tmp_path):
    """op_filter limits which ops are tuned."""
    from hyperonnx.compile.cutlass import annotate_cutlass_config

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

    result = annotate_cutlass_config(
        bundle_dir, manifest=manifest, op_filter={"convolution"}
    )
    updated = json.loads(result.read_text())
    step = updated["pipeline"][0]["steps"][0]
    assert "cutlass_config" not in step


def test_idempotent(tmp_path):
    """Running twice produces the same result."""
    from hyperonnx.compile.cutlass import annotate_cutlass_config

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

    annotate_cutlass_config(bundle_dir, manifest=manifest)
    first = json.loads((bundle_dir / "manifest.json").read_text())

    annotate_cutlass_config(bundle_dir)
    second = json.loads((bundle_dir / "manifest.json").read_text())

    assert first == second
