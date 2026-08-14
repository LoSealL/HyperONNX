"""CLI entry point for CUTLASS extern kernel config tuning.

Usage:
    python -m hyperonnx.compile.cutlass_bundle <bundle_dir> [--arch sm_90]
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate extern_kernel steps with CUTLASS tuning config"
    )
    parser.add_argument(
        "bundle_dir", type=Path, help="Path to .kernels/ bundle directory"
    )
    parser.add_argument("--arch", type=str, default=None, help="GPU arch (e.g. sm_90)")
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated op filter (e.g. mm,convolution)",
    )
    args = parser.parse_args()

    if not args.bundle_dir.is_dir():
        print(f"Error: {args.bundle_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    manifest_path = args.bundle_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    op_filter = set(args.ops.split(",")) if args.ops else None

    from .cutlass import annotate_cutlass_config

    result = annotate_cutlass_config(
        args.bundle_dir,
        arch=args.arch,
        op_filter=op_filter,
    )
    print(f"Updated manifest: {result}")


if __name__ == "__main__":
    main()
