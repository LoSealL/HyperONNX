# HyperONNX Instructions

## Workflow

- Check relevant skills in `.claude/skills/` before handling specialized tasks.
- Prefer minimal, targeted changes that preserve existing project structure and style.
- Validate the files you change with the smallest relevant command before finishing.

## Python Changes

- Follow the existing Python style already used in `hyperonnx/`.
- When Python files are modified, use the lint skill workflow and prefer `uv run ruff check hyperonnx`.
- Format Python changes with `uv run ruff format hyperonnx` when formatting is needed.

## Version Compatibility (Hard Constraints)

⚠️ **Supported Versions**: This project is currently tested and working on:
- **torch < 2.11** (torch 2.5–2.10 verified)
- **onnxscript < 0.6.0**

Both torch >= 2.11 and onnxscript >= 0.6.0 introduce breaking changes to module structure, dtype handling, and ONNX IR that are not yet resolved. **Do not use these newer versions** without explicit future work to support them.

**When users ask about newer versions**: Acknowledge the constraint, suggest using compatible versions from `pyproject.toml` optional dependencies, and note that torch >= 2.11 support is a known gap requiring future implementation.

## Commits

- If asked to create a commit, use a clear and concise commit message that describes the actual change.
