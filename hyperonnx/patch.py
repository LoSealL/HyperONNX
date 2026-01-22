"""
Copyright (C) 2025 The HYPERONNX Authors.

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

from contextlib import contextmanager
from unittest.mock import patch

from packaging.version import Version


@contextmanager
def patch_transformers():
    r"""Patch `transformers` package temporarily to export onnx successfully.

    :func:`~transformers.masking_utils.sdpa_mask_recent_torch` uses `torch.vmap`
    to index 4D mask, which is not supported well in torch onnx exporter.
    """
    # pylint: disable=import-outside-toplevel
    import transformers

    TRANSFORMERS_HAS_SDPA_ATTENTION = Version("4.48.0")
    TRANSFORMERS_HAS_SDPA_MASK = Version("4.53.0")
    TRANSFORMERS_FIX_VMAP = Version("5.0.0")
    CURR_VER = Version(transformers.__version__)

    patches = []
    if TRANSFORMERS_HAS_SDPA_MASK <= CURR_VER < TRANSFORMERS_FIX_VMAP:
        from transformers.masking_utils import (
            ALL_MASK_ATTENTION_FUNCTIONS,
            sdpa_mask_older_torch,
        )

        # need to patch sdpa_mask to older version
        # But after 5.0.0 use_vmap is False by default
        p1 = patch("transformers.masking_utils.sdpa_mask", sdpa_mask_older_torch)
        ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask_older_torch
        patches.append(p1)
    if CURR_VER >= TRANSFORMERS_HAS_SDPA_ATTENTION:

        def use_gqa_in_sdpa(*args, **kwargs):
            return False

        p2 = patch(
            "transformers.integrations.sdpa_attention.use_gqa_in_sdpa", use_gqa_in_sdpa
        )
        patches.append(p2)

    try:
        for p in patches:
            p.start()
        yield None
    finally:
        for p in patches:
            p.stop()
