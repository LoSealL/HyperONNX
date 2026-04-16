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

import importlib
from contextlib import contextmanager
from unittest.mock import patch

import torch
from packaging.version import Version


@contextmanager
def patch_transformers():
    r"""Patch `transformers` package temporarily to export onnx successfully.

    :func:`~transformers.masking_utils.sdpa_mask_recent_torch` uses `torch.vmap`
    to index 4D mask, which is not supported well in torch onnx exporter.
    """
    # pylint: disable=invalid-name, unused-argument
    transformers = importlib.import_module("transformers")

    TRANSFORMERS_HAS_SDPA_ATTENTION = Version("4.48.0")
    TRANSFORMERS_HAS_SDPA_MASK = Version("4.53.0")
    TRANSFORMERS_FIX_VMAP = Version("5.0.0")
    TRANSFORMERS_DEPRECATE_CACHE_POSITION = Version("5.4.0")
    CURR_VER = Version(transformers.__version__)

    patches = []
    if TRANSFORMERS_HAS_SDPA_MASK <= CURR_VER < TRANSFORMERS_FIX_VMAP:
        masking_utils = importlib.import_module(".masking_utils", "transformers")
        ALL_MASK_ATTENTION_FUNCTIONS = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
        sdpa_mask_older_torch = masking_utils.sdpa_mask_older_torch

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
    if CURR_VER >= TRANSFORMERS_DEPRECATE_CACHE_POSITION:
        original_sdpa_mask = transformers.masking_utils.sdpa_mask

        def patch_sdpa_mask(
            batch_size: int,
            q_length: int,
            kv_length: int,
            *args,
            **kwargs,
        ):
            # In onnx exporting (torch.onnx.is_in_onnx_export()==True), the shape of a
            # Tensor is still a tensor:
            # type(tensor([1]).shape[0]) == torch.Tensor
            # So the wrap in sdpa_mask is wrong in this case. We forcely convert the
            # shape to integer.
            def _to_int(x):
                if isinstance(x, torch.Tensor) and x.ndim == 0:
                    return int(x.item())
                return x

            q_length = _to_int(q_length)
            kv_length = _to_int(kv_length)

            return original_sdpa_mask(batch_size, q_length, kv_length, *args, **kwargs)

        p3 = patch("transformers.masking_utils.sdpa_mask", patch_sdpa_mask)
        patches.append(p3)

    try:
        for p in patches:
            p.start()
        yield None
    finally:
        for p in patches:
            p.stop()
