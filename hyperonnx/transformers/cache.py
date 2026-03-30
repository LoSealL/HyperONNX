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

from typing import Any

import torch
from torch import Tensor


class StaticCache(Tensor):
    """Mock all methods of :class:~`transformers.cache_utils.StaticCache`.

    Wrap into `torch.Tensor` to preserve the methods.

    Tensor shape: [layers, 2, batch, num_heads, seq_len, head_dim]
    2 stands for Key and Value states.

    Note:

        We should treat DynamicCache as static for onnx export.
    """

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer
        `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each
                subclass and allow new types of cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """

        layer_kv = self[layer_idx]
        keys, values = layer_kv.unbind()
        # Some old models give None for `cache_position` or even omit passing
        # `cache_kwargs` when used as cross-attention, in which case we should copy
        # the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        cache_position = (
            cache_position
            if cache_position is not None
            else torch.arange(key_states.shape[-2], device=self.device)
        )
        try:
            keys.index_copy_(2, cache_position, key_states)
            values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            keys[:, :, cache_position] = key_states
            values[:, :, cache_position] = value_states

        return keys, values

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        max_cache_len: int = 2048,
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized
        on the first `update` call). This is useful for our `export` recipes, as
        `export` needs everything in advance.
        """
        return self.new_zeros(
            (self.shape[0], 2, batch_size, num_heads, max_cache_len, head_dim),
            dtype=dtype,
            device=device,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= self.shape[0]:
            return 0
        return int(self[0, 0].any(dim=-1).sum().item())

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset
        that will be returned for the given layer at `layer_idx`. The masks are then
        prepared according to the given lengths (kv_length, kv_offset) and patterns
        for each layer.
        """
        if layer_idx >= self.shape[0]:
            return cache_position.shape[0], 0
        return self.shape[4], 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """
        Returns maximum sequence length of the cache object.
        Dynamic caches do not have a maximum length.
        """
        if layer_idx >= self.shape[0]:
            return -1
        return self.shape[4]

    def reset(self):
        """Recursively reset all layers tensors"""
        self.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search"""
        if self.get_seq_length() > 0:
            self.copy_(self.index_select(0, beam_idx.to(self.device)))

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        return self.shape[2]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        return self.shape[4]

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compilable"""
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        return True

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache data is initialized"""
        return True

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [False for _ in range(len(self))]

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        return self.shape[0]
