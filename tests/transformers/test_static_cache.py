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

from unittest.mock import patch

import onnx
import pytest
import torch
from torch import nn

from hyperonnx.transformers import StaticCache

# ---------------------------------------------------------------------------
# Constants used throughout
# Tensor shape: [layers, 2, batch, num_heads, seq_len, head_dim]
# ---------------------------------------------------------------------------
LAYERS = 2
NUM_HEADS = 2
BATCH = 1
HEAD_DIM = 4
MAX_CACHE_LEN = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cache(
    *,
    layers: int = LAYERS,
    num_heads: int = NUM_HEADS,
    batch: int = BATCH,
    head_dim: int = HEAD_DIM,
    max_cache_len: int = MAX_CACHE_LEN,
) -> StaticCache:
    t = torch.zeros(layers, 2, batch, num_heads, max_cache_len, head_dim)
    return t.as_subclass(StaticCache)


@pytest.fixture
def cache() -> StaticCache:
    return _make_cache()


# ---------------------------------------------------------------------------
# Helpers to build key/value state tensors
# ---------------------------------------------------------------------------


def _kv(
    seq: int,
    val: float = 1.0,
    *,
    batch: int = BATCH,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
) -> torch.Tensor:
    return torch.full((batch, num_heads, seq, head_dim), val)


# ===========================================================================
# __len__
# ===========================================================================


class TestLen:
    def test_returns_number_of_layers(self, cache: StaticCache):
        assert len(cache) == LAYERS

    def test_single_layer_cache(self):
        c = _make_cache(layers=1)
        assert len(c) == 1

    def test_many_layers(self):
        c = _make_cache(layers=8)
        assert len(c) == 8


# ===========================================================================
# Properties
# ===========================================================================


class TestProperties:
    def test_max_batch_size(self, cache: StaticCache):
        assert cache.max_batch_size == BATCH

    def test_max_batch_size_multi(self):
        c = _make_cache(batch=4)
        assert c.max_batch_size == 4

    def test_max_cache_len(self, cache: StaticCache):
        assert cache.max_cache_len == MAX_CACHE_LEN

    def test_max_cache_len_custom(self):
        c = _make_cache(max_cache_len=1024)
        assert c.max_cache_len == 1024

    def test_is_compileable(self, cache: StaticCache):
        assert cache.is_compileable is True

    def test_is_initialized(self, cache: StaticCache):
        assert cache.is_initialized is True

    def test_is_sliding_all_false(self, cache: StaticCache):
        result = cache.is_sliding
        assert result == [False] * LAYERS

    def test_is_sliding_length_matches_layers(self):
        c = _make_cache(layers=5)
        assert len(c.is_sliding) == 5
        assert all(v is False for v in c.is_sliding)


# ===========================================================================
# get_seq_length
# ===========================================================================


class TestGetSeqLength:
    def test_empty_cache_returns_zero(self, cache: StaticCache):
        assert cache.get_seq_length() == 0

    def test_empty_with_explicit_default_layer(self, cache: StaticCache):
        assert cache.get_seq_length(layer_idx=0) == 0

    def test_out_of_range_layer_returns_zero(self, cache: StaticCache):
        assert cache.get_seq_length(layer_idx=LAYERS) == 0

    def test_far_out_of_range_layer_returns_zero(self, cache: StaticCache):
        assert cache.get_seq_length(layer_idx=LAYERS + 99) == 0

    def test_returns_positive_after_update(self, cache: StaticCache):
        cache.update(
            _kv(3),
            _kv(3),
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(3)},
        )
        assert cache.get_seq_length() > 0

    def test_returns_zero_for_all_zero_data(self, cache: StaticCache):
        # Writing zero key/value states should not change the seq-length count.
        cache.update(
            torch.zeros(BATCH, NUM_HEADS, 3, HEAD_DIM),
            torch.zeros(BATCH, NUM_HEADS, 3, HEAD_DIM),
            layer_idx=0,
            cache_kwargs={"cache_position": torch.arange(3)},
        )
        assert cache.get_seq_length() == 0


# ===========================================================================
# get_mask_sizes
# ===========================================================================


class TestGetMaskSizes:
    def test_valid_layer_returns_max_cache_len_and_zero(self, cache: StaticCache):
        cp = torch.arange(3)
        kv_len, kv_offset = cache.get_mask_sizes(cp, layer_idx=0)
        assert kv_len == MAX_CACHE_LEN
        assert kv_offset == 0

    def test_last_valid_layer(self, cache: StaticCache):
        cp = torch.arange(1)
        kv_len, kv_offset = cache.get_mask_sizes(cp, layer_idx=LAYERS - 1)
        assert kv_len == MAX_CACHE_LEN
        assert kv_offset == 0

    def test_out_of_range_layer_returns_position_length(self, cache: StaticCache):
        cp = torch.arange(5)
        kv_len, kv_offset = cache.get_mask_sizes(cp, layer_idx=LAYERS)
        assert kv_len == cp.shape[0]
        assert kv_offset == 0

    def test_out_of_range_layer_with_single_position(self, cache: StaticCache):
        cp = torch.arange(1)
        kv_len, kv_offset = cache.get_mask_sizes(cp, layer_idx=LAYERS + 5)
        assert kv_len == 1
        assert kv_offset == 0


# ===========================================================================
# get_max_cache_shape
# ===========================================================================


class TestGetMaxCacheShape:
    def test_default_layer(self, cache: StaticCache):
        assert cache.get_max_cache_shape() == MAX_CACHE_LEN

    def test_explicit_valid_layer(self, cache: StaticCache):
        assert cache.get_max_cache_shape(0) == MAX_CACHE_LEN
        assert cache.get_max_cache_shape(LAYERS - 1) == MAX_CACHE_LEN

    def test_out_of_range_returns_minus_one(self, cache: StaticCache):
        assert cache.get_max_cache_shape(LAYERS) == -1

    def test_far_out_of_range_returns_minus_one(self, cache: StaticCache):
        assert cache.get_max_cache_shape(LAYERS + 100) == -1


# ===========================================================================
# update
# ===========================================================================


class TestUpdate:
    def test_explicit_cache_position_writes_keys(self, cache: StaticCache):
        ks = _kv(3, val=1.0)
        vs = _kv(3, val=2.0)
        cp = torch.tensor([0, 1, 2])
        keys, values = cache.update(
            ks, vs, layer_idx=0, cache_kwargs={"cache_position": cp}
        )

        assert torch.allclose(keys[:, :, :3, :], ks)
        assert torch.allclose(values[:, :, :3, :], vs)

    def test_explicit_cache_position_leaves_rest_zero(self, cache: StaticCache):
        ks = _kv(2, val=5.0)
        vs = _kv(2, val=7.0)
        cp = torch.tensor([0, 1])
        keys, values = cache.update(
            ks, vs, layer_idx=0, cache_kwargs={"cache_position": cp}
        )

        assert torch.all(keys[:, :, 2:, :] == 0)
        assert torch.all(values[:, :, 2:, :] == 0)

    def test_none_cache_kwargs_uses_arange(self, cache: StaticCache):
        ks = _kv(3, val=3.0)
        vs = _kv(3, val=4.0)
        keys, values = cache.update(ks, vs, layer_idx=0, cache_kwargs=None)

        assert torch.allclose(keys[:, :, :3, :], ks)
        assert torch.allclose(values[:, :, :3, :], vs)

    def test_empty_cache_kwargs_uses_arange(self, cache: StaticCache):
        ks = _kv(2, val=9.0)
        vs = _kv(2, val=8.0)
        keys, values = cache.update(ks, vs, layer_idx=0, cache_kwargs={})

        assert torch.allclose(keys[:, :, :2, :], ks)
        assert torch.allclose(values[:, :, :2, :], vs)

    def test_none_cache_position_in_kwargs_uses_arange(self, cache: StaticCache):
        ks = _kv(4, val=6.0)
        vs = _kv(4, val=6.0)
        keys, values = cache.update(
            ks, vs, layer_idx=0, cache_kwargs={"cache_position": None}
        )

        assert torch.allclose(keys[:, :, :4, :], ks)
        assert torch.allclose(values[:, :, :4, :], vs)

    def test_returns_full_cache_length(self, cache: StaticCache):
        ks = _kv(2)
        vs = _kv(2)
        keys, values = cache.update(ks, vs, layer_idx=0, cache_kwargs=None)

        assert keys.shape == (BATCH, NUM_HEADS, MAX_CACHE_LEN, HEAD_DIM)
        assert values.shape == (BATCH, NUM_HEADS, MAX_CACHE_LEN, HEAD_DIM)

    def test_non_contiguous_cache_position(self, cache: StaticCache):
        ks = _kv(3, val=1.5)
        vs = _kv(3, val=2.5)
        cp = torch.tensor([0, 3, 7])
        keys, values = cache.update(
            ks, vs, layer_idx=0, cache_kwargs={"cache_position": cp}
        )

        for i, pos in enumerate(cp):
            assert torch.allclose(keys[:, :, pos, :], ks[:, :, i, :])
            assert torch.allclose(values[:, :, pos, :], vs[:, :, i, :])

    def test_update_second_layer(self, cache: StaticCache):
        ks = _kv(2, val=11.0)
        vs = _kv(2, val=12.0)
        keys, values = cache.update(ks, vs, layer_idx=1, cache_kwargs=None)

        # Layer 0 should still be zero
        assert torch.all(cache[0] == 0)
        assert torch.allclose(keys[:, :, :2, :], ks)

    def test_index_copy_not_implemented_fallback(self, cache: StaticCache):
        ks = _kv(3, val=1.0)
        vs = _kv(3, val=2.0)
        cp = torch.tensor([0, 1, 2])

        with patch.object(torch.Tensor, "index_copy_", side_effect=NotImplementedError):
            keys, values = cache.update(
                ks, vs, layer_idx=0, cache_kwargs={"cache_position": cp}
            )

        assert torch.allclose(keys[:, :, :3, :], ks)
        assert torch.allclose(values[:, :, :3, :], vs)

    def test_index_copy_not_implemented_fallback_none_kwargs(self, cache: StaticCache):
        ks = _kv(2, val=5.0)
        vs = _kv(2, val=5.0)

        with patch.object(torch.Tensor, "index_copy_", side_effect=NotImplementedError):
            keys, values = cache.update(ks, vs, layer_idx=0, cache_kwargs=None)

        assert torch.allclose(keys[:, :, :2, :], ks)
        assert torch.allclose(values[:, :, :2, :], vs)


# ===========================================================================
# early_initialization
# ===========================================================================


class TestEarlyInitialization:
    def test_shape_uses_self_layers(self, cache: StaticCache):
        result = cache.early_initialization(
            batch_size=4,
            num_heads=8,
            head_dim=64,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_cache_len=512,
        )
        assert result.shape == (LAYERS, 2, 4, 8, 512, 64)

    def test_default_max_cache_len(self, cache: StaticCache):
        result = cache.early_initialization(
            batch_size=1,
            num_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert result.shape == (LAYERS, 2, 1, 2, 2048, 4)

    def test_result_is_all_zeros(self, cache: StaticCache):
        result = cache.early_initialization(
            batch_size=1,
            num_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_cache_len=16,
        )
        assert (result == 0).all()

    def test_dtype_is_respected(self, cache: StaticCache):
        result = cache.early_initialization(
            batch_size=1,
            num_heads=2,
            head_dim=4,
            dtype=torch.float16,
            device=torch.device("cpu"),
            max_cache_len=16,
        )
        assert result.dtype == torch.float16


# ===========================================================================
# reset
# ===========================================================================


class TestReset:
    def test_reset_zeros_a_filled_cache(self, cache: StaticCache):
        cache.update(_kv(3), _kv(3), layer_idx=0, cache_kwargs=None)
        assert cache.any()

        cache.reset()

        assert (cache == 0).all()

    def test_reset_on_already_zero_cache(self, cache: StaticCache):
        cache.reset()
        assert (cache == 0).all()

    def test_reset_all_layers(self, cache: StaticCache):
        cache[0].fill_(1.0)
        cache[1].fill_(2.0)
        cache.reset()
        assert torch.all(cache == 0)


# ===========================================================================
# reorder_cache
# ===========================================================================


class TestReorderCache:
    def test_no_op_when_cache_is_empty(self, cache: StaticCache):
        original = cache.clone()
        beam_idx = torch.LongTensor([0, 1])
        cache.reorder_cache(beam_idx)
        assert torch.equal(cache, original)

    def test_reorders_layers_when_data_present(self, cache: StaticCache):
        # Fill different values per layer so we can verify the swap.
        cache[0].fill_(1.0)
        cache[1].fill_(2.0)

        # beam_idx selects [layer-1, layer-0] along dim 0 → swaps layers
        beam_idx = torch.LongTensor([1, 0])
        cache.reorder_cache(beam_idx)

        assert torch.all(cache[0] == 2.0)
        assert torch.all(cache[1] == 1.0)

    def test_identity_reorder_leaves_cache_unchanged(self, cache: StaticCache):
        cache[0].fill_(3.0)
        cache[1].fill_(7.0)

        beam_idx = torch.LongTensor([0, 1])
        cache.reorder_cache(beam_idx)

        assert torch.all(cache[0] == 3.0)
        assert torch.all(cache[1] == 7.0)


# ===========================================================================
# torch.onnx.export
# ===========================================================================


class _StaticCacheInputModel(nn.Module):
    def forward(self, x: torch.Tensor, cache: StaticCache) -> torch.Tensor:
        return x + cache[0, 0, :, :, 0, :]


class TestTorchOnnxExport:
    @pytest.mark.parametrize("dynamo", [True, False])
    def test_export_with_static_cache_input_type(self, tmp_path, dynamo):
        model = _StaticCacheInputModel().eval()
        x = torch.randn(BATCH, NUM_HEADS, HEAD_DIM)
        cache = _make_cache()
        cache[0, 0, :, :, 0, :] = 1.0

        output_file = tmp_path / "static_cache_input.onnx"
        torch.onnx.export(
            model,
            (x, cache),
            output_file,
            input_names=["x", "cache"],
            output_names=["y"],
            dynamo=dynamo,
        )

        assert output_file.exists()
        onnx_model = onnx.load_model(output_file)
        onnx.checker.check_model(onnx_model, True)
