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

import onnxscript
import torch
from onnx import ModelProto
from torch.library import custom_op
from torch.onnx import symbolic_helper

from ..utils import HYPER_DOMAIN

HYPERONNX_ATTN_IMPL = "onnx_attention_opset24"


def _broadcast_kv_heads(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast key/value heads to query heads for grouped-query attention."""
    q_heads = query.shape[-3]
    kv_heads = key.shape[-3]
    if q_heads == kv_heads:
        return key, value
    if q_heads % kv_heads != 0:
        raise ValueError(
            f"query heads ({q_heads}) must be divisible by kv heads ({kv_heads})."
        )
    repeats = q_heads // kv_heads
    return (
        key.repeat_interleave(repeats, dim=-3),
        value.repeat_interleave(repeats, dim=-3),
    )


def _causal_mask(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    # Follow PyTorch SDPA causal semantics for non-square attention matrices.
    return torch.ones((q_len, kv_len), dtype=torch.bool, device=query.device).tril()


@custom_op("hyperonnx::attention_opset24", mutates_args=())
def _attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float = 1.0,
    softcap: float = 0.0,
) -> torch.Tensor:
    key, value = _broadcast_kv_heads(query, key, value)
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    if is_causal or attn_mask is None:
        attn_mask = _causal_mask(query, key)
    scores = scores.masked_fill(~attn_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value)


@_attention_impl.register_fake
def _attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float = 1.0,
    softcap: float = 0.0,
) -> torch.Tensor:
    del key, attn_mask, is_causal, scale, softcap
    out_shape = tuple(query.shape[:-1]) + (value.shape[-1],)
    return torch.empty(out_shape, device=query.device, dtype=query.dtype)


def attention_interface(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Attention interface aligned with object
    :class:`transformers.modeling_utils.AttentionInterface`.
    """
    del module, dropout, kwargs
    attn_output = _attention_impl(
        query, key, value, attn_mask=attention_mask, scale=scaling
    )
    attn_weights = None
    return attn_output, attn_weights


@symbolic_helper.parse_args("v", "v", "v", "v", "b", "f", "f")
def _symbolic_attention_opset18(
    g,
    query,
    key,
    value,
    attn_mask,
    is_causal,
    scale,
    softcap,
):
    if attn_mask is None:
        return g.op(
            "Attention",
            query,
            key,
            value,
            is_causal_i=int(is_causal),
            q_num_heads_i=1,
            kv_num_heads_i=1,
            scale_f=scale,
            softcap_f=softcap,
        )
    return g.op(
        "Attention",
        query,
        key,
        value,
        attn_mask,
        is_causal_i=int(is_causal),
        q_num_heads_i=1,
        kv_num_heads_i=1,
        scale_f=scale,
        softcap_f=softcap,
    )


def register_attention_opsets():
    """Register ONNX symbolic for torchscript export path (dynamo=False)."""

    register = getattr(torch.onnx, "register_custom_op_symbolic", None)
    if register is None:
        from torch.onnx import utils as onnx_utils

        register = onnx_utils.register_custom_op_symbolic
    try:
        register("hyperonnx::attention_opset24", _symbolic_attention_opset18, 18)
    except RuntimeError:
        # Already registered in current process.
        pass
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(HYPERONNX_ATTN_IMPL, attention_interface)


_HYPER_OPSET = onnxscript.values.Opset(HYPER_DOMAIN, 1)


@onnxscript.script(_HYPER_OPSET)
def onnx_attention_opset24(
    query: onnxscript.FLOAT,
    key: onnxscript.FLOAT,
    value: onnxscript.FLOAT,
    attn_mask: onnxscript.BOOL,
    is_causal: int = 0,
    scale: float = 1.0,
    softcap: float = 0.0,
) -> onnxscript.FLOAT:
    return _HYPER_OPSET.Attention(  # type: ignore[return-value]
        query,
        key,
        value,
        attn_mask,
        is_causal=is_causal,
        q_num_heads=1,
        kv_num_heads=1,
        scale=scale,
        softcap=softcap,
    )


def attention_translation_table() -> dict[str, onnxscript.OnnxFunction]:
    """Return custom translation table for dynamo ONNX export."""
    return {"hyperonnx::attention_opset24": onnx_attention_opset24}


def promote_onnx_model_to_opset24(model: ModelProto) -> ModelProto:
    """Promote default ONNX domain opset to 24 after export.

    Current torch/onnx converters cannot always auto-upgrade models containing
    Attention from opset 18 to 24 due missing version-converter adapters.
    """
    has_default_domain = False
    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = 24
            has_default_domain = True
            break
    if not has_default_domain:
        new_opset = model.opset_import.add()
        new_opset.domain = ""
        new_opset.version = 24

    # For 3D query tensors, ONNX Attention requires q_num_heads.
    for node in model.graph.node:
        if node.domain == HYPER_DOMAIN and node.op_type == "Attention":
            node.domain = ""
    return model
