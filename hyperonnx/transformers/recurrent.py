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

from tempfile import TemporaryDirectory

import onnxscript
import torch
import torch.nn.functional as F
from onnxifier.domain.trt.ops.recurrent_plugin import gated_delta_rule_schema
from torch.library import custom_op
from torch.onnx import symbolic_helper

HYPERONNX_GATED_DELTA_RULE = (
    f"{gated_delta_rule_schema.domain}::{gated_delta_rule_schema.name}"
)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(
        value
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


@custom_op(HYPERONNX_GATED_DELTA_RULE, mutates_args=())
def _gated_delta_rule_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    context_lengths: torch.Tensor,
    k_dim: int,
    v_dim: int,
    num_v_heads: int,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive GDN implementation for ONNX tracing."""
    del context_lengths, k_dim, v_dim, num_v_heads

    seq_len = q.shape[1]
    if seq_len > 1:
        out, last_recurrent_state = _chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=recurrent_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
    else:
        out, last_recurrent_state = _recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            recurrent_state,
            use_qk_l2norm,
        )

    return out, last_recurrent_state


@_gated_delta_rule_impl.register_fake
def _gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    context_lengths: torch.Tensor,
    k_dim: int,
    v_dim: int,
    num_v_heads: int,
    use_qk_l2norm: bool = True,
):
    del k, v, g, beta, context_lengths, k_dim, v_dim, num_v_heads, use_qk_l2norm
    return torch.empty_like(q), recurrent_state.clone()


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "v", "i", "i", "i", "b")
def _symbolic_gated_delta_rule(
    g,
    q,
    k,
    v,
    gate,
    beta,
    recurrent_state,
    context_lengths,
    k_dim,
    v_dim,
    num_v_heads,
    use_qk_l2norm,
):
    o, h0_out = g.op(
        HYPERONNX_GATED_DELTA_RULE,
        q,
        k,
        v,
        gate,
        beta,
        recurrent_state,
        context_lengths,
        k_dim_i=k_dim,
        v_dim_i=v_dim,
        num_v_heads_i=num_v_heads,
        use_qk_l2norm_i=int(use_qk_l2norm),
        outputs=2,
    )
    o.setType(v.type())
    h0_out.setType(recurrent_state.type())
    return o, h0_out


def register_recurrent_opsets():
    """Register ONNX symbolic for torchscript export path (dynamo=False)."""

    register = getattr(torch.onnx, "register_custom_op_symbolic", None)
    if register is None:
        from torch.onnx import utils as onnx_utils

        register = onnx_utils.register_custom_op_symbolic
    try:
        register(HYPERONNX_GATED_DELTA_RULE, _symbolic_gated_delta_rule, 1)
    except RuntimeError:
        # Already registered in current process.
        pass


HYPER_OPSET = onnxscript.values.Opset(gated_delta_rule_schema.domain, 1)
ONNX_GATED_DELTA_RULE_SCRIPT = f"""
@onnxscript.script(HYPER_OPSET)
def onnx_gated_delta_rule(
    q: onnxscript.FLOAT,
    k: onnxscript.FLOAT,
    v: onnxscript.FLOAT,
    g: onnxscript.FLOAT,
    beta: onnxscript.FLOAT,
    recurrent_state: onnxscript.FLOAT,
    context_lengths: onnxscript.INT64,
    k_dim: int,
    v_dim: int,
    num_v_heads: int,
    use_qk_l2norm: bool = True,
) -> tuple[onnxscript.FLOAT, onnxscript.FLOAT]:
    # ONNX implementation of causal conv1d for ONNX export.
    output, conv_state_out = HYPER_OPSET.{gated_delta_rule_schema.name}(
        q,
        k,
        v,
        g,
        beta,
        recurrent_state,
        context_lengths,
        k_dim=k_dim,
        v_dim=v_dim,
        num_v_heads=num_v_heads,
        use_qk_l2norm=use_qk_l2norm,
    )
    return output, conv_state_out
"""


def recurrent_translation_table() -> dict[str, onnxscript.OnnxFunction]:
    """Return custom translation table for dynamo ONNX export."""
    with TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/_onnxscript_recur.py", "w", encoding="utf-8") as source:
            source.write(ONNX_GATED_DELTA_RULE_SCRIPT)
            code = compile(ONNX_GATED_DELTA_RULE_SCRIPT, source.name, "exec")
        g = globals()
        meta: dict = {}
        # onnxscript requires a physical file readable to getsource
        eval(code, g, meta)  # pylint: disable=eval-used
    onnx_gated_delta_rule = meta["onnx_gated_delta_rule"]
    return {HYPERONNX_GATED_DELTA_RULE: onnx_gated_delta_rule}


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked Gated Delta Rule function"""
    del chunk_size, output_final_state
    if initial_state is None:
        initial_state = torch.zeros(
            [query.shape[0], value.shape[2], key.shape[-1], value.shape[-1]],
            dtype=value.dtype,
            device=value.device,
        )

    return _gated_delta_rule_impl(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        torch.zeros([1]).long(),
        key.shape[-1],
        value.shape[-1],
        value.shape[-2],
        use_qk_l2norm_in_kernel,
    )
