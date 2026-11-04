"""Single-token decode graph for the Qwen3.8 text model.

Mirrors `transformers.models.qwen3_5.modeling_qwen3_5` with these changes for NeuronCore-v2:

* Static shapes. One graph runs one token for each row of a fixed batch, and each row is an independent
  sequence at its own position. A prompt is fed token by token, so there is no prefill graph and no
  chunked gated-delta solve (`torch.linalg.solve_triangular` and its unrolled form do not suit this
  target). Token-by-token is the recurrent form of the same rule.
* Weights stay MXFP4 and are decoded in the graph by `FP4Linear`. Decoding is about 90% of the time per
  step and is shared by every row, so throughput grows with batch size until chip memory runs out.
* Caches are registered buffers that are reassigned on each step. The Neuron tracer turns that into
  on-device state. It does not detect `Tensor.copy_`, so do not use that here.
* Tensor parallelism. Each rank owns a contiguous slice of heads, and each head's computation is
  independent. Only the output projections need a collective (`all_reduce`).
* Embeddings, the final norm and lm_head live on the host (see `generate.py`), so the graph maps
  hidden to hidden. That also lets one compiled block of layers be chained with itself.
"""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from . import fp4
from .config import FULL, LINEAR, TextConfig

# Added to attention scores of unfilled cache slots. Scores are bounded by about sqrt(head_dim), so
# this underflows to exactly zero in the float32 softmax without needing an infinity on the device.
MASKED_SCORE = -1e4


def identity_all_reduce(x: torch.Tensor) -> torch.Tensor:
    return x


def silu(x: torch.Tensor) -> torch.Tensor:
    """Spelled out because torch-neuronx rewrites `F.silu` to a Neuron-only custom call. XLA on CPU
    cannot run that, and without it the whole graph can be checked with no Neuron hardware."""
    return x * torch.sigmoid(x)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    h = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + eps)
    return (h * (1.0 + weight.float())).type_as(x)


def l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def repeat_interleave(x: torch.Tensor, repeats: int, dim: int) -> torch.Tensor:
    """`Tensor.repeat_interleave` expressed as expand and reshape, which avoids a gather."""
    expanded = x.unsqueeze(dim + 1).expand(*x.shape[: dim + 1], repeats, *x.shape[dim + 1 :])
    return expanded.reshape(*x.shape[:dim], x.shape[dim] * repeats, *x.shape[dim + 1 :])


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rot = cos.shape[-1]
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    k_rot, k_pass = k[..., :rot], k[..., rot:]
    q_out = q_rot * cos + rotate_half(q_rot) * sin
    k_out = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat((q_out, q_pass), dim=-1), torch.cat((k_out, k_pass), dim=-1)


def _frozen(*shape: int, dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)


class FP4Linear(nn.Module):
    """Bias-free linear layer holding MXFP4 weights, expanded to BF16 on every forward."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_packed = _frozen(out_features, in_features // 2, dtype=torch.uint8)
        self.weight_scale = _frozen(out_features, in_features // fp4.GROUP_SIZE, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        even, odd = fp4.decode_planes(self.weight_packed, self.weight_scale)
        return F.linear(x[..., ::2], even.to(x.dtype)) + F.linear(x[..., 1::2], odd.to(x.dtype))


class FP8Linear(nn.Module):
    """Bias-free linear layer holding weights as FP8 bytes, so a chip runs it with no decoding at all.

    `weight_fp8` is ``[in_features, out_features]`` because the Tensor Engine contracts along the first dimension.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_fp8 = _frozen(in_features, out_features, dtype=torch.uint8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cpu":
            return F.linear(x, fp4.fp8_to_float(self.weight_fp8).T.to(x.dtype))
        from .kernels import fp8_linear

        return fp8_linear(x.reshape(x.shape[0], -1).transpose(0, 1), self.weight_fp8).unsqueeze(1)


def make_linear(cfg: TextConfig, in_features: int, out_features: int) -> nn.Module:
    return (FP8Linear if cfg.weight_format == "fp8" else FP4Linear)(in_features, out_features)


class MLP(nn.Module):
    def __init__(self, cfg: TextConfig, tp_degree: int, all_reduce: Callable):
        super().__init__()
        intermediate = cfg.intermediate_size // tp_degree
        self.gate_proj = make_linear(cfg, cfg.hidden_size, intermediate)
        self.up_proj = make_linear(cfg, cfg.hidden_size, intermediate)
        self.down_proj = make_linear(cfg, intermediate, cfg.hidden_size)
        self.all_reduce = all_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.all_reduce(self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x)))


class GatedDeltaNetStep(nn.Module):
    """One token of the gated delta rule (`torch_recurrent_gated_delta_rule` with a causal conv)."""

    def __init__(self, cfg: TextConfig, tp_degree: int, dtype: torch.dtype, all_reduce: Callable, batch_size: int = 1):
        super().__init__()
        self.cfg = cfg
        self.all_reduce = all_reduce
        self.key_dim = cfg.linear_key_dim // tp_degree
        self.value_dim = cfg.linear_value_dim // tp_degree
        self.num_v_heads = cfg.linear_num_value_heads // tp_degree
        self.num_k_heads = cfg.linear_num_key_heads // tp_degree
        conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkv = make_linear(cfg, cfg.hidden_size, conv_dim)
        self.in_proj_z = make_linear(cfg, cfg.hidden_size, self.value_dim)
        self.out_proj = make_linear(cfg, self.value_dim, cfg.hidden_size)
        # These stay float in the checkpoint: the quantization recipe excludes them.
        self.in_proj_b = _frozen(self.num_v_heads, cfg.hidden_size, dtype=dtype)
        self.in_proj_a = _frozen(self.num_v_heads, cfg.hidden_size, dtype=dtype)
        self.conv_weight = _frozen(conv_dim, cfg.linear_conv_kernel_dim, dtype=dtype)
        self.A_log = _frozen(self.num_v_heads, dtype=dtype)
        self.dt_bias = _frozen(self.num_v_heads, dtype=dtype)
        self.norm_weight = _frozen(cfg.linear_value_head_dim, dtype=dtype)

        self.register_buffer("conv_state", torch.zeros(batch_size, conv_dim, cfg.linear_conv_kernel_dim - 1, dtype=dtype))
        self.state_dtype = {"float32": torch.float32, "bf16": torch.bfloat16}[cfg.recurrent_state_format]
        self.register_buffer(
            "recurrent_state",
            torch.zeros(batch_size, self.num_v_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim, dtype=self.state_dtype),
        )

    def forward(self, x: torch.Tensor, keep_state: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        batch = x.shape[0]
        k_head, v_head = self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim

        # A depthwise causal conv over one new sample is a weighted sum of a sliding window, so no
        # conv op is needed. Newest sample is last, matching F.conv1d's kernel order.
        qkv = self.in_proj_qkv(x).transpose(1, 2)
        window = torch.cat((self.conv_state * keep_state.reshape(batch, 1, 1).to(dtype), qkv), dim=-1)
        self.conv_state = window[:, :, 1:]
        mixed = silu((window * self.conv_weight).sum(dim=-1))

        query = mixed[:, : self.key_dim].reshape(batch, self.num_k_heads, k_head)
        key = mixed[:, self.key_dim : 2 * self.key_dim].reshape(batch, self.num_k_heads, k_head)
        value = mixed[:, 2 * self.key_dim :].reshape(batch, self.num_v_heads, v_head)
        query = repeat_interleave(query, self.cfg.linear_key_repeat, dim=1)
        key = repeat_interleave(key, self.cfg.linear_key_repeat, dim=1)

        z = self.in_proj_z(x).reshape(batch, self.num_v_heads, v_head)
        beta = F.linear(x, self.in_proj_b).sigmoid().reshape(batch, self.num_v_heads, 1).float()
        a = F.linear(x, self.in_proj_a).reshape(batch, self.num_v_heads).float()
        log_decay = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())

        query = l2_norm(query.float()) / math.sqrt(k_head)
        key = l2_norm(key.float())
        value = value.float()

        state = self.recurrent_state.float() * keep_state.reshape(batch, 1, 1, 1) * log_decay.exp()[..., None, None]
        delta = (value - (state * key.unsqueeze(-1)).sum(dim=-2)) * beta
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        self.recurrent_state = state.to(self.state_dtype)
        out = (state * query.unsqueeze(-1)).sum(dim=-2).to(dtype)

        # Gated RMS norm, per value head.
        h = out.float()
        h = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        h = self.norm_weight * h.to(dtype)
        h = (h * silu(z.float())).to(dtype)
        return self.all_reduce(self.out_proj(h.reshape(batch, 1, self.value_dim)))


class FullAttentionStep(nn.Module):
    def __init__(
        self, cfg: TextConfig, tp_degree: int, max_seq_len: int, dtype: torch.dtype, all_reduce: Callable, batch_size: int = 1
    ):
        super().__init__()
        self.cfg = cfg
        self.all_reduce = all_reduce
        self.num_heads = cfg.num_attention_heads // tp_degree
        self.num_kv_heads = cfg.num_key_value_heads // tp_degree
        head = cfg.head_dim

        self.q_proj = make_linear(cfg, cfg.hidden_size, self.num_heads * head * 2)  # query plus output gate
        self.k_proj = make_linear(cfg, cfg.hidden_size, self.num_kv_heads * head)
        self.v_proj = make_linear(cfg, cfg.hidden_size, self.num_kv_heads * head)
        self.o_proj = make_linear(cfg, self.num_heads * head, cfg.hidden_size)
        self.q_norm_weight = _frozen(head, dtype=dtype)
        self.k_norm_weight = _frozen(head, dtype=dtype)

        self.register_buffer("k_cache", torch.zeros(batch_size, self.num_kv_heads, max_seq_len, head, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(batch_size, self.num_kv_heads, max_seq_len, head, dtype=dtype))

    def forward(self, x, cos, sin, write_mask, attn_mask) -> torch.Tensor:
        cfg = self.cfg
        head = cfg.head_dim
        batch = x.shape[0]

        q_and_gate = self.q_proj(x).reshape(batch, 1, self.num_heads, head * 2)
        query, gate = q_and_gate[..., :head], q_and_gate[..., head:]
        query = rms_norm(query, self.q_norm_weight, cfg.rms_norm_eps).transpose(1, 2)
        key = self.k_proj(x).reshape(batch, 1, self.num_kv_heads, head)
        key = rms_norm(key, self.k_norm_weight, cfg.rms_norm_eps).transpose(1, 2)
        value = self.v_proj(x).reshape(batch, 1, self.num_kv_heads, head).transpose(1, 2)
        query, key = apply_rotary(query, key, cos, sin)

        write = write_mask > 0.5
        self.k_cache = torch.where(write, key, self.k_cache)
        self.v_cache = torch.where(write, value, self.v_cache)

        keys = repeat_interleave(self.k_cache, cfg.num_key_value_groups, dim=1)
        values = repeat_interleave(self.v_cache, cfg.num_key_value_groups, dim=1)

        scores = torch.matmul(query, keys.transpose(2, 3)) * head**-0.5
        weights = torch.softmax(scores.float() + attn_mask, dim=-1).to(query.dtype)
        out = torch.matmul(weights, values).transpose(1, 2).reshape(batch, 1, self.num_heads * head)
        return self.all_reduce(self.o_proj(out * torch.sigmoid(gate.reshape(batch, 1, self.num_heads * head))))


class HeadStep(nn.Module):
    """Final norm and lm_head on the chip: the largest logit in each group of tokens, for every row.

    Each core holds a slice of the tokens. `rank_gate` is 1 for this core's own slot, so the sum across
    cores lays the two slices side by side in token order. `head_format` "fp8" halves the memory. It only
    picks candidates, because the host rechecks the best groups with the exact weights.
    """

    def __init__(
        self,
        cfg: TextConfig,
        tp_degree: int = 1,
        batch_size: int = 1,
        all_reduce: Callable = identity_all_reduce,
        group_rows: int = 32,
        head_format: str = "bf16",
    ):
        super().__init__()
        self.cfg = cfg
        self.tp_degree = tp_degree
        self.group_rows = group_rows
        self.all_reduce = all_reduce
        self.norm_weight = _frozen(cfg.hidden_size, dtype=torch.bfloat16)
        lm_head_dtype = torch.uint8 if head_format == "fp8" else torch.bfloat16
        self.lm_head = _frozen(cfg.hidden_size, cfg.vocab_size // tp_degree, dtype=lm_head_dtype)
        self.rank_gate = _frozen(tp_degree, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = rms_norm(hidden, self.norm_weight, self.cfg.rms_norm_eps).reshape(hidden.shape[0], -1)
        if x.device.type == "cpu":
            weight = fp4.fp8_to_float(self.lm_head) if self.lm_head.dtype == torch.uint8 else self.lm_head.float()
            logits = x.float() @ weight
            maxima = torch.stack([group.amax(dim=-1) for group in logits.split(self.group_rows, dim=-1)], dim=-1)
        else:
            from .kernels import lm_head_group_maxima

            maxima = lm_head_group_maxima(x.transpose(0, 1), self.lm_head, self.group_rows)
        return self.all_reduce(torch.cat([maxima * self.rank_gate[rank] for rank in range(self.tp_degree)], dim=-1))


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_type, tp_degree, max_seq_len, dtype, all_reduce, batch_size=1):
        super().__init__()
        self.cfg = cfg
        self.layer_type = layer_type
        if layer_type == LINEAR:
            self.linear_attn = GatedDeltaNetStep(cfg, tp_degree, dtype, all_reduce, batch_size)
        elif layer_type == FULL:
            self.self_attn = FullAttentionStep(cfg, tp_degree, max_seq_len, dtype, all_reduce, batch_size)
        else:
            raise ValueError(f"Unknown layer type {layer_type}")
        self.mlp = MLP(cfg, tp_degree, all_reduce)
        self.input_layernorm_weight = _frozen(cfg.hidden_size, dtype=dtype)
        self.post_attention_layernorm_weight = _frozen(cfg.hidden_size, dtype=dtype)

    def forward(self, x, cos, sin, write_mask, attn_mask, keep_state):
        eps = self.cfg.rms_norm_eps
        h = rms_norm(x, self.input_layernorm_weight, eps)
        if self.layer_type == LINEAR:
            h = self.linear_attn(h, keep_state)
        else:
            h = self.self_attn(h, cos, sin, write_mask, attn_mask)
        x = x + h
        return x + self.mlp(rms_norm(x, self.post_attention_layernorm_weight, eps))


class Qwen38Step(nn.Module):
    """The compiled graph: a run of decoder layers, for one token per row.

    `cfg.layer_types` may be the whole model or one repeating block of it (see `generate.BlockChain`).

    Inputs (all built by `generate.step_inputs`):
        inputs_embeds: ``[batch, 1, hidden]``, from the embedding table or the previous block.
        cos, sin: rotary terms for each row's position, ``[batch, 1, rotary_dim]``.
        write_mask: float32 ``[batch, 1, max_seq_len, 1]``, 1 at the row's position.
        attn_mask: float32 ``[batch, 1, 1, max_seq_len]``, 0 where filled and `MASKED_SCORE` elsewhere.
        keep_state: float32 ``[batch]``, 0 on a sequence's first token to clear recurrent state.
    """

    def __init__(
        self,
        cfg: TextConfig,
        max_seq_len: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        all_reduce: Callable = identity_all_reduce,
        batch_size: int = 1,
    ):
        super().__init__()
        cfg.validate_tp(tp_degree)
        self.cfg = cfg
        self.layers = nn.ModuleList(
            DecoderLayer(cfg, layer_type, tp_degree, max_seq_len, dtype, all_reduce, batch_size)
            for layer_type in cfg.layer_types
        )

    def forward(self, inputs_embeds, cos, sin, write_mask, attn_mask, keep_state):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, cos, sin, write_mask, attn_mask, keep_state)
        return x
