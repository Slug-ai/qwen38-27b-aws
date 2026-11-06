"""Decode and prefill graphs for the Qwen3.8 text model.

Mirrors `transformers.models.qwen3_5.modeling_qwen3_5` with these changes for NeuronCore-v2:

* Static shapes. The decode graph runs one token for each row of a fixed batch, and each row is an
  independent sequence at its own position. The prefill graph runs a chunk of one row's prompt at once,
  with the chunked form of the gated delta rule, and shares weights and caches with the decode graph.
  Its triangular solve is a product of doublings, so it is all matmuls (`torch.linalg.solve_triangular`
  and its unrolled form do not suit this target).
* Weights are FP8 bytes read straight by a kernel (`FP8Linear`), or MXFP4 decoded in the graph (`FP4Linear`).
* Caches are registered buffers that are reassigned on each step. The Neuron tracer turns that into
  on-device state. It does not detect `Tensor.copy_`, so do not use that here. Writes touch only the
  positions they change, through `index_copy`, so a step costs nothing for unused cache length.
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
# Tokens per piece of the chunked gated delta rule: one partition per token in the chip's kernel.
DELTA_CHUNK = 128
# Cache positions per piece of prefill attention, which bounds the size of its score matrix.
ATTENTION_BLOCK = 2048
# The largest FP8 e4m3 value, as the checkpoint's KV-cache scales assume.
FP8_E4M3_MAX = 448.0


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


def fp8_round(x: torch.Tensor) -> torch.Tensor:
    """Round to FP8 e5m2 and back, as the chip's matmul kernel does to its input."""
    return x.to(torch.float8_e5m2).to(x.dtype)


def chunked_gated_delta(query, key, value, log_decay, beta, state, chunk: int = DELTA_CHUNK):
    """The gated delta rule over a run of tokens, as `torch_chunk_gated_delta_rule` computes it.

    Args:
        query, key: float32 ``[heads, tokens, k_head]``, L2-normalized, with query also scaled.
        value: float32 ``[heads, tokens, v_head]``.
        log_decay, beta: float32 ``[heads, tokens]``. A token with both set to 0 leaves the state unchanged.
        state: float32 ``[heads, k_head, v_head]`` before the first token.

    Returns:
        (output ``[heads, tokens, v_head]``, state after the last token).
    """
    heads, tokens, k_head = key.shape
    pieces = tokens // chunk
    q = query.reshape(heads, pieces, chunk, k_head)
    k = key.reshape(heads, pieces, chunk, k_head)
    v = value.reshape(heads, pieces, chunk, -1)
    g = log_decay.reshape(heads, pieces, chunk).cumsum(-1)
    b = beta.reshape(heads, pieces, chunk, 1)

    device = key.device
    index = torch.arange(chunk, device=device)
    causal = (index.unsqueeze(-1) >= index.unsqueeze(0)).float()
    below = (index.unsqueeze(-1) > index.unsqueeze(0)).float()
    diff = g.unsqueeze(-1) - g.unsqueeze(-2)
    decay = (diff * causal).exp() * causal
    k_beta = k * b
    inverse = unit_lower_inverse(-(k_beta @ k.transpose(-1, -2)) * decay * below)
    new_values = inverse @ (v * b)
    cumulative_keys = inverse @ (k_beta * g.exp().unsqueeze(-1))

    outputs = []
    for i in range(pieces):
        attn = (q[:, i] @ k[:, i].transpose(-1, -2)) * decay[:, i]
        v_new = new_values[:, i] - cumulative_keys[:, i] @ state
        outputs.append((q[:, i] * g[:, i].exp().unsqueeze(-1)) @ state + attn @ v_new)
        last = g[:, i, -1]
        state = state * last.exp().reshape(heads, 1, 1) + (k[:, i] * (last.unsqueeze(-1) - g[:, i]).exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
    return torch.cat(outputs, dim=1), state


def unit_lower_inverse(lower: torch.Tensor, base: int = 8) -> torch.Tensor:
    """``(I - lower)^-1`` for strictly lower triangular ``lower`` ``[..., n, n]``, by blocked forward substitution.

    Blocks of `base` are inverted as ``(I + L)(I + L^2)(I + L^4)``, exact because ``L^base = 0``. That series
    cancels catastrophically when keys are strongly correlated and grows with block size, so larger blocks are built
    by merging neighbours with the block formula instead, as flash-linear-attention does: for
    ``[[A, 0], [C, B]]``, the inverse's lower block is ``B^-1 C A^-1``. All of it is matmuls.
    """
    n = lower.shape[-1]
    eye = torch.eye(n, device=lower.device)
    in_block = (torch.arange(n, device=lower.device) // base).unsqueeze(-1) == (torch.arange(n, device=lower.device) // base)
    local = lower * in_block
    inverse = eye + local
    power = local
    for _ in range(int(math.log2(base)) - 1):
        power = power @ power
        inverse = inverse + inverse @ power
    size = base
    while size < n:
        merged = inverse.clone()
        for first in range(0, n, 2 * size):
            top, bottom = slice(first, first + size), slice(first + size, first + 2 * size)
            merged[..., bottom, top] = inverse[..., bottom, bottom] @ lower[..., bottom, top] @ inverse[..., top, top]
        inverse, size = merged, 2 * size
    return inverse


def piece_masks(device, n: int = DELTA_CHUNK, base: int = 8) -> torch.Tensor:
    """The masks `kernels.chunked_delta_body` needs for `unit_lower_inverse`'s blocks, ``[10, n, n]`` float32."""
    i, j = torch.arange(n).unsqueeze(-1), torch.arange(n).unsqueeze(0)
    masks = [i // base == j // base]
    size = base
    while size < n:
        masks.append((i // (2 * size) == j // (2 * size)) & ((i // size) % 2 == 1) & ((j // size) % 2 == 0))
        size *= 2
    masks = torch.stack(masks).float()
    return torch.cat((masks, masks.transpose(1, 2))).to(device)


def blocked_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, visible: torch.Tensor, block: int) -> torch.Tensor:
    """Softmax attention over the cache in blocks of positions, keeping a running maximum and sum.

    Args:
        query: ``[kv_heads, queries, head]``, already scaled. Queries sharing a KV head are stacked.
        keys, values: ``[kv_heads, positions, head]``.
        visible: bool ``[queries, positions]``. Position 0 must be visible to every query.
    """
    device = query.device
    best = torch.full((*query.shape[:2], 1), 2 * MASKED_SCORE, device=device)
    total = torch.zeros(*query.shape[:2], 1, device=device)
    out = torch.zeros(*query.shape[:2], values.shape[-1], device=device)
    for first in range(0, keys.shape[1], block):
        scores = (query @ keys[:, first : first + block].transpose(-1, -2)).float()
        scores = scores + torch.where(visible[:, first : first + block], 0.0, MASKED_SCORE)
        new_best = torch.maximum(best, scores.amax(dim=-1, keepdim=True))
        weights = (scores - new_best).exp()
        rescale = (best - new_best).exp()
        total = total * rescale + weights.sum(dim=-1, keepdim=True)
        out = out * rescale + (weights.to(values.dtype) @ values[:, first : first + block]).float()
        best = new_best
    return out / total


def _scatter(cache: torch.Tensor, dim: int, index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """`Tensor.scatter`. PyTorch's CPU scatter has no FP8 kernel but XLA's does, so only the CPU path goes through bytes."""
    if cache.device.type == "cpu" and cache.dtype == torch.float8_e4m3fn:
        return cache.view(torch.uint8).scatter(dim, index, values.view(torch.uint8)).view(torch.float8_e4m3fn)
    return cache.scatter(dim, index, values)


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
            # The chip's kernel rounds x to FP8 e5m2 too, since its Tensor Engine multiplies FP8 only by FP8.
            x = fp8_round(x)
            return F.linear(x, fp4.fp8_to_float(self.weight_fp8).T.to(x.dtype))
        from .kernels import fp8_linear

        out = fp8_linear(x.reshape(-1, x.shape[-1]).transpose(0, 1), self.weight_fp8)
        return out.reshape(*x.shape[:-1], out.shape[-1])


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
    """The gated delta rule with a causal conv: one token per row (decode), or a chunk of one row (prefill)."""

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

    def _split(self, mixed: torch.Tensor):
        """Query and key per value head, and value, from the conv output ``[tokens, conv_dim]``, all float32."""
        tokens = mixed.shape[0]
        k_head, v_head = self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim
        query = mixed[:, : self.key_dim].reshape(tokens, self.num_k_heads, k_head)
        key = mixed[:, self.key_dim : 2 * self.key_dim].reshape(tokens, self.num_k_heads, k_head)
        value = mixed[:, 2 * self.key_dim :].reshape(tokens, self.num_v_heads, v_head)
        query = repeat_interleave(query, self.cfg.linear_key_repeat, dim=1)
        key = repeat_interleave(key, self.cfg.linear_key_repeat, dim=1)
        return l2_norm(query.float()) / math.sqrt(k_head), l2_norm(key.float()), value.float()

    def _gates(self, x: torch.Tensor):
        """beta and the log of the decay, float32 ``[tokens, v_heads]``, from ``x`` ``[tokens, hidden]``."""
        beta = F.linear(x, self.in_proj_b).sigmoid().float()
        a = F.linear(x, self.in_proj_a).float()
        return beta, -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())

    def _finish(self, out: torch.Tensor, z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Gated RMS norm per value head, then the output projection. ``out`` and ``z`` are ``[tokens, v_heads, v_head]``."""
        h = out.float()
        h = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        h = self.norm_weight * h.to(dtype)
        h = (h * silu(z.float())).to(dtype)
        return self.out_proj(h.reshape(out.shape[0], 1, self.value_dim))

    def forward(self, x: torch.Tensor, position: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """One token per row. ``x`` is ``[batch, 1, hidden]``. A row with `active` 0 keeps its state exactly."""
        dtype = x.dtype
        batch = x.shape[0]
        on = active.reshape(batch, 1, 1) > 0.5
        # A sequence's first token starts from empty state. Inactive rows keep theirs, whatever their position.
        keep = torch.where((position == 0) & (active > 0.5), 0.0, 1.0)

        # A depthwise causal conv over one new sample is a weighted sum of a sliding window, so no
        # conv op is needed. Newest sample is last, matching F.conv1d's kernel order.
        qkv = self.in_proj_qkv(x).transpose(1, 2)
        window = torch.cat((self.conv_state * keep.reshape(batch, 1, 1).to(dtype), qkv), dim=-1)
        self.conv_state = torch.where(on, window[:, :, 1:], self.conv_state)
        query, key, value = self._split(silu((window * self.conv_weight).sum(dim=-1)))

        z = self.in_proj_z(x).reshape(batch, self.num_v_heads, -1)
        beta, log_decay = self._gates(x.reshape(batch, -1))
        # No decay and no write leaves an inactive row's state as it was.
        beta = (beta * active.unsqueeze(-1)).unsqueeze(-1)
        log_decay = log_decay * active.unsqueeze(-1)

        state = self.recurrent_state.float() * keep.reshape(batch, 1, 1, 1) * log_decay.exp()[..., None, None]
        delta = (value - (state * key.unsqueeze(-1)).sum(dim=-2)) * beta
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        self.recurrent_state = state.to(self.state_dtype)
        out = (state * query.unsqueeze(-1)).sum(dim=-2).to(dtype)
        return self.all_reduce(self._finish(out, z, dtype))

    def prefill(self, x: torch.Tensor, slot: torch.Tensor, start: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        """A chunk of one row's tokens. ``x`` is ``[1, chunk, hidden]``. Tokens from `n_valid` on are padding.

        On the chip everything between the input and output projections is one kernel, `kernels.delta_prefill`.
        """
        dtype = x.dtype
        tokens = x.shape[1]
        kernel = self.cfg.linear_conv_kernel_dim
        slot, keep = slot.long(), (start > 0).to(dtype)
        valid = (torch.arange(tokens, device=x.device) < n_valid).float()

        qkv = self.in_proj_qkv(x)[0]
        history = self.conv_state.index_select(0, slot)[0] * keep
        # The last kernel - 1 real inputs carry over to the next chunk or to decode.
        window = torch.cat((history.transpose(0, 1), qkv))
        carried = window.index_select(0, n_valid.long() + torch.arange(kernel - 1, device=x.device)).transpose(0, 1)
        self.conv_state = self.conv_state.index_copy(0, slot, carried.unsqueeze(0))

        z = self.in_proj_z(x)[0]
        beta, log_decay = self._gates(x[0])
        beta, log_decay = beta * valid.unsqueeze(-1), log_decay * valid.unsqueeze(-1)
        state = self.recurrent_state.index_select(0, slot)[0].float() * keep.float()

        if x.device.type != "cpu":
            from .kernels import delta_prefill

            # The sum of log decays from each piece's start is a matmul with a block triangle of ones.
            index = torch.arange(tokens, device=x.device)
            same_piece = (index.unsqueeze(-1) // DELTA_CHUNK == index // DELTA_CHUNK) & (index.unsqueeze(-1) <= index)
            gated, state = delta_prefill(
                qkv, z, self.conv_weight, history, log_decay.transpose(0, 1) @ same_piece.float(), beta.transpose(0, 1), state,
                piece_masks(x.device), self.norm_weight.reshape(1, -1).expand(DELTA_CHUNK, -1),
                key_heads=self.num_k_heads, eps=self.cfg.rms_norm_eps,
            )
            self.recurrent_state = self.recurrent_state.index_copy(0, slot, state.unsqueeze(0).to(self.state_dtype))
            return self.all_reduce(self.out_proj(gated.reshape(tokens, 1, -1)).reshape(1, tokens, -1))

        conv = sum(window.transpose(0, 1)[:, j : j + tokens] * self.conv_weight[:, j : j + 1] for j in range(kernel))
        query, key, value = self._split(silu(conv).transpose(0, 1))
        out, state = chunked_gated_delta(
            query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), log_decay.transpose(0, 1), beta.transpose(0, 1), state,
            min(DELTA_CHUNK, tokens),
        )
        self.recurrent_state = self.recurrent_state.index_copy(0, slot, state.unsqueeze(0).to(self.state_dtype))
        z = z.reshape(tokens, self.num_v_heads, -1)
        return self.all_reduce(self._finish(out.transpose(0, 1).to(dtype), z, dtype).reshape(1, tokens, -1))


class FullAttentionStep(nn.Module):
    """Gated softmax attention with a KV cache of `max_seq_len` positions per row.

    Both graphs attend to the cache as it was plus this call's own keys and values, and write the cache only at the
    end, as NxD Inference does: the write then feeds nothing else, so the compiler does it in place instead of
    copying the cache. Each write index covers a whole key or value, 512 contiguous bytes. Rows that aren't being
    written write junk at the end instead: decode at the last position, prefill at the last `chunk` positions, so
    that no two writes in one call land on the same place. No sequence uses those positions, so a build with a
    prefill graph serves sequences of up to `max_seq_len - chunk` positions.

    `cfg.kv_cache_format` "fp8" stores the cache as FP8 e4m3 with the checkpoint's static per-layer scales, the
    format it was calibrated for, at half the memory. Only the CPU path runs it so far: the compiler accepts FP8
    only inside NKI kernels.
    """

    def __init__(
        self, cfg: TextConfig, tp_degree: int, max_seq_len: int, dtype: torch.dtype, all_reduce: Callable, batch_size: int = 1
    ):
        super().__init__()
        self.cfg = cfg
        self.all_reduce = all_reduce
        self.num_heads = cfg.num_attention_heads // tp_degree
        self.num_kv_heads = cfg.num_key_value_heads // tp_degree
        self.max_seq_len = max_seq_len
        head = cfg.head_dim

        self.q_proj = make_linear(cfg, cfg.hidden_size, self.num_heads * head * 2)  # query plus output gate
        self.k_proj = make_linear(cfg, cfg.hidden_size, self.num_kv_heads * head)
        self.v_proj = make_linear(cfg, cfg.hidden_size, self.num_kv_heads * head)
        self.o_proj = make_linear(cfg, self.num_heads * head, cfg.hidden_size)
        self.q_norm_weight = _frozen(head, dtype=dtype)
        self.k_norm_weight = _frozen(head, dtype=dtype)

        self.fp8_cache = cfg.kv_cache_format == "fp8"
        cache_dtype = torch.float8_e4m3fn if self.fp8_cache else dtype
        if self.fp8_cache:
            self.k_scale = _frozen(1, dtype=dtype)
            self.v_scale = _frozen(1, dtype=dtype)
        self.register_buffer("k_cache", torch.zeros(batch_size, self.num_kv_heads, max_seq_len, head, dtype=cache_dtype))
        self.register_buffer("v_cache", torch.zeros(batch_size, self.num_kv_heads, max_seq_len, head, dtype=cache_dtype))

    def _project(self, x, cos, sin):
        """Query ``[rows, heads, tokens, head]``, key and value ``[rows, kv_heads, tokens, head]``, and the output gate."""
        cfg = self.cfg
        head = cfg.head_dim
        rows, tokens = x.shape[:2]
        q_and_gate = self.q_proj(x).reshape(rows, tokens, self.num_heads, head * 2)
        query, gate = q_and_gate[..., :head], q_and_gate[..., head:]
        query = rms_norm(query, self.q_norm_weight, cfg.rms_norm_eps).transpose(1, 2)
        key = self.k_proj(x).reshape(rows, tokens, self.num_kv_heads, head)
        key = rms_norm(key, self.k_norm_weight, cfg.rms_norm_eps).transpose(1, 2)
        value = self.v_proj(x).reshape(rows, tokens, self.num_kv_heads, head).transpose(1, 2)
        query, key = apply_rotary(query, key, cos, sin)
        return query, key, value, gate.reshape(rows, tokens, self.num_heads * head)

    def _store(self, t: torch.Tensor, scale) -> torch.Tensor:
        if not self.fp8_cache:
            return t
        return (t.float() / scale.float()).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    def _load(self, t: torch.Tensor, scale, dtype: torch.dtype) -> torch.Tensor:
        return (t.to(dtype) * scale).to(dtype) if self.fp8_cache else t

    def _round_trip(self, t: torch.Tensor, scale, dtype: torch.dtype) -> torch.Tensor:
        """A key or value as it reads back from the cache, so a step sees its own the way later steps will."""
        return self._load(self._store(t, scale), scale, dtype)

    def _write_both(self, key, value, positions, scales):
        """Write keys and values ``[rows, kv_heads, n, head]`` at each row's ``positions`` ``[rows, n]``."""
        k_scale, v_scale = scales
        rows, kv_heads, n, head = key.shape
        at = positions.reshape(rows, 1, n, 1).expand(-1, kv_heads, -1, head)
        self.k_cache = _scatter(self.k_cache, 2, at, self._store(key, k_scale))
        self.v_cache = _scatter(self.v_cache, 2, at, self._store(value, v_scale))

    def forward(self, x, cos, sin, position, active) -> torch.Tensor:
        """One token per row at `position`. A row with `active` 0 leaves its cache unchanged."""
        cfg = self.cfg
        head = cfg.head_dim
        batch = x.shape[0]
        device = x.device
        query, key, value, gate = self._project(x, cos, sin)
        scales = (self.k_scale, self.v_scale) if self.fp8_cache else (None, None)

        groups = cfg.num_key_value_groups
        grouped = query.reshape(batch, self.num_kv_heads, groups, head)
        keys = self._load(self.k_cache, scales[0], x.dtype)
        values = self._load(self.v_cache, scales[1], x.dtype)
        earlier = torch.arange(self.max_seq_len, device=device) < position.unsqueeze(-1)
        cached = torch.matmul(grouped, keys.transpose(2, 3)).float() + torch.where(earlier, 0.0, MASKED_SCORE).reshape(batch, 1, 1, -1)
        own_key = self._round_trip(key, scales[0], x.dtype)
        own_value = self._round_trip(value, scales[1], x.dtype)
        own = torch.matmul(grouped, own_key.transpose(2, 3)).float()
        weights = torch.softmax(torch.cat((cached, own), dim=-1) * head**-0.5, dim=-1).to(x.dtype)
        out = torch.matmul(weights[..., :-1], values) + weights[..., -1:] * own_value
        out = out.reshape(batch, 1, self.num_heads * head)

        where = torch.where(active > 0.5, position.long(), self.max_seq_len - 1).unsqueeze(-1)
        self._write_both(key, value, where, scales)
        return self.all_reduce(self.o_proj(out * torch.sigmoid(gate)))

    def prefill(self, x, cos, sin, slot, start, n_valid) -> torch.Tensor:
        """A chunk of one row's tokens from position `start`. Padding past `n_valid` is written but never read
        before decode overwrites it, because each step reads only positions up to its own."""
        cfg = self.cfg
        head = cfg.head_dim
        tokens = x.shape[1]
        device = x.device
        query, key, value, gate = self._project(x, cos, sin)
        scales = (self.k_scale, self.v_scale) if self.fp8_cache else (None, None)
        groups = cfg.num_key_value_groups

        # A KV head's query heads stacked, so query g * tokens + i is head g's token i.
        grouped = (query[0] * head**-0.5).reshape(self.num_kv_heads, groups * tokens, head)
        own_keys = self._round_trip(key[0], scales[0], x.dtype)
        own_values = self._round_trip(value[0], scales[1], x.dtype)
        if device.type == "cpu":
            cache_keys = self._load(self.k_cache.index_select(0, slot.long())[0], scales[0], x.dtype)
            cache_values = self._load(self.v_cache.index_select(0, slot.long())[0], scales[1], x.dtype)
            # The same visibility as the kernel: cached positions before `start`, then the chunk causally.
            index = torch.arange(groups * tokens) % tokens
            visible = torch.cat(
                (
                    (torch.arange(self.max_seq_len) < start).expand(groups * tokens, -1),
                    torch.arange(tokens).unsqueeze(0) <= index.unsqueeze(-1),
                ),
                dim=-1,
            )
            all_keys = torch.cat((cache_keys, own_keys), dim=1)
            all_values = torch.cat((cache_values, own_values), dim=1)
            out = blocked_attention(grouped, all_keys, all_values, visible, ATTENTION_BLOCK)
        else:
            from .kernels import prefill_attention

            starts = start.float().reshape(1, 1).expand(128, 1)
            first_row = (slot.int() * self.num_kv_heads * self.max_seq_len).reshape(1, 1).expand(128, 1)
            out = prefill_attention(
                grouped.transpose(1, 2), self.k_cache.reshape(-1, head), self.v_cache.reshape(-1, head),
                own_keys.transpose(1, 2), own_values, starts, first_row, positions=self.max_seq_len,
            )
        out = out.to(x.dtype).reshape(self.num_heads, tokens, head).transpose(0, 1).reshape(1, tokens, self.num_heads * head)

        # Every row gets a chunk of positions: `slot` its own, the others the junk positions at the end.
        rows = self.k_cache.shape[0]
        positions = start.long() + torch.arange(tokens, device=device)
        junk = self.max_seq_len - tokens + torch.arange(tokens, device=device)
        others = torch.arange(rows, device=device) != slot.long()
        where = torch.where(others.unsqueeze(-1), junk.unsqueeze(0), positions.unsqueeze(0))
        self._write_both(key.expand(rows, -1, -1, -1), value.expand(rows, -1, -1, -1), where, scales)
        return self.all_reduce(self.o_proj(out * torch.sigmoid(gate)))


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

    def forward(self, x, cos, sin, position=None, active=None, slot=None, start=None, n_valid=None):
        eps = self.cfg.rms_norm_eps
        h = rms_norm(x, self.input_layernorm_weight, eps)
        if slot is not None:
            mixer = self.linear_attn.prefill(h, slot, start, n_valid) if self.layer_type == LINEAR else self.self_attn.prefill(h, cos, sin, slot, start, n_valid)
        elif self.layer_type == LINEAR:
            mixer = self.linear_attn(h, position, active)
        else:
            mixer = self.self_attn(h, cos, sin, position, active)
        x = x + mixer
        return x + self.mlp(rms_norm(x, self.post_attention_layernorm_weight, eps))


class Qwen38Step(nn.Module):
    """The compiled graph: a run of decoder layers, called in one of two ways that share weights and caches.

    `cfg.layer_types` may be the whole model or one repeating block of it (see `generate.BlockChain`).

    Decode, one token per row (inputs built by `generate.batch_step_inputs`):
        inputs_embeds: ``[batch, 1, hidden]``, from the embedding table or the previous block.
        cos, sin: rotary terms for each row's position, ``[batch, 1, rotary_dim]``.
        position: int32 ``[batch]``, where each row's token goes. Position 0 starts a new sequence.
        active: float32 ``[batch]``. A row with 0 keeps its state and cache exactly as they were.

    Prefill, one chunk of one row (inputs built by `generate.prefill_inputs`):
        inputs_embeds: ``[1, chunk, hidden]``.
        cos, sin: ``[1, chunk, rotary_dim]``, for positions `start` onward.
        slot: int32 ``[1]``, the row whose state and cache the chunk continues.
        start: int32 ``[1]``, the chunk's first position, a multiple of the chunk. 0 starts a new sequence.
        n_valid: int32 ``[1]``, how many of the chunk's tokens are real. The rest are padding.
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

    def forward(self, inputs_embeds, cos, sin, position=None, active=None, slot=None, start=None, n_valid=None):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, cos, sin, position, active, slot, start, n_valid)
        return x
