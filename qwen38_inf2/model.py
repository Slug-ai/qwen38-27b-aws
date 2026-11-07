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
# The KV cache's FP8 is NeuronCore-v2's e4m3, whose largest finite value is 240, unlike OCP e4m3fn's 448. Up to
# 240 the two share every byte, so the CPU path encodes with torch's e4m3fn and gets the chip's bytes. The
# checkpoint's scales, calibrated for 448, are stretched by 448 / 240 when loaded (`checkpoint.KV_SCALE_STRETCH`).
FP8_E4M3_MAX = 240.0


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


# From this many tokens the MLP's gate and up projections run as `kernels.mlp_up`, with weights stationary. With a
# few tokens, as in decode, streaming the weights through the Tensor Engine is faster.
MLP_UP_MIN_TOKENS = 128


class MLP(nn.Module):
    def __init__(self, cfg: TextConfig, tp_degree: int, all_reduce: Callable):
        super().__init__()
        intermediate = cfg.intermediate_size // tp_degree
        self.gate_proj = make_linear(cfg, cfg.hidden_size, intermediate)
        self.up_proj = make_linear(cfg, cfg.hidden_size, intermediate)
        self.down_proj = make_linear(cfg, intermediate, cfg.hidden_size)
        self.all_reduce = all_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.numel() // x.shape[-1]
        if x.device.type != "cpu" and isinstance(self.gate_proj, FP8Linear) and tokens >= MLP_UP_MIN_TOKENS:
            # A prefill chunk: gate, up and SiLU in one kernel whose output is already the down projection's input
            # layout, so it skips a transpose and two round trips through HBM.
            from .kernels import fp8_linear, mlp_up

            inner = mlp_up(x.reshape(tokens, -1).transpose(0, 1), self.gate_proj.weight_fp8, self.up_proj.weight_fp8)
            return self.all_reduce(fp8_linear(inner, self.down_proj.weight_fp8).reshape(*x.shape[:-1], -1))
        return self.all_reduce(self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x)))


def _commit_and_read(state, scale, weighted, deltas, reads):
    """`kernels.delta_verify_state` on a chip, and the same in float32 ops on a CPU."""
    if state.device.type == "cpu":
        committed = (state.float() * scale + weighted.transpose(1, 2) @ deltas).to(state.dtype)
        return committed, reads.transpose(1, 2) @ committed.float(), torch.zeros(1, 1)
    from .kernels import delta_verify_state

    # While NxD traces, tensors made from a state buffer are placeholders, and the kernel's tracer can't take a call
    # whose every input is one: their data_ptr() reads an original only the buffer itself has.
    return delta_verify_state(*(t.as_subclass(torch.Tensor) for t in (state, scale, weighted, deltas, reads)))


class GatedDeltaNetStep(nn.Module):
    """The gated delta rule with a causal conv: one token per row (decode), or a chunk of one row (prefill)."""

    def __init__(
        self, cfg: TextConfig, tp_degree: int, dtype: torch.dtype, all_reduce: Callable, batch_size: int = 1, verify_tokens: int = 0,
        saved_states: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
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

        # With `saved_states`, rows from `batch_size` on hold each row's states as they were at the end of
        # its prompt (`copy_states`), so a later prompt extending that one can start from there: the KV cache still
        # holds those positions. They are rows of the same buffers, so every graph touching them writes them too,
        # and NxD keeps them as the shared state they are rather than taking them for weights.
        rows = batch_size * (2 if saved_states else 1)
        self.register_buffer("conv_state", torch.zeros(rows, conv_dim, cfg.linear_conv_kernel_dim - 1, dtype=dtype))
        self.state_dtype = {"float32": torch.float32, "bf16": torch.bfloat16}[cfg.recurrent_state_format]
        self.register_buffer(
            "recurrent_state",
            torch.zeros(rows, self.num_v_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim, dtype=self.state_dtype),
        )
        if verify_tokens:
            # A verify step's tokens leave the states as they were, and keep what each would add, so the next step
            # commits only the ones accepted: each token's update to the state is rank 1, key times delta, after decay.
            k_head, v_head = cfg.linear_key_head_dim, cfg.linear_value_head_dim
            self.register_buffer("pending_key_state", torch.zeros(batch_size, verify_tokens, self.num_v_heads, k_head))
            self.register_buffer("pending_delta_state", torch.zeros(batch_size, verify_tokens, self.num_v_heads, v_head))
            self.register_buffer("pending_decay_state", torch.zeros(batch_size, verify_tokens, self.num_v_heads))
            self.register_buffer("pending_input_state", torch.zeros(batch_size, verify_tokens, conv_dim, dtype=dtype))

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
        conv_state = self._live(self.conv_state)
        window = torch.cat((conv_state * keep.reshape(batch, 1, 1).to(dtype), qkv), dim=-1)
        self.conv_state = self._with_live(self.conv_state, torch.where(on, window[:, :, 1:], conv_state))
        query, key, value = self._split(silu((window * self.conv_weight).sum(dim=-1)))

        z = self.in_proj_z(x).reshape(batch, self.num_v_heads, -1)
        beta, log_decay = self._gates(x.reshape(batch, -1))
        # No decay and no write leaves an inactive row's state as it was.
        beta = (beta * active.unsqueeze(-1)).unsqueeze(-1)
        log_decay = log_decay * active.unsqueeze(-1)

        state = self._live(self.recurrent_state).float() * keep.reshape(batch, 1, 1, 1) * log_decay.exp()[..., None, None]
        delta = (value - (state * key.unsqueeze(-1)).sum(dim=-2)) * beta
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        self.recurrent_state = self._with_live(self.recurrent_state, state.to(self.state_dtype))
        out = (state * query.unsqueeze(-1)).sum(dim=-2).to(dtype)
        return self.all_reduce(self._finish(out, z, dtype))

    def verify(self, x: torch.Tensor, position: torch.Tensor, active: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
        """K tokens per row from `position`, ``x`` ``[batch, K, hidden]``, for speculative decoding.

        First the last verify step's first `accepted` tokens per row are committed to the states, from the updates
        it kept. Then the K new tokens run from those states without changing them, each token's update kept for
        the next step to commit as many as are accepted. A row with `active` 0 keeps its states exactly.
        """
        dtype = x.dtype
        batch, tokens = x.shape[:2]
        kernel = self.cfg.linear_conv_kernel_dim
        on = (active > 0.5).reshape(batch, 1)
        # A sequence's first token starts from empty states. Inactive rows keep theirs, whatever their position.
        keep = torch.where((position.reshape(batch, 1) == 0) & on, 0.0, 1.0)
        # The state after the first `accepted` tokens of the last step, in closed form: their decays applied to the
        # state they started from, plus each one's key times delta, decayed by the tokens after it. Masks compare
        # against `accepted` and loops run over the K tokens: the compiler fails on masks from iota.
        taken = [(accepted.reshape(batch, 1) > s).float() for s in range(self.pending_key_state.shape[1])]
        decays = [self.pending_decay_state[:, s] * taken[s] for s in range(len(taken))]
        remaining, weighted = sum(decays), []
        scale = torch.where(on, remaining.exp() * keep, 1.0)
        for s in range(len(taken)):
            remaining = remaining - decays[s]
            weighted.append(self.pending_key_state[:, s] * (remaining.exp() * taken[s] * keep * on).unsqueeze(-1))
        # The conv's history is the last kernel - 1 inputs before the first token not committed.
        window = torch.cat((self._live(self.conv_state), self.pending_input_state.transpose(1, 2)), dim=-1)
        index = (accepted.long().reshape(batch, 1, 1) + torch.arange(kernel - 1, device=x.device)).expand(-1, window.shape[1], -1)
        history = window.gather(2, index) * keep.reshape(batch, 1, 1).to(dtype)
        history = torch.where(on.reshape(batch, 1, 1), history, self._live(self.conv_state))
        self.conv_state = self._with_live(self.conv_state, history)

        qkv = self.in_proj_qkv(x)
        window = torch.cat((history, qkv.transpose(1, 2)), dim=-1)
        mixed = torch.stack([(window[:, :, t : t + kernel] * self.conv_weight).sum(dim=-1) for t in range(tokens)], dim=1)
        query, key, value = (part.reshape(batch, tokens, *part.shape[1:]) for part in self._split(silu(mixed).reshape(batch * tokens, -1)))
        z = self.in_proj_z(x).reshape(batch * tokens, self.num_v_heads, -1)
        beta, log_decay = self._gates(x.reshape(batch * tokens, -1))
        beta = beta.reshape(batch, tokens, -1) * active.reshape(batch, 1, 1)
        log_decay = log_decay.reshape(batch, tokens, -1) * active.reshape(batch, 1, 1)

        # The commit, and the new keys and queries times the committed state, one row and head at a time. What this
        # step keeps for the next has the kernel's zero added, so it overwrites the last step's only once read.
        heads, k_head, v_head = self.recurrent_state.shape[1:]
        pairs = batch * heads
        committed, products, zero = _commit_and_read(
            self._live(self.recurrent_state).reshape(pairs, k_head, v_head),
            scale.reshape(pairs, 1, 1).expand(-1, k_head, -1),
            torch.stack(weighted, dim=2).reshape(pairs, -1, k_head),
            self.pending_delta_state.transpose(1, 2).reshape(pairs, -1, v_head),
            torch.cat((key, query), dim=1).permute(0, 2, 3, 1).reshape(pairs, k_head, 2 * tokens),
        )
        zero = zero.reshape(())
        self.recurrent_state = self._with_live(self.recurrent_state, committed.reshape(batch, heads, k_head, v_head))
        products = products.reshape(batch, heads, 2 * tokens, v_head).transpose(1, 2)
        key_state, query_state = products[:, :tokens], products[:, tokens:]

        # The K tokens as one small chunk of the chunked delta rule from the committed state, so the state appears
        # only in those products and the tokens' dependence on each other is a K x K triangular solve.
        cumulative = [log_decay[:, 0]]
        for t in range(1, tokens):
            cumulative.append(cumulative[-1] + log_decay[:, t])
        deltas, outs = [], []
        for t in range(tokens):
            delta = value[:, t] - cumulative[t].exp().unsqueeze(-1) * key_state[:, t]
            for s in range(t):
                coupling = (cumulative[t] - cumulative[s]).exp() * (key[:, t] * key[:, s]).sum(dim=-1)
                delta = delta - coupling.unsqueeze(-1) * deltas[s]
            deltas.append(delta * beta[:, t].unsqueeze(-1))
            out = cumulative[t].exp().unsqueeze(-1) * query_state[:, t]
            for s in range(t + 1):
                attend = (cumulative[t] - cumulative[s]).exp() * (query[:, t] * key[:, s]).sum(dim=-1)
                out = out + attend.unsqueeze(-1) * deltas[s]
            outs.append(out)
        delta, out = torch.stack(deltas, dim=1), torch.stack(outs, dim=1)
        self.pending_key_state = key.float() + zero
        self.pending_delta_state = delta
        self.pending_decay_state = log_decay.float() + zero
        self.pending_input_state = qkv + zero.to(qkv.dtype)
        out = out.reshape(batch * tokens, self.num_v_heads, -1).to(dtype)
        return self.all_reduce(self._finish(out, z, dtype).reshape(batch, tokens, -1))

    def copy_states(self, slot: torch.Tensor, to_saved: bool) -> None:
        """Save a row's recurrent and conv states, or restore them from what was saved."""
        if self.recurrent_state.shape[0] == self.batch_size:
            raise ValueError("this layer keeps no saved states")
        slot = slot.long()
        saved = slot + self.batch_size
        source, target = (slot, saved) if to_saved else (saved, slot)
        self.recurrent_state = self.recurrent_state.index_copy(0, target, self.recurrent_state.index_select(0, source))
        self.conv_state = self.conv_state.index_copy(0, target, self.conv_state.index_select(0, source))

    def _live(self, buffer: torch.Tensor) -> torch.Tensor:
        """The rows of a state buffer that running sequences use, without the saved ones."""
        return buffer[: self.batch_size]

    def _with_live(self, buffer: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        """`buffer` with its running rows replaced by `live`, the saved rows left as they are."""
        if buffer.shape[0] == live.shape[0]:
            return live
        return buffer.index_copy(0, torch.arange(live.shape[0], device=live.device), live)

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


# Positions per page of a paged KV cache, the attention kernels' blocks.
PAGE = 2048


class FullAttentionStep(nn.Module):
    """Gated softmax attention with a KV cache of `max_seq_len` positions per row.

    Both graphs attend to the cache as it was plus this call's own keys and values, and write the cache only at the
    end, as NxD Inference does: the write then feeds nothing else, so the compiler does it in place instead of
    copying the cache. Each write index covers a whole key or value, 512 contiguous bytes. Rows that aren't being
    written write junk at the end instead: decode at the last position, prefill at the last `chunk` positions, so
    that no two writes in one call land on the same place. No sequence uses those positions, so a build with a
    prefill graph serves sequences of up to `max_seq_len - chunk` positions.

    `cfg.kv_cache_format` "fp8" stores the cache as FP8 e4m3 with the checkpoint's static per-layer scales, the
    format it was calibrated for, at half the memory. The compiler accepts FP8 only inside NKI kernels, so the cache
    holds the bytes as uint8: on the chip `kernels.fp8_encode` makes them and the attention kernels read them.

    With `pages`, the cache is that many pages of `PAGE` positions, ``[kv_heads, pages * PAGE, head]``, shared by
    every row through a table per row (the calls' `pages` input), so rows take only the positions they use and
    more rows fit. Page 0 takes the writes no sequence reads, each at its own position: a prefill chunk's padding
    and rows not running. Each row can still hold `max_seq_len` positions.
    """

    def __init__(
        self, cfg: TextConfig, tp_degree: int, max_seq_len: int, dtype: torch.dtype, all_reduce: Callable, batch_size: int = 1,
        pages: int = 0,
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
        cache_dtype = torch.uint8 if self.fp8_cache else dtype
        if self.fp8_cache:
            self.k_scale = _frozen(1, dtype=dtype)
            self.v_scale = _frozen(1, dtype=dtype)
        self.pages = pages
        if pages and max_seq_len % PAGE:
            raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of the page's {PAGE} positions")
        shape = (self.num_kv_heads, pages * PAGE, head) if pages else (batch_size, self.num_kv_heads, max_seq_len, head)
        self.register_buffer("k_cache", torch.zeros(*shape, dtype=cache_dtype))
        self.register_buffer("v_cache", torch.zeros(*shape, dtype=cache_dtype))

    def _tables(self, pages, device, slot=None):
        """For the attention kernels: the cache block holding each row's `PAGE` positions for KV head 0, ``[rows,
        max_seq_len / PAGE]`` or the one row `slot`, and how many blocks on KV head h's are."""
        blocks = self.max_seq_len // PAGE
        if self.pages:
            table, stride = pages.int(), self.pages
        else:
            # Each row's positions in order, KV head by KV head: a constant made on the CPU.
            rows = torch.arange(self.k_cache.shape[0], dtype=torch.int32).unsqueeze(-1) * (self.num_kv_heads * blocks)
            table, stride = (rows + torch.arange(blocks, dtype=torch.int32)).to(device), blocks
        return (table if slot is None else table.index_select(0, slot.long())), stride

    def _cached(self, cache, pages, rows=None):
        """Every row's cache, or those in `rows`, as ``[rows, kv_heads, max_seq_len, head]``."""
        if not self.pages:
            return cache if rows is None else cache.index_select(0, rows.long())
        table = pages if rows is None else pages.index_select(0, rows.long())
        index = (table.long().unsqueeze(-1) * PAGE + torch.arange(PAGE, device=cache.device)).reshape(-1)
        gathered = cache.index_select(1, index).reshape(self.num_kv_heads, table.shape[0], self.max_seq_len, -1)
        return gathered.transpose(0, 1)

    def _write_pages(self, key, value, positions, valid, pages):
        """Into a paged cache, `_store`d keys and values ``[rows, kv_heads, n, head]`` at each row's ``positions``
        ``[rows, n]`` where `valid`, and elsewhere each at its own position of page 0."""
        rows, kv_heads, n, head = key.shape
        at = pages.long().gather(1, positions // PAGE) * PAGE + positions % PAGE
        at = torch.where(valid, at, torch.arange(rows * n, device=key.device).reshape(rows, n))
        at = at.reshape(1, rows * n, 1).expand(kv_heads, -1, head)
        self.k_cache = self.k_cache.scatter(1, at, key.transpose(0, 1).reshape(kv_heads, rows * n, head))
        self.v_cache = self.v_cache.scatter(1, at, value.transpose(0, 1).reshape(kv_heads, rows * n, head))

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
        """What the cache holds for `t`: itself, or FP8 bytes of ``t / scale``."""
        if not self.fp8_cache:
            return t
        if t.device.type != "cpu":
            return self._encode(t, scale)[0]
        return (t.float() / scale.float()).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)

    def _load(self, t: torch.Tensor, scale, dtype: torch.dtype) -> torch.Tensor:
        """Cached values back from `_store`. CPU only for FP8: on the chip the kernels read the bytes."""
        return (t.view(torch.float8_e4m3fn).to(dtype) * scale).to(dtype) if self.fp8_cache else t

    def _round_trip(self, t: torch.Tensor, scale, dtype: torch.dtype) -> torch.Tensor:
        """A key or value as it reads back from the cache, so a step sees its own the way later steps will."""
        return self._load(self._store(t, scale), scale, dtype)

    def _encode(self, t: torch.Tensor, scale) -> tuple[torch.Tensor, torch.Tensor]:
        """On the chip, FP8 bytes of ``t / scale`` ``[..., head]`` and the values they hold, in units of `scale`."""
        from .kernels import fp8_encode

        inverse = (1.0 / scale.float()).reshape(1, 1).expand(128, 1)
        coded, held = fp8_encode(t.reshape(-1, t.shape[-1]), inverse)
        return coded.reshape(t.shape), held.reshape(t.shape)

    def _write_both(self, key, value, positions):
        """Write cached keys and values ``[rows, kv_heads, n, head]``, from `_store`, at each row's ``positions`` ``[rows, n]``."""
        rows, kv_heads, n, head = key.shape
        at = positions.reshape(rows, 1, n, 1).expand(-1, kv_heads, -1, head)
        self.k_cache = self.k_cache.scatter(2, at, key)
        self.v_cache = self.v_cache.scatter(2, at, value)

    def forward(self, x, cos, sin, position, active, pages=None) -> torch.Tensor:
        """One token per row at `position`. A row with `active` 0 leaves its cache unchanged."""
        cfg = self.cfg
        head = cfg.head_dim
        batch = x.shape[0]
        device = x.device
        query, key, value, gate = self._project(x, cos, sin)
        scales = (self.k_scale, self.v_scale) if self.fp8_cache else (None, None)

        groups = cfg.num_key_value_groups
        grouped = query.reshape(batch, self.num_kv_heads, groups, head)
        where = torch.where(active > 0.5, position.long(), self.max_seq_len - 1).unsqueeze(-1)
        if self.pages:
            write = lambda k, v: self._write_pages(k, v, position.long().unsqueeze(-1), (active > 0.5).unsqueeze(-1), pages)  # noqa: E731
        else:
            write = lambda k, v: self._write_both(k, v, where)  # noqa: E731
        if self.fp8_cache and device.type != "cpu":
            out, k_coded, v_coded = self._decode_fp8(grouped, key, value, position, scales, pages)
            write(k_coded, v_coded)
            return self.all_reduce(self.o_proj(out * torch.sigmoid(gate)))

        keys = self._load(self._cached(self.k_cache, pages), scales[0], x.dtype)
        values = self._load(self._cached(self.v_cache, pages), scales[1], x.dtype)
        earlier = torch.arange(self.max_seq_len, device=device) < position.unsqueeze(-1)
        cached = torch.matmul(grouped, keys.transpose(2, 3)).float() + torch.where(earlier, 0.0, MASKED_SCORE).reshape(batch, 1, 1, -1)
        own_key = self._round_trip(key, scales[0], x.dtype)
        own_value = self._round_trip(value, scales[1], x.dtype)
        own = torch.matmul(grouped, own_key.transpose(2, 3)).float()
        weights = torch.softmax(torch.cat((cached, own), dim=-1) * head**-0.5, dim=-1).to(x.dtype)
        out = torch.matmul(weights[..., :-1], values) + weights[..., -1:] * own_value
        out = out.reshape(batch, 1, self.num_heads * head)

        write(self._store(key, scales[0]), self._store(value, scales[1]))
        return self.all_reduce(self.o_proj(out * torch.sigmoid(gate)))

    def verify(self, x, cos, sin, position, active, pages=None) -> torch.Tensor:
        """K tokens per row from `position`, ``x`` ``[batch, K, hidden]``, for speculative decoding: each against
        the row's cache before `position` and the step's own tokens up to itself. Their keys and values are written
        from `position` on; the next step overwrites those of tokens not accepted."""
        cfg = self.cfg
        head = cfg.head_dim
        batch, tokens = x.shape[:2]
        device = x.device
        query, key, value, gate = self._project(x, cos, sin)
        scales = (self.k_scale, self.v_scale) if self.fp8_cache else (None, None)
        groups = cfg.num_key_value_groups
        # Rows not running write junk at the cache's last positions, as in decode.
        steps = torch.arange(tokens, device=device)
        where = torch.where(active.unsqueeze(-1) > 0.5, position.long().unsqueeze(-1) + steps, self.max_seq_len - tokens + steps)
        if self.pages:
            valid = (active.unsqueeze(-1) > 0.5).expand(-1, tokens)
            write = lambda k, v: self._write_pages(k, v, position.long().unsqueeze(-1) + steps, valid, pages)  # noqa: E731
        else:
            write = lambda k, v: self._write_both(k, v, where)  # noqa: E731
        if self.fp8_cache and device.type != "cpu":
            out, k_coded, v_coded = self._verify_fp8(query, key, value, position, scales, pages)
            write(k_coded, v_coded)
            return self.all_reduce(self.o_proj(out * torch.sigmoid(gate)))
        grouped = query.reshape(batch, self.num_kv_heads, groups, tokens, head)
        keys = self._load(self._cached(self.k_cache, pages), scales[0], x.dtype).unsqueeze(2)
        values = self._load(self._cached(self.v_cache, pages), scales[1], x.dtype).unsqueeze(2)
        earlier = torch.arange(self.max_seq_len, device=device) < position.unsqueeze(-1)
        cached = torch.matmul(grouped, keys.transpose(-1, -2)).float() + torch.where(earlier, 0.0, MASKED_SCORE).reshape(batch, 1, 1, 1, -1)
        own_key = self._round_trip(key, scales[0], x.dtype).unsqueeze(2)
        own_value = self._round_trip(value, scales[1], x.dtype).unsqueeze(2)
        causal = steps.unsqueeze(0) <= steps.unsqueeze(-1)
        own = torch.matmul(grouped, own_key.transpose(-1, -2)).float() + torch.where(causal, 0.0, MASKED_SCORE)
        weights = torch.softmax(torch.cat((cached, own), dim=-1) * head**-0.5, dim=-1).to(x.dtype)
        out = torch.matmul(weights[..., : self.max_seq_len], values) + torch.matmul(weights[..., self.max_seq_len :], own_value)
        out = out.permute(0, 3, 1, 2, 4).reshape(batch, tokens, self.num_heads * head)
        write(self._store(key, scales[0]), self._store(value, scales[1]))
        return self.all_reduce(self.o_proj(out * torch.sigmoid(gate)))

    def _verify_fp8(self, query, key, value, position, scales, pages):
        """Verify attention on the chip over the FP8 cache, through `kernels.verify_attention`. A row's K tokens are
        query heads of its pairs, and calls take as many rows as fit 128 query rows. Returns the output
        ``[batch, K, heads * head]`` and the tokens' keys and values as cache bytes."""
        from .kernels import DECODE_STEP, verify_attention

        k_scale, v_scale = scales
        batch, heads, tokens, head = query.shape
        kv_heads, device = self.num_kv_heads, query.device
        groups = heads // kv_heads
        per_pair = groups * tokens
        k_coded, k_held = self._encode(key, k_scale)
        v_coded, v_held = self._encode(value, v_scale)
        # Pair (row, KV head), then query head, then token, as the kernel takes them.
        q = (query.float() * (head**-0.5 * k_scale.float())).to(query.dtype).reshape(batch, kv_heads, groups, tokens, head)
        limit = position.float().reshape(batch, 1).expand(-1, kv_heads * per_pair)
        blocks = torch.div(position.max().int() + DECODE_STEP - 1, DECODE_STEP, rounding_mode="floor").reshape(1, 1)
        rows_per_call = max(1, 128 // (kv_heads * per_pair))
        table, stride = self._tables(pages, device)
        outs = []
        for first in range(0, batch, rows_per_call):
            rows = min(rows_per_call, batch - first)
            pairs = rows * kv_heads
            # A query sees its own pair's tokens up to its own.
            index_q = torch.arange(pairs * per_pair, device=device)
            index_k = torch.arange(pairs * tokens, device=device)
            visible = (index_q.unsqueeze(-1) // per_pair == index_k // tokens) & (index_k % tokens <= index_q.unsqueeze(-1) % tokens)
            own_mask = torch.where(visible, 0.0, MASKED_SCORE).float()
            outs.append(
                verify_attention(
                    q[first : first + rows].reshape(-1, head), self.k_cache.reshape(-1, head), self.v_cache.reshape(-1, head),
                    k_held[first : first + rows].reshape(-1, head), v_held[first : first + rows].reshape(-1, head), own_mask,
                    limit[first : first + rows].reshape(-1, 1), blocks, table, groups=per_pair, kv_heads=kv_heads, head_stride=stride,
                    pair_offset=first * kv_heads,
                )
            )
        out = (torch.cat(outs).float() * v_scale.float()).to(query.dtype).reshape(batch, kv_heads, groups, tokens, head)
        return out.permute(0, 3, 1, 2, 4).reshape(batch, tokens, heads * head), k_coded, v_coded

    def _decode_fp8(self, grouped, key, value, position, scales, pages):
        """Decode attention on the chip over the FP8 cache, in the kernel since XLA can't read FP8. Returns the output
        ``[batch, 1, heads * head]`` and the step's key and value as cache bytes."""
        from .kernels import DECODE_STEP, decode_attention

        k_scale, v_scale = scales
        batch, kv_heads, groups, head = grouped.shape
        k_coded, k_held = self._encode(key, k_scale)
        v_coded, v_held = self._encode(value, v_scale)
        # Cached keys are in units of k_scale, so the queries take it instead.
        q = (grouped.float() * (head**-0.5 * k_scale.float())).to(grouped.dtype).reshape(-1, head)
        per_query = lambda t: t.reshape(batch * kv_heads, 1, head).expand(-1, groups, -1).reshape(-1, head)  # noqa: E731
        limit = position.float().reshape(batch, 1).expand(-1, kv_heads * groups).reshape(-1, 1)
        blocks = torch.div(position.max().int() + DECODE_STEP - 1, DECODE_STEP, rounding_mode="floor").reshape(1, 1)
        table, stride = self._tables(pages, q.device)
        out = decode_attention(
            q, self.k_cache.reshape(-1, head), self.v_cache.reshape(-1, head), per_query(k_held), per_query(v_held), limit, blocks, table,
            groups=groups, kv_heads=kv_heads, head_stride=stride,
        )
        out = (out.float() * v_scale.float()).to(grouped.dtype).reshape(batch, 1, kv_heads * groups * head)
        return out, k_coded, v_coded

    def prefill(self, x, cos, sin, slot, start, n_valid, pages=None) -> torch.Tensor:
        """A chunk of one row's tokens from position `start`. Padding past `n_valid` is written but never read
        before decode overwrites it, because each step reads only positions up to its own; with a paged cache it is
        written to page 0 instead."""
        cfg = self.cfg
        head = cfg.head_dim
        tokens = x.shape[1]
        device = x.device
        query, key, value, gate = self._project(x, cos, sin)
        scales = (self.k_scale, self.v_scale) if self.fp8_cache else (None, None)
        groups = cfg.num_key_value_groups

        # A KV head's query heads stacked, so query g * tokens + i is head g's token i.
        grouped = (query[0] * head**-0.5).reshape(self.num_kv_heads, groups * tokens, head)
        if self.fp8_cache and device.type != "cpu":
            # The kernel works in units of the scales: cached bytes and the chunk's own values as the bytes hold them.
            k_coded, own_keys = self._encode(key, scales[0])
            v_coded, own_values = self._encode(value, scales[1])
            grouped = (grouped.float() * scales[0].float()).to(x.dtype)
            own_keys, own_values = own_keys[0], own_values[0]
        else:
            k_coded, v_coded = self._store(key, scales[0]), self._store(value, scales[1])
            own_keys = self._round_trip(key[0], scales[0], x.dtype)
            own_values = self._round_trip(value[0], scales[1], x.dtype)
        if device.type == "cpu":
            cache_keys = self._load(self._cached(self.k_cache, pages, slot)[0], scales[0], x.dtype)
            cache_values = self._load(self._cached(self.v_cache, pages, slot)[0], scales[1], x.dtype)
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
            from .kernels import KEY_BLOCK, prefill_attention

            starts = start.float().reshape(1, 1).expand(128, 1)
            table, stride = self._tables(pages, device, slot)
            # Only the cache blocks before `start` hold this row's keys, and the kernel visits only those.
            blocks = torch.div(start.int() + KEY_BLOCK - 1, KEY_BLOCK, rounding_mode="floor").reshape(1, 1)
            out = prefill_attention(
                grouped.transpose(1, 2), self.k_cache.reshape(-1, head), self.v_cache.reshape(-1, head),
                own_keys.transpose(1, 2), own_values, starts, table, blocks, head_stride=stride,
            )
            if self.fp8_cache:
                out = out.float() * scales[1].float()
        out = out.to(x.dtype).reshape(self.num_heads, tokens, head).transpose(0, 1).reshape(1, tokens, self.num_heads * head)

        if self.pages:
            positions = (start.long() + torch.arange(tokens, device=device)).unsqueeze(0)
            valid = (torch.arange(tokens, device=device) < n_valid).unsqueeze(0)
            self._write_pages(k_coded, v_coded, positions, valid, pages.index_select(0, slot.long()))
            return self.all_reduce(self.o_proj(out * torch.sigmoid(gate)))
        # Every row gets a chunk of positions: `slot` its own, the others the junk positions at the end.
        rows = self.k_cache.shape[0]
        positions = start.long() + torch.arange(tokens, device=device)
        junk = self.max_seq_len - tokens + torch.arange(tokens, device=device)
        others = torch.arange(rows, device=device) != slot.long()
        where = torch.where(others.unsqueeze(-1), junk.unsqueeze(0), positions.unsqueeze(0))
        self._write_both(k_coded.expand(rows, -1, -1, -1), v_coded.expand(rows, -1, -1, -1), where)
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
    def __init__(self, cfg, layer_type, tp_degree, max_seq_len, dtype, all_reduce, batch_size=1, verify_tokens=0, saved_states=False, pages=0):
        super().__init__()
        self.cfg = cfg
        self.layer_type = layer_type
        if layer_type == LINEAR:
            self.linear_attn = GatedDeltaNetStep(cfg, tp_degree, dtype, all_reduce, batch_size, verify_tokens, saved_states)
        elif layer_type == FULL:
            self.self_attn = FullAttentionStep(cfg, tp_degree, max_seq_len, dtype, all_reduce, batch_size, pages)
        else:
            raise ValueError(f"Unknown layer type {layer_type}")
        self.mlp = MLP(cfg, tp_degree, all_reduce)
        self.input_layernorm_weight = _frozen(cfg.hidden_size, dtype=dtype)
        self.post_attention_layernorm_weight = _frozen(cfg.hidden_size, dtype=dtype)

    def forward(self, x, cos, sin, position=None, active=None, slot=None, start=None, n_valid=None, accepted=None, pages=None):
        eps = self.cfg.rms_norm_eps
        h = rms_norm(x, self.input_layernorm_weight, eps)
        if accepted is not None:
            mixer = self.linear_attn.verify(h, position, active, accepted) if self.layer_type == LINEAR else self.self_attn.verify(h, cos, sin, position, active, pages)
        elif slot is not None:
            mixer = self.linear_attn.prefill(h, slot, start, n_valid) if self.layer_type == LINEAR else self.self_attn.prefill(h, cos, sin, slot, start, n_valid, pages)
        elif self.layer_type == LINEAR:
            mixer = self.linear_attn(h, position, active)
        else:
            mixer = self.self_attn(h, cos, sin, position, active, pages)
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
        start: int32 ``[1]``, the chunk's first position. 0 starts a new sequence.
        n_valid: int32 ``[1]``, how many of the chunk's tokens are real. The rest are padding.

    Verify, K tokens per row for speculative decoding (inputs built by `generate.verify_inputs`):
        inputs_embeds: ``[batch, K, hidden]``: each row's last chosen token, then K - 1 drafted ones.
        cos, sin: ``[batch, K, rotary_dim]``, for each row's positions `position` onward.
        position, active: as for decode, for the first of the K tokens.
        accepted: int32 ``[batch]``, how many of the row's last verify step's tokens to commit first, 0 after a
            prefill or decode step.

    Saving or restoring one row's recurrent states (inputs built by `generate.state_copy_inputs`), so a prompt that
    extends one already prefilled can start where that one ended:
        inputs_embeds: ``[1, 1, hidden]``, returned as it is, so blocks chain as for the other calls.
        save_slot or restore_slot: int32 ``[1]``, the row.

    With `pages`, the KV cache is a pool of that many pages (`FullAttentionStep`), and decode, prefill and verify
    also take pages: int32 ``[batch, max_seq_len / PAGE]``, each row's page for each `PAGE` of its positions, 0
    where it has none.
    """

    def __init__(
        self,
        cfg: TextConfig,
        max_seq_len: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        all_reduce: Callable = identity_all_reduce,
        batch_size: int = 1,
        verify_tokens: int = 0,
        saved_states: bool = False,
        pages: int = 0,
    ):
        super().__init__()
        cfg.validate_tp(tp_degree)
        self.cfg = cfg
        self.layers = nn.ModuleList(
            DecoderLayer(cfg, layer_type, tp_degree, max_seq_len, dtype, all_reduce, batch_size, verify_tokens, saved_states, pages)
            for layer_type in cfg.layer_types
        )

    def forward(
        self, inputs_embeds, cos=None, sin=None, position=None, active=None, slot=None, start=None, n_valid=None, save_slot=None, restore_slot=None,
        accepted=None,
        pages=None,
    ):
        if save_slot is not None or restore_slot is not None:
            for layer in self.layers:
                if layer.layer_type == LINEAR:
                    layer.linear_attn.copy_states(save_slot if save_slot is not None else restore_slot, to_saved=save_slot is not None)
            return inputs_embeds
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, cos, sin, position, active, slot, start, n_valid, accepted, pages)
        return x
