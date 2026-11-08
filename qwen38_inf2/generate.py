"""Host side of inference: embeddings, lm_head and the greedy loop.

The compiled graph only maps hidden to hidden, so the same loop drives the CPU model in tests and the
Neuron model in production. Anything callable with the `Qwen38Step` inputs works as `step`.
"""

from collections.abc import Callable, Iterator, Sequence

import torch

from . import checkpoint
from .config import TextConfig
from .model import rms_norm

# From the checkpoint's generation_config.json: <|im_end|> and <|endoftext|>.
EOS_TOKEN_IDS = (248046, 248044)


def chat_prompt(user_message: str, thinking: bool = False) -> str:
    """The checkpoint's chat template, for a single user message."""
    opening = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{opening}"


def rotary(cfg: TextConfig, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin for each position, ``[len(positions), rotary_dim]``. With text-only input the T/H/W grids
    coincide, so MRoPE is plain RoPE."""
    dim = cfg.rotary_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq
    return freqs.cos().repeat(1, 2).to(dtype), freqs.sin().repeat(1, 2).to(dtype)


def step_inputs(cfg: TextConfig, position: int, max_seq_len: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return batch_step_inputs(cfg, [position], max_seq_len, dtype)


def batch_step_inputs(
    cfg: TextConfig, positions: Sequence[int], max_seq_len: int, dtype: torch.dtype, active: Sequence[bool] | None = None
) -> dict[str, torch.Tensor]:
    """Decode inputs with one row per sequence, each at its own position. Rows with `active` False are left alone.

    The cache's last position is where inactive rows write, so a sequence can use `max_seq_len - 1` positions.
    """
    position = torch.as_tensor(positions, dtype=torch.int32)
    on = torch.ones(len(positions), dtype=torch.bool) if active is None else torch.as_tensor(active, dtype=torch.bool)
    if bool((position[on] >= max_seq_len - 1).any()):
        raise ValueError(f"position {int(position[on].max())} is past the last usable one of max_seq_len {max_seq_len}")
    cos, sin = rotary(cfg, position, dtype)
    return {
        "cos": cos.unsqueeze(1),
        "sin": sin.unsqueeze(1),
        "position": position,
        "active": on.float(),
    }


def prefill_inputs(cfg: TextConfig, slot: int, start: int, n_valid: int, chunk: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Prefill inputs for `n_valid` tokens of one row from position `start`, padded to a full chunk."""
    if start < 0 or not 0 < n_valid <= chunk:
        raise ValueError(f"a chunk must start at a position and hold 1 to {chunk} tokens")
    cos, sin = rotary(cfg, start + torch.arange(chunk), dtype)
    return {
        "cos": cos.unsqueeze(0),
        "sin": sin.unsqueeze(0),
        "slot": torch.tensor([slot], dtype=torch.int32),
        "start": torch.tensor([start], dtype=torch.int32),
        "n_valid": torch.tensor([n_valid], dtype=torch.int32),
    }


def verify_inputs(
    cfg: TextConfig, positions: Sequence[int], tokens: int, max_seq_len: int, dtype: torch.dtype, active: Sequence[bool], accepted: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Verify inputs for `tokens` tokens per row from each row's position (`model.Qwen38Step`)."""
    position = torch.as_tensor(positions, dtype=torch.int32)
    on = torch.as_tensor(active, dtype=torch.bool)
    if bool((position[on] + tokens > max_seq_len - 1).any()):
        raise ValueError(f"a row's {tokens} tokens from position {int(position[on].max())} pass max_seq_len {max_seq_len}")
    cos, sin = rotary(cfg, (position.unsqueeze(-1) + torch.arange(tokens)).reshape(-1), dtype)
    return {
        "cos": cos.reshape(len(positions), tokens, -1),
        "sin": sin.reshape(len(positions), tokens, -1),
        "position": position,
        "active": on.float(),
        "accepted": torch.as_tensor(accepted, dtype=torch.int32),
    }


def mtp_step_inputs(
    cfg: TextConfig, indices: Sequence[int], tokens: int, max_seq_len: int, dtype: torch.dtype, active: Sequence[bool]
) -> dict[str, torch.Tensor]:
    """Inputs for `model.MTPStep` steps: `tokens` per row from each row's MTP index, at positions one on."""
    inputs = verify_inputs(cfg, indices, tokens, max_seq_len, dtype, active, [0] * len(indices))
    del inputs["accepted"]
    at = (torch.as_tensor(indices, dtype=torch.int32).unsqueeze(-1) + 1 + torch.arange(tokens)).reshape(-1)
    cos, sin = rotary(cfg, at, dtype)
    inputs["cos"], inputs["sin"] = cos.reshape(len(indices), tokens, -1), sin.reshape(len(indices), tokens, -1)
    return inputs


def mtp_prefill_inputs(cfg: TextConfig, slot: int, start: int, n_valid: int, chunk: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Inputs for a `model.MTPStep` prefill chunk of one row from MTP index `start`, at positions one on."""
    inputs = prefill_inputs(cfg, slot, start, n_valid, chunk, dtype)
    cos, sin = rotary(cfg, start + 1 + torch.arange(chunk), dtype)
    inputs["cos"], inputs["sin"] = cos.unsqueeze(0), sin.unsqueeze(0)
    return inputs


def state_copy_inputs(cfg: TextConfig, slot: int, save: bool, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Inputs that save one row's recurrent states, or restore them (`model.Qwen38Step`)."""
    return {
        "inputs_embeds": torch.zeros(1, 1, cfg.hidden_size, dtype=dtype),
        ("save_slot" if save else "restore_slot"): torch.tensor([slot], dtype=torch.int32),
    }


def prefill_prompt(prefill, head, cfg: TextConfig, slot: int, prompt_ids: Sequence[int], chunk: int, dtype: torch.dtype) -> torch.Tensor:
    """Write a whole prompt into `slot` a chunk at a time, and return the last token's hidden state ``[1, 1, hidden]``."""
    hidden = None
    for start in range(0, len(prompt_ids), chunk):
        ids = list(prompt_ids[start : start + chunk])
        embeds = head.embed_batch(ids + [0] * (chunk - len(ids)), dtype).reshape(1, chunk, -1)
        hidden = prefill(inputs_embeds=embeds, **prefill_inputs(cfg, slot, start, len(ids), chunk, dtype))
    last = (len(prompt_ids) - 1) % chunk
    return hidden[:, last : last + 1]


class HostHead:
    """Embedding lookup, final norm and lm_head, kept off the device to save 5 GB of its memory.

    Built to fit a small host such as inf2.xlarge (16 GiB). Embeddings are read a row at a time from
    the mapped checkpoint, and lm_head is widened in pieces, so peak RAM is about 5 GB.
    """

    def __init__(self, source, eps: float, chunk_rows: int = 32768):
        self.source = source
        self.eps = eps
        self.norm_weight = source.get(checkpoint.FINAL_NORM)
        rows, cols = source.shape(checkpoint.LM_HEAD)
        # float32 because CPU BF16 matmul is far slower.
        self.lm_head = torch.empty(rows, cols, dtype=torch.float32)
        for first in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - first)
            self.lm_head[first : first + count] = source.get_row_range(checkpoint.LM_HEAD, first, count).float()

    def embed(self, token_id: int, dtype: torch.dtype) -> torch.Tensor:
        return self.source.get_rows(checkpoint.EMBED_TOKENS, [token_id]).reshape(1, 1, -1).to(dtype)

    def embed_batch(self, token_ids: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        return self.source.get_rows(checkpoint.EMBED_TOKENS, list(token_ids)).unsqueeze(1).to(dtype)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden.float(), self.norm_weight, self.eps).reshape(-1) @ self.lm_head.T

    def next_tokens(self, hidden: torch.Tensor, recheck_groups: int | None = None) -> list[int]:
        """Greedy next token for each row of ``[rows, 1, hidden]``. `recheck_groups` is for `ChipHead`'s call."""
        norm = rms_norm(hidden.float(), self.norm_weight, self.eps).squeeze(1)
        return torch.argmax(norm @ self.lm_head.T, dim=-1).tolist()


class ChipHead:
    """lm_head on the chip, which is far faster than the host at a large batch and needs no host copy of it.

    The chip reports the largest logit in each group of tokens, and the host picks the exact token inside
    the best few groups. The chip's input is rounded to BF16, so its best group alone could lose a near tie.
    `run_graph` is the compiled `HeadStep`, or anything else with the same call.
    """

    def __init__(
        self, source, eps: float, run_graph: Callable[..., torch.Tensor], batch_size: int, group_rows: int = 32, recheck_groups: int = 4
    ):
        self.source = source
        self.eps = eps
        self.run_graph = run_graph
        self.batch_size = batch_size
        self.group_rows = group_rows
        self.recheck_groups = recheck_groups
        self.norm_weight = source.get(checkpoint.FINAL_NORM)
        # lm_head ``[vocab, hidden]`` over the checkpoint's memory map: rows are read as the recheck slices them.
        self.lm_head = source.matrix_view(checkpoint.LM_HEAD)

    def embed_batch(self, token_ids: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        return self.source.get_rows(checkpoint.EMBED_TOKENS, list(token_ids)).unsqueeze(1).to(dtype)

    def next_tokens(self, hidden: torch.Tensor, recheck_groups: int | None = None) -> list[int]:
        """Greedy next token for each row of ``[rows, 1, hidden]``. Fewer rows than the batch are padded for the graph.

        `recheck_groups` overrides how many of the chip's best groups the host rechecks, as for drafts, which need
        not be exact. Each group is a slice of lm_head straight from the checkpoint's memory map."""
        rows = hidden.shape[0]
        full = torch.zeros(self.batch_size, 1, hidden.shape[-1], dtype=hidden.dtype)
        full[:rows] = hidden
        maxima = self.run_graph(hidden=full)[:rows].float()
        best_groups = maxima.topk(min(recheck_groups or self.recheck_groups, maxima.shape[-1]), dim=-1).indices.tolist()
        norm = rms_norm(hidden.float(), self.norm_weight, self.eps).squeeze(1)
        tokens = []
        for row, groups in enumerate(best_groups):
            best_token, best_logit = -1, -torch.inf
            for group in groups:
                first = group * self.group_rows
                logit, index = (self.lm_head[first : first + self.group_rows].float() @ norm[row]).max(dim=0)
                if logit > best_logit:
                    best_token, best_logit = first + int(index), logit
            if best_token < 0:
                raise FloatingPointError(f"row {row}'s logits are not finite, so the graph produced NaN or infinity")
            tokens.append(best_token)
        return tokens


class BlockChain:
    """Run compiled blocks of layers in sequence, as one step."""

    def __init__(self, blocks: Sequence[Callable[..., torch.Tensor]]):
        self.blocks = blocks

    def __call__(self, inputs_embeds: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs_embeds
        for block in self.blocks:
            hidden = block(inputs_embeds=hidden, **inputs)
        return hidden


def greedy_generate(
    step: Callable[..., torch.Tensor],
    head: HostHead,
    cfg: TextConfig,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    max_seq_len: int,
    eos_token_ids: Sequence[int] = (),
    dtype: torch.dtype = torch.bfloat16,
    prefill: Callable[..., torch.Tensor] | None = None,
    chunk: int | None = None,
) -> Iterator[int]:
    """Yield greedily decoded token ids, one at a time.

    With `prefill`, the prompt goes through the prefill graph `chunk` tokens at a time, into row 0. Otherwise
    each prompt token is a decode step.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids is empty")
    if prefill is not None:
        hidden = prefill_prompt(prefill, head, cfg, 0, prompt_ids, chunk, dtype)
        position = len(prompt_ids)
    else:
        hidden = None
        position = 0
        for token_id in prompt_ids:
            hidden = step(inputs_embeds=head.embed(token_id, dtype), **step_inputs(cfg, position, max_seq_len, dtype))
            position += 1

    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(head.logits(hidden)))
        yield next_id
        if next_id in eos_token_ids or position >= max_seq_len - 1:
            return
        hidden = step(inputs_embeds=head.embed(next_id, dtype), **step_inputs(cfg, position, max_seq_len, dtype))
        position += 1


def batched_greedy_generate(
    step: Callable[..., torch.Tensor],
    head: HostHead,
    cfg: TextConfig,
    prompts: Sequence[Sequence[int]],
    batch_size: int,
    max_new_tokens: int,
    max_seq_len: int,
    eos_token_ids: Sequence[int] = (),
    dtype: torch.dtype = torch.bfloat16,
) -> Iterator[tuple[int, int]]:
    """Run many prompts through a fixed batch, and yield (prompt index, token id) as tokens are chosen.

    A row that finishes takes the next waiting prompt on the following step. Its first token clears the
    row's state, and an idle row is inactive, so rows never affect each other.
    """
    if any(not prompt for prompt in prompts):
        raise ValueError("a prompt is empty")
    waiting = list(range(len(prompts)))[::-1]
    rows: list[dict | None] = [None] * batch_size

    while waiting or any(rows):
        for r in range(batch_size):
            if rows[r] is None and waiting:
                index = waiting.pop()
                rows[r] = {"index": index, "ids": list(prompts[index]), "position": 0, "new": 0}

        positions = [row["position"] if row else 0 for row in rows]
        token_ids = [row["ids"][row["position"]] if row else 0 for row in rows]
        active = [row is not None for row in rows]
        hidden = step(inputs_embeds=head.embed_batch(token_ids, dtype), **batch_step_inputs(cfg, positions, max_seq_len, dtype, active))

        # Only rows that have fed their last known token need lm_head, which is the costly part on the host.
        ready = []
        for r, row in enumerate(rows):
            if row:
                row["position"] += 1
                if row["position"] == len(row["ids"]):
                    ready.append(r)
        if ready:
            for r, token_id in zip(ready, head.next_tokens(hidden[ready])):
                row = rows[r]
                row["ids"].append(token_id)
                row["new"] += 1
                yield row["index"], token_id
                if token_id in eos_token_ids or row["new"] >= max_new_tokens or row["position"] >= max_seq_len - 1:
                    rows[r] = None
