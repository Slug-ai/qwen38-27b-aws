"""Host side of inference: embeddings, lm_head and the greedy loop.

The compiled graph only maps hidden to hidden, so the same loop drives the CPU model in tests and the
Neuron model in production. Anything callable with the `Qwen38Step` inputs works as `step`.
"""

from collections.abc import Callable, Iterator, Sequence

import torch

from . import checkpoint
from .config import TextConfig
from .model import MASKED_SCORE, rms_norm

# From the checkpoint's generation_config.json: <|im_end|> and <|endoftext|>.
EOS_TOKEN_IDS = (248046, 248044)


def chat_prompt(user_message: str, thinking: bool = False) -> str:
    """The checkpoint's chat template, for a single user message."""
    opening = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{opening}"


def rotary_terms(cfg: TextConfig, position: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin for one position. With text-only input the T/H/W grids coincide, so MRoPE is plain RoPE."""
    dim = cfg.rotary_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = position * inv_freq
    cos, sin = freqs.cos(), freqs.sin()
    return (
        torch.cat((cos, cos)).reshape(1, 1, -1).to(dtype),
        torch.cat((sin, sin)).reshape(1, 1, -1).to(dtype),
    )


def step_inputs(cfg: TextConfig, position: int, max_seq_len: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    if position >= max_seq_len:
        raise ValueError(f"position {position} is past the compiled max_seq_len of {max_seq_len}")
    cos, sin = rotary_terms(cfg, position, dtype)
    write_mask = torch.zeros(1, 1, max_seq_len, 1, dtype=torch.float32)
    write_mask[0, 0, position, 0] = 1.0
    attn_mask = torch.full((1, 1, 1, max_seq_len), MASKED_SCORE, dtype=torch.float32)
    attn_mask[..., : position + 1] = 0.0
    return {
        "cos": cos,
        "sin": sin,
        "write_mask": write_mask,
        "attn_mask": attn_mask,
        "keep_state": torch.tensor([0.0 if position == 0 else 1.0], dtype=torch.float32),
    }


def batch_step_inputs(cfg: TextConfig, positions: Sequence[int], max_seq_len: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """`step_inputs` with one row per sequence, each at its own position."""
    rows = [step_inputs(cfg, position, max_seq_len, dtype) for position in positions]
    return {name: torch.cat([row[name] for row in rows]) for name in rows[0]}


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

    def next_tokens(self, hidden: torch.Tensor) -> list[int]:
        """Greedy next token for each row of ``[rows, 1, hidden]``."""
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

    def embed_batch(self, token_ids: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        return self.source.get_rows(checkpoint.EMBED_TOKENS, list(token_ids)).unsqueeze(1).to(dtype)

    def next_tokens(self, hidden: torch.Tensor) -> list[int]:
        """Greedy next token for each row of ``[rows, 1, hidden]``. Fewer rows than the batch are padded for the graph."""
        rows = hidden.shape[0]
        full = torch.zeros(self.batch_size, 1, hidden.shape[-1], dtype=hidden.dtype)
        full[:rows] = hidden
        maxima = self.run_graph(hidden=full)[:rows].float()
        best_groups = maxima.topk(min(self.recheck_groups, maxima.shape[-1]), dim=-1).indices.tolist()
        norm = rms_norm(hidden.float(), self.norm_weight, self.eps).squeeze(1)
        total = self.source.shape(checkpoint.LM_HEAD)[0]
        tokens = []
        for row, groups in enumerate(best_groups):
            best_token, best_logit = -1, -torch.inf
            for group in groups:
                first = group * self.group_rows
                candidates = self.source.get_row_range(checkpoint.LM_HEAD, first, min(self.group_rows, total - first)).float()
                logit, index = (candidates @ norm[row]).max(dim=0)
                if logit > best_logit:
                    best_token, best_logit = first + int(index), logit
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
) -> Iterator[int]:
    """Yield greedily decoded token ids, one at a time."""
    if not prompt_ids:
        raise ValueError("prompt_ids is empty")
    hidden = None
    position = 0
    for token_id in prompt_ids:
        hidden = step(inputs_embeds=head.embed(token_id, dtype), **step_inputs(cfg, position, max_seq_len, dtype))
        position += 1

    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(head.logits(hidden)))
        yield next_id
        if next_id in eos_token_ids or position >= max_seq_len:
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
    row's state, and an idle row just rewrites position 0 of its own cache, so rows never affect each other.
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
        hidden = step(inputs_embeds=head.embed_batch(token_ids, dtype), **batch_step_inputs(cfg, positions, max_seq_len, dtype))

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
                if token_id in eos_token_ids or row["new"] >= max_new_tokens or row["position"] >= max_seq_len:
                    rows[r] = None
