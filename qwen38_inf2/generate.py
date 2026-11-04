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

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden.float(), self.norm_weight, self.eps).reshape(-1) @ self.lm_head.T


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
