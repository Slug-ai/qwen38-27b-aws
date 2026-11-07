"""Run many requests through a prefill graph and a decode graph that share one set of slots.

The decode graph runs one token for every slot. The prefill graph runs one chunk of one slot's prompt, so long
prompts cost far less per token than feeding them through decode. Both read and write the same per-slot
state and KV cache on the device.

Policy, for throughput: fill every free slot with a fully prefilled request before running a decode step. Prefill
work is fixed by the prompts, so total time is lowest when each decode step has as many rows as possible.
Waiting rows lose latency, not throughput.
"""

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import torch

from .config import TextConfig
from .generate import batch_step_inputs, prefill_inputs


@dataclass
class _Row:
    index: int
    ids: list[int]
    prompt_len: int
    new: int = 0
    # Tokens already written to the slot's cache.
    written: int = 0
    tokens: list[int] = field(default_factory=list)


class Scheduler:
    """Drive `prefill` and `decode` for a fixed number of slots.

    Args:
        decode: the decode graph, ``(inputs_embeds [B, 1, H], cos, sin, position, active) -> [B, 1, H]``.
        prefill: the prefill graph, ``(inputs_embeds [1, C, H], cos, sin, slot, start, n_valid) -> [1, C, H]``.
        head: has ``embed_batch(ids, dtype)`` and ``next_tokens(hidden [rows, 1, H])``, as `generate.ChipHead`.
        chunk: C, the prefill graph's tokens per call. Chunks start at multiples of it.
    """

    def __init__(
        self,
        decode: Callable[..., torch.Tensor],
        prefill: Callable[..., torch.Tensor],
        head,
        cfg: TextConfig,
        batch_size: int,
        max_seq_len: int,
        chunk: int,
        eos_token_ids: Sequence[int] = (),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.decode = decode
        self.prefill = prefill
        self.head = head
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.chunk = chunk
        self.eos = set(eos_token_ids)
        self.dtype = dtype
        self.prefill_calls = 0
        self.decode_calls = 0
        # Where the time goes: prefill calls including the first token's head, and decode steps including theirs.
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self.decode_rows = 0

    def _prefill(self, slot: int, row: _Row) -> int:
        """Write the whole prompt to `slot` in chunks, and return the first new token."""
        hidden = None
        started = time.perf_counter()
        for start in range(0, row.prompt_len, self.chunk):
            ids = row.ids[start : start + self.chunk]
            n_valid = len(ids)
            embeds = self.head.embed_batch(ids + [0] * (self.chunk - n_valid), self.dtype).reshape(1, self.chunk, -1)
            hidden = self.prefill(inputs_embeds=embeds, **prefill_inputs(self.cfg, slot, start, n_valid, self.chunk, self.dtype))
            self.prefill_calls += 1
        row.written = row.prompt_len
        token_id = self.head.next_tokens(hidden[:, n_valid - 1 : n_valid])[0]
        self.prefill_seconds += time.perf_counter() - started
        return token_id

    def _finished(self, row: _Row, token_id: int, max_new_tokens: int) -> bool:
        # The next decode step would write position row.written. The cache's last `chunk` positions take junk writes.
        return token_id in self.eos or row.new >= max_new_tokens or row.written >= self.max_seq_len - self.chunk

    def run(self, prompts: Sequence[Sequence[int]], max_new_tokens: int) -> Iterator[tuple[int, int]]:
        """Yield (prompt index, token id) as tokens are chosen."""
        for prompt in prompts:
            if not prompt:
                raise ValueError("a prompt is empty")
            if len(prompt) >= self.max_seq_len - self.chunk:
                raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in max_seq_len {self.max_seq_len}")
        waiting = list(range(len(prompts)))[::-1]
        slots: list[_Row | None] = [None] * self.batch_size

        while waiting or any(slots):
            # Prefill first: every free slot gets a request before the next decode step.
            for slot in range(self.batch_size):
                while slots[slot] is None and waiting:
                    index = waiting.pop()
                    row = _Row(index, list(prompts[index]), len(prompts[index]))
                    token_id = self._prefill(slot, row)
                    row.new = 1
                    row.ids.append(token_id)
                    yield index, token_id
                    if not self._finished(row, token_id, max_new_tokens):
                        slots[slot] = row

            if not any(slots):
                continue
            positions = [row.written if row else 0 for row in slots]
            active = [row is not None for row in slots]
            last = [row.ids[-1] if row else 0 for row in slots]
            started = time.perf_counter()
            hidden = self.decode(inputs_embeds=self.head.embed_batch(last, self.dtype), **batch_step_inputs(self.cfg, positions, self.max_seq_len, self.dtype, active))
            self.decode_calls += 1
            ready = [slot for slot, row in enumerate(slots) if row]
            chosen = self.head.next_tokens(hidden[ready])
            self.decode_seconds += time.perf_counter() - started
            self.decode_rows += len(ready)
            for slot, token_id in zip(ready, chosen):
                row = slots[slot]
                row.written += 1
                row.new += 1
                row.ids.append(token_id)
                yield row.index, token_id
                if self._finished(row, token_id, max_new_tokens):
                    slots[slot] = None
