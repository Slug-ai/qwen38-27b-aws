"""Run many requests through a prefill graph and a decode graph that share one set of slots.

The decode graph runs one token for every slot. The prefill graph runs one chunk of one slot's prompt, so long
prompts cost far less per token than feeding them through decode. Both read and write the same per-slot
state and KV cache on the device.

Policy, for throughput: fill every free slot with a fully prefilled request before running a decode step. Prefill
work is fixed by the prompts, so total time is lowest when each decode step has as many rows as possible.
Waiting rows lose latency, not throughput.

Sessions, as a coding agent sends them, are turns whose prompts each extend the last. A session's next turn runs
in the slot its last turn used, and when a slot's saved states match the start of a new prompt, its prefill starts
from there: the KV cache still holds those positions, and the recurrent states were saved at the end of the prompt
they belong to.
"""

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import torch

from .config import TextConfig
from .drafting import Drafter
from .generate import batch_step_inputs, mtp_prefill_inputs, mtp_step_inputs, prefill_inputs, state_copy_inputs, verify_inputs


@dataclass
class _Row:
    index: tuple[int, int]
    ids: list[int]
    prompt_len: int
    new: int = 0
    # Tokens already written to the slot's cache.
    written: int = 0
    tokens: list[int] = field(default_factory=list)
    # Drafts for speculative decoding, from the row's history so far.
    drafter: Drafter | None = None
    # With an MTP drafter, where its next drafts start: the main model's hidden states ``[n, hidden]`` at the
    # row's last tokens, the tokens after them, the first one's MTP index, and which of them to draft from.
    mtp_seed: tuple | None = None


class Scheduler:
    """Drive `prefill` and `decode` for a fixed number of slots.

    Args:
        decode: the decode graph, ``(inputs_embeds [B, 1, H], cos, sin, position, active) -> [B, 1, H]``.
        prefill: the prefill graph, ``(inputs_embeds [1, C, H], cos, sin, slot, start, n_valid) -> [1, C, H]``.
        head: has ``embed_batch(ids, dtype)`` and ``next_tokens(hidden [rows, 1, H])``, as `generate.ChipHead`.
        chunk: C, the prefill graph's tokens per call.
        copy_states: the graph that saves or restores one slot's recurrent states, called with
            `generate.state_copy_inputs`. Without one, or with `reuse_prefixes` False, every prompt is prefilled whole.
        mtp: the MTP drafter's graph (`model.MTPStep`): with it and `verify`, drafts come from it instead of from the
            row's history, and every prompt chunk also goes through it, to fill its cache.
        verify, verify_tokens: with these, every decode step is a verify step of `verify_tokens` tokens per row
            (`generate.verify_inputs`): each row's last token and the next ones drafted from its own history by
            `drafting.lookup_draft`. The drafted tokens the model would have chosen itself are kept, with the
            model's next choice after them, so each step adds 1 to `verify_tokens` tokens per row and the tokens
            are the same as plain greedy decoding gives. Its head must take ``batch * verify_tokens`` rows.
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
        copy_states: Callable[..., torch.Tensor] | None = None,
        reuse_prefixes: bool = True,
        verify: Callable[..., torch.Tensor] | None = None,
        verify_tokens: int = 1,
        mtp: Callable[..., torch.Tensor] | None = None,
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
        self.verify = verify
        self.step_tokens = verify_tokens if verify else 1
        # Per slot, how many of its last verify step's tokens the next one commits.
        self.accepted = [0] * batch_size
        self.drafted_tokens = 0
        self.copy_states = copy_states
        self.reuse_prefixes = reuse_prefixes and copy_states is not None
        # Per slot, the prompt whose recurrent states at its end it has saved.
        self.saved: list[list[int]] = [[] for _ in range(batch_size)]
        self.reused_tokens = 0
        self.prefilled_tokens = 0
        self.prefill_calls = 0
        self.decode_calls = 0
        # Where the time goes: prefill calls including the first token's head, and decode steps including theirs.
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self.decode_rows = 0
        self.mtp = mtp if verify else None
        # Per slot, the main model's hidden state at its prompt's last token, for a later turn that extends it.
        self.mtp_last: list[torch.Tensor | None] = [None] * batch_size

    def _prefill(self, slot: int, row: _Row) -> int:
        """Write the prompt to `slot` in chunks, from the slot's saved states when they match its start, and return
        the first new token."""
        hidden = None
        started = time.perf_counter()
        saved = self.saved[slot]
        begin = len(saved) if self.reuse_prefixes and saved and len(saved) < row.prompt_len and row.ids[: len(saved)] == saved else 0
        self.reused_tokens += begin
        if begin:
            self.copy_states(**state_copy_inputs(self.cfg, slot, False, self.dtype))
        last = self.mtp_last[slot] if begin else None
        for start in range(begin, row.prompt_len, self.chunk):
            ids = row.ids[start : start + self.chunk]
            n_valid = len(ids)
            embeds = self.head.embed_batch(ids + [0] * (self.chunk - n_valid), self.dtype).reshape(1, self.chunk, -1)
            hidden = self.prefill(inputs_embeds=embeds, **prefill_inputs(self.cfg, slot, start, n_valid, self.chunk, self.dtype))
            self.prefill_calls += 1
            self.prefilled_tokens += n_valid
            if self.mtp:
                last = self._mtp_prefill(slot, start, n_valid, embeds, hidden, last)
        # The states at the prompt's end are saved, for a later prompt that extends this one. Padding past a
        # chunk's real tokens leaves them as they were, so the last chunk may be part full.
        if self.reuse_prefixes:
            self.copy_states(**state_copy_inputs(self.cfg, slot, True, self.dtype))
            self.saved[slot] = row.ids[: row.prompt_len]
        self.accepted[slot] = 0
        row.written = row.prompt_len
        token_id = self.head.next_tokens(hidden[:, n_valid - 1 : n_valid])[0]
        if self.mtp:
            # The first drafts start from the prompt's last hidden state and the token chosen after it.
            self.mtp_last[slot] = last
            row.mtp_seed = (last[0], [token_id], row.prompt_len - 1, 0)
        self.prefill_seconds += time.perf_counter() - started
        return token_id

    def _mtp_prefill(self, slot: int, start: int, n_valid: int, embeds: torch.Tensor, hidden: torch.Tensor, last) -> torch.Tensor:
        """Put a prompt chunk through the MTP drafter, to fill its cache: its index t takes the main model's hidden
        state at t and token t + 1, so the chunk at `start` covers indices from `start - 1`, starting from the last
        chunk's final hidden state `last`, and its tokens are the chunk's own (`embeds`). The first chunk has no
        index before it, so it covers one token fewer. Positions past the real ones take junk, as padding does in
        prefill. Returns this chunk's final hidden state."""
        if start == 0 or last is None:
            first, states, embeds = start, hidden[:, : n_valid - 1], torch.cat((embeds[:, 1:], embeds[:, :1]), dim=1)
        else:
            first, states = start - 1, torch.cat((last, hidden[:, : n_valid - 1]), dim=1)
        if states.shape[1]:
            states = torch.cat((states, torch.zeros(1, self.chunk - states.shape[1], states.shape[-1], dtype=states.dtype)), dim=1)
            self.mtp(hidden=states, inputs_embeds=embeds, **mtp_prefill_inputs(self.cfg, slot, first, n_valid - (first == start), self.chunk, self.dtype))
        return hidden[:, n_valid - 1 : n_valid]

    def _mtp_drafts(self, slots) -> dict[int, list[int]]:
        """K - 1 drafts for every running slot from the MTP drafter: one step over each row's last tokens with the
        main model's hidden states, which also caches them, whose output at the last accepted one drafts the next
        token, then a step per further draft on its own output and the token it drafted."""
        k, batch = self.step_tokens, self.batch_size
        size = self.cfg.hidden_size
        states = torch.zeros(batch, k, size, dtype=self.dtype)
        tokens = [[0] * k for _ in range(batch)]
        index, take, active = [0] * batch, [0] * batch, [False] * batch
        for slot, row in enumerate(slots):
            if row and row.mtp_seed:
                hidden, ids, index[slot], take[slot] = row.mtp_seed
                states[slot, : hidden.shape[0]] = hidden
                tokens[slot][: len(ids)] = ids
                active[slot] = True
        ready = [slot for slot in range(batch) if active[slot]]
        drafts = {slot: [] for slot in ready}
        rows = torch.arange(batch)
        embeds = self.head.embed_batch([t for row in tokens for t in row], self.dtype).reshape(batch, k, -1)
        out = self.mtp(hidden=states, inputs_embeds=embeds, **mtp_step_inputs(self.cfg, index, k, self.max_seq_len, self.dtype, active))
        current = out[rows, torch.tensor(take)]
        for j in range(k - 1):
            # A draft needn't be exact, since verify keeps only what the model picks, so the chip's best group serves.
            for slot, token in zip(ready, self.head.next_tokens(current[ready].unsqueeze(1), recheck_groups=1)):
                drafts[slot].append(token)
            if j == k - 2:
                break
            states = torch.zeros(batch, k, size, dtype=self.dtype)
            states[:, 0] = current
            ids = [[drafts[slot][-1] if active[slot] else 0] + [0] * (k - 1) for slot in range(batch)]
            embeds = self.head.embed_batch([t for row in ids for t in row], self.dtype).reshape(batch, k, -1)
            at = [index[slot] + take[slot] + 1 + j for slot in range(batch)]
            current = self.mtp(hidden=states, inputs_embeds=embeds, **mtp_step_inputs(self.cfg, at, k, self.max_seq_len, self.dtype, active))[:, 0]
        return drafts

    def _finished(self, row: _Row, token_id: int, max_new_tokens: int) -> bool:
        # The next step would write positions from row.written. The cache's last `chunk` positions take junk writes.
        return token_id in self.eos or row.new >= max_new_tokens or row.written + self.step_tokens > self.max_seq_len - self.chunk

    def run(self, prompts: Sequence[Sequence[int]], max_new_tokens: int) -> Iterator[tuple[int, int]]:
        """Yield (prompt index, token id) as tokens are chosen."""
        for session, _, token_id in self.run_sessions([[prompt] for prompt in prompts], max_new_tokens):
            yield session, token_id

    def run_sessions(self, sessions: Sequence[Sequence[Sequence[int]]], max_new_tokens: int) -> Iterator[tuple[int, int, int]]:
        """Yield (session, turn, token id) as tokens are chosen. Each session is a list of turn prompts, and a turn
        starts once the one before it has finished, in the same slot."""
        for turns in sessions:
            for prompt in turns:
                if not prompt:
                    raise ValueError("a prompt is empty")
                if len(prompt) >= self.max_seq_len - self.chunk:
                    raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in max_seq_len {self.max_seq_len}")
        waiting = list(range(len(sessions)))[::-1]
        slots: list[_Row | None] = [None] * self.batch_size
        # Per slot, the next turn of the session that last ran there.
        follow: list[tuple[int, int] | None] = [None] * self.batch_size

        def done(slot: int, row: _Row) -> None:
            session, turn = row.index
            if turn + 1 < len(sessions[session]):
                follow[slot] = (session, turn + 1)

        while waiting or any(slots) or any(follow):
            # Prefill first: every free slot gets a request before the next decode step, its own session's next
            # turn before a new session.
            for slot in range(self.batch_size):
                while slots[slot] is None and (follow[slot] or waiting):
                    if follow[slot]:
                        (session, turn), follow[slot] = follow[slot], None
                    else:
                        session, turn = waiting.pop(), 0
                    row = _Row((session, turn), list(sessions[session][turn]), len(sessions[session][turn]))
                    token_id = self._prefill(slot, row)
                    row.new = 1
                    row.ids.append(token_id)
                    yield session, turn, token_id
                    if self._finished(row, token_id, max_new_tokens):
                        done(slot, row)
                    else:
                        slots[slot] = row

            if not any(slots):
                continue
            positions = [row.written if row else 0 for row in slots]
            active = [row is not None for row in slots]
            ready = [slot for slot, row in enumerate(slots) if row]
            started = time.perf_counter()
            if self.verify:
                emitted = self._verify_step(slots, positions, active, ready)
            else:
                last = [row.ids[-1] if row else 0 for row in slots]
                hidden = self.decode(inputs_embeds=self.head.embed_batch(last, self.dtype), **batch_step_inputs(self.cfg, positions, self.max_seq_len, self.dtype, active))
                emitted = [[token] for token in self.head.next_tokens(hidden[ready])]
            self.decode_calls += 1
            self.decode_seconds += time.perf_counter() - started
            self.decode_rows += len(ready)
            for slot, tokens in zip(ready, emitted):
                row = slots[slot]
                for token_id in tokens:
                    row.written += 1
                    row.new += 1
                    row.ids.append(token_id)
                    yield *row.index, token_id
                    if self._finished(row, token_id, max_new_tokens):
                        slots[slot] = None
                        done(slot, row)
                        break

    def _verify_step(self, slots, positions, active, ready) -> list[list[int]]:
        """One verify step for every running slot. Returns each ready slot's new tokens: the drafted ones the model
        chose too, then its own next choice."""
        k = self.step_tokens
        drafts, tokens = (self._mtp_drafts(slots) if self.mtp else {}), []
        for slot, row in enumerate(slots):
            if row:
                if not self.mtp:
                    if row.drafter is None:
                        row.drafter = Drafter(row.ids)
                    else:
                        row.drafter.extend(row.ids[len(row.drafter.tokens) :])
                    draft = row.drafter.draft(k - 1)
                    drafts[slot] = draft + [row.ids[-1]] * (k - 1 - len(draft))
                tokens += [row.ids[-1]] + drafts[slot]
            else:
                tokens += [0] * k
        accepted = [self.accepted[slot] if slots[slot] else 0 for slot in range(self.batch_size)]
        embeds = self.head.embed_batch(tokens, self.dtype).reshape(self.batch_size, k, -1)
        hidden = self.verify(inputs_embeds=embeds, **verify_inputs(self.cfg, positions, k, self.max_seq_len, self.dtype, active, accepted))
        chosen = self.head.next_tokens(hidden[ready].reshape(len(ready) * k, 1, -1))
        out = []
        for i, slot in enumerate(ready):
            picks, draft = chosen[i * k : (i + 1) * k], drafts[slot]
            match = 0
            while match < k - 1 and draft[match] == picks[match]:
                match += 1
            self.accepted[slot] = match + 1
            self.drafted_tokens += match
            out.append(picks[: match + 1])
            if self.mtp:
                slots[slot].mtp_seed = (hidden[slot], picks, positions[slot], match)
        return out
