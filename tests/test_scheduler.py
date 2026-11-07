"""The scheduler must give every request exactly the tokens it would get running alone."""

import pytest
import torch

from qwen38_inf2.config import tiny_config
from qwen38_inf2.scheduler import Scheduler

CHUNK = 8
EOS = 0


def next_token(history: list[int]) -> int:
    """A deterministic stand-in for the model: depends on every token so far and their order."""
    value = len(history)
    for token in history:
        value = (value * 31 + token) % 1009
    return value % 50


class FakeDevice:
    """Keeps each slot's cached tokens, as the real graphs keep state and KV, and checks every write."""

    def __init__(self, batch_size: int, max_seq_len: int):
        self.cache = [[] for _ in range(batch_size)]
        self.saved = [None] * batch_size
        self.max_seq_len = max_seq_len

    def _hidden(self, slot: int, position: int) -> torch.Tensor:
        return torch.tensor([slot, position, 0.0])

    def prefill(self, inputs_embeds, cos, sin, slot, start, n_valid):
        slot, start, n_valid = int(slot[0]), int(start[0]), int(n_valid[0])
        assert inputs_embeds.shape[:2] == (1, CHUNK) and cos.shape[:2] == (1, CHUNK)
        assert start % CHUNK == 0 and 0 < n_valid <= CHUNK
        if start == 0:
            self.cache[slot] = []
        assert len(self.cache[slot]) == start, "a chunk must continue exactly where the last one ended"
        self.cache[slot] += [int(t) for t in inputs_embeds[0, :n_valid, 0]]
        hidden = torch.zeros(1, CHUNK, 3)
        for i in range(n_valid):
            hidden[0, i] = self._hidden(slot, start + i)
        return hidden

    def copy_states(self, inputs_embeds, save_slot=None, restore_slot=None):
        """Saved states stand for exactly the tokens up to where they were saved, at a chunk boundary."""
        if save_slot is not None:
            slot = int(save_slot[0])
            assert len(self.cache[slot]) % CHUNK == 0, "states are saved at a chunk boundary"
            self.saved[slot] = list(self.cache[slot])
        else:
            slot = int(restore_slot[0])
            assert self.saved[slot] is not None, "nothing was saved"
            self.cache[slot] = list(self.saved[slot])
        return inputs_embeds

    def decode(self, inputs_embeds, cos, sin, position, active):
        hidden = torch.zeros(len(self.cache), 1, 3)
        for slot, (pos, on) in enumerate(zip(position.tolist(), active.tolist())):
            if not on:
                continue
            assert pos == len(self.cache[slot]) < self.max_seq_len, "a decode step must append at the next position"
            self.cache[slot].append(int(inputs_embeds[slot, 0, 0]))
            hidden[slot, 0] = self._hidden(slot, pos)
        return hidden


class FakeHead:
    def __init__(self, device: FakeDevice):
        self.device = device

    def embed_batch(self, ids, dtype):
        out = torch.zeros(len(ids), 1, 3, dtype=dtype)
        out[:, 0, 0] = torch.tensor(ids, dtype=dtype)
        return out

    def next_tokens(self, hidden):
        tokens = []
        for slot, position, _ in hidden.reshape(-1, 3).tolist():
            tokens.append(next_token(self.device.cache[int(slot)][: int(position) + 1]))
        return tokens


def alone(prompt: list[int], max_new_tokens: int, max_seq_len: int) -> list[int]:
    ids, out = list(prompt), []
    while True:
        token = next_token(ids)
        out.append(token)
        ids.append(token)
        # Mirrors the scheduler's stop: the next step would write position len(ids) - 1, and the cache's last
        # chunk of positions takes junk writes.
        if token == EOS or len(out) >= max_new_tokens or len(ids) - 1 >= max_seq_len - CHUNK:
            return out


@pytest.mark.parametrize("batch_size", [1, 3])
def test_every_request_gets_the_tokens_it_would_get_alone(batch_size):
    max_seq_len = 40
    torch.manual_seed(0)
    lengths = [1, 7, 8, 9, 16, 17, 30, 31, 3, 25]
    prompts = [torch.randint(1, 50, (n,)).tolist() for n in lengths]
    device = FakeDevice(batch_size, max_seq_len)
    scheduler = Scheduler(device.decode, device.prefill, FakeHead(device), tiny_config(), batch_size, max_seq_len, CHUNK, [EOS], torch.float32)

    got = {i: [] for i in range(len(prompts))}
    for index, token in scheduler.run(prompts, max_new_tokens=12):
        got[index].append(token)

    assert got == {i: alone(p, 12, max_seq_len) for i, p in enumerate(prompts)}
    # Each prompt is prefilled in whole chunks, never fed through decode.
    assert scheduler.prefill_calls == sum(-(-n // CHUNK) for n in lengths)


def test_decode_steps_run_with_every_slot_filled_while_requests_wait():
    device = FakeDevice(4, 64)
    scheduler = Scheduler(device.decode, device.prefill, FakeHead(device), tiny_config(), 4, 64, CHUNK, [], torch.float32)
    rows_per_step = []
    original = device.decode

    def counting(**inputs):
        rows_per_step.append(int(inputs["active"].sum()))
        return original(**inputs)

    scheduler.decode = counting
    list(scheduler.run([[1, 2, 3]] * 12, max_new_tokens=5))
    # 12 requests of 5 tokens through 4 slots: three full rounds of 4 decode steps each.
    assert rows_per_step == [4] * 12


def test_prompts_that_leave_no_room_are_refused():
    device = FakeDevice(2, 16)
    scheduler = Scheduler(device.decode, device.prefill, FakeHead(device), tiny_config(), 2, 16, CHUNK, [], torch.float32)
    with pytest.raises(ValueError):
        list(scheduler.run([[1] * (16 - CHUNK)], max_new_tokens=4))


def session_prompts(seed: int, turns: int) -> list[list[int]]:
    """A session's turn prompts, each the last one plus a scripted reply and a new tool result, as an agent sends them."""
    rng = torch.Generator().manual_seed(seed)
    prompts, ids = [], torch.randint(1, 50, (5 + seed * 3,), generator=rng).tolist()
    for _ in range(turns):
        prompts.append(list(ids))
        ids = ids + torch.randint(1, 50, (int(torch.randint(2, 21, (1,), generator=rng)),), generator=rng).tolist()
    return prompts


@pytest.mark.parametrize("batch_size", [1, 2])
def test_sessions_reuse_saved_prefixes_and_get_the_tokens_they_would_alone(batch_size):
    max_seq_len, max_new = 160, 6
    sessions = [session_prompts(seed, 5) for seed in range(3)]
    got = {}
    calls = {}
    for reuse in (True, False):
        device = FakeDevice(batch_size, max_seq_len)
        scheduler = Scheduler(
            device.decode, device.prefill, FakeHead(device), tiny_config(), batch_size, max_seq_len, CHUNK, [EOS], torch.float32, device.copy_states, reuse
        )
        out = {}
        for session, turn, token in scheduler.run_sessions(sessions, max_new):
            out.setdefault((session, turn), []).append(token)
        got[reuse], calls[reuse] = out, (scheduler.prefill_calls, scheduler.reused_tokens)
    want = {(s, t): alone(p, max_new, max_seq_len) for s, turns in enumerate(sessions) for t, p in enumerate(turns)}
    assert got[True] == want and got[False] == want
    assert calls[False][1] == 0 and calls[True][1] > 0 and calls[True][0] < calls[False][0]


def test_a_slot_never_reuses_states_saved_for_a_different_prompt():
    device = FakeDevice(1, 64)
    scheduler = Scheduler(device.decode, device.prefill, FakeHead(device), tiny_config(), 1, 64, CHUNK, [EOS], torch.float32, device.copy_states)
    first = list(range(1, 20))
    other = [7] + first[1:] + [3, 4]  # differs at its first token, so nothing saved for `first` applies
    got = [t for _, _, t in scheduler.run_sessions([[first], [other], [first + [9, 9]]], 3)]
    assert scheduler.reused_tokens == 0
    assert got == alone(first, 3, 64) + alone(other, 3, 64) + alone(first + [9, 9], 3, 64)



def test_the_drafter_drafts_what_a_scan_of_the_history_drafts():
    from qwen38_inf2.drafting import Drafter, lookup_draft

    rng = torch.Generator().manual_seed(0)
    history = torch.randint(0, 6, (400,), generator=rng).tolist()
    drafter = Drafter(history[:50])
    for end in range(50, 400):
        assert drafter.draft(3) == lookup_draft(history[:end], 3), end
        drafter.extend([history[end]])
