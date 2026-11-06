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
