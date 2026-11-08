"""Requests arriving while the chip works must get exactly the tokens they would get alone."""

import asyncio

import pytest
import torch

from qwen38_inf2.config import tiny_config
from qwen38_inf2.scheduler import Scheduler
from qwen38_inf2.serving import ChipBackend
from .test_scheduler import CHUNK, EOS, FakeDevice, FakeHead, alone


@pytest.mark.parametrize("batch_size", [1, 3])
def test_requests_arriving_mid_run_get_the_tokens_they_would_alone(batch_size):
    max_seq_len = 40
    torch.manual_seed(0)
    prompts = [torch.randint(1, 50, (n,)).tolist() for n in [1, 7, 9, 16, 17, 30, 3, 25]]
    device = FakeDevice(batch_size, max_seq_len)
    backend = ChipBackend(Scheduler(device.decode, device.prefill, FakeHead(device), tiny_config(), batch_size, max_seq_len, CHUNK, [EOS], torch.float32), 12)

    async def run():
        running = asyncio.create_task(backend.run())
        seqs = []
        for prompt in prompts:  # a request every few steps, as an API would hand them over
            seqs.append(backend.queue.add(prompt))
            await backend.fill()
            await asyncio.sleep(0.01)
        while not all(s.done for s in seqs):
            await asyncio.sleep(0.01)
        running.cancel()
        return [s.tokens for s in seqs]

    assert asyncio.run(asyncio.wait_for(run(), 30)) == [alone(p, 12, max_seq_len) for p in prompts]
