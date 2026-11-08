"""Serve requests as they arrive, through slug-api's OpenAI-compatible API (https://github.com/Slug-ai/slug-api).

`ChipBackend` is slug-api's backend for a `scheduler.Scheduler`. It runs `Scheduler.run_sessions`' loop with a queue
that fills while it runs: each round prefills waiting requests into free slots, then runs one decode or verify step
over every busy slot, with the scheduler's own prefill and verify steps, so prefix reuse and speculation work as in
the benchmarks. A request goes to the free slot whose saved prompt it extends, when one does, as a session's next turn
would. The chip runs in a worker thread, so the API keeps answering while it works.
"""

import asyncio

from slug_api import Queue

from .generate import batch_step_inputs
from .scheduler import Scheduler, _Row


def chat_prompt(messages: list[dict]) -> str:
    """The checkpoint's chat template (`generate.chat_prompt`) for any list of messages, thinking off."""
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return turns + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class ChipBackend:
    def __init__(self, scheduler: Scheduler, max_tokens: int):
        self.chip = scheduler
        self.queue = Queue(max_tokens, scheduler.eos)
        self.max_tokens = max_tokens
        # A step writes up to step_tokens positions, and the cache's last chunk of positions takes junk writes.
        self.context = scheduler.max_seq_len - scheduler.chunk - scheduler.step_tokens + 1
        self.ready = True
        self.on_step = lambda seqs: None
        self.rows: list[_Row | None] = [None] * scheduler.batch_size
        self.seqs = [None] * scheduler.batch_size
        self.work = asyncio.Event()

    async def fill(self) -> None:
        self.work.set()

    async def run(self) -> None:
        while True:
            for slot, seq in enumerate(self.seqs):
                if seq and seq.done:  # cancelled
                    self.rows[slot] = self.seqs[slot] = None
            new = []
            while self.queue.waiting and None in self.rows:
                seq = self.queue.waiting.popleft()
                slot = self.slot_for(seq.prompt)
                self.rows[slot], self.seqs[slot] = _Row((seq.id, 0), list(seq.prompt), len(seq.prompt)), seq
                new.append(slot)
            if not any(self.rows):
                self.work.clear()
                await self.work.wait()
                continue
            limits = {slot: seq.max_tokens for slot, seq in enumerate(self.seqs) if seq}
            results = await asyncio.to_thread(self.step, new, limits)
            stepped = []
            for slot, (tokens, finished) in results.items():
                seq = self.seqs[slot]
                if seq.done:  # cancelled while the chip worked
                    continue
                seq.tokens += tokens
                seq.done = finished
                stepped.append(seq)
                if finished:
                    self.rows[slot] = self.seqs[slot] = None
            self.on_step(stepped)

    def slot_for(self, prompt: list[int]) -> int:
        """A free slot, the one whose saved prompt is the longest start of `prompt` if any is."""
        free = [slot for slot, row in enumerate(self.rows) if row is None]
        saved = self.chip.saved

        def reused(slot: int) -> int:
            n = len(saved[slot])
            return n if self.chip.reuse_prefixes and 0 < n < len(prompt) and prompt[:n] == saved[slot] else 0

        return max(free, key=reused)

    def step(self, new: list[int], limits: dict[int, int]) -> dict[int, tuple[list[int], bool]]:
        """Prefill the `new` slots, then one decode or verify step over every slot still going: each slot's new
        tokens, and whether it has finished. Runs in a worker thread."""
        import torch

        chip, rows, out = self.chip, self.rows, {}
        with torch.no_grad():
            for slot in new:
                row = rows[slot]
                token = chip._prefill(slot, row)
                row.new = 1
                row.ids.append(token)
                out[slot] = ([token], chip._finished(row, token, limits[slot]))
            ready = [slot for slot, row in enumerate(rows) if row and not self.seqs[slot].done and not out.get(slot, ([], False))[1]]
            if not ready:
                return out
            slots = [rows[slot] if slot in ready else None for slot in range(len(rows))]
            positions = [row.written if row else 0 for row in slots]
            active = [row is not None for row in slots]
            if chip.verify:
                emitted = chip._verify_step(slots, positions, active, ready)
            else:
                last = [row.ids[-1] if row else 0 for row in slots]
                hidden = chip.decode(inputs_embeds=chip.head.embed_batch(last, chip.dtype), **batch_step_inputs(chip.cfg, positions, chip.max_seq_len, chip.dtype, active))
                emitted = [[token] for token in chip.head.next_tokens(hidden[ready])]
            for slot, tokens in zip(ready, emitted):
                row, kept, finished = rows[slot], [], False
                for token in tokens:
                    row.written += 1
                    row.new += 1
                    row.ids.append(token)
                    kept.append(token)
                    if finished := chip._finished(row, token, limits[slot]):
                        break
                earlier = out.get(slot, ([], False))[0]
                out[slot] = (earlier + kept, finished)
        return out

