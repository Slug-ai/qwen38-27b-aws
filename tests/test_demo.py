"""scripts/demo.py keeps one loaded model and runs prompt after prompt through it."""

import io
from types import SimpleNamespace

import pytest
import torch

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import tiny_config
from qwen38_inf2.generate import HostHead
from qwen38_inf2.model import Qwen38Step
from scripts.demo import encode, generate, repl

from .reference import build_reference

# Long enough for the chat template, which the fake tokenizer splits into one token per character.
MAX_SEQ_LEN = 128
PROMPT = [3, 17, 42, 99, 5, 8, 120, 7, 64, 33]


def batch_model(tensors, cfg, batch_size):
    dtype = torch.bfloat16
    model = Qwen38Step(cfg, MAX_SEQ_LEN, dtype=dtype, batch_size=batch_size).eval()
    model.load_state_dict(checkpoint.rank_weights(checkpoint.DictSource(tensors), cfg, 0, 1, dtype), strict=False)
    return model


class FakeTokenizer:
    """One token per character, kept below the tiny vocabulary, written back as numbers."""

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(c) % 128 for c in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(f"<{i}>" for i in ids)


class Recorded:
    """Keep what each step returns, and raise KeyboardInterrupt on one call if asked, as Ctrl-C would."""

    def __init__(self, step, stop_at: int | None = None):
        self.step = step
        self.stop_at = stop_at
        self.calls = 0
        self.outputs: list[torch.Tensor] = []

    def __call__(self, **inputs):
        self.calls += 1
        if self.calls == self.stop_at:
            raise KeyboardInterrupt
        hidden = self.step(**inputs)
        self.outputs.append(hidden.clone())
        return hidden


@pytest.fixture(scope="module")
def tiny():
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    return cfg, tensors, HostHead(checkpoint.DictSource(tensors), cfg.rms_norm_eps, chunk_rows=50)


def test_later_prompts_get_the_same_answers_as_on_a_fresh_model(tiny):
    """Nothing may carry over from earlier prompts, including a run stopped partway.

    Tokens alone could hide a leak, since the tiny model's top token rarely moves, so every row's hidden
    state at every step must match too.
    """
    cfg, tensors, head = tiny
    first = [PROMPT, [9, 9, 9, 1, 2], [4]]
    later = [[77, 78, 79], PROMPT[::-1], [5, 6]]
    with torch.no_grad():
        fresh = Recorded(batch_model(tensors, cfg, 2))
        want, _ = generate(fresh, head, cfg, later, 2, 6, MAX_SEQ_LEN)
        model = Recorded(batch_model(tensors, cfg, 2), stop_at=30)
        generate(model, head, cfg, first, 2, 6, MAX_SEQ_LEN)
        with pytest.raises(KeyboardInterrupt):
            generate(model, head, cfg, first, 2, 6, MAX_SEQ_LEN)
        assert model.calls == 30
        model.outputs.clear()
        got, _ = generate(model, head, cfg, later, 2, 6, MAX_SEQ_LEN)
    assert got == want
    assert len(model.outputs) == len(fresh.outputs)
    assert all(torch.equal(a, b) for a, b in zip(model.outputs, fresh.outputs))


def test_the_prompt_loop_answers_messages_and_files(tiny, tmp_path):
    cfg, tensors, head = tiny
    tokenizer = FakeTokenizer()
    lines = ["sky", "sea", "sun"]
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("sky\n\nsea\nsun\n")
    with torch.no_grad():
        (want_message,), _ = generate(batch_model(tensors, cfg, 2), head, cfg, encode(tokenizer, ["hi"]), 2, 5, MAX_SEQ_LEN)
        want_file, _ = generate(batch_model(tensors, cfg, 2), head, cfg, encode(tokenizer, lines), 2, 5, MAX_SEQ_LEN)

        typed = iter(["hi", "", "hi", f"@{prompts_file}", "x" * MAX_SEQ_LEN, "@missing.txt", "quit", "never read"])
        out = io.StringIO()
        # The first message is stopped partway, so the second shows the loop recovers from Ctrl-C.
        step = Recorded(batch_model(tensors, cfg, 2), stop_at=3)
        repl(step, head, cfg, tokenizer, 2, MAX_SEQ_LEN, 5, read_line=lambda _: next(typed), out=out)

    text = out.getvalue()
    assert "(stopped)" in text
    assert f"{tokenizer.decode(want_message)}\n\n[5 tokens in" in text
    for number, (line, tokens) in enumerate(zip(lines, want_file), start=1):
        assert f"[{number}] {line}\n{tokenizer.decode(tokens)}\n" in text
    assert "[3 prompts, 15 tokens in" in text
    assert "these prompts don't: [1]" in text
    assert "missing.txt" in text
    assert next(typed) == "never read"
