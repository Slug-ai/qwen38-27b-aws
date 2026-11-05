"""scripts/benchmark.py's numbers, on the tiny CPU model."""

import pytest
import torch

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import tiny_config
from qwen38_inf2.generate import HostHead
from qwen38_inf2.runner import generate
from scripts.benchmark import measure, per_user_rate, report

from .reference import build_reference
from .test_demo import MAX_SEQ_LEN, PROMPT, batch_model


def test_per_user_rate_averages_each_sequence_from_its_first_token_to_its_last():
    # 2 tokens a second, then 1 a second, and a sequence with one token has no rate to count.
    times = [[10.0, 10.5, 11.0, 11.5], [0.0, 1.0, 2.0], [7.0]]
    assert per_user_rate(times) == pytest.approx(1.5)
    assert per_user_rate([[1.0]]) == 0.0


def test_measure_counts_every_generated_token():
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    head = HostHead(checkpoint.DictSource(tensors), cfg.rms_norm_eps, chunk_rows=50)
    prompts = [PROMPT, [9, 9, 9, 1, 2], [4], [77, 78, 79]]
    with torch.no_grad():
        want, _ = generate(batch_model(tensors, cfg, 2), head, cfg, prompts, 2, 6, MAX_SEQ_LEN)
        outputs, numbers = measure(batch_model(tensors, cfg, 2), head, cfg, prompts, 2, 6, MAX_SEQ_LEN)

    assert outputs == want
    assert numbers["batch_size"] == 2
    assert numbers["sequences"] == 4
    assert numbers["generated_tokens"] == 4 * 6
    assert numbers["tokens_per_second"] == pytest.approx(24 / numbers["seconds"])
    # Each row adds one token per step, and two rows share a step, so one user sees under half the total rate.
    assert 0 < numbers["per_user_tokens_per_second"] < numbers["tokens_per_second"]

    text = report("hello", "world", numbers, 190.3, correct=True)
    assert "Input:  hello\nOutput: world\n" in text
    assert "Batch size        2\n" in text
    assert "after 190 s of loading" in text
    assert text.endswith("Correct           yes")
