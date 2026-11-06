"""The agentic benchmark's prompts and numbers."""

from types import SimpleNamespace

import pytest

from qwen38_inf2.agentic import NEEDLE, Request, benchmark_requests, prompt_digest, stdlib_files
from scripts.agentic_benchmark import summarize


class ByteTokenizer:
    """One token per byte, so lengths are easy to reason about without the real tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(text.encode()))

    def decode(self, ids):
        return bytes(ids).decode()


@pytest.fixture(scope="module")
def requests():
    return benchmark_requests(ByteTokenizer(), 6, 20000, 120000)


def test_lengths_are_spread_from_the_minimum_to_the_maximum(requests):
    lengths = [len(r.ids) for r in requests]
    targets = [20000 + 100000 * i // 5 for i in range(6)]
    assert lengths == sorted(lengths)
    for got, want in zip(lengths, targets):
        assert want - 2000 < got <= want


def test_only_the_longest_request_carries_the_needle_and_asks_for_it(requests):
    text = [ByteTokenizer().decode(r.ids) for r in requests]
    assert [r.has_needle for r in requests] == [False] * 5 + [True]
    assert [NEEDLE in t for t in text] == [False] * 5 + [True]
    # Planted in the first file read, so it sits near the start of the context.
    assert text[-1].index(NEEDLE) < len(text[-1]) // 4
    assert text[-1].rstrip().endswith("<think>\n\n</think>")


def test_requests_read_different_files_and_are_reproducible(requests):
    assert len({r.files[0] for r in requests}) == len(requests)
    again = benchmark_requests(ByteTokenizer(), 6, 20000, 120000)
    assert prompt_digest(again) == prompt_digest(requests)
    assert len(stdlib_files()) > 50


def test_summary_counts_input_and_output_over_the_same_time():
    reqs = [Request([1] * 1000, [], False), Request([1] * 3000, [], True)]
    numbers = summarize(reqs, [[5] * 10, [5] * 30], 2.0, True, True, batch_size=4, max_seq_len=8192)
    assert numbers["prompt_tokens"] == 4000 and numbers["generated_tokens"] == 40
    assert numbers["input_tokens_per_second"] == 2000
    assert numbers["output_tokens_per_second"] == 20
    assert numbers["total_tokens_per_second"] == 2020
    assert numbers["requests_per_hour"] == 3600
