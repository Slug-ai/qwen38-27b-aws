#!/usr/bin/env python3
"""The throughput benchmark with a short report: one example answer, then batch size, tokens, time and tok/s.

Runs the same work as batch_generate.py, which writes the full results: two full batches of distinct
prompts, 32 new tokens each, greedy, with the saved greedy prompt in row 0, whose output has to match.
Load time is reported but not counted in the throughput.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch

from qwen38_inf2.runner import encode, generate, load
from scripts.batch_generate import benchmark_prompts


def per_user_rate(times: list[list[float]]) -> float:
    """Each sequence's own tokens per second, from its first new token to its last, averaged over sequences.

    This is the speed one user sees once their answer starts. The wait for the first token isn't in it.
    """
    rates = [(len(t) - 1) / (t[-1] - t[0]) for t in times if len(t) > 1 and t[-1] > t[0]]
    return sum(rates) / len(rates) if rates else 0.0


def measure(step, head, cfg, prompts, batch_size: int, max_new_tokens: int, max_seq_len: int) -> tuple[list[list[int]], dict]:
    """Run the prompts through the batch, and return each one's new tokens and the numbers for the report."""
    times = [[] for _ in prompts]
    outputs, seconds = generate(
        step, head, cfg, prompts, batch_size, max_new_tokens, max_seq_len, on_token=lambda index, _: times[index].append(time.perf_counter())
    )
    generated = sum(len(tokens) for tokens in outputs)
    return outputs, {
        "batch_size": batch_size,
        "sequences": len(prompts),
        "generated_tokens": generated,
        "seconds": seconds,
        "tokens_per_second": generated / seconds,
        "per_user_tokens_per_second": per_user_rate(times),
    }


def report(example_input: str, example_output: str, numbers: dict, load_seconds: float, correct: bool) -> str:
    return "\n".join(
        [
            "",
            f"Input:  {example_input}",
            f"Output: {example_output}",
            "",
            f"Batch size        {numbers['batch_size']}",
            f"Sequences         {numbers['sequences']}",
            f"Generated tokens  {numbers['generated_tokens']}",
            f"Total time        {numbers['seconds']:.2f} s, after {load_seconds:.0f} s of loading",
            f"Throughput        {numbers['tokens_per_second']:.2f} tok/s",
            f"Per user          {numbers['per_user_tokens_per_second']:.2f} tok/s",
            f"Correct           {'yes' if correct else 'NO: the saved greedy prompt in row 0 gave different tokens'}",
        ]
    )


def main():
    # The Neuron libraries print pages of deprecation warnings while loading, and none of them matter here.
    warnings.showwarning = lambda *args, **kwargs: None
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--head", help="Compiled lm_head for the chip. Its weights go in --weights too. Omit to run lm_head on the host.")
    parser.add_argument("--prompts-file", type=Path, help="One prompt per line. Defaults to the fixed benchmark set.")
    parser.add_argument("--sequences", type=int, help="Total sequences to run. Defaults to two full batches.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--golden", type=Path, default=Path("tests/golden/model.json"), help="Saved greedy result, checked in row 0.")
    parser.add_argument("--simulate", action="store_true", help="Run on CPU instead of a chip.")
    args = parser.parse_args()

    loaded = load(args.model_dir, args.compiled, args.weights, args.head, args.simulate)
    lines = args.prompts_file.read_text().splitlines() if args.prompts_file else benchmark_prompts()
    lines = [line for line in lines if line.strip()]
    sequences = args.sequences or 2 * loaded.batch_size
    if sequences < 2 or sequences > len(lines) + 1:
        parser.error(f"--sequences must be from 2 to {len(lines) + 1}, the number of prompts plus row 0's saved prompt")
    golden = json.loads(args.golden.read_text())
    prompts = [golden["prompt"]] + encode(loaded.tokenizer, lines[: sequences - 1])

    with torch.no_grad():
        outputs, numbers = measure(loaded.step, loaded.head, loaded.cfg, prompts, loaded.batch_size, args.max_new_tokens, loaded.max_seq_len)
    correct = outputs[0][: len(golden["output"])] == golden["output"]
    example = loaded.tokenizer.decode(outputs[1], skip_special_tokens=True)
    print(report(lines[0], example, numbers, loaded.load_seconds, correct))


if __name__ == "__main__":
    main()
