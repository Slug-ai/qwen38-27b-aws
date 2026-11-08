#!/usr/bin/env python3
"""The agentic benchmark: coding-agent requests with long contexts and short answers, as Claude Code sends.

Each request is a system prompt with tools, a task, several standard library files read back as tool results,
and a question about them (see `qwen38_inf2.agentic`). By default 16 requests run, from 8K to 30K prompt tokens,
with up to 256 new tokens each, greedy. Input, output and total tokens per second are all reported, because
agent traffic is mostly input.

Two checks must pass, so a fast but broken build can't score: the saved greedy prompt must give its saved
tokens, and the longest request must find a key planted in its first file.
"""

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import torch

from qwen38_inf2.agentic import NEEDLE, Request, benchmark_requests, prompt_digest
from qwen38_inf2.generate import EOS_TOKEN_IDS
from qwen38_inf2.runner import generate, load
from qwen38_inf2.scheduler import Scheduler


def run(loaded, prompts, max_new_tokens: int, use_prefill: bool = True, eos=EOS_TOKEN_IDS, speculate: bool = True) -> tuple[list[list[int]], float, dict]:
    """With a prefill graph, prompts go through it in chunks. Without one, they're fed token by token through decode."""
    if not use_prefill or getattr(loaded, "prefill", None) is None:
        outputs, seconds = generate(loaded.step, loaded.head, loaded.cfg, prompts, loaded.batch_size, max_new_tokens, loaded.max_seq_len)
        return outputs, seconds, {"prompt_path": "decode"}
    scheduler = Scheduler(
        loaded.step, loaded.prefill, loaded.head, loaded.cfg, loaded.batch_size, loaded.max_seq_len, loaded.chunk, eos,
        verify=loaded.verify if speculate else None, verify_tokens=loaded.verify_tokens, mtp=getattr(loaded, "mtp", None),
    )
    outputs = [[] for _ in prompts]
    started = time.perf_counter()
    for index, token_id in scheduler.run(prompts, max_new_tokens):
        outputs[index].append(token_id)
    seconds = time.perf_counter() - started
    return outputs, seconds, {
        "prompt_path": "prefill",
        "prefill_calls": scheduler.prefill_calls,
        "decode_calls": scheduler.decode_calls,
        "prefill_seconds": scheduler.prefill_seconds,
        "decode_seconds": scheduler.decode_seconds,
        "mean_rows_per_decode_step": scheduler.decode_rows / max(scheduler.decode_calls, 1),
        "verify_tokens": scheduler.step_tokens,
        "drafted_tokens_accepted": scheduler.drafted_tokens,
    }


def summarize(requests, outputs, seconds: float, golden_ok: bool, needle_found: bool, batch_size: int, max_seq_len: int) -> dict:
    prompt_tokens = sum(len(r.ids) for r in requests)
    generated = sum(len(o) for o in outputs)
    return {
        "requests": len(requests),
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_min": min(len(r.ids) for r in requests),
        "prompt_tokens_max": max(len(r.ids) for r in requests),
        "generated_tokens": generated,
        "wall_seconds": seconds,
        "input_tokens_per_second": prompt_tokens / seconds,
        "output_tokens_per_second": generated / seconds,
        "total_tokens_per_second": (prompt_tokens + generated) / seconds,
        "requests_per_hour": len(requests) * 3600 / seconds,
        "golden_row_matches": golden_ok,
        "needle_found": needle_found,
        # Greedy answers end where the model stops, so a numerics change can change how much is generated.
        "generated_per_request": [len(o) for o in outputs],
        "outputs": [list(o) for o in outputs],
    }


def report(numbers: dict, load_seconds: float, answer: str) -> str:
    ok = numbers["golden_row_matches"] and numbers["needle_found"]
    return "\n".join(
        [
            "",
            f"Longest request's answer: {answer!r}",
            "",
            f"Requests          {numbers['requests']}, prompts of {numbers['prompt_tokens_min']} to {numbers['prompt_tokens_max']} tokens",
            f"Batch size        {numbers['batch_size']}, max {numbers['max_seq_len']} tokens per sequence",
            f"Tokens            {numbers['prompt_tokens']} in, {numbers['generated_tokens']} out",
            f"Total time        {numbers['wall_seconds']:.1f} s, after {load_seconds:.0f} s of loading",
            f"Input             {numbers['input_tokens_per_second']:.1f} tok/s",
            f"Output            {numbers['output_tokens_per_second']:.2f} tok/s",
            f"Total             {numbers['total_tokens_per_second']:.1f} tok/s",
            f"Requests          {numbers['requests_per_hour']:.0f} per hour",
            *(
                [f"Split             prefill {numbers['prefill_seconds']:.1f} s over {numbers['prefill_calls']} calls, decode {numbers['decode_seconds']:.1f} s over {numbers['decode_calls']} steps of {numbers['mean_rows_per_decode_step']:.2f} rows"]
                if "prefill_seconds" in numbers
                else []
            ),
            f"Correct           {'yes' if ok else 'NO: ' + ('saved greedy prompt differs' if not numbers['golden_row_matches'] else 'needle not found')}",
        ]
    )


def main():
    warnings.showwarning = lambda *args, **kwargs: None
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--head", help="Compiled lm_head for the chip. Its weights go in --weights too.")
    parser.add_argument("--mtp", help="Compiled MTP drafter (scripts/compile.py --mtp): speculate with its drafts. Its weights go in --weights too.")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--min-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-prompt-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--golden", type=Path, default=Path("tests/golden/model.json"), help="Saved greedy result, run as an extra request.")
    parser.add_argument("--output", type=Path, help="Write the results as JSON here.")
    parser.add_argument("--simulate", action="store_true", help="Run on CPU instead of a chip.")
    parser.add_argument("--no-prefill", action="store_true", help="Feed prompts token by token through decode even if the build has a prefill graph.")
    parser.add_argument("--no-speculation", action="store_true", help="Decode one token per step even if the build has a verify graph.")
    parser.add_argument(
        "--ignore-eos", action="store_true",
        help="Generate exactly --max-new-tokens for every request, so two builds do the same work even where their greedy answers stop differently.",
    )
    args = parser.parse_args()

    loaded = load(args.model_dir, args.compiled, args.weights, args.head, args.simulate, mtp_dir=args.mtp)
    longest = args.max_prompt_tokens + args.max_new_tokens
    # The cache's last positions take junk writes: one for decode, a chunk with a prefill graph.
    usable = loaded.max_seq_len - (loaded.chunk or 1)
    if longest > usable:
        parser.error(f"prompts of {args.max_prompt_tokens} tokens plus {args.max_new_tokens} new need {longest} positions, the build has {usable}")

    requests = benchmark_requests(loaded.tokenizer, args.requests, args.min_prompt_tokens, args.max_prompt_tokens)
    golden = json.loads(args.golden.read_text())
    prompts = [r.ids for r in requests] + [golden["prompt"]]

    with torch.no_grad():
        outputs, seconds, path = run(
            loaded, prompts, args.max_new_tokens, not args.no_prefill, () if args.ignore_eos else EOS_TOKEN_IDS, not args.no_speculation
        )
    golden_ok = outputs[-1][: len(golden["output"])] == golden["output"]
    needle_index = next(i for i, r in enumerate(requests) if r.has_needle)
    answer = loaded.tokenizer.decode(outputs[needle_index], skip_special_tokens=True)
    # The golden request is counted too, since its tokens were really processed.
    counted = requests + [Request(golden["prompt"], [], False)]
    numbers = summarize(counted, outputs, seconds, golden_ok, NEEDLE in answer, loaded.batch_size, loaded.max_seq_len)
    numbers |= path | {
        "load_seconds": loaded.load_seconds, "prompt_digest": prompt_digest(requests), "python": platform.python_version(), "ignore_eos": args.ignore_eos,
    }
    print(report(numbers, loaded.load_seconds, answer))
    if args.output:
        args.output.write_text(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
