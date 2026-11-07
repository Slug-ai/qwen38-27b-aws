#!/usr/bin/env python3
"""The multi-turn agentic benchmark: coding-agent sessions whose turns each send the whole conversation again.

As Claude Code does, each turn's prompt is the last one plus the agent's tool call and the file it read back. By
default 8 sessions run, of up to 12 turns and 30K tokens each, with up to 128 new tokens per turn, greedy. The
agent's replies are scripted reads of the next file rather than what the model wrote, so every build sees the
same prompts; the model's own replies are generated and counted all the same.

A session's turns run in one slot, and a turn's prefill starts from the slot's states saved at the end of the
last turn's prompt, so only new tokens are prefilled. `--no-reuse` prefills every prompt whole, which gives the
same tokens up to rounding, since the chunks then start at other positions. Session 0 plants a key in its first file and its last turn asks for it.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch

from qwen38_inf2.agentic import NEEDLE, multi_turn_sessions
from qwen38_inf2.generate import EOS_TOKEN_IDS
from qwen38_inf2.runner import load
from qwen38_inf2.scheduler import Scheduler


def main():
    warnings.showwarning = lambda *args, **kwargs: None
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--head", help="Compiled lm_head for the chip. Its weights go in --weights too.")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-prompt-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-reuse", action="store_true", help="Prefill every turn's prompt from the start.")
    parser.add_argument("--no-speculation", action="store_true", help="Decode one token per step even if the build has a verify graph.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    loaded = load(args.model_dir, args.compiled, args.weights, args.head)
    usable = loaded.max_seq_len - loaded.chunk
    if args.max_prompt_tokens + args.max_new_tokens > usable:
        parser.error(f"prompts of {args.max_prompt_tokens} tokens plus {args.max_new_tokens} new need more than the build's {usable} positions")
    sessions = multi_turn_sessions(loaded.tokenizer, args.sessions, args.max_prompt_tokens, args.max_turns)
    scheduler = Scheduler(
        loaded.step, loaded.prefill, loaded.head, loaded.cfg, loaded.batch_size, loaded.max_seq_len, loaded.chunk, EOS_TOKEN_IDS,
        copy_states=loaded.step, reuse_prefixes=not args.no_reuse,
        verify=None if args.no_speculation else loaded.verify, verify_tokens=loaded.verify_tokens, pages=loaded.pages,
    )
    outputs: dict[tuple[int, int], list[int]] = {}
    started = time.perf_counter()
    with torch.no_grad():
        for session, turn, token in scheduler.run_sessions(sessions, args.max_new_tokens):
            outputs.setdefault((session, turn), []).append(token)
    seconds = time.perf_counter() - started

    prompt_tokens = sum(len(p) for turns in sessions for p in turns)
    generated = sum(len(o) for o in outputs.values())
    answer = loaded.tokenizer.decode(outputs[(0, len(sessions[0]) - 1)], skip_special_tokens=True)
    numbers = {
        "sessions": len(sessions),
        "turns": sum(len(t) for t in sessions),
        "batch_size": loaded.batch_size,
        "max_seq_len": loaded.max_seq_len,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_prefilled": scheduler.prefilled_tokens,
        "prompt_tokens_reused": scheduler.reused_tokens,
        "generated_tokens": generated,
        "wall_seconds": seconds,
        "input_tokens_per_second": prompt_tokens / seconds,
        "output_tokens_per_second": generated / seconds,
        "total_tokens_per_second": (prompt_tokens + generated) / seconds,
        "prefill_calls": scheduler.prefill_calls,
        "decode_calls": scheduler.decode_calls,
        "prefill_seconds": scheduler.prefill_seconds,
        "decode_seconds": scheduler.decode_seconds,
        "mean_rows_per_decode_step": scheduler.decode_rows / max(scheduler.decode_calls, 1),
        "verify_tokens": scheduler.step_tokens,
        "drafted_tokens_accepted": scheduler.drafted_tokens,
        "needle_found": NEEDLE in answer,
        "reuse": not args.no_reuse,
        "outputs": {f"{s},{t}": o for (s, t), o in sorted(outputs.items())},
    }
    print("\n".join([
        "",
        f"Last turn of session 0: {answer!r}",
        "",
        f"Sessions          {numbers['sessions']}, {numbers['turns']} turns, batch {numbers['batch_size']} x {numbers['max_seq_len']}",
        f"Prompt tokens     {prompt_tokens} sent, {scheduler.prefilled_tokens} prefilled, {scheduler.reused_tokens} reused from saved states",
        f"Generated tokens  {generated}",
        f"Total time        {seconds:.1f} s, after {loaded.load_seconds:.0f} s of loading",
        f"Input             {numbers['input_tokens_per_second']:.1f} tok/s",
        f"Output            {numbers['output_tokens_per_second']:.2f} tok/s",
        f"Total             {numbers['total_tokens_per_second']:.1f} tok/s",
        f"Split             prefill {scheduler.prefill_seconds:.1f} s over {scheduler.prefill_calls} calls, decode {scheduler.decode_seconds:.1f} s over {scheduler.decode_calls} steps of {numbers['mean_rows_per_decode_step']:.2f} rows",
        f"Key recalled      {'yes' if numbers['needle_found'] else 'no'}",
    ]))
    if args.output:
        args.output.write_text(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
