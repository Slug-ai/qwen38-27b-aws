#!/usr/bin/env python3
"""Load the model once, then keep taking prompts, on Inf2 or with --simulate on CPU.

Loading takes about 3 minutes, so this keeps one process running instead of paying for it on every run.
At the prompt, type a message to see the answer as it's generated, or `@prompts.txt` to run a file of
prompts, one per line, together. `quit` or Ctrl-D exits, and Ctrl-C stops a generation in progress.

The batch has a fixed number of rows. One message at a time uses one row, so it runs at the chip's step
rate, about 3.5 tokens/s for the batch 63 build, and each prompt token takes a step too. A file of prompts
fills the rows, which is where the throughput comes from.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from qwen38_inf2.runner import encode, generate, load


class Streamer:
    """Write one sequence's text as its tokens arrive, holding back a character that isn't complete yet."""

    def __init__(self, tokenizer, out):
        self.tokenizer = tokenizer
        self.out = out
        self.ids: list[int] = []
        self.written = 0

    def __call__(self, index: int, token_id: int) -> None:
        self.ids.append(token_id)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if not text.endswith("�") and len(text) > self.written:
            self.out.write(text[self.written :])
            self.out.flush()
            self.written = len(text)


def repl(
    step,
    head,
    cfg,
    tokenizer,
    batch_size: int,
    max_seq_len: int,
    max_new_tokens: int,
    read_line: Callable[[str], str] = input,
    out=sys.stdout,
) -> None:
    """Take prompts until `quit` or the end of input."""
    while True:
        try:
            line = read_line("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            out.write("\n")
            return
        if not line:
            continue
        if line in ("quit", "exit"):
            return

        if line.startswith("@"):
            path = Path(line[1:].strip()).expanduser()
            try:
                lines = [text for text in path.read_text().splitlines() if text.strip()]
            except OSError as error:
                out.write(f"{error}\n")
                continue
        else:
            lines = [line]
        if not lines:
            out.write(f"no prompts in {path}\n")
            continue
        prompts = encode(tokenizer, lines)
        too_long = [i + 1 for i, prompt in enumerate(prompts) if len(prompt) >= max_seq_len]
        if too_long:
            out.write(f"prompt plus answer must fit in {max_seq_len} tokens, and these prompts don't: {too_long}\n")
            continue

        streamer = Streamer(tokenizer, out) if len(lines) == 1 else None
        try:
            outputs, seconds = generate(step, head, cfg, prompts, batch_size, max_new_tokens, max_seq_len, streamer)
        except KeyboardInterrupt:
            out.write("\n(stopped)\n")
            continue

        generated = sum(len(tokens) for tokens in outputs)
        if streamer:
            out.write(f"\n\n[{generated} tokens in {seconds:.1f}s]\n")
            continue
        for number, (text, tokens) in enumerate(zip(lines, outputs), start=1):
            out.write(f"\n[{number}] {text}\n{tokenizer.decode(tokens, skip_special_tokens=True)}\n")
        rate = generated / seconds if seconds else 0.0
        out.write(f"\n[{len(lines)} prompts, {generated} tokens in {seconds:.1f}s: {rate:.2f} generated tok/s]\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--head", help="Compiled lm_head for the chip. Its weights go in --weights too. Omit to run lm_head on the host.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Most tokens to generate per prompt. Prompt plus answer stops at the compiled maximum length anyway.")
    parser.add_argument("--simulate", action="store_true", help="Run on CPU instead of a chip.")
    args = parser.parse_args()

    loaded = load(args.model_dir, args.compiled, args.weights, args.head, args.simulate)
    print("ready. Type a message, or @file.txt for a file of prompts, one per line. quit or Ctrl-D exits.", flush=True)
    with torch.no_grad():
        repl(loaded.step, loaded.head, loaded.cfg, loaded.tokenizer, loaded.batch_size, loaded.max_seq_len, args.max_new_tokens)


if __name__ == "__main__":
    main()
