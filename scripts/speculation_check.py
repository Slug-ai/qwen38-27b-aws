#!/usr/bin/env python3
"""Whether a build's verify graph picks the tokens decode would, teacher-forced on real text.

Each row's prompt, Python source from the standard library, is prefilled, then the same continuation goes through
decode one token at a time and, prefilled again, through verify steps that each commit all K tokens of the last.
Both paths' greedy picks are compared with each other and with the text's actual next tokens.

Verify does the same arithmetic in another order, so the two agree up to rounding: at 184 of 192 picks for the
first verify formulation and 183 for the current one. A verify step that commits the last one's tokens wrongly
agrees less from its second step on, and predicts the text worse than decode does.
"""

import argparse
import json
import sysconfig
import warnings
from pathlib import Path

import torch

from qwen38_inf2.generate import batch_step_inputs, page_table, prefill_prompt, verify_inputs
from qwen38_inf2.runner import load


def texts(tokenizer, rows: int, length: int) -> list[list[int]]:
    """`rows` token sequences of `length` from standard library modules, one module each."""
    out = []
    for path in sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*/*.py")):
        ids = tokenizer.encode(path.read_text(errors="ignore")).ids
        if len(ids) >= length:
            out.append(ids[:length])
        if len(out) == rows:
            return out
    raise ValueError(f"fewer than {rows} standard library modules have {length} tokens")


def main():
    warnings.showwarning = lambda *args, **kwargs: None
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True, help="A build with a verify graph (scripts/compile.py --verify-tokens).")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--head", help="Compiled lm_head for the chip, taking batch x K rows. Its weights go in --weights too.")
    parser.add_argument("--prompt-tokens", type=int, default=1500)
    parser.add_argument("--tokens", type=int, default=48, help="Continuation tokens per row, a multiple of K.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    loaded = load(args.model_dir, args.compiled, args.weights, args.head)
    if not loaded.verify:
        raise SystemExit(f"{args.compiled} has no verify graph")
    cfg, rows, k, dtype = loaded.cfg, loaded.batch_size, loaded.verify_tokens, torch.bfloat16
    prompt, n = args.prompt_tokens, args.tokens - args.tokens % k
    seqs = texts(loaded.tokenizer, rows, prompt + n + 1)
    # A paged cache's rows each take the pages they write.
    paged = {"pages": page_table([prompt + n + k] * rows, loaded.max_seq_len, loaded.pages)} if loaded.pages else {}
    prefill = lambda **inputs: loaded.prefill(**inputs, **paged)  # noqa: E731

    def prefill_all():
        for slot, ids in enumerate(seqs):
            prefill_prompt(prefill, loaded.head, cfg, slot, ids[:prompt], loaded.chunk, dtype)

    with torch.no_grad():
        prefill_all()
        decode = []
        for i in range(n):
            embeds = loaded.head.embed_batch([ids[prompt + i] for ids in seqs], dtype)
            hidden = loaded.step(inputs_embeds=embeds, **batch_step_inputs(cfg, [prompt + i] * rows, loaded.max_seq_len, dtype), **paged)
            decode.append(loaded.head.next_tokens(hidden))
        prefill_all()
        verify = [[] for _ in range(n)]
        for step in range(n // k):
            start = prompt + step * k
            embeds = loaded.head.embed_batch([t for ids in seqs for t in ids[start : start + k]], dtype).reshape(rows, k, -1)
            inputs = verify_inputs(cfg, [start] * rows, k, loaded.max_seq_len, dtype, [True] * rows, [0 if step == 0 else k] * rows)
            picks = loaded.head.next_tokens(loaded.verify(inputs_embeds=embeds, **inputs, **paged).reshape(rows * k, 1, -1))
            for r in range(rows):
                for t in range(k):
                    verify[step * k + t].append(picks[r * k + t])

    truth = [[ids[prompt + i + 1] for ids in seqs] for i in range(n)]
    count = lambda a, b, positions: sum(x == y for i in positions for x, y in zip(a[i], b[i]))  # noqa: E731
    summary = {
        "build": args.compiled,
        "picks": n * rows,
        "same_picks": count(decode, verify, range(n)),
        "same_picks_first_step": count(decode, verify, range(k)),
        "decode_right": count(decode, truth, range(n)),
        "verify_right": count(verify, truth, range(n)),
    }
    print(f"Verify picks what decode picks at {summary['same_picks']} of {summary['picks']} positions "
          f"({summary['same_picks_first_step']} of {k * rows} in the first step); the text's next token: "
          f"decode {summary['decode_right']}, verify {summary['verify_right']}")
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
