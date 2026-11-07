#!/usr/bin/env python3
"""How surely a build recalls facts from long prompts: a steadier measure than one greedy answer.

Each request plants its own key in a coding-agent prompt of 8K to 30K tokens, near the start or about halfway,
and asks for it. The build reads the prompt, then the key's tokens are fed back one at a time, and the exact
log-probability of each is computed on the host over the whole vocabulary. The summary is the mean over
requests of the key's total log-probability, and how many requests would answer with the key's first token.

A change to rounding, such as fusing ops differently, can flip one greedy answer either way while being no less
accurate. Its mean log-probability should stay within noise of the build it replaces.
"""

import argparse
import json
import warnings
from pathlib import Path

import torch

from qwen38_inf2 import checkpoint
from qwen38_inf2.agentic import long_context_requests
from qwen38_inf2.generate import batch_step_inputs, page_table, prefill_prompt
from qwen38_inf2.model import rms_norm
from qwen38_inf2.runner import load

LENGTHS = (8192, 12288, 16384, 20480, 24576, 27648, 29952, 29952)


def host_logits(head, cfg, hidden: torch.Tensor) -> torch.Tensor:
    """Exact float32 logits over the whole vocabulary for one hidden state ``[1, 1, hidden]``."""
    norm = rms_norm(hidden.float(), head.norm_weight, cfg.rms_norm_eps).reshape(-1)
    rows = head.source.shape(checkpoint.LM_HEAD)[0]
    logits = torch.empty(rows)
    for first in range(0, rows, 16384):
        part = head.source.get_row_range(checkpoint.LM_HEAD, first, min(16384, rows - first)).float()
        logits[first : first + part.shape[0]] = part @ norm
    return logits


def key_logprob(loaded, ids: list[int], answer: list[int]) -> dict:
    """Prefill `ids` into row 0, then feed `answer` through decode, scoring each of its tokens."""
    cfg, dtype = loaded.cfg, torch.bfloat16
    # With a paged cache, row 0 takes the pages the prompt and answer need.
    paged = {"pages": page_table([len(ids) + len(answer) + 1] + [0] * (loaded.batch_size - 1), loaded.max_seq_len, loaded.pages)} if loaded.pages else {}
    hidden = prefill_prompt(lambda **inputs: loaded.prefill(**inputs, **paged), loaded.head, cfg, 0, ids, loaded.chunk, dtype)
    position, total, first = len(ids), 0.0, None
    others = loaded.batch_size - 1
    for index, token in enumerate(answer):
        logits = host_logits(loaded.head, cfg, hidden)
        total += logits.log_softmax(0)[token].item()
        if index == 0:
            first = {"rank": int((logits > logits[token]).sum()) + 1, "p": logits.softmax(0)[token].item()}
        embeds = loaded.head.embed_batch([token] + [0] * others, dtype)
        inputs = batch_step_inputs(cfg, [position] + [0] * others, loaded.max_seq_len, dtype, [True] + [False] * others)
        hidden = loaded.step(inputs_embeds=embeds, **inputs, **paged)[0:1]
        position += 1
    return {"key_logprob": total, "first_token_rank": first["rank"], "first_token_p": first["p"]}


def main():
    warnings.showwarning = lambda *args, **kwargs: None
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--head", help="Compiled lm_head for the chip. Its weights go in --weights too.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    loaded = load(args.model_dir, args.compiled, args.weights, args.head)
    usable = loaded.max_seq_len - (loaded.chunk or 1)
    rows = []
    with torch.no_grad():
        for request, key in long_context_requests(loaded.tokenizer, [n for n in LENGTHS if n + 32 <= usable]):
            answer = loaded.tokenizer.encode(key, add_special_tokens=False).ids
            result = key_logprob(loaded, request.ids, answer) | {"prompt_tokens": len(request.ids), "key": key}
            rows.append(result)
            print(f"{result['prompt_tokens']:6d} tokens  {key:16s}  log p(key) {result['key_logprob']:8.3f}  "
                  f"first token rank {result['first_token_rank']}, p {result['first_token_p']:.3f}", flush=True)
    summary = {
        "build": args.compiled,
        "requests": len(rows),
        "mean_key_logprob": sum(r["key_logprob"] for r in rows) / len(rows),
        "first_token_top": sum(r["first_token_rank"] == 1 for r in rows),
        "rows": rows,
    }
    print(f"\nMean log p(key) {summary['mean_key_logprob']:.3f}; key's first token on top in {summary['first_token_top']} of {len(rows)}")
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
