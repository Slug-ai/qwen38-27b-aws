#!/usr/bin/env python3
"""Check BF16 rounding on real weights, using the simulated cores.

Runs the passage from verify_real_weights.py in BF16, the chip's dtype, and compares each next-token
prediction with that script's float32 results. Takes about 50 seconds per token on a Mac.
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from verify_real_weights import DEFAULT_TEXT

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import TextConfig
from qwen38_inf2.cpu_sim import SimulatedCoresStep
from qwen38_inf2.generate import HostHead, step_inputs

DTYPE = torch.bfloat16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", type=Path, default=Path("docs/real_weights_check.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    info = checkpoint.read_build_info(args.compiled)
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    tokens = tokenizer.encode(DEFAULT_TEXT, add_special_tokens=False).ids
    reference = json.loads(args.reference.read_text())["predictions"]
    if len(reference) != len(tokens):
        raise ValueError(f"{len(reference)} reference predictions for {len(tokens)} tokens")

    head = HostHead(checkpoint.TensorSource(f"{args.model_dir}/model.safetensors"), cfg.rms_norm_eps)
    step = SimulatedCoresStep(args.weights, cfg, info, DTYPE)

    results = []
    for position, token_id in enumerate(tokens):
        hidden = step(inputs_embeds=head.embed(token_id, DTYPE), **step_inputs(cfg, position, info["max_seq_len"], DTYPE))
        predicted = tokenizer.decode([int(torch.argmax(head.logits(hidden)))])
        results.append({"position": position, "bf16": predicted, "float32": reference[position]["predicted"]})
        match = "same" if predicted == reference[position]["predicted"] else "DIFFERENT"
        print(f"{position:3d} {tokenizer.decode([token_id])!r:14} bf16={predicted!r:14} float32={reference[position]['predicted']!r:14} {match}", flush=True)

    agree = sum(r["bf16"] == r["float32"] for r in results)
    print(f"BF16 and float32 agree at {agree} of {len(results)} positions")
    if args.output:
        args.output.write_text(json.dumps({"agree": agree, "positions": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
