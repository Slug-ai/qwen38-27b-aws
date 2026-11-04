#!/usr/bin/env python3
"""Greedy generation. Runs on Inf2, or with --simulate on CPU from the same weight files."""

import argparse
import sys
import time

import torch
from tokenizers import Tokenizer

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import TextConfig
from qwen38_inf2.generate import EOS_TOKEN_IDS, HostHead, chat_prompt, greedy_generate


def timed(step):
    """Log each step's time, since per-token latency is the number that matters during bring-up."""
    count = [0]

    def run(**inputs):
        started = time.perf_counter()
        hidden = step(**inputs)
        print(f"[step {count[0]}: {time.perf_counter() - started:.2f}s]", file=sys.stderr, flush=True)
        count[0] += 1
        return hidden

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory holding model.safetensors, config.json, tokenizer.json.")
    parser.add_argument("--compiled", required=True, help="Output directory of scripts/compile.py.")
    parser.add_argument("--weights", required=True, help="Output directory of scripts/prepare_weights.py.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--raw", action="store_true", help="Use the prompt as is, without the chat template.")
    parser.add_argument("--thinking", action="store_true", help="Let the model think before answering.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--simulate", action="store_true", help="Run on CPU, one thread per core. Needs no Neuron SDK.")
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16", help="Only float32 needs --simulate.")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if dtype != torch.bfloat16 and not args.simulate:
        parser.error("the compiled graph is BF16, so --dtype float32 needs --simulate")

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    info = checkpoint.read_build_info(args.compiled)
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    text = args.prompt if args.raw else chat_prompt(args.prompt, args.thinking)
    prompt_ids = tokenizer.encode(text, add_special_tokens=False).ids

    source = checkpoint.TensorSource(f"{args.model_dir}/model.safetensors")
    head = HostHead(source, cfg.rms_norm_eps)

    started = time.perf_counter()
    if args.simulate:
        from qwen38_inf2.cpu_sim import SimulatedCoresStep

        step = SimulatedCoresStep(args.weights, cfg, info, dtype)
    else:
        from qwen38_inf2.neuron import NeuronStep

        step = NeuronStep(args.compiled, args.weights, cfg)
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    first_token_at = None
    generated = []
    tokens = greedy_generate(timed(step), head, cfg, prompt_ids, args.max_new_tokens, info["max_seq_len"], EOS_TOKEN_IDS, dtype)
    for token_id in tokens:
        if first_token_at is None:
            first_token_at = time.perf_counter()
        generated.append(token_id)
        print(tokenizer.decode([token_id], skip_special_tokens=False), end="", flush=True)
    finished = time.perf_counter()

    print()
    print(f"prompt: {len(prompt_ids)} tokens in {first_token_at - started:.1f}s")
    if len(generated) > 1:
        print(f"decode: {(len(generated) - 1) / (finished - first_token_at):.3f} tokens/s")


if __name__ == "__main__":
    main()
