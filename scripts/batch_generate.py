#!/usr/bin/env python3
"""Run many prompts through a batched graph and report throughput. Runs on Inf2, or with --simulate on CPU."""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import TextConfig
from qwen38_inf2.generate import EOS_TOKEN_IDS, HostHead, batched_greedy_generate, chat_prompt

DEFAULT_PROMPTS = [
    "What is the capital of France? Answer in one short sentence, then name one famous landmark there.",
    "Explain why the sky is blue in two sentences.",
    "Write a haiku about a lighthouse in winter.",
    "List three prime numbers greater than 100.",
    "Give one tip for learning to play the piano.",
    "What does a compiler do? Answer briefly.",
    "Name the largest planet in the solar system and one of its moons.",
    "Translate 'good morning, my friend' into Spanish.",
]


class TimedStep:
    """Add up time on the chip, so the rest of the wall time can be put down to the host."""

    def __init__(self, step):
        self.step = step
        self.seconds = 0.0
        self.count = 0

    def __call__(self, **inputs):
        started = time.perf_counter()
        hidden = self.step(**inputs)
        self.seconds += time.perf_counter() - started
        self.count += 1
        return hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prompts-file", type=Path, help="One prompt per line. Defaults to a small built-in set.")
    parser.add_argument("--sequences", type=int, help="Total sequences to run, repeating the prompts. Defaults to one full batch.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--raw", action="store_true", help="Don't apply the chat template.")
    parser.add_argument("--show", type=int, default=8, help="How many outputs to print.")
    parser.add_argument("--simulate", action="store_true", help="Run on CPU instead of a chip.")
    args = parser.parse_args()

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    info = checkpoint.read_build_info(args.compiled)
    batch_size = info.get("batch_size", 1)
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    lines = args.prompts_file.read_text().splitlines() if args.prompts_file else DEFAULT_PROMPTS
    texts = [line if args.raw else chat_prompt(line) for line in lines if line.strip()]
    encoded = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    sequences = args.sequences or batch_size
    prompts = [encoded[i % len(encoded)] for i in range(sequences)]

    head = HostHead(checkpoint.TensorSource(f"{args.model_dir}/model.safetensors"), cfg.rms_norm_eps)
    started = time.perf_counter()
    if args.simulate:
        from qwen38_inf2.cpu_sim import SimulatedCoresStep

        inner = SimulatedCoresStep(args.weights, cfg, info, torch.bfloat16)
    else:
        from qwen38_inf2.neuron import NeuronStep

        inner = NeuronStep(args.compiled, args.weights, cfg)
    print(f"loaded {batch_size}-row graph in {time.perf_counter() - started:.1f}s", flush=True)

    step = TimedStep(inner)
    outputs = defaultdict(list)
    started = time.perf_counter()
    for index, token_id in batched_greedy_generate(
        step, head, cfg, prompts, batch_size, args.max_new_tokens, info["max_seq_len"], EOS_TOKEN_IDS
    ):
        outputs[index].append(token_id)
    wall = time.perf_counter() - started

    for index in range(min(args.show, sequences)):
        print(f"\n[{index}] input:  {lines[index % len(lines)]}")
        print(f"[{index}] output: {tokenizer.decode(outputs[index], skip_special_tokens=True)!r}")

    # A prompt repeated across rows must give the same text every time, unless rows interfere.
    same_prompt = defaultdict(set)
    for index in range(sequences):
        same_prompt[index % len(encoded)].add(tuple(outputs[index]))
    mismatched = sum(len(variants) > 1 for variants in same_prompt.values())

    generated = sum(len(tokens) for tokens in outputs.values())
    fed = sum(len(p) for p in prompts)
    print(f"\nbatch size:           {batch_size}")
    print(f"sequences:            {sequences}")
    print(f"prompt tokens:        {fed}")
    print(f"generated tokens:     {generated}")
    print(f"steps:                {step.count}")
    print(f"chip per step:        {step.seconds / step.count:.3f}s")
    print(f"host per step:        {(wall - step.seconds) / step.count:.3f}s")
    print(f"wall time:            {wall:.1f}s")
    print(f"generated tokens/s:   {generated / wall:.2f}")
    print(f"with every row generating: {batch_size * step.count / wall:.2f} tokens/s")
    print(f"all tokens/s:         {(generated + fed) / wall:.2f}")
    print(f"prompts with differing outputs across rows: {mismatched}")


if __name__ == "__main__":
    main()
