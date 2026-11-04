#!/usr/bin/env python3
"""Greedy generation on Inf2. Needs Neuron hardware."""

import argparse
import time

from tokenizers import Tokenizer

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import TextConfig
from qwen38_inf2.generate import EOS_TOKEN_IDS, HostHead, chat_prompt, greedy_generate
from qwen38_inf2.neuron import DTYPE, NeuronStep, read_build_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory holding model.safetensors, config.json, tokenizer.json.")
    parser.add_argument("--compiled", required=True, help="Output directory of scripts/compile.py.")
    parser.add_argument("--weights", required=True, help="Output directory of scripts/prepare_weights.py.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--raw", action="store_true", help="Use the prompt as is, without the chat template.")
    parser.add_argument("--thinking", action="store_true", help="Let the model think before answering.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    max_seq_len = read_build_info(args.compiled)["max_seq_len"]
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    text = args.prompt if args.raw else chat_prompt(args.prompt, args.thinking)
    prompt_ids = tokenizer.encode(text, add_special_tokens=False).ids

    source = checkpoint.TensorSource(f"{args.model_dir}/model.safetensors")
    head = HostHead(source.get(checkpoint.EMBED_TOKENS), source.get(checkpoint.LM_HEAD))

    started = time.perf_counter()
    step = NeuronStep(args.compiled, args.weights)
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    first_token_at = None
    generated = []
    for token_id in greedy_generate(step, head, cfg, prompt_ids, args.max_new_tokens, max_seq_len, EOS_TOKEN_IDS, DTYPE):
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
