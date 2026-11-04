#!/usr/bin/env python3
"""Save the CPU results that `chip_check.py` compares a chip against.

Runs on any machine with the prepared weights, using the simulated cores. The `model` level takes about
40 minutes on a Mac, and the others take a minute or two.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from verify_real_weights import DEFAULT_TEXT

from qwen38_inf2 import checkpoint, probes
from qwen38_inf2.config import TextConfig
from qwen38_inf2.cpu_sim import SimulatedCoresStep
from qwen38_inf2.generate import HostHead, greedy_generate, step_inputs

BF16, FLOAT32 = torch.bfloat16, torch.float32
LEVEL_TOKENS = 6
GREEDY_PROMPT = "The capital of France is"
GREEDY_TOKENS = 8
TP_DEGREE = 2


def run_tokens(step, cfg, embeds, max_seq_len, dtype):
    return torch.cat(
        [
            step(inputs_embeds=embeds[t].reshape(1, 1, -1).to(dtype), **step_inputs(cfg, t, max_seq_len, dtype)).float()
            for t in range(embeds.shape[0])
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--block-layers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("tests/golden"))
    parser.add_argument("--levels", nargs="+", default=["decode", "matrix", "layer", "block", "model"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = TextConfig.from_file(args.model_dir / "config.json")
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    source = checkpoint.TensorSource(args.model_dir / "model.safetensors")
    tokens = tokenizer.encode(DEFAULT_TEXT, add_special_tokens=False).ids
    embeds = source.get_rows(checkpoint.EMBED_TOKENS, tokens[:LEVEL_TOKENS]).unsqueeze(1)

    if "decode" in args.levels:
        weights, expected = probes.decode_probe_data()
        save_file({**weights, "expected": expected}, args.output / "decode.safetensors")
        print("saved decode", flush=True)

    if "matrix" in args.levels:
        probe = probes.matrix_probe()
        probe.load_state_dict(probes.matrix_probe_weights(args.weights), assign=True)
        x = embeds[:1]
        with torch.no_grad():
            bf16, float32 = probe(x.to(BF16)).float(), probe(x.to(FLOAT32))
        save_file({"x": x.contiguous(), "bf16": bf16, "float32": float32}, args.output / "matrix.safetensors")
        print("saved matrix", flush=True)

    for name, layers in (("layer", 1), ("block", args.block_layers)):
        if name not in args.levels:
            continue
        info = {"layers": layers, "tp_degree": TP_DEGREE, "max_seq_len": args.max_seq_len}
        results = {
            label: run_tokens(SimulatedCoresStep(args.weights, cfg, info, dtype, num_blocks=1), cfg, embeds, args.max_seq_len, dtype)
            for label, dtype in (("bf16", BF16), ("float32", FLOAT32))
        }
        save_file({"inputs": embeds.contiguous(), **results}, args.output / f"{name}.safetensors")
        print(f"saved {name}", flush=True)

    if "model" in args.levels:
        info = {"layers": args.block_layers, "tp_degree": TP_DEGREE, "max_seq_len": args.max_seq_len}
        head = HostHead(source, cfg.rms_norm_eps)
        step = SimulatedCoresStep(args.weights, cfg, info, BF16)
        top, margin = [], []
        for position, token_id in enumerate(tokens):
            hidden = step(inputs_embeds=head.embed(token_id, BF16), **step_inputs(cfg, position, args.max_seq_len, BF16))
            values, indices = head.logits(hidden).topk(2)
            top.append(int(indices[0]))
            margin.append(float(values[0] - values[1]))
            print(f"position {position}: {tokenizer.decode([top[-1]])!r} margin {margin[-1]:.2f}", flush=True)
        prompt = tokenizer.encode(GREEDY_PROMPT, add_special_tokens=False).ids
        output = list(greedy_generate(step, head, cfg, prompt, GREEDY_TOKENS, args.max_seq_len, (), BF16))
        (args.output / "model.json").write_text(
            json.dumps(
                {
                    "tokens": tokens,
                    "top": top,
                    "margin": margin,
                    "prompt": prompt,
                    "output": output,
                    "output_text": tokenizer.decode(output),
                },
                indent=1,
            )
        )
        print(f"saved model. greedy: {tokenizer.decode(output)!r}", flush=True)


if __name__ == "__main__":
    main()
