#!/usr/bin/env python3
"""Compile and time one real gated delta or full attention layer's mixer on one NeuronCore.

Results at batch 56: gated delta 3.39 ms and attention 2.96 ms, of which reading weights is about 0.9 ms. Tried with
no gain: gated delta contracting state with a matmul (3.75 ms in float32, 3.52 ms in BF16), and attention without
repeating the KV cache for each head group (2.94 ms).
"""

import argparse
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import FULL, LINEAR, TextConfig
from qwen38_inf2.generate import batch_step_inputs
from qwen38_inf2.model import FullAttentionStep, GatedDeltaNetStep, identity_all_reduce
from qwen38_inf2.neuron import DTYPE, compile_graph, load_graph


class GatedDelta(nn.Module):
    def __init__(self, cfg, batch_size, max_seq_len):
        super().__init__()
        self.linear_attn = GatedDeltaNetStep(cfg, 2, DTYPE, identity_all_reduce, batch_size)

    def forward(self, x, keep_state):
        return self.linear_attn(x, keep_state)


class Attention(nn.Module):
    def __init__(self, cfg, batch_size, max_seq_len):
        super().__init__()
        self.self_attn = FullAttentionStep(cfg, 2, max_seq_len, DTYPE, identity_all_reduce, batch_size)

    def forward(self, x, cos, sin, write_mask, attn_mask):
        return self.self_attn(x, cos, sin, write_mask, attn_mask)


# Layer 0 is gated delta and layer 3 is full attention in the repeating block.
VARIANTS = {"gated_delta": (GatedDelta, 0, "linear_attn"), "attention": (Attention, 3, "self_attn")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="tests/fixtures/config.json")
    parser.add_argument("--weights", default="weights_fp8_16")
    parser.add_argument("--output", type=Path, default=Path("artifacts/bench_mixer"))
    parser.add_argument("--batch-size", type=int, default=56)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--tag", default="")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()

    cfg = replace(TextConfig.from_file(args.config), weight_format="fp8", recurrent_state_format="bf16")
    batch = args.batch_size
    step = batch_step_inputs(cfg, [1] * batch, args.max_seq_len, DTYPE)
    x = torch.zeros(batch, 1, cfg.hidden_size, dtype=DTYPE)

    for name in args.variants:
        make, layer, attribute = VARIANTS[name]
        inputs = {"x": x, "keep_state": step["keep_state"]} if name == "gated_delta" else {
            "x": x, "cos": step["cos"], "sin": step["sin"], "write_mask": step["write_mask"], "attn_mask": step["attn_mask"],
        }
        directory = args.output / f"{name}{args.tag}_b{batch}"
        if not args.run_only:
            started = time.time()
            compile_graph(
                lambda: make(cfg, batch, args.max_seq_len), inputs, directory, {}, tp_degree=1, workdir=directory / "compiler-work"
            )
            print(f"{name}: compiled in {time.time() - started:.0f}s", flush=True)
        if args.compile_only:
            continue
        block = checkpoint.load_rank_block(args.weights, 0, 0)
        prefix = f"layers.{layer}.{attribute}."
        weights = {f"{attribute}.{k.removeprefix(prefix)}": t for k, t in block.items() if k.startswith(prefix)}
        run = load_graph(directory, [weights])
        torch.manual_seed(0)
        inputs = dict(inputs, x=(torch.randn(batch, 1, cfg.hidden_size) * 0.5).to(DTYPE))
        times = []
        for _ in range(14):
            started = time.perf_counter()
            run(**inputs)
            times.append(time.perf_counter() - started)
        print(f"{name}{args.tag} batch {batch}: {statistics.median(times[4:]) * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
