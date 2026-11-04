#!/usr/bin/env python3
"""Check lm_head on the chip against the host's full matmul, and time both. Takes about a minute."""

import argparse
import statistics
import time

import torch

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import TextConfig
from qwen38_inf2.generate import ChipHead, HostHead
from qwen38_inf2.neuron import load_head_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--head", required=True, help="Output directory of scripts/compile.py --head.")
    parser.add_argument("--weights", required=True, help="Directory holding the head weight files.")
    args = parser.parse_args()

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    batch = checkpoint.read_build_info(args.head)["batch_size"]
    source = checkpoint.TensorSource(f"{args.model_dir}/model.safetensors")
    graph = load_head_graph(args.head, args.weights)
    chip = ChipHead(source, cfg.rms_norm_eps, graph, batch)
    host = HostHead(source, cfg.rms_norm_eps)

    # Real embeddings give a real spread of inputs, and each one's best token is clear on the host.
    hidden = source.get_rows(checkpoint.EMBED_TOKENS, list(range(1000, 1000 + batch))).unsqueeze(1).to(torch.bfloat16)
    want = host.next_tokens(hidden)

    def timed(fn, runs=7):
        times = []
        for _ in range(runs):
            started = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - started)
        return result, statistics.median(times)

    _, graph_seconds = timed(lambda: graph(hidden=hidden))
    got, chip_seconds = timed(lambda: chip.next_tokens(hidden))
    _, host_seconds = timed(lambda: host.next_tokens(hidden), runs=3)
    matches = sum(a == b for a, b in zip(got, want))
    print(f"batch {batch}: {matches} of {batch} tokens match the host")
    print(f"chip graph alone {graph_seconds * 1e3:.1f} ms, chip head in total {chip_seconds * 1e3:.1f} ms, host head {host_seconds * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
