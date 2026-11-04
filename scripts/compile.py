#!/usr/bin/env python3
"""Compile the single-token graph. Needs the Neuron SDK but no Neuron hardware."""

import argparse
import time

from qwen38_inf2.config import TextConfig, tiny_config
from qwen38_inf2.neuron import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to the model's config.json. Omit with --tiny.")
    parser.add_argument("--tiny", action="store_true", help="Compile the tiny test model instead.")
    parser.add_argument("--layers", type=int, help="Keep only the first N layers, to check lowering cheaply.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--tp-degree", type=int, default=2)
    parser.add_argument("--target", default="inf2")
    parser.add_argument("--compiler-args", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.tiny == bool(args.config):
        parser.error("pass exactly one of --config or --tiny")
    cfg = tiny_config() if args.tiny else TextConfig.from_file(args.config)
    if args.layers:
        cfg = _truncated(cfg, args.layers)

    started = time.time()
    path = build(cfg, args.max_seq_len, args.output, args.tp_degree, args.target, args.compiler_args)
    print(f"compiled {cfg.num_hidden_layers} layers for {args.target} in {time.time() - started:.0f}s: {path}")


def _truncated(cfg: TextConfig, layers: int) -> TextConfig:
    from dataclasses import replace

    return replace(cfg, layer_types=cfg.layer_types[:layers])


if __name__ == "__main__":
    main()
