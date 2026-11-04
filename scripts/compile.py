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
    parser.add_argument("--layers", type=int, help="Compile only the first N layers. 4 is one repeating block, loaded 16 times at run time.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--tp-degree", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1, help="Sequences run together per step. Decoding is shared by all of them.")
    parser.add_argument("--target", default="inf2")
    parser.add_argument("--compiler-args", default="")
    parser.add_argument("--optimize-layout", action="store_true", help="Use NxD's weight layout optimization.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workdir", help="Compiler scratch and logs. Defaults to <output>/compiler-work.")
    args = parser.parse_args()

    if args.tiny == bool(args.config):
        parser.error("pass exactly one of --config or --tiny")
    cfg = tiny_config() if args.tiny else TextConfig.from_file(args.config)
    if args.layers:
        cfg = _truncated(cfg, args.layers)

    started = time.time()
    workdir = args.workdir or f"{args.output}/compiler-work"
    path = build(cfg, args.max_seq_len, args.output, args.tp_degree, args.target, args.compiler_args, workdir, args.optimize_layout, args.batch_size)
    print(f"compiled {cfg.num_hidden_layers} layers for {args.target} in {time.time() - started:.0f}s: {path}")


def _truncated(cfg: TextConfig, layers: int) -> TextConfig:
    from dataclasses import replace

    return replace(cfg, layer_types=cfg.layer_types[:layers])


if __name__ == "__main__":
    main()
