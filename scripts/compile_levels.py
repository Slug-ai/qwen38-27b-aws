#!/usr/bin/env python3
"""Compile the small graphs that `chip_check.py` runs before the full block. Needs no Neuron hardware."""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from qwen38_inf2 import probes
from qwen38_inf2.config import TextConfig
from qwen38_inf2.neuron import DTYPE, build, compile_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    compile_graph(
        probes.DecodeProbe,
        {"gain": torch.ones(1, dtype=DTYPE)},
        args.output / "decode",
        {},
        tp_degree=1,
        workdir=args.output / "decode" / "compiler-work",
    )
    print("compiled decode", flush=True)
    compile_graph(
        probes.matrix_probe,
        {"x": torch.zeros(1, 1, probes.MATRIX_IN_FEATURES, dtype=DTYPE)},
        args.output / "matrix",
        {},
        tp_degree=1,
        workdir=args.output / "matrix" / "compiler-work",
    )
    print("compiled matrix", flush=True)
    cfg = TextConfig.from_file(args.config)
    build(
        replace(cfg, layer_types=cfg.layer_types[:1]),
        args.max_seq_len,
        args.output / "layer",
        workdir=args.output / "layer" / "compiler-work",
    )
    print("compiled layer", flush=True)


if __name__ == "__main__":
    main()
