#!/usr/bin/env python3
"""Cut the checkpoint into one weights file per NeuronCore. Needs no Neuron hardware.

Vision and MTP tensors are never read. Each rank's file is about 6.9 GB.
"""

import argparse

from qwen38_inf2.checkpoint import TensorSource
from qwen38_inf2.config import TextConfig
from qwen38_inf2.neuron import write_rank_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory holding model.safetensors and config.json.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp-degree", type=int, default=2)
    args = parser.parse_args()

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    write_rank_weights(TensorSource(f"{args.model_dir}/model.safetensors"), cfg, args.output, args.tp_degree)
    print(f"wrote {args.tp_degree} weight files to {args.output}")


if __name__ == "__main__":
    main()
