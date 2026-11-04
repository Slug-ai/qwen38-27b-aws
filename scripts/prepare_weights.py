#!/usr/bin/env python3
"""Cut the checkpoint into one weights file per block per NeuronCore. Needs no Neuron SDK or hardware.

Vision and MTP tensors are never read. In total this is about 6.9 GB per core.
"""

import argparse

from qwen38_inf2.checkpoint import TensorSource, read_build_info, write_rank_weights
from qwen38_inf2.config import TextConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory holding model.safetensors and config.json.")
    parser.add_argument("--compiled", required=True, help="Output directory of scripts/compile.py.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    info = read_build_info(args.compiled)
    write_rank_weights(TensorSource(f"{args.model_dir}/model.safetensors"), cfg, args.output, info["layers"], info["tp_degree"])
    print(f"wrote weights to {args.output}")


if __name__ == "__main__":
    main()
