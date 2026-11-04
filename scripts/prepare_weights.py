#!/usr/bin/env python3
"""Cut the checkpoint into one weights file per block per NeuronCore. Needs no Neuron SDK or hardware.

Vision and MTP tensors are never read. In total this is about 6.9 GB per core for FP4 and 12.2 GB for FP8.
"""

import argparse
from dataclasses import replace

from qwen38_inf2.checkpoint import TensorSource, read_build_info, write_head_weights, write_rank_weights
from qwen38_inf2.config import TextConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory holding model.safetensors and config.json.")
    parser.add_argument("--compiled", required=True, help="Output directory of scripts/compile.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--head-only", action="store_true", help="Write only the weights for lm_head on the chip, 1.3 GB per core, or 0.6 GB for FP8.")
    args = parser.parse_args()

    if args.head_only:
        info = read_build_info(args.compiled)
        write_head_weights(TensorSource(f"{args.model_dir}/model.safetensors"), args.output, info["tp_degree"], info.get("head_format", "bf16"))
        print(f"wrote head weights to {args.output}")
        return

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    info = read_build_info(args.compiled)
    cfg = replace(cfg, weight_format=info.get("weight_format", "fp4"))
    write_rank_weights(TensorSource(f"{args.model_dir}/model.safetensors"), cfg, args.output, info["layers"], info["tp_degree"])
    print(f"wrote weights to {args.output}")


if __name__ == "__main__":
    main()
