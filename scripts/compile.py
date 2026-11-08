#!/usr/bin/env python3
"""Compile the single-token graph. Needs the Neuron SDK but no Neuron hardware."""

import argparse
import time
from dataclasses import replace

from qwen38_inf2.config import TextConfig, tiny_config
from qwen38_inf2.neuron import build, build_head, build_mtp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to the model's config.json. Omit with --tiny.")
    parser.add_argument("--tiny", action="store_true", help="Compile the tiny test model instead.")
    parser.add_argument("--layers", type=int, help="Compile only the first N layers. 4 is one repeating block, loaded 16 times at run time.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--tp-degree", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1, help="Sequences run together per step. Decoding is shared by all of them.")
    parser.add_argument("--target", default="inf2")
    parser.add_argument("--weight-format", choices=("fp4", "fp8"), default="fp4", help="fp8 uses more memory and needs no decoding on the chip.")
    parser.add_argument("--recurrent-state", choices=("float32", "bf16"), default="float32", help="bf16 halves the largest cost per row.")
    parser.add_argument("--kv-cache", choices=("bf16", "fp8"), default="bf16", help="fp8 uses the checkpoint's KV-cache scales and half the memory.")
    parser.add_argument("--chunk", type=int, help="Also compile a prefill graph taking this many prompt tokens of one row per call.")
    parser.add_argument("--verify-tokens", type=int, default=0, help="Also compile a verify graph of this many tokens per row, for speculative decoding.")
    parser.add_argument("--no-decode", action="store_true", help="With --verify-tokens, leave out the one-token graph, which speculation never uses.")
    parser.add_argument("--mtp", action="store_true", help="Compile the MTP drafter instead, with --batch-size, --max-seq-len, --chunk and --verify-tokens.")
    parser.add_argument("--compiler-args", default="")
    parser.add_argument("--optimize-layout", action="store_true", help="Use NxD's weight layout optimization.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--head", action="store_true", help="Compile lm_head for the chip instead of decoder layers. Uses --batch-size.")
    parser.add_argument("--head-format", choices=("bf16", "fp8"), default="bf16", help="fp8 halves lm_head's memory, and the host rechecks its picks.")
    parser.add_argument("--workdir", help="Compiler scratch and logs. Defaults to <output>/compiler-work.")
    args = parser.parse_args()

    if args.tiny == bool(args.config):
        parser.error("pass exactly one of --config or --tiny")
    cfg = tiny_config() if args.tiny else TextConfig.from_file(args.config)
    if args.layers:
        cfg = _truncated(cfg, args.layers)
    cfg = replace(cfg, weight_format=args.weight_format, recurrent_state_format=args.recurrent_state, kv_cache_format=args.kv_cache)

    started = time.time()
    workdir = args.workdir or f"{args.output}/compiler-work"
    if args.head:
        path = build_head(cfg, args.batch_size, args.output, args.tp_degree, args.target, workdir, args.head_format)
        print(f"compiled lm_head for {args.target} in {time.time() - started:.0f}s: {path}")
        return
    if args.mtp:
        path = build_mtp(cfg, args.max_seq_len, args.output, args.batch_size, args.chunk, args.verify_tokens, args.tp_degree, args.target, workdir)
        print(f"compiled the MTP drafter for {args.target} in {time.time() - started:.0f}s: {path}")
        return

    path = build(
        cfg, args.max_seq_len, args.output, args.tp_degree, args.target, args.compiler_args, workdir, args.optimize_layout, args.batch_size, args.chunk,
        args.verify_tokens, not args.no_decode,
    )
    print(f"compiled {cfg.num_hidden_layers} layers for {args.target} in {time.time() - started:.0f}s: {path}")


def _truncated(cfg: TextConfig, layers: int) -> TextConfig:
    return replace(cfg, layer_types=cfg.layer_types[:layers])


if __name__ == "__main__":
    main()
