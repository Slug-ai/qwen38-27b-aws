#!/usr/bin/env python3
"""Compile and time FP4 decode variants on a real matrix, on one NeuronCore.

Decoding is most of the time per step, so these numbers decide which decode the model uses. This is the
fastest test there is, at under a minute for each variant. Each variant does the same matmul, and its output
must match the current decode. Results so far on a real matrix: floor and interleave, 3.59 G values/s. Split
without interleave, 3.70. Shift and mask, 5.97. Sign-free codes as well, 7.87. Table lookups, 3.7 to 4.2.
Masking the high nibble rather than the integer divide that `>> 4` compiles to, 7.89, so no gain. -O2 changes
nothing. No decode at all, with FP8 weights read by a kernel, 72.9. At batch 32 that kernel does 59, and loading
each output tile's weights with one static DMA does 44.5, so it stays with one load per matmul. Its time from batch
1 to 128 only goes from 0.62 ms to 1.00 ms.
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen38_inf2 import checkpoint, fp4, probes
from qwen38_inf2.model import FP8Linear
from qwen38_inf2.neuron import DTYPE, compile_graph, load_graph

ROWS, COLS = probes.MATRIX_OUT_FEATURES, probes.MATRIX_IN_FEATURES


class Planes(probes.FP4Linear):
    """The decode the model uses today. New candidates go next to this one and must beat it."""


VARIANTS = {"planes": Planes, "fp8_direct": FP8Linear}


def fp8_weights(model_dir: str) -> dict[str, torch.Tensor]:
    """The probe matrix as FP8, from the checkpoint's own bytes, in the layout the kernel reads."""
    source = checkpoint.TensorSource(f"{model_dir}/model.safetensors")
    name = f"{checkpoint.TEXT_PREFIX}{probes.MATRIX_NAME}"
    packed = source.get(f"{name}.weight_packed")[:ROWS]
    scale = source.get(f"{name}.weight_scale")[:ROWS]
    return {"weight_fp8": fp4.to_fp8(packed, scale).transpose(0, 1).clone()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--output", type=Path, default=Path("artifacts/bench"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=1, help="Rows run together, to see how a kernel's time grows with batch.")
    parser.add_argument("--compiler-args", default="")
    parser.add_argument("--tag", default="", help="Added to the directory name, to keep compiles with other flags.")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()

    weights, x, expected = None, None, None
    if not args.compile_only:
        weights = probes.matrix_probe_weights(args.weights)
        gains = torch.linspace(1, 2, args.batch_size).reshape(-1, 1, 1)
        x = (gains * torch.linspace(-1, 1, COLS).reshape(1, 1, COLS)).to(DTYPE)
        even, odd = fp4.decode_planes(weights["weight_packed"], weights["weight_scale"])
        expected = F.linear(x[..., ::2].float(), even.float()) + F.linear(x[..., 1::2].float(), odd.float())

    for name in args.variants:
        directory = args.output / f"{name}{args.tag}"
        if not args.run_only:
            started = time.time()
            compile_graph(
                lambda variant=VARIANTS[name]: variant(COLS, ROWS),
                {"x": torch.zeros(args.batch_size, 1, COLS, dtype=DTYPE)},
                directory,
                {},
                tp_degree=1,
                extra_compiler_args=args.compiler_args,
                workdir=directory / "compiler-work",
            )
            print(f"{name}: compiled in {time.time() - started:.0f}s", flush=True)
        if args.compile_only:
            continue
        run = load_graph(directory, [fp8_weights(args.model_dir)] if name.startswith("fp8") else [weights])
        times = []
        for _ in range(14):
            started = time.perf_counter()
            output = run(x=x)
            times.append(time.perf_counter() - started)
        median = statistics.median(times[4:])
        error = float((output.float() - expected).abs().max() / expected.abs().max())
        print(
            f"{name}{args.tag} batch {args.batch_size}: {median * 1e3:.2f} ms, "
            f"{ROWS * COLS / median / 1e9:.2f} G values/s, {ROWS * COLS * args.batch_size / median / 1e12:.2f} T MAC/s, relative error {error:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
