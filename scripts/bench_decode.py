#!/usr/bin/env python3
"""Compile and time FP4 decode variants on a real matrix, on one NeuronCore.

Decoding is most of the time per step, so these numbers decide which decode the model uses. Each variant
returns F.linear of the same matrix, and the output must match the current decode.
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from qwen38_inf2 import fp4, probes
from qwen38_inf2.neuron import DTYPE, compile_graph, load_graph

ROWS, COLS = probes.MATRIX_OUT_FEATURES, probes.MATRIX_IN_FEATURES


def _frozen(*shape, dtype):
    return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_packed = _frozen(ROWS, COLS // 2, dtype=torch.uint8)
        self.weight_scale = _frozen(ROWS, COLS // fp4.GROUP_SIZE, dtype=torch.bfloat16)

    def half_scale(self):
        rows, groups = self.weight_scale.shape
        half = fp4.GROUP_SIZE // 2
        return self.weight_scale.unsqueeze(-1).expand(rows, groups, half).reshape(rows, groups * half)


def magnitude(mag):
    return 0.5 * (mag + torch.clamp(mag - 4.0, 0.0, 3.0)) + torch.clamp(mag - 6.0, 0.0, 1.0)


def signed(nibble):
    sign_bit = torch.clamp(nibble - 7.0, 0.0, 1.0)
    return magnitude(nibble - 8.0 * sign_bit) * (1.0 - 2.0 * sign_bit)


class Current(Base):
    """What the model uses today: split, interleave, then decode."""

    def forward(self, x):
        return F.linear(x, fp4.decode_in_graph(self.weight_packed, self.weight_scale).to(x.dtype))


class Split(Base):
    """Even and odd columns stay separate, so there is no interleave, and the matmul is done in two halves."""

    def forward(self, x):
        byte = self.weight_packed.to(torch.bfloat16)
        high = torch.floor(byte / 16.0)
        low = byte - high * 16.0
        scale = self.half_scale()
        return F.linear(x[..., ::2], signed(low) * scale) + F.linear(x[..., 1::2], signed(high) * scale)


class SplitBits(Base):
    """As Split, separating nibbles with integer shift and mask rather than floor."""

    def forward(self, x):
        low = (self.weight_packed & 15).to(torch.bfloat16)
        high = (self.weight_packed >> 4).to(torch.bfloat16)
        scale = self.half_scale()
        return F.linear(x[..., ::2], signed(low) * scale) + F.linear(x[..., 1::2], signed(high) * scale)


class Reencoded(Base):
    """As Split, with bytes rewritten when weights are prepared so a nibble is sign-free: value code v = n - 7."""

    def forward(self, x):
        byte = self.weight_packed.to(torch.bfloat16)
        high = torch.floor(byte / 16.0)
        low = byte - high * 16.0
        scale = self.half_scale()
        return F.linear(x[..., ::2], odd_value(low - 7.0) * scale) + F.linear(x[..., 1::2], odd_value(high - 7.0) * scale)


def odd_value(v):
    outer = v - torch.clamp(v, -4.0, 4.0)
    return 0.5 * (v + outer) + (outer - torch.clamp(outer, -2.0, 2.0))


def reencode(packed):
    """Map each nibble n to a sign-free code, where -6 to 6 become 0 to 14. Negative zero becomes zero."""
    table = torch.tensor([7, 8, 9, 10, 11, 12, 13, 14, 7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.uint8)
    return table[(packed & 15).long()] | (table[(packed >> 4).long()] << 4)


class ReencodedBits(Base):
    """Sign-free codes, separated with shift and mask."""

    def forward(self, x):
        low = (self.weight_packed & 15).to(torch.bfloat16)
        high = (self.weight_packed >> 4).to(torch.bfloat16)
        scale = self.half_scale()
        return F.linear(x[..., ::2], odd_value(low - 7.0) * scale) + F.linear(x[..., 1::2], odd_value(high - 7.0) * scale)


NIBBLE_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


class GatherNibble(Base):
    """A 16-entry table lookup for each nibble."""

    def forward(self, x):
        table = torch.tensor(NIBBLE_VALUES, dtype=torch.bfloat16, device=x.device)
        scale = self.half_scale()
        low = table[(self.weight_packed & 15).int()]
        high = table[(self.weight_packed >> 4).int()]
        return F.linear(x[..., ::2], low * scale) + F.linear(x[..., 1::2], high * scale)


class GatherByte(Base):
    """Two 256-entry table lookups on the byte, with no shift, mask or arithmetic."""

    def forward(self, x):
        low_table = torch.tensor([NIBBLE_VALUES[b & 15] for b in range(256)], dtype=torch.bfloat16, device=x.device)
        high_table = torch.tensor([NIBBLE_VALUES[b >> 4] for b in range(256)], dtype=torch.bfloat16, device=x.device)
        scale = self.half_scale()
        index = self.weight_packed.int()
        return F.linear(x[..., ::2], low_table[index] * scale) + F.linear(x[..., 1::2], high_table[index] * scale)


VARIANTS = {
    "current": Current,
    "split": Split,
    "split_bits": SplitBits,
    "reencoded": Reencoded,
    "reencoded_bits": ReencodedBits,
    "gather_nibble": GatherNibble,
    "gather_byte": GatherByte,
}
REENCODED = {"reencoded", "reencoded_bits"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--output", type=Path, default=Path("artifacts/bench"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--compiler-args", default="")
    parser.add_argument("--tag", default="", help="Added to the directory name, to keep compiles with other flags.")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()

    weights = probes.matrix_probe_weights(args.weights)
    x = torch.linspace(-1, 1, COLS).reshape(1, 1, COLS).to(DTYPE)
    expected = F.linear(x.float(), fp4.decode_in_graph(weights["weight_packed"], weights["weight_scale"]).float())

    for name in args.variants:
        directory = args.output / f"{name}{args.tag}"
        if not args.run_only:
            started = time.time()
            compile_graph(
                VARIANTS[name],
                {"x": torch.zeros(1, 1, COLS, dtype=DTYPE)},
                directory,
                {},
                tp_degree=1,
                extra_compiler_args=args.compiler_args,
                workdir=directory / "compiler-work",
            )
            print(f"{name}: compiled in {time.time() - started:.0f}s", flush=True)
        if args.compile_only:
            continue
        data = dict(weights)
        if name in REENCODED:
            data["weight_packed"] = reencode(data["weight_packed"])
        run = load_graph(directory, [data])
        times = []
        for _ in range(14):
            started = time.perf_counter()
            output = run(x=x)
            times.append(time.perf_counter() - started)
        median = statistics.median(times[4:])
        error = float((output.float() - expected).abs().max() / expected.abs().max())
        print(f"{name}{args.tag}: {median * 1e3:.2f} ms, {ROWS * COLS / median / 1e9:.2f} G values/s, relative error {error:.4f}", flush=True)


if __name__ == "__main__":
    main()
