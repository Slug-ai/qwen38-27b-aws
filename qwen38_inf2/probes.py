"""Small graphs for bringing up a chip one level at a time, so a failure points at its cause."""

from pathlib import Path

import torch
from torch import nn

from . import checkpoint, fp4
from .model import FP4Linear

# One real matrix from the model, as cut for one core.
MATRIX_NAME = "layers.0.mlp.gate_proj"
MATRIX_IN_FEATURES = 5120
MATRIX_OUT_FEATURES = 8704
DECODE_ROWS = 8
DECODE_GROUPS = 2


class DecodeProbe(nn.Module):
    """FP4 decode and nothing else. `gain` is always 1, and exists because a graph needs an input."""

    def __init__(self):
        super().__init__()
        half_cols = fp4.GROUP_SIZE // 2 * DECODE_GROUPS
        self.weight_packed = nn.Parameter(torch.zeros(DECODE_ROWS, half_cols, dtype=torch.uint8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.zeros(DECODE_ROWS, DECODE_GROUPS, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, gain: torch.Tensor) -> torch.Tensor:
        even, odd = fp4.decode_planes(self.weight_packed, self.weight_scale)
        return torch.stack((even, odd), dim=-1).reshape(even.shape[0], -1) * gain


def decode_probe_data() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Every byte value, across a spread of exponents, and the exact answer from the reference decoder."""
    packed = torch.arange(256, dtype=torch.uint8).reshape(DECODE_ROWS, -1)
    # The real checkpoint's scales seen so far span bytes 117 to 122. Go well past that on both sides.
    exponents = torch.linspace(90, 150, DECODE_ROWS * DECODE_GROUPS).to(torch.uint8).reshape(DECODE_ROWS, DECODE_GROUPS)
    weights = {"weight_packed": fp4.to_sign_free(packed), "weight_scale": fp4.scale_to_float(exponents)}
    return weights, fp4.decode_host(packed, exponents)


def matrix_probe_weights(weights_dir: str | Path) -> dict[str, torch.Tensor]:
    block = checkpoint.load_rank_block(weights_dir, 0, 0)
    return {name: block[f"{MATRIX_NAME}.{name}"] for name in ("weight_packed", "weight_scale")}


def matrix_probe() -> FP4Linear:
    return FP4Linear(MATRIX_IN_FEATURES, MATRIX_OUT_FEATURES)
