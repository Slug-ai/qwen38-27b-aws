"""MXFP4 (OCP microscaling E2M1 + E8M0) decoding.

Checkpoint format, as written by compressed-tensors `mxfp4-pack-quantized`:

* ``weight_packed``: uint8, shape ``[rows, cols / 2]``. Each byte holds two elements. The low
  nibble is the even-indexed element and the high nibble is the odd-indexed element.
* ``weight_scale``: uint8, shape ``[rows, cols / 32]``. One E8M0 exponent per group of 32 elements
  along the last dimension, so a group's scale is ``2 ** (byte - 127)``.
* Each nibble is sign (bit 3) plus a magnitude index (bits 0-2) into
  ``[0, 0.5, 1, 1.5, 2, 3, 4, 6]``.

NeuronCore-v2 has no FP4 type (native MX compute is Trn3 only), and 27B parameters at BF16 do not
fit on one 32 GB Inf2 chip. So weights stay packed on the device and `decode_in_graph` expands them
to BF16 inside the compiled graph.

`decode_in_graph` deliberately uses only float add, multiply, floor and clamp. It needs no integer
ops and no gather, so it does not depend on those lowering well for this target. Every intermediate
has at most 8 significant bits, which BF16 represents exactly, so the result is exact.
"""

import torch

GROUP_SIZE = 32
E8M0_BIAS = 127
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def scale_to_float(scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Convert E8M0 exponent bytes to float. Run once on the host when weights are prepared."""
    return torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.to(torch.int32) - E8M0_BIAS).to(dtype)


def decode_host(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Reference LUT decode for host-side use and for checking `decode_in_graph`."""
    lut = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32)
    lut = torch.cat((lut, -lut))
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    values = lut[nibbles.long()]
    return (values * scale_to_float(scale, torch.float32).repeat_interleave(GROUP_SIZE, dim=-1)).to(dtype)


def decode_in_graph(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Expand packed FP4 to BF16 using arithmetic the Neuron compiler can lower.

    Args:
        packed: uint8 ``[rows, cols / 2]``.
        scale: BF16 ``[rows, cols / 32]``, from `scale_to_float`.

    Returns:
        BF16 ``[rows, cols]``.
    """
    rows, half_cols = packed.shape
    byte = packed.to(torch.bfloat16)
    hi = torch.floor(byte / 16.0)
    lo = byte - hi * 16.0
    nibble = torch.stack((lo, hi), dim=-1).reshape(rows, half_cols * 2)

    sign_bit = torch.clamp(nibble - 7.0, 0.0, 1.0)
    mag = nibble - 8.0 * sign_bit
    # Piecewise-linear form of the E2M1 table: 0 .5 1 1.5 2 3 4 6.
    value = 0.5 * mag + 0.5 * torch.clamp(mag - 4.0, 0.0, 2.0) + 1.5 * torch.clamp(mag - 6.0, 0.0, 1.0)
    value = value * (1.0 - 2.0 * sign_bit)

    groups = scale.shape[1]
    group_scale = scale.unsqueeze(-1).expand(rows, groups, GROUP_SIZE).reshape(rows, groups * GROUP_SIZE)
    return value * group_scale


def encode_host(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float matrix to (packed, scale bytes). Used only to build test checkpoints."""
    rows, cols = weight.shape
    if cols % GROUP_SIZE:
        raise ValueError(f"cols must be a multiple of {GROUP_SIZE}, got {cols}")
    groups = weight.float().reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    peak = groups.abs().amax(dim=-1).clamp_min(2.0**-100)
    # A scale of 2**(floor(log2(peak)) - 2) maps the group peak into [4, 8), then clamped to 6.
    exponent = torch.floor(torch.log2(peak)) - 2
    scale_bytes = (exponent + E8M0_BIAS).to(torch.uint8)

    lut = torch.tensor(E2M1_MAGNITUDES)
    scaled = groups / torch.ldexp(torch.ones_like(exponent), exponent.to(torch.int32)).unsqueeze(-1)
    mag_index = (scaled.abs().unsqueeze(-1) - lut).abs().argmin(dim=-1)
    nibbles = (mag_index + 8 * (scaled < 0)).to(torch.uint8).reshape(rows, cols)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return packed, scale_bytes
