"""MXFP4 (OCP microscaling E2M1 + E8M0) decoding.

Checkpoint format, as written by compressed-tensors `mxfp4-pack-quantized`:

* ``weight_packed``: uint8, shape ``[rows, cols / 2]``. Each byte holds two elements. The low
  nibble is the even-indexed element and the high nibble is the odd-indexed element.
* ``weight_scale``: uint8, shape ``[rows, cols / 32]``. One E8M0 exponent per group of 32 elements
  along the last dimension, so a group's scale is ``2 ** (byte - 127)``.
* Each nibble is sign (bit 3) plus a magnitude index (bits 0-2) into
  ``[0, 0.5, 1, 1.5, 2, 3, 4, 6]``.

NeuronCore-v2 has no FP4 type (native MX compute is Trn3 only), and 27B parameters at BF16 do not
fit on one 32 GB Inf2 chip. So weights stay packed on the device and `decode_planes` expands them
to BF16 inside the compiled graph.

Decoding is most of the time per step on a chip, so `decode_planes` is built for speed. Weights are
rewritten once, by `to_sign_free`, so a nibble needs no sign handling. Nibbles are split with a shift and
a mask, because that is far faster than floor on this chip, and the even and odd columns stay in separate
matrices, because interleaving them costs time. Every intermediate has at most 8 significant bits, which
BF16 represents exactly, so the result is exact.

`decode_in_graph` is the slower decode of the checkpoint's own encoding, kept as a reference.
"""

import math

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


def to_sign_free(packed: torch.Tensor) -> torch.Tensor:
    """Rewrite each nibble as a code from 0 to 14, for values -6 to 6. Run once when weights are prepared.

    Negative zero becomes zero, which changes nothing in a matmul.
    """
    return _sign_free_nibble(packed & 15) | (_sign_free_nibble(packed >> 4) << 4)


def _sign_free_nibble(nibble: torch.Tensor) -> torch.Tensor:
    return torch.where(nibble < 8, nibble + 7, 15 - nibble)


def _code_to_value(code: torch.Tensor) -> torch.Tensor:
    v = code - 7.0
    outer = v - torch.clamp(v, -4.0, 4.0)
    return 0.5 * (v + outer) + (outer - torch.clamp(outer, -2.0, 2.0))


def decode_planes(packed: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand `to_sign_free` bytes to BF16, returning the even and the odd columns as separate matrices."""
    rows, groups = scale.shape
    half = GROUP_SIZE // 2
    plane_scale = scale.unsqueeze(-1).expand(rows, groups, half).reshape(rows, groups * half)
    even = _code_to_value((packed & 15).to(torch.bfloat16)) * plane_scale
    odd = _code_to_value((packed >> 4).to(torch.bfloat16)) * plane_scale
    return even, odd


def _fp8_e5m2_byte(value: float) -> int:
    """FP8 e5m2 byte for a value: 1 sign bit, 5 exponent bits with bias 15, 2 mantissa bits, rounded to nearest."""
    sign = 0x80 if value < 0 else 0
    magnitude = abs(value)
    if magnitude == 0:
        return 0
    exponent = math.floor(math.log2(magnitude))
    if exponent < -14:
        return sign | round(magnitude / 2.0**-16)  # subnormal, in steps of 2**-16
    mantissa = round((magnitude / 2.0**exponent - 1.0) * 4)
    if mantissa == 4:
        exponent, mantissa = exponent + 1, 0
    return sign | ((exponent + 15) << 2) | mantissa


def to_fp8(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Expand checkpoint bytes and exponents to one FP8 e5m2 byte per element, ``[rows, cols]``. Run on the host.

    FP8 holds a value with 1 mantissa bit exactly. In the real checkpoint that covers all but 0.0003% of
    scale groups, and the rest hold weights under 1e-4 that round to the nearest step of 2**-16.
    """
    table = torch.tensor(
        [
            [_fp8_e5m2_byte(sign * magnitude * 2.0 ** (byte - E8M0_BIAS)) for sign in (1, -1) for magnitude in E2M1_MAGNITUDES]
            for byte in range(E8M0_BIAS + 1)  # The checkpoint's largest scale is 1. A larger one fails loudly below.
        ],
        dtype=torch.uint8,
    )
    rows, half_cols = packed.shape
    groups = scale.shape[1]
    expanded = scale.long().unsqueeze(-1).expand(rows, groups, GROUP_SIZE).reshape(rows, groups * GROUP_SIZE)
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, half_cols * 2).long()
    return table[expanded, nibbles]


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
