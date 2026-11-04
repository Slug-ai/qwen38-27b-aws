import pytest
import torch
from compressed_tensors.compressors.mx_utils import decompress_mx_scale
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

from qwen38_inf2 import fp4


def all_bytes(rows=8):
    return torch.arange(256, dtype=torch.uint8).reshape(rows, 256 // rows)


def test_in_graph_decode_is_exact_for_every_byte_and_exponent():
    packed = all_bytes()  # 8 rows of 32 packed bytes -> 64 columns, 2 groups per row
    for exponent in (1, 60, 100, 120, 127, 130, 150, 200, 254):
        scale_bytes = torch.full((8, 2), exponent, dtype=torch.uint8)
        got = fp4.decode_in_graph(packed, fp4.scale_to_float(scale_bytes))
        want = fp4.decode_host(packed, scale_bytes)
        assert got.dtype == torch.bfloat16
        assert torch.equal(got, want), f"mismatch at exponent {exponent}"


def test_sign_free_decode_is_exact_for_every_byte_and_exponent():
    packed = all_bytes()
    for exponent in (1, 60, 100, 120, 127, 130, 150, 200, 254):
        scale_bytes = torch.full((8, 2), exponent, dtype=torch.uint8)
        even, odd = fp4.decode_planes(fp4.to_sign_free(packed), fp4.scale_to_float(scale_bytes))
        want = fp4.decode_host(packed, scale_bytes)
        assert even.dtype == torch.bfloat16 and odd.dtype == torch.bfloat16
        assert torch.equal(even, want[:, ::2]), f"even columns differ at exponent {exponent}"
        assert torch.equal(odd, want[:, 1::2]), f"odd columns differ at exponent {exponent}"


def fp8_to_float(byte: torch.Tensor) -> torch.Tensor:
    """Independent reading of FP8 e5m2 bytes, by their bit fields."""
    b = byte.long()
    exponent, mantissa = (b >> 2) & 31, b & 3
    magnitude = torch.where(
        exponent == 0,
        mantissa.double() * 2.0**-16,
        (1.0 + mantissa.double() / 4.0) * torch.pow(2.0, exponent.double() - 15.0),
    )
    return torch.where(b >= 128, -magnitude, magnitude)


def test_fp8_is_exact_at_the_checkpoints_common_scales():
    packed = all_bytes()
    for exponent in range(112, 128):  # 99.9997% of the real checkpoint's scale groups
        scale_bytes = torch.full((8, 2), exponent, dtype=torch.uint8)
        got = fp8_to_float(fp4.to_fp8(packed, scale_bytes))
        assert torch.equal(got, fp4.decode_host(packed, scale_bytes, torch.float64)), f"differs at exponent {exponent}"


def test_fp8_rounds_the_rarest_scales_by_under_one_step():
    packed = all_bytes()
    for exponent in range(106, 112):
        scale_bytes = torch.full((8, 2), exponent, dtype=torch.uint8)
        error = (fp8_to_float(fp4.to_fp8(packed, scale_bytes)) - fp4.decode_host(packed, scale_bytes, torch.float64)).abs()
        assert error.max() <= 2.0**-17, f"too far at exponent {exponent}"


def test_matches_compressed_tensors():
    """compressed-tensors wrote this checkpoint, so its decoder defines correct."""
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (37, 64), dtype=torch.uint8)
    scale_bytes = torch.randint(90, 135, (37, 4), dtype=torch.uint8)

    ct_values = unpack_fp4_from_uint8(packed, 37, 128)
    ct_scale = decompress_mx_scale(scale_bytes).to(torch.bfloat16)
    want = ct_values * ct_scale.repeat_interleave(32, dim=-1)

    assert torch.equal(fp4.decode_host(packed, scale_bytes), want)
    assert torch.equal(fp4.decode_in_graph(packed, fp4.scale_to_float(scale_bytes)), want)


def test_nibble_order_low_is_even_index():
    packed = torch.tensor([[0x72] + [0] * 15], dtype=torch.uint8)  # low=2 (1.0), high=7 (6.0)
    out = fp4.decode_host(packed, torch.tensor([[127]], dtype=torch.uint8))
    assert out[0, 0] == 1.0 and out[0, 1] == 6.0


def test_sign_bit():
    packed = torch.tensor([[0xF9] + [0] * 15], dtype=torch.uint8)  # low=9 (-0.5), high=15 (-6.0)
    out = fp4.decode_in_graph(packed, fp4.scale_to_float(torch.tensor([[127]], dtype=torch.uint8)))
    assert out[0, 0] == -0.5 and out[0, 1] == -6.0


def test_encode_round_trip_error_is_bounded():
    torch.manual_seed(1)
    weight = torch.randn(16, 64) * 0.03
    packed, scale_bytes = fp4.encode_host(weight)
    assert packed.shape == (16, 32) and scale_bytes.shape == (16, 2)
    decoded = fp4.decode_host(packed, scale_bytes, torch.float32)
    assert (decoded - weight).abs().max() < weight.abs().max()
    assert torch.corrcoef(torch.stack((decoded.flatten(), weight.flatten())))[0, 1] > 0.9


def test_encode_rejects_unaligned_columns():
    with pytest.raises(ValueError):
        fp4.encode_host(torch.zeros(2, 33))
