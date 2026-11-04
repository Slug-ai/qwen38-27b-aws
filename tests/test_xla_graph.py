"""Run the model through XLA on CPU, which uses the same HLO lowering the Neuron compiler receives.

This is the closest available check to running the compiled graph without Neuron hardware. It needs
torch_xla, so run it in the Neuron environment: pytest tests/test_xla_graph.py
"""

import os

import pytest
import torch

os.environ["PJRT_DEVICE"] = "CPU"
torch_xla = pytest.importorskip("torch_xla")
import torch_xla.core.xla_model as xm  # noqa: E402

from qwen38_inf2 import fp4  # noqa: E402
from qwen38_inf2.config import tiny_config  # noqa: E402
from qwen38_inf2.generate import step_inputs  # noqa: E402
from qwen38_inf2.model import Qwen38Step  # noqa: E402

MAX_SEQ_LEN = 16
TOKENS = 6


def random_model(dtype):
    torch.manual_seed(0)
    model = Qwen38Step(tiny_config(), MAX_SEQ_LEN, dtype=dtype).eval()
    for name, param in model.named_parameters():
        if name.endswith("weight_packed"):
            param.data = torch.randint(0, 256, param.shape, dtype=torch.uint8)
        elif name.endswith("weight_scale"):
            exponents = torch.randint(115, 125, param.shape, dtype=torch.uint8)
            param.data = fp4.scale_to_float(exponents)
        else:
            param.data = (torch.randn(param.shape) * 0.1).to(param.dtype)
    return model


def run(model, device, dtype):
    cfg = model.cfg
    torch.manual_seed(1)
    outputs = []
    with torch.no_grad():
        for position in range(TOKENS):
            inputs = {k: v.to(device) for k, v in step_inputs(cfg, position, MAX_SEQ_LEN, dtype).items()}
            embed = (torch.randn(1, 1, cfg.hidden_size) * 0.1).to(dtype).to(device)
            outputs.append(model(inputs_embeds=embed, **inputs))
            if device != "cpu":
                xm.mark_step()
    return [o.cpu().float() for o in outputs]


def test_fp4_decode_is_exact_through_xla():
    device = xm.xla_device()
    packed = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    for exponent in (100, 120, 127, 140):
        scale_bytes = torch.full((8, 2), exponent, dtype=torch.uint8)
        got = fp4.decode_in_graph(packed.to(device), fp4.scale_to_float(scale_bytes).to(device)).cpu()
        assert got.dtype == torch.bfloat16
        assert torch.equal(got, fp4.decode_host(packed, scale_bytes))


def test_float32_tokens_match_eager_pytorch():
    """The strict check that the traced graph computes the right thing, state included."""
    want = run(random_model(torch.float32), "cpu", torch.float32)
    device = xm.xla_device()
    got = run(random_model(torch.float32).to(device), device, torch.float32)
    for position, (g, w) in enumerate(zip(got, want)):
        assert (g - w).abs().max() <= 1e-4 * w.abs().max(), position
    # State must carry between tokens, or later tokens could not be matched at all.
    assert not torch.equal(got[0], got[-1])


def relative_error(outputs, truth):
    return max(float((o - t).abs().max() / t.abs().max()) for o, t in zip(outputs, truth))


def test_bf16_tracks_float32_about_as_well_as_eager_pytorch():
    """Two BF16 runs differ from each other by rounding alone, so compare each to float32 instead."""
    truth = run(random_model(torch.float32), "cpu", torch.float32)
    eager = run(random_model(torch.bfloat16), "cpu", torch.bfloat16)
    device = xm.xla_device()
    traced = run(random_model(torch.bfloat16).to(device), device, torch.bfloat16)

    assert all(torch.isfinite(o).all() for o in traced)
    assert relative_error(traced, truth) <= 1.5 * relative_error(eager, truth)
