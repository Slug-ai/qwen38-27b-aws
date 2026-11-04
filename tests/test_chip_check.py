"""A checker that always passes is worse than none, so these feed it broken results on purpose."""

import pytest
import torch
from safetensors.torch import save_file

from qwen38_inf2 import chip_check, fp4, probes
from qwen38_inf2.config import tiny_config
from qwen38_inf2.generate import step_inputs
from qwen38_inf2.model import Qwen38Step

MAX_SEQ_LEN = 16
TOKENS = 6
DTYPE = chip_check.DTYPE


def random_step(dtype):
    torch.manual_seed(0)
    model = Qwen38Step(tiny_config(), MAX_SEQ_LEN, dtype=dtype).eval()
    for name, param in model.named_parameters():
        if name.endswith("weight_packed"):
            param.data = torch.randint(0, 256, param.shape, dtype=torch.uint8)
        elif name.endswith("weight_scale"):
            param.data = fp4.scale_to_float(torch.randint(115, 125, param.shape, dtype=torch.uint8))
        else:
            param.data = (torch.randn(param.shape) * 0.1).to(param.dtype)
    return model


def run_all(model, inputs, dtype):
    cfg = tiny_config()
    with torch.no_grad():
        return torch.cat(
            [
                model(inputs_embeds=inputs[t].reshape(1, 1, -1).to(dtype), **step_inputs(cfg, t, MAX_SEQ_LEN, dtype))
                for t in range(TOKENS)
            ]
        ).float()


@pytest.fixture(scope="module")
def token_golden():
    torch.manual_seed(1)
    inputs = (torch.randn(TOKENS, 1, tiny_config().hidden_size) * 0.1).to(DTYPE)
    return {
        "inputs": inputs,
        "bf16": run_all(random_step(DTYPE), inputs, DTYPE),
        "float32": run_all(random_step(torch.float32), inputs, torch.float32),
    }


def healthy_step():
    model = random_step(DTYPE)
    return lambda **inputs: model(**inputs)


def check(step, golden):
    with torch.no_grad():
        return chip_check.check_tokens(step, tiny_config(), golden, MAX_SEQ_LEN, "block")


def test_decode_passes_when_exact_and_fails_on_a_single_wrong_value():
    weights, expected = probes.decode_probe_data()
    golden = {"expected": expected}
    probe = probes.DecodeProbe()
    probe.load_state_dict(weights)
    assert chip_check.check_decode(probe, golden).passed

    wrong = expected.clone()
    wrong[3, 7] = wrong[3, 7] * 2
    result = chip_check.check_decode(lambda gain: wrong, golden)
    assert not result.passed and result.details["wrong_values"] == 1
    assert "decode" in result.hint


def test_decode_probe_covers_every_byte_and_nibble_order():
    weights, expected = probes.decode_probe_data()
    assert sorted(weights["weight_packed"].flatten().tolist()) == list(range(256))
    assert len(set(weights["weight_scale"].float().flatten().tolist())) > 8
    assert torch.equal(fp4.decode_in_graph(weights["weight_packed"], weights["weight_scale"]), expected)


def test_tokens_pass_on_healthy_results(token_golden):
    assert check(healthy_step(), token_golden).passed


def test_tokens_allow_small_differences(token_golden):
    step = healthy_step()
    result = check(lambda **i: step(**i) * (1 + 1e-3), token_golden)
    assert result.passed, result.details


def test_tokens_catch_wrong_math_at_the_first_token(token_golden):
    step = healthy_step()
    result = check(lambda **i: step(**i) * 1.5, token_golden)
    assert not result.passed and result.details["failed_at_token"] == 0
    assert "all-reduce" in result.hint


def test_tokens_catch_state_lost_between_tokens(token_golden):
    """The first token is correct and every later one has forgotten history: the bug that compiles cleanly."""
    model = random_step(DTYPE)

    def forgetful(**inputs):
        inputs = {**inputs, "keep_state": torch.zeros(1)}
        for layer in model.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.k_cache = torch.zeros_like(layer.self_attn.k_cache)
                layer.self_attn.v_cache = torch.zeros_like(layer.self_attn.v_cache)
        return model(**inputs)

    result = check(forgetful, token_golden)
    assert not result.passed and result.details["failed_at_token"] > 0
    assert "persisting" in result.hint


def test_tokens_catch_nan(token_golden):
    step = healthy_step()
    result = check(lambda **i: step(**i) * float("nan"), token_golden)
    assert not result.passed


class FixedHead:
    """Returns chosen top tokens in order, standing in for real logits."""

    def __init__(self, tops):
        self.tops = iter(tops)

    def embed(self, token_id, dtype):
        return torch.zeros(1, 1, 4, dtype=dtype)

    def logits(self, hidden):
        logits = torch.zeros(10)
        logits[next(self.tops)] = 1.0
        return logits


MODEL_GOLDEN = {"tokens": [1, 2, 3, 4], "top": [5, 6, 7, 8], "margin": [3.0, 0.1, 0.2, 4.0]}


def model_check(tops):
    return chip_check.check_model(lambda **i: torch.zeros(1), FixedHead(tops), tiny_config(), MODEL_GOLDEN, MAX_SEQ_LEN)


def test_model_passes_when_top_tokens_match():
    assert model_check([5, 6, 7, 8]).passed


def test_model_forgives_a_flip_only_at_a_near_tie():
    result = model_check([5, 9, 7, 8])
    assert result.passed and len(result.details["flips"]) == 1


def test_model_forgives_every_near_tie_flipping_at_once():
    result = model_check([5, 9, 9, 8])
    assert result.passed and len(result.details["flips"]) == 2


def test_model_fails_on_a_flip_where_the_cpu_was_confident():
    assert not model_check([9, 6, 7, 8]).passed


def test_matrix_catches_wrong_output():
    x = torch.ones(1, 1, 8, dtype=DTYPE)
    golden = {"x": x, "bf16": torch.ones(3), "float32": torch.ones(3)}
    assert chip_check.check_matrix(lambda x: torch.ones(3), golden).passed
    assert not chip_check.check_matrix(lambda x: torch.ones(3) * 1.1, golden).passed


def test_speed_reports_numbers_and_ignores_the_first_step():
    calls = []

    def step(**inputs):
        calls.append(1)
        return torch.zeros(1)

    class Head(FixedHead):
        def __init__(self):
            pass

        def logits(self, hidden):
            return torch.zeros(10)

    result = chip_check.measure_speed(step, Head(), tiny_config(), [1, 2], 3, MAX_SEQ_LEN)
    assert result.passed and len(calls) == 5
    assert result.details["tokens_per_second"] > 0


def test_blocks_can_keep_only_their_first_layers(tmp_path):
    from qwen38_inf2 import checkpoint

    save_file(
        {"layers.0.a": torch.zeros(1), "layers.1.a": torch.zeros(1), "layers.10.a": torch.zeros(1)},
        checkpoint.shard_path(tmp_path, 0, 0),
    )
    assert set(checkpoint.load_rank_block(tmp_path, 0, 0)) == {"layers.0.a", "layers.1.a", "layers.10.a"}
    assert set(checkpoint.load_rank_block(tmp_path, 0, 0, layers=1)) == {"layers.0.a"}
