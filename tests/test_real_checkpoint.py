"""Check the full 27B model against the real checkpoint header. No weights are read."""

import json
from pathlib import Path

import pytest
import torch

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import TextConfig
from qwen38_inf2.model import Qwen38Step

FIXTURES = Path(__file__).parent / "fixtures"
GB = 1e9


@pytest.fixture(scope="module")
def cfg():
    return TextConfig.from_file(FIXTURES / "config.json")


@pytest.fixture(scope="module")
def header():
    return json.loads((FIXTURES / "model.header.json").read_text())


def test_real_config_parses(cfg):
    assert cfg.num_hidden_layers == 64
    assert cfg.layer_types.count("full_attention") == 16
    assert cfg.rotary_dim == 64
    cfg.validate_tp(2)


@pytest.mark.parametrize("tp_degree", [1, 2])
@pytest.mark.parametrize("rank_of", ["first", "last"])
def test_every_parameter_loads_with_the_right_shape(cfg, header, tp_degree, rank_of):
    rank = 0 if rank_of == "first" else tp_degree - 1
    weights = checkpoint.rank_weights(checkpoint.ShapeSource(header), cfg, rank, tp_degree)
    with torch.device("meta"):
        model = Qwen38Step(cfg, max_seq_len=128, tp_degree=tp_degree)
    params = dict(model.named_parameters())
    assert set(params) == set(weights)
    for name, param in params.items():
        assert param.shape == weights[name].shape, name
        assert param.dtype == weights[name].dtype, name


def test_only_vision_mtp_and_fp8_scales_go_unread(cfg, header):
    source = checkpoint.RecordingSource(checkpoint.ShapeSource(header))
    checkpoint.rank_weights(source, cfg, 0, 1)
    unread = set(header) - source.read - {checkpoint.EMBED_TOKENS, checkpoint.LM_HEAD}
    assert all(n.startswith("model.visual.") or n.endswith(checkpoint.UNUSED_LAYER_SUFFIXES) for n in unread), unread
    assert not [n for n in source.read if "visual" in n or "mtp" in n]


def test_two_rank_weights_fit_a_single_inf2_chip(cfg, header):
    """Inf2 has 32 GB per chip, 16 GB per NeuronCore. Leave room for caches and scratch."""
    source = checkpoint.ShapeSource(header)
    per_rank = [checkpoint.device_bytes_per_rank(checkpoint.rank_weights(source, cfg, r, 2)) for r in range(2)]
    assert per_rank[0] == per_rank[1]
    assert 6.5 * GB < per_rank[0] < 8 * GB, per_rank
