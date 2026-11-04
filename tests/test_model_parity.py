"""The single-token graph must reproduce the Hugging Face model on the same FP4 weights."""

import threading
from dataclasses import replace

import pytest
import torch

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import tiny_config
from qwen38_inf2.generate import BlockChain, HostHead, greedy_generate, step_inputs
from qwen38_inf2.model import Qwen38Step

from .reference import build_reference

MAX_SEQ_LEN = 64
PROMPT = [3, 17, 42, 99, 5, 8, 120, 7, 64, 33]


def load_model(tensors, cfg, dtype, tp_degree=1, rank=0, all_reduce=None):
    kwargs = {"all_reduce": all_reduce} if all_reduce else {}
    model = Qwen38Step(cfg, MAX_SEQ_LEN, tp_degree=tp_degree, dtype=dtype, **kwargs).eval()
    weights = checkpoint.rank_weights(checkpoint.DictSource(tensors), cfg, rank, tp_degree, dtype)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert not unexpected
    assert all(name.endswith(("_cache", "_state")) for name in missing), missing
    return model


def run_tokens(model, cfg, head, token_ids, dtype):
    outputs = []
    for position, token_id in enumerate(token_ids):
        inputs = step_inputs(cfg, position, MAX_SEQ_LEN, dtype)
        outputs.append(model(inputs_embeds=head.embed(token_id, dtype), **inputs))
    return outputs


def host_head(tensors):
    return HostHead(
        tensors[checkpoint.EMBED_TOKENS], tensors[checkpoint.FINAL_NORM], tensors[checkpoint.LM_HEAD], tiny_config().rms_norm_eps
    )


def reference_and_ours(cfg, dtype):
    reference, tensors = build_reference(cfg, dtype=dtype)
    model = load_model(tensors, cfg, dtype)
    head = host_head(tensors)
    with torch.no_grad():
        want = reference(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0].float()
        got = torch.stack([head.logits(h) for h in run_tokens(model, cfg, head, PROMPT, dtype)])
    return got, want


def test_float32_logits_match_reference_exactly():
    got, want = reference_and_ours(tiny_config(), torch.float32)
    assert (got - want).abs().max() < 1e-5
    assert torch.equal(got.argmax(-1), want.argmax(-1))


def test_bf16_logits_match_reference_within_rounding():
    """BF16 has 8 significant bits, so an untrained model's near-flat logits can tie."""
    got, want = reference_and_ours(tiny_config(), torch.bfloat16)
    assert (got - want).abs().max() < 0.02
    chosen = got.argmax(-1, keepdim=True)
    assert (want.max(-1, keepdim=True).values - want.gather(-1, chosen)).max() < 0.02


def test_greedy_output_matches_hf_generate():
    cfg = tiny_config()
    reference, tensors = build_reference(cfg)
    model = load_model(tensors, cfg, torch.float32)
    head = host_head(tensors)

    with torch.no_grad():
        ids = torch.tensor([PROMPT])
        want = reference.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=20, do_sample=False,
            top_k=None, top_p=None, temperature=None, pad_token_id=0, eos_token_id=None,
        )[0, len(PROMPT):].tolist()
        got = list(greedy_generate(model, head, cfg, PROMPT, 20, MAX_SEQ_LEN, dtype=torch.float32))

    assert got == want


def test_keep_state_zero_starts_a_fresh_sequence():
    """A second generation on the same model must not see state left by the first."""
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    model = load_model(tensors, cfg, torch.float32)
    head = host_head(tensors)
    with torch.no_grad():
        first = list(greedy_generate(model, head, cfg, PROMPT, 8, MAX_SEQ_LEN, dtype=torch.float32))
        list(greedy_generate(model, head, cfg, [9, 9, 9, 1, 2], 8, MAX_SEQ_LEN, dtype=torch.float32))
        again = list(greedy_generate(model, head, cfg, PROMPT, 8, MAX_SEQ_LEN, dtype=torch.float32))
    assert first == again


def test_chained_blocks_match_one_whole_graph():
    """One compiled block loaded per repeat must equal the model compiled as a single graph."""
    cfg = tiny_config(num_blocks=3)
    _, tensors = build_reference(cfg)
    head = host_head(tensors)
    dtype = torch.float32
    source = checkpoint.DictSource(tensors)
    block_layers = 4

    block_cfg = replace(cfg, layer_types=cfg.layer_types[:block_layers])
    blocks = []
    for block in range(cfg.num_hidden_layers // block_layers):
        model = Qwen38Step(block_cfg, MAX_SEQ_LEN, dtype=dtype).eval()
        weights = checkpoint.block_weights(source, cfg, block, block_layers, 0, 1, dtype)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        assert not unexpected and all(name.endswith(("_cache", "_state")) for name in missing)
        blocks.append(model)

    with torch.no_grad():
        whole = load_model(tensors, cfg, dtype)
        want = list(greedy_generate(whole, head, cfg, PROMPT, 12, MAX_SEQ_LEN, dtype=dtype))
        got = list(greedy_generate(BlockChain(blocks), head, cfg, PROMPT, 12, MAX_SEQ_LEN, dtype=dtype))
    assert got == want


def test_block_weights_reject_a_non_repeating_layout():
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    with pytest.raises(ValueError):
        checkpoint.block_weights(checkpoint.DictSource(tensors), cfg, 1, 2, 0, 1)  # layers 2, 3 are linear, full


class ThreadedAllReduce:
    """Two ranks running in lock-step threads, summing tensors like the real collective."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, action=self._finish)
        self.values = {}
        self.result = None
        self.rank_of = {}
        self.count = 0

    def _finish(self):
        self.result = sum(self.values.values())
        self.values = {}
        self.count += 1

    def bind(self, rank):
        def all_reduce(x):
            self.values[rank] = x
            self.barrier.wait()
            return self.result
        return all_reduce


@pytest.mark.parametrize("tp_degree", [2])
def test_tensor_parallel_ranks_match_single_rank(tp_degree):
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    head = host_head(tensors)
    dtype = torch.float32

    with torch.no_grad():
        single = load_model(tensors, cfg, dtype)
        want = run_tokens(single, cfg, head, PROMPT, dtype)

        reducer = ThreadedAllReduce(tp_degree)
        results = {}

        def worker(rank):
            torch.set_num_threads(1)
            model = load_model(tensors, cfg, dtype, tp_degree, rank, reducer.bind(rank))
            with torch.no_grad():
                results[rank] = run_tokens(model, cfg, head, PROMPT, dtype)

        threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(tp_degree)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # One reduce per layer for the mixer, one per layer for the MLP, per token.
    assert reducer.count == 2 * cfg.num_hidden_layers * len(PROMPT)
    for rank in range(tp_degree):
        for got, expected in zip(results[rank], want):
            assert (got - expected).abs().max() < 1e-5


def test_all_text_tensors_are_consumed():
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    source = checkpoint.RecordingSource(checkpoint.DictSource(tensors))
    checkpoint.rank_weights(source, cfg, 0, 1)
    host_only = {checkpoint.EMBED_TOKENS, checkpoint.FINAL_NORM, checkpoint.LM_HEAD}
    assert set(tensors) - source.read == host_only
