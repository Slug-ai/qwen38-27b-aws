"""The single-token graph must reproduce the Hugging Face model on the same FP4 weights."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import tiny_config
from qwen38_inf2.cpu_sim import SimulatedCoresStep, ThreadedAllReduce
from qwen38_inf2.generate import BlockChain, ChipHead, HostHead, batched_greedy_generate, greedy_generate, step_inputs
from qwen38_inf2.model import HeadStep, Qwen38Step, rms_norm

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
    return HostHead(checkpoint.DictSource(tensors), tiny_config().rms_norm_eps, chunk_rows=50)


def test_host_head_never_loads_the_full_tables():
    """Whole-table reads of embed_tokens or lm_head would not fit a 16 GiB inf2.xlarge host."""
    _, tensors = build_reference(tiny_config())
    source = checkpoint.RecordingSource(checkpoint.DictSource(tensors))
    head = HostHead(source, tiny_config().rms_norm_eps, chunk_rows=50)
    head.embed(3, torch.float32)
    assert not {checkpoint.EMBED_TOKENS, checkpoint.LM_HEAD} & source.read
    assert torch.equal(head.lm_head, tensors[checkpoint.LM_HEAD].float())


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


def test_simulated_cores_from_weight_files_match_hf_generate(tmp_path):
    """The whole offline path: per-core files on disk, chained blocks, two threads, greedy loop."""
    cfg = tiny_config(num_blocks=3)
    reference, tensors = build_reference(cfg)
    source = checkpoint.DictSource(tensors)
    block_layers, tp_degree = 4, 2
    checkpoint.write_rank_weights(source, cfg, tmp_path, block_layers, tp_degree)
    info = {"layers": block_layers, "tp_degree": tp_degree, "max_seq_len": MAX_SEQ_LEN}

    step = SimulatedCoresStep(tmp_path, cfg, info, torch.float32)
    got = list(greedy_generate(step, host_head(tensors), cfg, PROMPT, 12, MAX_SEQ_LEN, dtype=torch.float32))

    with torch.no_grad():
        ids = torch.tensor([PROMPT])
        want = reference.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=12, do_sample=False,
            top_k=None, top_p=None, temperature=None, pad_token_id=0, eos_token_id=None,
        )[0, len(PROMPT):].tolist()
    assert got == want


def test_batch_rows_match_separate_runs_and_refill_cleanly():
    """Rows at different positions must not affect each other, including a row reused for a new prompt."""
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    head = host_head(tensors)
    dtype = torch.float32
    prompts = [PROMPT, [9, 9, 9, 1, 2], [4], PROMPT[::-1], [77, 78, 79]]
    new_tokens = [6, 9, 3, 7, 5]

    single = load_model(tensors, cfg, dtype)
    with torch.no_grad():
        want = {
            i: list(greedy_generate(single, head, cfg, prompt, new_tokens[i], MAX_SEQ_LEN, dtype=dtype))
            for i, prompt in enumerate(prompts)
        }

        batch_size = 2
        model = Qwen38Step(cfg, MAX_SEQ_LEN, dtype=dtype, batch_size=batch_size).eval()
        weights = checkpoint.rank_weights(checkpoint.DictSource(tensors), cfg, 0, 1, dtype)
        model.load_state_dict(weights, strict=False)

        got = {i: [] for i in range(len(prompts))}
        # A stop token per prompt isn't available here, so every row runs to the longest count and is compared up to its own.
        for index, token_id in batched_greedy_generate(model, head, cfg, prompts, batch_size, max(new_tokens), MAX_SEQ_LEN, dtype=dtype):
            got[index].append(token_id)

    for i, tokens in want.items():
        assert got[i][: len(tokens)] == tokens


def test_batched_graph_matches_two_simulated_cores(tmp_path):
    """The batched path through per-core files, chained blocks and the two-core sum."""
    cfg = tiny_config(num_blocks=3)
    _, tensors = build_reference(cfg)
    source = checkpoint.DictSource(tensors)
    checkpoint.write_rank_weights(source, cfg, tmp_path, 4, 2)
    dtype = torch.float32
    head = host_head(tensors)
    prompts = [PROMPT, [9, 9, 9, 1, 2], [4]]

    one = SimulatedCoresStep(tmp_path, cfg, {"layers": 4, "tp_degree": 2, "max_seq_len": MAX_SEQ_LEN}, dtype)
    want = {i: list(greedy_generate(one, head, cfg, p, 6, MAX_SEQ_LEN, dtype=dtype)) for i, p in enumerate(prompts)}

    info = {"layers": 4, "tp_degree": 2, "max_seq_len": MAX_SEQ_LEN, "batch_size": 3}
    batched = SimulatedCoresStep(tmp_path, cfg, info, dtype)
    got = {i: [] for i in range(len(prompts))}
    for index, token_id in batched_greedy_generate(batched, head, cfg, prompts, 3, 6, MAX_SEQ_LEN, dtype=dtype):
        got[index].append(token_id)
    assert got == want


def test_fp8_weights_match_the_reference():
    cfg = replace(tiny_config(), weight_format="fp8")
    reference, tensors = build_reference(cfg)
    model = load_model(tensors, cfg, torch.float32)
    head = host_head(tensors)
    with torch.no_grad():
        want = reference(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0].float()
        got = torch.stack([head.logits(h) for h in run_tokens(model, cfg, head, PROMPT, torch.float32)])
    assert (got - want).abs().max() < 1e-4
    assert torch.equal(got.argmax(-1), want.argmax(-1))


@pytest.mark.parametrize("tp_degree", [1, 2])
def test_chip_head_picks_the_same_tokens_as_the_host(tp_degree):
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    source = checkpoint.DictSource(tensors)
    host = host_head(tensors)
    torch.manual_seed(1)
    hidden = torch.randn(5, 1, cfg.hidden_size)
    want = host.next_tokens(hidden)

    reducer = ThreadedAllReduce(tp_degree)
    graphs = []
    for rank in range(tp_degree):
        graph = HeadStep(cfg, tp_degree, batch_size=8, all_reduce=reducer.bind(rank), group_rows=8).eval()
        graph.load_state_dict(checkpoint.head_weights(source, rank, tp_degree))
        graphs.append(graph)

    def run_graph(hidden):
        with ThreadPoolExecutor(tp_degree) as pool:
            return list(pool.map(lambda g: g(hidden), graphs))[0]

    with torch.no_grad():
        chip = ChipHead(source, cfg.rms_norm_eps, run_graph, batch_size=8, group_rows=8)
        assert chip.next_tokens(hidden) == want
        assert chip.next_tokens(hidden[:2]) == want[:2]
        # The chip rounds its input to BF16, so a tiny model's nearly flat logits must still give a near-best token.
        rounded = hidden.to(torch.bfloat16)
        norm = rms_norm(rounded.float(), host.norm_weight, cfg.rms_norm_eps).squeeze(1)
        logits = norm @ host.lm_head.T
        chosen = logits.gather(1, torch.tensor(chip.next_tokens(rounded)).unsqueeze(1))
        assert (logits.max(1, keepdim=True).values - chosen).max() < 0.02


def test_bf16_recurrent_state_stays_close_to_float32():
    cfg = tiny_config()
    reference, tensors = build_reference(cfg)
    head = host_head(tensors)
    with torch.no_grad():
        want = torch.stack([head.logits(h) for h in run_tokens(load_model(tensors, cfg, torch.float32), cfg, head, PROMPT, torch.float32)])
        bf16_cfg = replace(cfg, recurrent_state_format="bf16")
        model = load_model(tensors, bf16_cfg, torch.float32)
        assert all(layer.linear_attn.recurrent_state.dtype == torch.bfloat16 for layer in model.layers[:3])
        got = torch.stack([head.logits(h) for h in run_tokens(model, bf16_cfg, head, PROMPT, torch.float32)])
    assert (got - want).abs().max() < 0.02


def test_all_text_tensors_are_consumed():
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    source = checkpoint.RecordingSource(checkpoint.DictSource(tensors))
    checkpoint.rank_weights(source, cfg, 0, 1)
    host_only = {checkpoint.EMBED_TOKENS, checkpoint.FINAL_NORM, checkpoint.LM_HEAD}
    assert set(tensors) - source.read == host_only
