"""The prefill graph must give the same model as decoding token by token, and as Hugging Face."""

from dataclasses import replace

import torch

from qwen38_inf2.config import tiny_config
from qwen38_inf2.generate import batch_step_inputs, greedy_generate, prefill_inputs, prefill_prompt
from qwen38_inf2.model import chunked_gated_delta

from .reference import build_reference
from .test_model_parity import MAX_SEQ_LEN, host_head, load_model

CHUNK = 16
PROMPT = [3, 17, 42, 99, 5, 8, 120, 7, 64, 33, 2, 91, 14, 77, 50, 6, 23, 88, 1, 45, 12]


def recurrent_gated_delta(query, key, value, log_decay, beta, state):
    """The rule one token at a time, as the decode graph runs it."""
    outputs = []
    for t in range(key.shape[1]):
        state = state * log_decay[:, t].exp().reshape(-1, 1, 1)
        delta = (value[:, t] - (state * key[:, t].unsqueeze(-1)).sum(-2)) * beta[:, t].unsqueeze(-1)
        state = state + key[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * query[:, t].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, dim=1), state


def test_chunked_rule_matches_the_recurrent_rule_and_padding_changes_nothing():
    torch.manual_seed(0)
    heads, tokens, dim = 3, 128, 32
    query = torch.nn.functional.normalize(torch.randn(heads, tokens, dim), dim=-1) / dim**0.5
    key = torch.nn.functional.normalize(torch.randn(heads, tokens, dim), dim=-1)
    value = torch.randn(heads, tokens, dim)
    log_decay = -torch.rand(heads, tokens) * 0.5
    beta = torch.rand(heads, tokens)
    state = torch.randn(heads, dim, dim)

    want_out, want_state = recurrent_gated_delta(query, key, value, log_decay, beta, state)
    got_out, got_state = chunked_gated_delta(query, key, value, log_decay, beta, state, chunk=64)
    assert (got_out - want_out).abs().max() < 1e-3
    assert (got_state - want_state).abs().max() < 1e-3

    # Tokens past the real ones, with no decay and no write, must leave the state alone.
    real = 100
    log_decay[:, real:], beta[:, real:] = 0, 0
    _, want_state = recurrent_gated_delta(query[:, :real], key[:, :real], value[:, :real], log_decay[:, :real], beta[:, :real], state)
    _, got_state = chunked_gated_delta(query, key, value, log_decay, beta, state, chunk=64)
    assert (got_state - want_state).abs().max() < 1e-3


def test_chunked_rule_stays_exact_with_strongly_correlated_keys():
    """The triangular solve inside a piece must not cancel catastrophically when keys nearly coincide. A doubling
    series over the whole 64-token piece gave errors of 1e9 here."""
    torch.manual_seed(0)
    heads, tokens, dim = 2, 256, 64
    shared = torch.randn(heads, 1, dim)
    key = torch.nn.functional.normalize(0.99 * shared + 0.01 * torch.randn(heads, tokens, dim), dim=-1)
    query = torch.nn.functional.normalize(torch.randn(heads, tokens, dim), dim=-1) / dim**0.5
    value = torch.randn(heads, tokens, dim)
    log_decay, beta = -torch.rand(heads, tokens) * 0.01, torch.rand(heads, tokens)
    state = torch.randn(heads, dim, dim) * 0.1
    want_out, want_state = recurrent_gated_delta(*(t.double() for t in (query, key, value, log_decay, beta, state)))
    for chunk in (64, 128):
        got_out, got_state = chunked_gated_delta(query, key, value, log_decay, beta, state, chunk=chunk)
        assert (got_out - want_out).abs().max() < 1e-4
        assert (got_state - want_state).abs().max() < 1e-4


def prefill_of(model):
    return lambda **inputs: model(**inputs)


def test_prefill_logits_match_hugging_face_at_every_prompt_position():
    cfg = tiny_config()
    reference, tensors = build_reference(cfg)
    model = load_model(tensors, cfg, torch.float32)
    head = host_head(tensors)
    with torch.no_grad():
        want = reference(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0].float()
        got = []
        for start in range(0, len(PROMPT), CHUNK):
            ids = PROMPT[start : start + CHUNK]
            embeds = head.embed_batch(ids + [0] * (CHUNK - len(ids)), torch.float32).reshape(1, CHUNK, -1)
            hidden = model(inputs_embeds=embeds, **prefill_inputs(cfg, 0, start, len(ids), CHUNK, torch.float32))
            got.extend(head.logits(hidden[:, i]) for i in range(len(ids)))
    assert (torch.stack(got) - want).abs().max() < 1e-4


def test_prefill_then_decode_matches_hf_generate():
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
        got = list(greedy_generate(model, head, cfg, PROMPT, 20, MAX_SEQ_LEN, dtype=torch.float32, prefill=prefill_of(model), chunk=CHUNK))
    assert got == want


def batched_model(cfg, tensors, batch):
    from qwen38_inf2 import checkpoint
    from qwen38_inf2.model import Qwen38Step

    model = Qwen38Step(cfg, MAX_SEQ_LEN, dtype=torch.float32, batch_size=batch).eval()
    model.load_state_dict(checkpoint.rank_weights(checkpoint.DictSource(tensors), cfg, 0, 1, torch.float32), strict=False)
    return model


def caches(model):
    """Every state buffer, without the KV caches' last chunk of positions, where junk goes and nothing reads."""
    return {name: (buffer[:, :, :-CHUNK] if name.endswith("_cache") else buffer).clone() for name, buffer in model.named_buffers()}


def test_prefill_and_inactive_rows_touch_only_their_own_slot():
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    head = host_head(tensors)
    model = batched_model(cfg, tensors, 3)
    with torch.no_grad():
        torch.manual_seed(2)
        for buffer in model.buffers():
            buffer.copy_(torch.randn_like(buffer.float()).to(buffer.dtype))
        before = caches(model)

        prefill_prompt(prefill_of(model), head, cfg, 1, PROMPT, CHUNK, torch.float32)
        after = caches(model)
        for name in before:
            assert torch.equal(before[name][[0, 2]], after[name][[0, 2]]), name
            assert not torch.equal(before[name][1], after[name][1]), name

        # A decode step with every row inactive must change nothing at all, wherever the rows say they are.
        embeds = head.embed_batch([5, 6, 7], torch.float32)
        model(inputs_embeds=embeds, **batch_step_inputs(cfg, [0, 30, 0], MAX_SEQ_LEN, torch.float32, [False, False, False]))
        for name, buffer in caches(model).items():
            assert torch.equal(buffer, after[name]), name


def test_a_prefilled_row_decodes_like_a_row_fed_token_by_token():
    """Row 1 is prefilled while rows 0 and 2 decode other prompts, and must agree with a model fed one token at a time."""
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    head = host_head(tensors)
    alone = load_model(tensors, cfg, torch.float32)
    with torch.no_grad():
        want = list(greedy_generate(alone, head, cfg, PROMPT, 10, MAX_SEQ_LEN, dtype=torch.float32))

        model = batched_model(cfg, tensors, 3)
        others = [[9, 9, 4], [11, 2]]
        for position in range(2):
            ids = [others[0][position], 0, others[1][position]]
            model(inputs_embeds=head.embed_batch(ids, torch.float32), **batch_step_inputs(cfg, [position, 0, position], MAX_SEQ_LEN, torch.float32, [True, False, True]))
        hidden = prefill_prompt(prefill_of(model), head, cfg, 1, PROMPT, CHUNK, torch.float32)
        got = [head.next_tokens(hidden)[0]]
        position = len(PROMPT)
        for step in range(9):
            ids = [7, got[-1], 8]
            out = model(inputs_embeds=head.embed_batch(ids, torch.float32), **batch_step_inputs(cfg, [2 + step, position, 2 + step], MAX_SEQ_LEN, torch.float32))
            got.append(head.next_tokens(out[1:2])[0])
            position += 1
    assert got == want


def test_fp8_kv_cache_stays_close_to_bf16():
    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    for index, kind in enumerate(cfg.layer_types):
        if kind == "full_attention":
            # Calibrated scales map the largest cached value to FP8's largest. The tiny model's are about 4.
            for name in ("k_scale", "v_scale"):
                tensors[f"model.language_model.layers.{index}.self_attn.{name}"] = torch.tensor([4 / 448], dtype=torch.bfloat16)
    head = host_head(tensors)
    exact = load_model(tensors, cfg, torch.float32)
    fp8 = load_model(tensors, replace(cfg, kv_cache_format="fp8"), torch.float32)
    with torch.no_grad():
        want = head.logits(prefill_prompt(prefill_of(exact), head, cfg, 0, PROMPT, CHUNK, torch.float32))
        got = head.logits(prefill_prompt(prefill_of(fp8), head, cfg, 0, PROMPT, CHUNK, torch.float32))
    assert (got - want).abs().max() < 0.05 * want.abs().max()


def test_scheduler_on_the_real_model_matches_hf_generate_for_every_request():
    from qwen38_inf2.scheduler import Scheduler

    cfg = tiny_config()
    reference, tensors = build_reference(cfg)
    head = host_head(tensors)
    model = batched_model(cfg, tensors, 2)
    prompts = [PROMPT, [9, 9, 4, 1], PROMPT[5:] + [3] * 20, [11, 2, 60, 7, 7]]
    with torch.no_grad():
        want = []
        for prompt in prompts:
            ids = torch.tensor([prompt])
            out = reference.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False,
                top_k=None, top_p=None, temperature=None, pad_token_id=0, eos_token_id=None,
            )
            want.append(out[0, len(prompt):].tolist())
        scheduler = Scheduler(prefill_of(model), prefill_of(model), head, cfg, 2, MAX_SEQ_LEN, CHUNK, (), torch.float32)
        got = [[] for _ in prompts]
        for index, token_id in scheduler.run(prompts, 8):
            got[index].append(token_id)
    assert got == want
