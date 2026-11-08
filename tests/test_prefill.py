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


def batched_model(cfg, tensors, batch, verify_tokens=0, saved_states=False):
    from qwen38_inf2 import checkpoint
    from qwen38_inf2.model import Qwen38Step

    model = Qwen38Step(cfg, MAX_SEQ_LEN, dtype=torch.float32, batch_size=batch, verify_tokens=verify_tokens, saved_states=saved_states).eval()
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


def test_sessions_on_the_real_model_give_the_same_tokens_with_saved_prefixes_as_without():
    """A turn continuing a slot's saved states must give exactly what prefilling its whole prompt gives."""
    from qwen38_inf2.scheduler import Scheduler

    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    head = host_head(tensors)
    # Within MAX_SEQ_LEN less a chunk. Each later turn continues from where the last turn's prompt ended, mid-chunk.
    base = PROMPT + [5, 9] * 2
    sessions = [
        [base, base + [7, 8, 9, 10, 11, 12, 13], base + [7, 8, 9, 10, 11, 12, 13] + [21] * 9],
        [[60, 61, 62] * 6, [60, 61, 62] * 6 + [4] * 15],
    ]
    got = {}
    for reuse in (True, False):
        model = batched_model(cfg, tensors, 2, saved_states=True)
        scheduler = Scheduler(prefill_of(model), prefill_of(model), head, cfg, 2, MAX_SEQ_LEN, CHUNK, (), torch.float32, prefill_of(model), reuse)
        out = {}
        with torch.no_grad():
            for session, turn, token in scheduler.run_sessions(sessions, 6):
                out.setdefault((session, turn), []).append(token)
        got[reuse] = (out, scheduler.reused_tokens)
    assert got[True][0] == got[False][0]
    assert got[False][1] == 0 and got[True][1] == sum(len(turns[t - 1]) for turns in sessions for t in range(1, len(turns)))


def test_verify_steps_give_what_decoding_the_same_tokens_one_by_one_gives():
    """Verify steps commit only the tokens accepted, so after any mix of accepted counts each position's hidden
    state matches decoding the accepted tokens one at a time."""
    from qwen38_inf2.generate import verify_inputs

    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    head = host_head(tensors)
    tokens, k = [5, 77, 13, 2, 91, 40, 8, 66, 31, 12, 3], 4
    with torch.no_grad():
        plain = batched_model(cfg, tensors, 2)
        prefill_prompt(prefill_of(plain), head, cfg, 0, PROMPT, CHUNK, torch.float32)
        want = []
        for i, token in enumerate(tokens):
            embeds = head.embed_batch([token, 0], torch.float32)
            out = plain(inputs_embeds=embeds, **batch_step_inputs(cfg, [len(PROMPT) + i, 0], MAX_SEQ_LEN, torch.float32, [True, False]))
            want.append(out[0, 0])

        spec = batched_model(cfg, tensors, 2, verify_tokens=k)
        prefill_prompt(prefill_of(spec), head, cfg, 0, PROMPT, CHUNK, torch.float32)
        done, accepted = 0, 0
        # Each step commits a different number of the last one's tokens: drafts after the accepted ones are junk.
        for take in [2, 4, 1, 3, 1]:
            batch_tokens = tokens[done : done + take] + [7] * (k - take)
            embeds = head.embed_batch(batch_tokens + [0] * k, torch.float32).reshape(2, k, -1)
            out = spec(inputs_embeds=embeds, **verify_inputs(cfg, [len(PROMPT) + done, 0], k, MAX_SEQ_LEN, torch.float32, [True, False], [accepted, 0]))
            for j in range(take):
                assert torch.allclose(out[0, j], want[done + j], atol=1e-4, rtol=1e-4), (done, j, (out[0, j] - want[done + j]).abs().max())
            done, accepted = done + take, take



def test_speculative_decoding_gives_the_tokens_plain_greedy_decoding_gives():
    from qwen38_inf2.scheduler import Scheduler

    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    head = host_head(tensors)
    # Repeats in the prompts give the drafts something to match.
    prompts = [PROMPT + PROMPT[:12], [9, 9, 4, 1] * 5, PROMPT[5:] + [3] * 9, [11, 2, 60, 7] * 4]
    got = {}
    for k in (1, 4):
        model = batched_model(cfg, tensors, 2, verify_tokens=k if k > 1 else 0)
        scheduler = Scheduler(
            prefill_of(model), prefill_of(model), head, cfg, 2, MAX_SEQ_LEN, CHUNK, (), torch.float32,
            verify=prefill_of(model) if k > 1 else None, verify_tokens=k,
        )
        out = [[] for _ in prompts]
        with torch.no_grad():
            for index, token in scheduler.run(prompts, 12):
                out[index].append(token)
        got[k] = (out, scheduler.decode_calls, scheduler.drafted_tokens)
    assert got[4][0] == got[1][0]
    # The tiny model's tokens are near random, so few drafts match, but some must, and they never cost steps.
    assert got[4][2] > 0 and got[4][1] <= got[1][1]


def test_attention_cache_chunk_writes_what_prefill_writes():
    """`FullAttentionStep.cache_chunk`, which the MTP layer's prefill uses, must leave the cache exactly as
    `prefill` does, in both cache formats."""
    from qwen38_inf2 import fp4
    from qwen38_inf2.model import FullAttentionStep, identity_all_reduce

    for kv in ("bf16", "fp8"):
        cfg = replace(tiny_config(), weight_format="fp8", kv_cache_format=kv)
        torch.manual_seed(2)
        attn = FullAttentionStep(cfg, 1, MAX_SEQ_LEN, torch.float32, identity_all_reduce, 2).eval()
        for name, param in attn.named_parameters():
            param.data = fp4.float_to_fp8(torch.randn(param.shape) * 0.2) if name.endswith("weight_fp8") else torch.rand(param.shape) * 0.1 + 0.02
        other = FullAttentionStep(cfg, 1, MAX_SEQ_LEN, torch.float32, identity_all_reduce, 2).eval()
        other.load_state_dict(attn.state_dict())
        with torch.no_grad():
            for start in (0, CHUNK):
                x = torch.randn(1, CHUNK, cfg.hidden_size)
                inputs = prefill_inputs(cfg, 1, start, CHUNK - 3, CHUNK, torch.float32)
                attn.prefill(x, inputs["cos"], inputs["sin"], inputs["slot"], inputs["start"], inputs["n_valid"])
                other.cache_chunk(x, inputs["cos"], inputs["sin"], inputs["slot"], inputs["start"])
        assert torch.equal(attn.k_cache, other.k_cache) and torch.equal(attn.v_cache, other.v_cache)
        assert attn.k_cache[1, :, : 2 * CHUNK].abs().sum() > 0


def test_attention_score_bound_holds():
    """The prefill attention kernel subtracts `FullAttentionStep._score_bound` less a headroom from its scores, so
    no score may exceed it, here with norm weights large in the same few dims and inputs along those dims."""
    from qwen38_inf2 import fp4
    from qwen38_inf2.generate import rotary
    from qwen38_inf2.model import FullAttentionStep, identity_all_reduce

    cfg = replace(tiny_config(), weight_format="fp8")
    torch.manual_seed(3)
    attn = FullAttentionStep(cfg, 1, MAX_SEQ_LEN, torch.float32, identity_all_reduce, 1).eval()
    for name, param in attn.named_parameters():
        param.data = fp4.float_to_fp8(torch.randn(param.shape) * 0.2) if name.endswith("weight_fp8") else torch.randn(param.shape) * 0.5
    attn.q_norm_weight.data[:2] = 4.0
    attn.k_norm_weight.data[:2] = -6.0
    x = torch.randn(1, 64, cfg.hidden_size)
    cos, sin = rotary(cfg, torch.arange(64), torch.float32)
    with torch.no_grad():
        query, key, _, _ = attn._project(x, cos.unsqueeze(0), sin.unsqueeze(0))
    scores = torch.einsum("hqd,kd->hqk", query[0], key[0, 0]) / cfg.head_dim**0.5
    bound = attn._score_bound()
    assert scores.abs().max() <= bound
    # The bound is reached by a normed query and key with all their mass on the dim where both weights are largest.
    q_norm, k_norm = (1 + attn.q_norm_weight).abs(), (1 + attn.k_norm_weight).abs()
    reach = cfg.head_dim**0.5 * q_norm.max() * k_norm.max()
    assert reach <= bound <= 1.081 * reach


def test_mtp_prefill_and_steps_draft_alike():
    """The MTP layer's prefill and its steps must agree: a row's indices run through prefill chunks, then the next
    ones as steps of K, give what running them all as steps gives, and its output is the next step's input."""
    from qwen38_inf2 import fp4
    from qwen38_inf2.generate import mtp_prefill_inputs, mtp_step_inputs
    from qwen38_inf2.model import MTPStep, identity_all_reduce

    cfg = replace(tiny_config(), weight_format="fp8")
    torch.manual_seed(0)
    mtp = MTPStep(cfg, 1, MAX_SEQ_LEN, torch.float32, identity_all_reduce, 2).eval()
    for name, param in mtp.named_parameters():
        if name.endswith("weight_fp8"):
            param.data = fp4.float_to_fp8(torch.randn(param.shape) * 0.1)
        else:
            param.data = torch.randn(param.shape) * 0.1
    k, n = 4, 28
    hidden = torch.randn(1, n, cfg.hidden_size)
    embeds = torch.randn(1, n, cfg.hidden_size)
    with torch.no_grad():
        stepped = []
        for start in range(0, n, k):
            pad = lambda t: torch.cat((t[:, start : start + k], torch.zeros(1, k, cfg.hidden_size)), 0)  # noqa: E731
            out = mtp(hidden=pad(hidden), inputs_embeds=pad(embeds), **mtp_step_inputs(cfg, [start, 0], k, MAX_SEQ_LEN, torch.float32, [True, False]))
            stepped.append(out[0])
        stepped = torch.cat(stepped)

        fresh = MTPStep(cfg, 1, MAX_SEQ_LEN, torch.float32, identity_all_reduce, 2).eval()
        fresh.load_state_dict(mtp.state_dict())
        for start in range(0, 16, CHUNK):
            fresh(hidden=hidden[:, start : start + CHUNK], inputs_embeds=embeds[:, start : start + CHUNK], **mtp_prefill_inputs(cfg, 0, start, CHUNK, CHUNK, torch.float32))
        mixed = []
        for start in range(16, n, k):
            pad = lambda t: torch.cat((t[:, start : start + k], torch.zeros(1, k, cfg.hidden_size)), 0)  # noqa: E731
            mixed.append(fresh(hidden=pad(hidden), inputs_embeds=pad(embeds), **mtp_step_inputs(cfg, [start, 0], k, MAX_SEQ_LEN, torch.float32, [True, False]))[0])
    assert torch.allclose(torch.cat(mixed), stepped[16:], atol=1e-5, rtol=1e-5)


def test_speculative_decoding_with_mtp_drafts_gives_greedy_tokens_and_drafts_from_the_true_history():
    """With MTP drafts, speculation must still give plain greedy decoding's tokens, and each round's first draft
    must be what the MTP layer gives over the sequence's true history: its cache filled through prefill and steps
    at the right indices, from the main model's hidden states."""
    from qwen38_inf2 import fp4
    from qwen38_inf2.generate import mtp_step_inputs
    from qwen38_inf2.model import MTPStep, identity_all_reduce
    from qwen38_inf2.scheduler import Scheduler

    cfg = tiny_config()
    _, tensors = build_reference(cfg)
    head = host_head(tensors)
    mtp_cfg = replace(cfg, weight_format="fp8")
    torch.manual_seed(1)
    mtp = MTPStep(mtp_cfg, 1, MAX_SEQ_LEN, torch.float32, identity_all_reduce, 2).eval()
    for name, param in mtp.named_parameters():
        param.data = fp4.float_to_fp8(torch.randn(param.shape) * 0.2) if name.endswith("weight_fp8") else torch.randn(param.shape) * 0.1
    state = {k: v.clone() for k, v in mtp.state_dict().items()}
    prompts = [PROMPT + PROMPT[:12], [9, 9, 4, 1] * 5, PROMPT[5:] + [3] * 9]
    out = {}
    with torch.no_grad():
        for spec in (False, True):
            model = batched_model(cfg, tensors, 2, verify_tokens=4 if spec else 0)
            scheduler = Scheduler(
                prefill_of(model), prefill_of(model), head, cfg, 2, MAX_SEQ_LEN, CHUNK, (), torch.float32,
                verify=prefill_of(model) if spec else None, verify_tokens=4 if spec else 1, mtp=(lambda **inputs: mtp(**inputs)) if spec else None,
            )
            seen = []
            if spec:
                real = scheduler._mtp_drafts

                def recording(slots, real=real):
                    drafts = real(slots)
                    for slot, row in enumerate(slots):
                        if row and row.mtp_seed:
                            seen.append((row.index[0], row.mtp_seed[2] + row.mtp_seed[3], drafts[slot][0]))
                    return drafts

                scheduler._mtp_drafts = recording
            got = [[] for _ in prompts]
            for index, token in scheduler.run(prompts, 12):
                got[index].append(token)
            out[spec] = (got, seen)
    assert out[True][0] == out[False][0]

    # The reference: the MTP layer run in steps of 4 over each finished sequence, from the main model's hidden states.
    plain = batched_model(cfg, tensors, 1)
    reference = {}
    with torch.no_grad():
        for p, prompt in enumerate(prompts):
            ids = prompt + out[False][0][p]
            hidden = []
            for start in range(0, len(ids), CHUNK):
                chunk = ids[start : start + CHUNK]
                embeds = head.embed_batch(chunk + [0] * (CHUNK - len(chunk)), torch.float32).reshape(1, CHUNK, -1)
                hidden.append(plain(inputs_embeds=embeds, **prefill_inputs(cfg, 0, start, len(chunk), CHUNK, torch.float32))[0, : len(chunk)])
            hidden = torch.cat(hidden)
            fresh = MTPStep(mtp_cfg, 1, MAX_SEQ_LEN, torch.float32, identity_all_reduce, 2).eval()
            fresh.load_state_dict(state)
            picks = []
            for start in range(0, len(ids) - 1, 4):
                n = min(4, len(ids) - 1 - start)
                pad = lambda t: torch.stack((torch.cat((t, torch.zeros(4 - n, t.shape[-1]))), torch.zeros(4, t.shape[-1])))  # noqa: E731
                embeds = head.embed_batch(ids[start + 1 : start + 1 + n], torch.float32)[:, 0]
                y = fresh(hidden=pad(hidden[start : start + n]), inputs_embeds=pad(embeds), **mtp_step_inputs(cfg, [start, 0], 4, MAX_SEQ_LEN, torch.float32, [True, False]))
                picks += head.next_tokens(y[0, :n].unsqueeze(1))
            reference[p] = picks
    checked = [(p, t, d, reference[p][t]) for p, t, d in out[True][1] if t < len(reference[p])]
    assert len(checked) > 10 and all(d == r for _, _, d, r in checked), [c for c in checked if c[2] != c[3]][:5]
