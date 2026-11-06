"""Hand-written Neuron kernels. Requires the Neuron SDK, so import this only in that environment."""

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

# The Tensor Engine contracts at most 128 elements per instruction, holds at most 128 stationary columns and
# produces at most 512 output columns.
CONTRACT_TILE = 128
STATIONARY_TILE = 128
OUTPUT_TILE = 512
# Weight columns read per DMA. Each partition then gets 2 KB at a time, against 512 bytes for one output tile,
# which reads weights about 20% faster.
LOAD_TILE = 2048


def _tiles(total: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(size, total - start)) for start in range(0, total, size)]


def fp8_linear_body(x: nl.ndarray, weight: nl.ndarray) -> nl.ndarray:
    """``x.T @ weight``, with weights read straight from memory as FP8, so nothing is decoded on the chip.

    `x` is rounded to FP8 e5m2 too, because the Tensor Engine on NeuronCore-v2 only multiplies FP8 by FP8.
    With BF16 `x` the compiler widens every weight tile to BF16 first, which halved the speed. e5m2 keeps
    2 mantissa bits and an exponent per value, which is finer than the MXFP4 activations, 1 mantissa bit per value
    and one exponent per 32, that the checkpoint was calibrated for.

    Args:
        x: BF16 ``[in_features, tokens]``. Stationary in the Tensor Engine, 128 tokens at a time, so every weight
            tile read serves all tokens.
        weight: uint8 ``[in_features, out_features]``, holding FP8 e5m2 bytes from `fp4.to_fp8`.

    Returns:
        BF16 ``[tokens, out_features]``.
    """
    in_features, tokens = x.shape
    _, out_features = weight.shape
    out = nl.ndarray((tokens, out_features), dtype=x.dtype, buffer=nl.shared_hbm)
    if tokens > STATIONARY_TILE and not (tokens % STATIONARY_TILE or in_features % CONTRACT_TILE or out_features % OUTPUT_TILE):
        _fp8_linear_looped(x, weight, out)
        return out
    contract_tiles = _tiles(in_features, CONTRACT_TILE)
    token_tiles = _tiles(tokens, STATIONARY_TILE)
    # Each output tile needs one PSUM bank per token tile, and half of the 8 banks are left for overlap.
    load_width = OUTPUT_TILE * max(1, (LOAD_TILE // OUTPUT_TILE) // len(token_tiles))

    x_tiles = []
    for start, size in contract_tiles:
        x_tiles.append(
            [nisa.tensor_copy(nl.load(x[start : start + size, t0 : t0 + ts]), dtype=nl.float8_e5m2) for t0, ts in token_tiles]
        )

    for load_start, load_size in _tiles(out_features, load_width):
        outputs = _tiles(load_size, OUTPUT_TILE)
        psums = [[nl.zeros((nl.par_dim(ts), size), nl.float32, buffer=nl.psum) for _, size in outputs] for _, ts in token_tiles]
        for x_row, (start, size) in zip(x_tiles, contract_tiles):
            moving = nl.load(weight[start : start + size, load_start : load_start + load_size]).view(nl.float8_e5m2)
            for x_tile, row in zip(x_row, psums):
                for psum, (out_start, out_size) in zip(row, outputs):
                    psum += nisa.nc_matmul(x_tile, moving[:, out_start : out_start + out_size])
        for (t0, ts), row in zip(token_tiles, psums):
            for psum, (out_start, out_size) in zip(row, outputs):
                result = nl.ndarray((nl.par_dim(ts), out_size), dtype=x.dtype, buffer=nl.sbuf)
                result[:, :] = nisa.tensor_copy(psum, engine=nisa.vector_engine)
                nl.store(out[t0 : t0 + ts, load_start + out_start : load_start + out_start + out_size], result)
    return out


def lm_head_group_maxima_body(x: nl.ndarray, weight: nl.ndarray, group_rows: int) -> nl.ndarray:
    """The largest logit in each group of `group_rows` tokens, for every row of a batch.

    The host then only has to look inside one group, so it needs no copy of lm_head.

    Args:
        x: BF16 ``[hidden, batch]``, already normalized.
        weight: ``[hidden, tokens]``, BF16 or else uint8 holding FP8 e5m2 bytes.

    Returns:
        float32 ``[batch, ceil(tokens / group_rows)]``.
    """
    hidden, batch = x.shape
    _, tokens = weight.shape
    out = nl.ndarray((batch, (tokens + group_rows - 1) // group_rows), dtype=nl.float32, buffer=nl.shared_hbm)
    contract_tiles = _tiles(hidden, CONTRACT_TILE)

    x_tiles = []
    for start, size in contract_tiles:
        x_tiles.append(nl.load(x[start : start + size, :]))

    # A multiple of group_rows, so no group straddles two tiles.
    for out_start, out_size in _tiles(tokens, OUTPUT_TILE - OUTPUT_TILE % group_rows):
        psum = nl.zeros((nl.par_dim(batch), out_size), nl.float32, buffer=nl.psum)
        for x_tile, (start, size) in zip(x_tiles, contract_tiles):
            moving = nl.load(weight[start : start + size, out_start : out_start + out_size])
            psum += nisa.nc_matmul(x_tile, moving if weight.dtype == nl.bfloat16 else moving.view(nl.float8_e5m2))
        logits = nl.ndarray((nl.par_dim(batch), out_size), dtype=nl.float32, buffer=nl.sbuf)
        logits[:, :] = nisa.tensor_copy(psum, engine=nisa.vector_engine)
        groups = _tiles(out_size, group_rows)
        maxima = nl.ndarray((nl.par_dim(batch), len(groups)), dtype=nl.float32, buffer=nl.sbuf)
        for index, (start, size) in enumerate(groups):
            maxima[:, index : index + 1] = nisa.tensor_reduce(nl.maximum, logits[:, start : start + size], axis=[1], keepdims=True)
        first = out_start // group_rows
        nl.store(out[:, first : first + len(groups)], maxima)
    return out


def _fp8_linear_looped(x: nl.ndarray, weight: nl.ndarray, out: nl.ndarray) -> None:
    """`fp8_linear_body` for many tokens, with device loops instead of unrolled ones, writing into `out`.

    At 512 tokens every weight tile serves 4 matmuls, so the Tensor Engine is the limit: this reaches 56 TFLOPS on
    the real projection shapes, its peak, and compiles about 3 times faster than the unrolled form.
    """
    in_features, tokens = x.shape
    _, out_features = weight.shape
    contract = in_features // CONTRACT_TILE
    x_sb = nl.ndarray((nl.par_dim(CONTRACT_TILE), contract, tokens), dtype=nl.float8_e5m2, buffer=nl.sbuf)
    for k in nl.affine_range(contract):
        x_sb[:, k, :] = nisa.tensor_copy(nl.load(x[k * CONTRACT_TILE : (k + 1) * CONTRACT_TILE, :]), dtype=nl.float8_e5m2)
    for n in nl.affine_range(out_features // OUTPUT_TILE):
        psums = [nl.zeros((nl.par_dim(STATIONARY_TILE), OUTPUT_TILE), nl.float32, buffer=nl.psum) for _ in list(range(tokens // STATIONARY_TILE))]
        for k in nl.affine_range(contract):
            moving = nl.load(weight[k * CONTRACT_TILE : (k + 1) * CONTRACT_TILE, n * OUTPUT_TILE : (n + 1) * OUTPUT_TILE]).view(nl.float8_e5m2)
            for t, psum in enumerate(psums):
                psum += nisa.nc_matmul(x_sb[:, k, t * STATIONARY_TILE : (t + 1) * STATIONARY_TILE], moving)
        for t, psum in enumerate(psums):
            result = nl.ndarray((nl.par_dim(STATIONARY_TILE), OUTPUT_TILE), dtype=x.dtype, buffer=nl.sbuf)
            result[...] = nisa.tensor_copy(psum, engine=nisa.vector_engine)
            nl.store(out[t * STATIONARY_TILE : (t + 1) * STATIONARY_TILE, n * OUTPUT_TILE : (n + 1) * OUTPUT_TILE], result)


# Added to scores of keys a query must not see. exp() of it, against any real score, is exactly zero in float32.
MASKED = -30000.0
KEY_BLOCK = 512


def prefill_attention_body(
    q: nl.ndarray, k_cache: nl.ndarray, v_cache: nl.ndarray, k_new: nl.ndarray, v_new: nl.ndarray, start: nl.ndarray, row: nl.ndarray,
    positions: int,
) -> nl.ndarray:
    """Flash attention for a prefill chunk of one row: its queries against the row's cache and against itself.

    Scores stay in SBUF and PSUM, so nothing grows with the cache length but time. Cached positions before
    `start` are all visible and the rest are not, so one mask per key block serves every query. The chunk's own
    keys are visible causally.

    Args:
        q: BF16 ``[kv_heads, head, queries]``, already scaled. A KV head's query heads are stacked, so query
            ``g * chunk + i`` is head g's token i.
        k_cache, v_cache: BF16 ``[rows * kv_heads * positions, head]``, every row's whole cache. The chunk's row is
            read with indirect DMA, so it is never copied, and key blocks are transposed on the Tensor Engine, once
            per block for all query tiles.
        k_new: BF16 ``[kv_heads, head, chunk]``. v_new: BF16 ``[kv_heads, chunk, head]``. The chunk's own keys and values.
        start: float32 ``[128, 1]``, the chunk's first position on every partition.
        row: int32 ``[128, 1]``, ``slot * kv_heads * positions``, where the chunk's row begins, on every partition.
        positions: cache positions per row and KV head.

    Returns:
        BF16 ``[kv_heads, queries, head]``.
    """
    kv_heads, head, queries = q.shape
    chunk = k_new.shape[2]
    d_tiles = _tiles(head, CONTRACT_TILE)
    q_tiles = _tiles(queries, STATIONARY_TILE)
    out = nl.ndarray((kv_heads, queries, head), dtype=q.dtype, buffer=nl.shared_hbm)
    start_tile = nl.load(start)
    row_tile = nl.load(row)
    i_p = nl.arange(STATIONARY_TILE)[:, None]
    i_f = nl.arange(KEY_BLOCK)[None, :]
    i_h = nl.arange(head)[None, :]

    for h in list(range(kv_heads)):
        q_sb = [nl.load(q[h, d0 : d0 + ds, :]) for d0, ds in d_tiles]
        best = [nl.full((nl.par_dim(STATIONARY_TILE), 1), MASKED, dtype=nl.float32, buffer=nl.sbuf) for _ in q_tiles]
        total = [nl.zeros((nl.par_dim(STATIONARY_TILE), 1), dtype=nl.float32, buffer=nl.sbuf) for _ in q_tiles]
        acc = [nl.zeros((nl.par_dim(STATIONARY_TILE), head), dtype=nl.float32, buffer=nl.sbuf) for _ in q_tiles]

        def attend(keys, values, biases, block):
            """One block of keys for every query tile. `biases[t]` is the mask added to query tile t's scores."""
            for t, (q0, qs) in enumerate(q_tiles):
                scores = nl.zeros((nl.par_dim(qs), block), nl.float32, buffer=nl.psum)
                for d, (d0, ds) in enumerate(d_tiles):
                    scores += nisa.nc_matmul(q_sb[d][:, q0 : q0 + qs], keys[d])
                masked = nisa.tensor_tensor(scores, biases[t], op=nl.add, dtype=nl.float32)
                new_best = nisa.tensor_tensor(best[t], nisa.tensor_reduce(nl.maximum, masked, axis=[1], keepdims=True), op=nl.maximum)
                negative = nisa.tensor_scalar(new_best, nl.multiply, -1.0)
                row_sum = nl.ndarray((nl.par_dim(qs), 1), dtype=nl.float32, buffer=nl.sbuf)
                weights = nisa.activation_reduce(nl.exp, masked, reduce_op=nl.add, reduce_res=row_sum, bias=negative, dtype=nl.bfloat16)
                rescale = nisa.activation(nl.exp, best[t], bias=negative)
                total[t][...] = nisa.scalar_tensor_tensor(data=total[t], op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=row_sum)
                mixed = nl.zeros((nl.par_dim(qs), head), nl.float32, buffer=nl.psum)
                for k, k0 in enumerate(list(range(0, block, STATIONARY_TILE))):
                    flipped = nl.ndarray((nl.par_dim(STATIONARY_TILE), qs), dtype=nl.bfloat16, buffer=nl.sbuf)
                    flipped[...] = nisa.nc_transpose(weights[:, k0 : k0 + STATIONARY_TILE])
                    mixed += nisa.nc_matmul(flipped, values[k])
                acc[t][...] = nisa.scalar_tensor_tensor(data=acc[t], op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=mixed)
                best[t][...] = new_best

        # A device loop, so code size and compile time don't grow with the cache.
        for b in nl.sequential_range(positions // KEY_BLOCK):
            b0 = b * KEY_BLOCK
            keys = [nl.ndarray((nl.par_dim(ds), KEY_BLOCK), dtype=k_cache.dtype, buffer=nl.sbuf) for _, ds in d_tiles]
            values = []
            for k0 in list(range(0, KEY_BLOCK, STATIONARY_TILE)):
                index = nisa.tensor_tensor(row_tile, nisa.iota(i_p + h * positions + b0 + k0, dtype=nl.int32), op=nl.add)
                rows = nl.load(k_cache[index, i_h])
                values.append(nl.load(v_cache[index, i_h]))
                for d, (d0, ds) in enumerate(d_tiles):
                    keys[d][:, k0 : k0 + STATIONARY_TILE] = nisa.nc_transpose(rows[:, d0 : d0 + ds])
            # Positions from `start` on hold no keys for this chunk yet.
            later = nisa.iota(i_f + b0 + 0 * i_p, dtype=nl.float32)
            bias = nisa.tensor_scalar(later, nl.greater_equal, start_tile, op1=nl.multiply, operand1=MASKED)
            attend(keys, values, [bias] * len(q_tiles), KEY_BLOCK)

        keys = [nl.load(k_new[h, d0 : d0 + ds, :]) for d0, ds in d_tiles]
        values = [nl.load(v_new[h, k0 : k0 + STATIONARY_TILE, :]) for k0 in list(range(0, chunk, STATIONARY_TILE))]
        i_c = nl.arange(chunk)[None, :]
        zero = nl.zeros((nl.par_dim(STATIONARY_TILE), chunk), dtype=nl.float32, buffer=nl.sbuf)
        # Query tile t holds tokens (q0 % chunk) + p, which see the chunk's keys up to themselves.
        causal = [nisa.affine_select((q0 % chunk) + i_p - i_c >= 0, zero, MASKED) for q0, _ in q_tiles]
        attend(keys, values, causal, chunk)

        for t, (q0, qs) in enumerate(q_tiles):
            inverse = nisa.reciprocal(total[t])
            result = nisa.tensor_scalar(acc[t], nl.multiply, inverse, dtype=q.dtype)
            nl.store(out[h, q0 : q0 + qs, :], result)
    return out


PIECE = 128


def _matmul(stationary, moving, rows: int, cols: int):
    """``stationary.T @ moving`` into a fresh float32 PSUM tile."""
    product = nl.zeros((nl.par_dim(rows), cols), nl.float32, buffer=nl.psum)
    product += nisa.nc_matmul(stationary, moving)
    return product


def _to_sbuf(tile, rows: int, cols: int):
    copied = nl.ndarray((nl.par_dim(rows), cols), dtype=nl.float32, buffer=nl.sbuf)
    copied[...] = nisa.tensor_copy(tile)
    return copied


def chunked_delta_body(
    q_t: nl.ndarray, k_t: nl.ndarray, v: nl.ndarray, decay: nl.ndarray, decay_t: nl.ndarray,
    beta: nl.ndarray, beta_t: nl.ndarray, state: nl.ndarray, masks: nl.ndarray,
):
    """The gated delta rule over a chunk of one row, as `model.chunked_gated_delta` computes it, in 128-token pieces.

    Each piece's triangular solve is done in SBUF by blocked forward substitution, and the state stays in SBUF from
    piece to piece. Every product is arranged so its left operand is already transposed, which the Tensor Engine
    needs, so only keys are transposed, once per piece. All matmuls are float32.

    Keys come in one layout only. Given the same keys in two layouts, as two views of one tensor, the compiler
    handed one of them over in a layout the kernel doesn't expect, which corrupted the state from the second piece on.

    Args:
        q_t, k_t: float32 ``[key_heads, k_head, tokens]``. Query normalized and scaled, key normalized. Each key head
            serves ``value_heads // key_heads`` value heads.
        v: float32 ``[value_heads, tokens, v_head]``.
        decay: float32 ``[value_heads, tokens]``, the log decay summed from the start of each piece. decay_t is
            ``[tokens, value_heads]``.
        beta: float32 ``[value_heads, tokens]``, and beta_t ``[tokens, value_heads]``.
        state: float32 ``[value_heads, k_head, v_head]`` before the chunk.
        masks: float32 ``[10, 128, 128]``: the 8-block diagonal, then for blocks of 8, 16, 32 and 64 the places
            where a lower block meets the upper block beside it, then the same five transposed.

    Returns:
        (output ``[value_heads, tokens, v_head]``, state after the chunk), both float32.
    """
    key_heads, k_head, tokens = q_t.shape
    value_heads, _, v_head = v.shape
    repeat = value_heads // key_heads
    outputs = nl.ndarray((value_heads, tokens, v_head), dtype=nl.float32, buffer=nl.shared_hbm)
    final = nl.ndarray((value_heads, k_head, v_head), dtype=nl.float32, buffer=nl.shared_hbm)
    i_p = nl.arange(PIECE)[:, None]
    i_f = nl.arange(PIECE)[None, :]
    ones = nl.ones((1, PIECE), dtype=nl.float32, buffer=nl.sbuf)
    zero = nl.zeros((nl.par_dim(PIECE), PIECE), dtype=nl.float32, buffer=nl.sbuf)
    eye = nisa.affine_select(i_p - i_f == 0, nl.ones((nl.par_dim(PIECE), PIECE), dtype=nl.float32, buffer=nl.sbuf), 0.0)
    mask = [nl.load(masks[m]) for m in list(range(10))]

    for h in list(range(value_heads)):
        kh = h // repeat
        s = nl.load(state[h])
        for p0 in list(range(0, tokens, PIECE)):
            cols = slice(p0, p0 + PIECE)
            q_tp = nl.load(q_t[kh, :, cols])
            k_tp = nl.load(k_t[kh, :, cols])
            k_p = nl.ndarray((nl.par_dim(PIECE), k_head), dtype=nl.float32, buffer=nl.sbuf)
            k_p[...] = nisa.nc_transpose(k_tp)
            v_p = nl.load(v[h, cols, :])
            g_col = nl.load(decay_t[cols, h : h + 1])
            b_col = nl.load(beta_t[cols, h : h + 1])
            # Rows of the piece's log decays and betas, broadcast to every partition by a one-row matmul.
            g_row = _to_sbuf(_matmul(ones, nl.load(decay[h : h + 1, cols]), PIECE, PIECE), PIECE, PIECE)
            b_row = _to_sbuf(_matmul(ones, nl.load(beta[h : h + 1, cols]), PIECE, PIECE), PIECE, PIECE)
            g_last = g_row[:, PIECE - 1 : PIECE]

            # exp(g_i - g_j), selected before it can overflow: rows i, and transposed with rows j.
            between = nisa.activation(nl.exp, g_row, bias=g_col, scale=-1.0)
            between_t = nisa.activation(nl.exp, g_row, bias=nisa.tensor_scalar(g_col, nl.multiply, -1.0))
            below = nisa.affine_select(i_p - i_f > 0, between, 0.0)
            above_t = nisa.affine_select(i_f - i_p > 0, between_t, 0.0)
            causal_t = nisa.affine_select(i_f - i_p >= 0, between_t, 0.0)

            kk = _matmul(k_tp, k_tp, PIECE, PIECE)
            negative_beta = nisa.tensor_scalar(b_col, nl.multiply, -1.0)
            lower = nisa.scalar_tensor_tensor(data=kk, op0=nl.multiply, operand0=negative_beta, op1=nl.multiply, operand1=below)
            lower_t = nisa.tensor_tensor(nisa.tensor_tensor(kk, above_t, op=nl.multiply), b_row, op=nl.multiply)
            lower_t = nisa.tensor_scalar(lower_t, nl.multiply, -1.0)

            # (I - lower)^-1: 8x8 diagonal blocks by doubling, then neighbouring blocks merged. Kept with its transpose.
            power = nisa.tensor_tensor(lower, mask[0], op=nl.multiply)
            power_t = nisa.tensor_tensor(lower_t, mask[5], op=nl.multiply)
            inverse = nisa.tensor_tensor(eye, power, op=nl.add)
            inverse_t = nisa.tensor_tensor(eye, power_t, op=nl.add)
            for _ in list(range(2)):
                squared = _to_sbuf(_matmul(power_t, power, PIECE, PIECE), PIECE, PIECE)
                squared_t = _to_sbuf(_matmul(power, power_t, PIECE, PIECE), PIECE, PIECE)
                inverse = nisa.tensor_tensor(inverse, _matmul(inverse_t, squared, PIECE, PIECE), op=nl.add)
                inverse_t = nisa.tensor_tensor(inverse_t, _matmul(squared, inverse_t, PIECE, PIECE), op=nl.add)
                power, power_t = squared, squared_t
            for level in list(range(4)):
                beside = nisa.tensor_tensor(lower, mask[1 + level], op=nl.multiply)
                beside_t = nisa.tensor_tensor(lower_t, mask[6 + level], op=nl.multiply)
                # inverse @ beside @ inverse, and its transpose.
                right = _to_sbuf(_matmul(beside_t, inverse, PIECE, PIECE), PIECE, PIECE)
                left_t = _to_sbuf(_matmul(beside, inverse_t, PIECE, PIECE), PIECE, PIECE)
                inverse_new = nisa.tensor_tensor(inverse, _matmul(inverse_t, right, PIECE, PIECE), op=nl.add)
                inverse_t = nisa.tensor_tensor(inverse_t, _matmul(inverse, left_t, PIECE, PIECE), op=nl.add)
                inverse = inverse_new

            v_beta = nisa.tensor_scalar(v_p, nl.multiply, b_col)
            new_values = _to_sbuf(_matmul(inverse_t, v_beta, PIECE, v_head), PIECE, v_head)
            k_scaled = nisa.tensor_scalar(k_p, nl.multiply, b_col, op1=nl.multiply, operand1=nisa.activation(nl.exp, g_col))
            cumulative_t = _to_sbuf(_matmul(k_scaled, inverse_t, k_head, PIECE), k_head, PIECE)

            # From here on the piece depends on the state.
            v_new = nisa.tensor_tensor(new_values, _matmul(cumulative_t, s, PIECE, v_head), op=nl.subtract)
            q_scaled_t = nisa.tensor_tensor(q_tp, nisa.activation(nl.exp, g_row[0:k_head, :]), op=nl.multiply)
            attn_t = nisa.tensor_tensor(_matmul(k_tp, q_tp, PIECE, PIECE), causal_t, op=nl.multiply)
            result = nl.zeros((nl.par_dim(PIECE), v_head), nl.float32, buffer=nl.psum)
            result += nisa.nc_matmul(q_scaled_t, s)
            result += nisa.nc_matmul(attn_t, v_new)
            nl.store(outputs[h, cols, :], _to_sbuf(result, PIECE, v_head))

            k_decayed = nisa.tensor_scalar(k_p, nl.multiply, nisa.activation(nl.exp, g_col, bias=g_last, scale=-1.0))
            s = nisa.scalar_tensor_tensor(
                data=s, op0=nl.multiply, operand0=nisa.activation(nl.exp, g_last), op1=nl.add, operand1=_matmul(k_decayed, v_new, k_head, v_head)
            )
        nl.store(final[h], s)
    return outputs, final


fp8_linear = nki.jit(fp8_linear_body)
chunked_delta = nki.jit(chunked_delta_body)
prefill_attention = nki.jit(prefill_attention_body)
lm_head_group_maxima = nki.jit(lm_head_group_maxima_body)
