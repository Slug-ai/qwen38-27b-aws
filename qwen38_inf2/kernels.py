"""Hand-written Neuron kernels. Requires the Neuron SDK, so import this only in that environment."""

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import scalar

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
# Cache positions per iteration of the attention kernel's device loop. Each iteration waits for its loads, so
# fewer, larger blocks hide more of that latency: at 32K positions 512 took 10.4 ms per layer, 2048 8.2 ms and
# 4096 9.0 ms.
KEY_BLOCK = 2048


def _transposed(tile, rows: int, cols: int, dtype, identity):
    """``tile.T``, ``[rows, cols]``, in SBUF, as `nc_transpose` does it: `tile` stationary against an identity.

    In a loop with a run-time trip count `nc_transpose` fails to compile, because it reloads its identity from a
    constant in HBM every iteration. So the caller makes `identity`, ``[cols, cols]`` of `tile`'s dtype, once in
    SBUF. With middle-end transformations skipped, the copy out of PSUM is explicit too.
    """
    flipped = nl.zeros((nl.par_dim(rows), cols), nl.float32, buffer=nl.psum)
    flipped += nisa.nc_matmul(tile, identity)
    return nisa.tensor_copy(flipped, dtype=dtype)


def prefill_attention_body(
    q: nl.ndarray, k_cache: nl.ndarray, v_cache: nl.ndarray, k_new: nl.ndarray, v_new: nl.ndarray, start: nl.ndarray, row: nl.ndarray,
    blocks: nl.ndarray, positions: int,
) -> nl.ndarray:
    """Flash attention for a prefill chunk of one row: its queries against the row's cache and against itself.

    Scores stay in SBUF and PSUM, so nothing grows with the cache length but time. The loop over cache blocks has a
    trip count read at run time, `blocks`, so a chunk only visits the blocks holding its row's earlier keys: early
    chunks of a long prompt cost what they would with a short cache. Cached positions before `start` are all
    visible and the rest are not, so one mask per key block serves every query. The chunk's own keys are visible
    causally.

    Args:
        q: BF16 ``[kv_heads, head, queries]``, already scaled. A KV head's query heads are stacked, so query
            ``g * chunk + i`` is head g's token i.
        k_cache, v_cache: BF16 ``[rows * kv_heads * positions, head]``, every row's whole cache. The chunk's row is
            read with indirect DMA, so it is never copied, and key blocks are transposed on the Tensor Engine, once
            per block for all query tiles.
        k_new: BF16 ``[kv_heads, head, chunk]``. v_new: BF16 ``[kv_heads, chunk, head]``. The chunk's own keys and values.
        start: float32 ``[128, 1]``, the chunk's first position on every partition.
        row: int32 ``[128, 1]``, ``slot * kv_heads * positions``, where the chunk's row begins, on every partition.
        blocks: int32 ``[1, 1]``, ``ceil(start / KEY_BLOCK)``, the cache blocks to visit.
        positions: cache positions per row and KV head.

    Returns:
        BF16 ``[kv_heads, queries, head]``.
    """
    out = nl.ndarray((q.shape[0], q.shape[2], q.shape[1]), dtype=q.dtype, buffer=nl.shared_hbm)
    _prefill_attention(q, k_cache, v_cache, k_new, v_new, start, row, nisa.tensor_reduce(nl.add, nl.load(blocks), axis=[1]), positions, out)
    return out


def _prefill_attention(q, k_cache, v_cache, k_new, v_new, start, row, count, positions: int, out) -> None:
    """`prefill_attention_body` with the block count as an SBUF scalar, `count`, writing into `out`."""
    kv_heads, head, queries = q.shape
    chunk = k_new.shape[2]
    assert positions % KEY_BLOCK == 0, f"the cache's {positions} positions must be a multiple of {KEY_BLOCK}"
    d_tiles = _tiles(head, CONTRACT_TILE)
    q_tiles = _tiles(queries, STATIONARY_TILE)
    assert all(qs == STATIONARY_TILE for _, qs in q_tiles), "queries must fill whole tiles"
    start_tile = nl.load(start)
    row_tile = nl.load(row)
    i_p = nl.arange(STATIONARY_TILE)[:, None]
    i_f = nl.arange(KEY_BLOCK)[None, :]
    i_h = nl.arange(head)[None, :]
    i_s = nl.arange(STATIONARY_TILE)[None, :]
    identity = nisa.affine_select(i_p - i_s == 0, nl.ones((nl.par_dim(STATIONARY_TILE), STATIONARY_TILE), dtype=nl.bfloat16, buffer=nl.sbuf), 0.0)

    for h in list(range(kv_heads)):
        q_sb = [nl.load(q[h, d0 : d0 + ds, :]) for d0, ds in d_tiles]
        best = [nl.full((nl.par_dim(STATIONARY_TILE), 1), MASKED, dtype=nl.float32, buffer=nl.sbuf) for _ in q_tiles]
        total = [nl.zeros((nl.par_dim(STATIONARY_TILE), 1), dtype=nl.float32, buffer=nl.sbuf) for _ in q_tiles]
        acc = [nl.zeros((nl.par_dim(STATIONARY_TILE), head), dtype=nl.float32, buffer=nl.sbuf) for _ in q_tiles]

        def attend(keys, values, biases, block):
            """One block of keys for every query tile. `biases[t]` is the mask added to query tile t's scores."""
            for t, (q0, qs) in enumerate(q_tiles):
                masked = nl.ndarray((nl.par_dim(qs), block), dtype=nl.float32, buffer=nl.sbuf)
                for j0 in list(range(0, block, OUTPUT_TILE)):
                    scores = nl.zeros((nl.par_dim(qs), OUTPUT_TILE), nl.float32, buffer=nl.psum)
                    for d, (d0, ds) in enumerate(d_tiles):
                        scores += nisa.nc_matmul(q_sb[d][:, q0 : q0 + qs], keys[d][:, j0 : j0 + OUTPUT_TILE])
                    masked[:, j0 : j0 + OUTPUT_TILE] = nisa.tensor_tensor(scores, biases[t][:, j0 : j0 + OUTPUT_TILE], op=nl.add, dtype=nl.float32)
                new_best = nisa.tensor_tensor(best[t], nisa.tensor_reduce(nl.maximum, masked, axis=[1], keepdims=True), op=nl.maximum)
                negative = nisa.tensor_scalar(new_best, nl.multiply, -1.0)
                row_sum = nl.ndarray((nl.par_dim(qs), 1), dtype=nl.float32, buffer=nl.sbuf)
                weights = nisa.activation_reduce(nl.exp, masked, reduce_op=nl.add, reduce_res=row_sum, bias=negative, dtype=nl.bfloat16)
                rescale = nisa.activation(nl.exp, best[t], bias=negative)
                total[t][...] = nisa.scalar_tensor_tensor(data=total[t], op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=row_sum)
                mixed = nl.zeros((nl.par_dim(qs), head), nl.float32, buffer=nl.psum)
                for k, k0 in enumerate(list(range(0, block, STATIONARY_TILE))):
                    mixed += nisa.nc_matmul(_transposed(weights[:, k0 : k0 + STATIONARY_TILE], STATIONARY_TILE, qs, nl.bfloat16, identity), values[k])
                acc[t][...] = nisa.scalar_tensor_tensor(data=acc[t], op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=mixed)
                best[t][...] = new_best

        # A device loop with a run-time trip count, so code size doesn't grow with the cache and time grows only
        # with the row's keys. Where the block's rows begin and its positions are carried from block to block.
        offset = nisa.tensor_tensor(row_tile, nisa.iota(i_p + h * positions, dtype=nl.int32), op=nl.add)
        later = nisa.iota(i_f + 0 * i_p, dtype=nl.float32)
        for _ in range(scalar(count)):
            keys = [nl.ndarray((nl.par_dim(ds), KEY_BLOCK), dtype=k_cache.dtype, buffer=nl.sbuf) for _, ds in d_tiles]
            values = []
            for k0 in list(range(0, KEY_BLOCK, STATIONARY_TILE)):
                index = nisa.tensor_scalar(offset, nl.add, k0)
                rows = nl.load(k_cache[index, i_h])
                values.append(nl.load(v_cache[index, i_h]))
                for d, (d0, ds) in enumerate(d_tiles):
                    keys[d][:, k0 : k0 + STATIONARY_TILE] = _transposed(rows[:, d0 : d0 + ds], ds, STATIONARY_TILE, k_cache.dtype, identity)
            # Positions from `start` on hold no keys for this chunk yet.
            bias = nisa.tensor_scalar(later, nl.greater_equal, start_tile, op1=nl.multiply, operand1=MASKED)
            attend(keys, values, [bias] * len(q_tiles), KEY_BLOCK)
            offset[...] = nisa.tensor_scalar(offset, nl.add, KEY_BLOCK)
            later[...] = nisa.tensor_scalar(later, nl.add, float(KEY_BLOCK))

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


def _delta_head(h, c):
    """One value head's chunked gated delta rule for `delta_prefill_body`, piece by piece from its state.

    A generator that yields between dependent steps, so `_interleave` can emit several heads' steps in turn: the
    engines take instructions in order, and one head alone kept each engine waiting on the others at every step.
    `c` holds the kernel's inputs, constants and conv outputs.
    """
    (state, decay, beta, z, gated, final, q_t, k_t, v_p, mask, eye, ones, epsilon_column, weight_row, i_p, i_f,
     repeat, pieces, k_head, v_head) = c
    kh = h // repeat
    s = nl.load(state[h])
    for t in list(range(pieces)):
        cols = slice(t * PIECE, (t + 1) * PIECE)
        q_tp = q_t[kh][:, cols]
        k_tp = k_t[kh][:, cols]
        k_p = nl.ndarray((nl.par_dim(PIECE), k_head), dtype=nl.float32, buffer=nl.sbuf)
        k_p[...] = nisa.nc_transpose(k_tp)
        v_now = v_p[(h * pieces) + t]
        # Rows of the piece's log decays and betas, broadcast to every partition by a one-row matmul, and their
        # diagonals as columns, one value per token.
        g_row = _to_sbuf(_matmul(ones, nl.load(decay[h : h + 1, cols]), PIECE, PIECE), PIECE, PIECE)
        b_row = _to_sbuf(_matmul(ones, nl.load(beta[h : h + 1, cols]), PIECE, PIECE), PIECE, PIECE)
        g_col = nisa.tensor_reduce(nl.add, nisa.tensor_tensor(g_row, eye, op=nl.multiply), axis=[1], keepdims=True)
        b_col = nisa.tensor_reduce(nl.add, nisa.tensor_tensor(b_row, eye, op=nl.multiply), axis=[1], keepdims=True)
        g_last = g_row[:, PIECE - 1 : PIECE]
        yield

        # exp(g_i - g_j), selected before it can overflow: rows i, and transposed with rows j.
        between = nisa.activation(nl.exp, g_row, bias=g_col, scale=-1.0)
        between_t = nisa.activation(nl.exp, g_row, bias=nisa.tensor_scalar(g_col, nl.multiply, -1.0))
        below = nisa.affine_select(i_p - i_f > 0, between, 0.0)
        above_t = nisa.affine_select(i_f - i_p > 0, between_t, 0.0)
        causal_t = nisa.affine_select(i_f - i_p >= 0, between_t, 0.0)
        yield

        kk = _matmul(k_tp, k_tp, PIECE, PIECE)
        negative_beta = nisa.tensor_scalar(b_col, nl.multiply, -1.0)
        lower = nisa.scalar_tensor_tensor(data=kk, op0=nl.multiply, operand0=negative_beta, op1=nl.multiply, operand1=below)
        lower_t = nisa.tensor_tensor(nisa.tensor_tensor(kk, above_t, op=nl.multiply), b_row, op=nl.multiply)
        lower_t = nisa.tensor_scalar(lower_t, nl.multiply, -1.0)
        yield

        # (I - lower)^-1: 8x8 diagonal blocks by doubling, then neighbouring blocks merged. Kept with its transpose.
        power = nisa.tensor_tensor(lower, mask[0], op=nl.multiply)
        power_t = nisa.tensor_tensor(lower_t, mask[5], op=nl.multiply)
        inverse = nisa.tensor_tensor(eye, power, op=nl.add)
        inverse_t = nisa.tensor_tensor(eye, power_t, op=nl.add)
        for _ in list(range(2)):
            squared = _to_sbuf(_matmul(power_t, power, PIECE, PIECE), PIECE, PIECE)
            squared_t = _to_sbuf(_matmul(power, power_t, PIECE, PIECE), PIECE, PIECE)
            yield
            inverse = nisa.tensor_tensor(inverse, _matmul(inverse_t, squared, PIECE, PIECE), op=nl.add)
            inverse_t = nisa.tensor_tensor(inverse_t, _matmul(squared, inverse_t, PIECE, PIECE), op=nl.add)
            yield
            power, power_t = squared, squared_t
        for level in list(range(4)):
            beside = nisa.tensor_tensor(lower, mask[1 + level], op=nl.multiply)
            beside_t = nisa.tensor_tensor(lower_t, mask[6 + level], op=nl.multiply)
            # inverse @ beside @ inverse, and its transpose.
            right = _to_sbuf(_matmul(beside_t, inverse, PIECE, PIECE), PIECE, PIECE)
            left_t = _to_sbuf(_matmul(beside, inverse_t, PIECE, PIECE), PIECE, PIECE)
            yield
            inverse_new = nisa.tensor_tensor(inverse, _matmul(inverse_t, right, PIECE, PIECE), op=nl.add)
            inverse_t = nisa.tensor_tensor(inverse_t, _matmul(inverse, left_t, PIECE, PIECE), op=nl.add)
            inverse = inverse_new
            yield

        v_beta = nisa.tensor_scalar(v_now, nl.multiply, b_col)
        new_values = _to_sbuf(_matmul(inverse_t, v_beta, PIECE, v_head), PIECE, v_head)
        yield
        k_scaled = nisa.tensor_scalar(k_p, nl.multiply, b_col, op1=nl.multiply, operand1=nisa.activation(nl.exp, g_col))
        cumulative_t = _to_sbuf(_matmul(k_scaled, inverse_t, k_head, PIECE), k_head, PIECE)
        yield

        # From here on the piece depends on the state.
        v_new = nisa.tensor_tensor(new_values, _matmul(cumulative_t, s, PIECE, v_head), op=nl.subtract)
        yield
        q_scaled_t = nisa.tensor_tensor(q_tp, nisa.activation(nl.exp, g_row[0:k_head, :]), op=nl.multiply)
        attn_t = nisa.tensor_tensor(_matmul(k_tp, q_tp, PIECE, PIECE), causal_t, op=nl.multiply)
        result = nl.zeros((nl.par_dim(PIECE), v_head), nl.float32, buffer=nl.psum)
        result += nisa.nc_matmul(q_scaled_t, s)
        result += nisa.nc_matmul(attn_t, v_new)
        yield

        # Gated RMS norm of the head's output: rounded to BF16 before the weight, as the model does.
        out = _to_sbuf(result, PIECE, v_head)
        mean_square = nisa.tensor_reduce(nl.add, nisa.tensor_tensor(out, out, op=nl.multiply), axis=[1], keepdims=True)
        scale = nisa.activation(nl.rsqrt, mean_square, scale=1.0 / v_head, bias=epsilon_column)
        normed = nisa.tensor_scalar(out, nl.multiply, scale, dtype=nl.bfloat16)
        yield
        gate = nl.load(z[cols, h * v_head : (h + 1) * v_head], dtype=nl.float32)
        gate = nisa.tensor_tensor(gate, nisa.activation(nl.sigmoid, gate), op=nl.multiply)
        weighted = nisa.tensor_tensor(normed, weight_row, op=nl.multiply, dtype=nl.bfloat16)
        nl.store(gated[cols, h * v_head : (h + 1) * v_head], nisa.tensor_tensor(weighted, gate, op=nl.multiply, dtype=z.dtype))

        k_decayed = nisa.tensor_scalar(k_p, nl.multiply, nisa.activation(nl.exp, g_col, bias=g_last, scale=-1.0))
        s = nisa.scalar_tensor_tensor(
            data=s, op0=nl.multiply, operand0=nisa.activation(nl.exp, g_last), op1=nl.add, operand1=_matmul(k_decayed, v_new, k_head, v_head)
        )
        yield
    nl.store(final[h], s)


def _interleave(steps) -> None:
    """Run generators round-robin, one step each, until all finish."""
    active = list(steps)
    while active:
        for g in list(active):
            try:
                next(g)
            except StopIteration:
                active.remove(g)


# Value heads whose steps the delta prefill kernel interleaves.
DELTA_HEAD_GROUP = 4


def delta_prefill_body(
    qkv: nl.ndarray, z: nl.ndarray, conv_weight: nl.ndarray, conv_history: nl.ndarray, decay: nl.ndarray,
    beta: nl.ndarray, state: nl.ndarray, masks: nl.ndarray, norm_weight: nl.ndarray, key_heads: int, eps: float,
):
    """A gated delta layer's prefill between its input and output projections, for a chunk of one row.

    The causal conv, SiLU, query and key L2 norms, the chunked gated delta rule as `model.chunked_gated_delta`
    computes it in 128-token pieces, and the gated RMS norm all run here in SBUF, so none of it makes a round trip
    through HBM. Each piece's triangular solve is blocked forward substitution, and the state stays in SBUF from
    piece to piece. Every product is arranged so its left operand is already transposed, which the Tensor Engine
    needs. The rule's matmuls are float32.

    Each tensor comes in one layout only, and the kernel transposes on the Tensor Engine where it needs another.
    Given two views of one tensor, the compiler handed one over in a layout the kernel didn't expect.

    Args:
        qkv: BF16 ``[tokens, conv_dim]``, the input projection: queries, keys, then values, by head.
        z: BF16 ``[tokens, value_heads * v_head]``, the output gate's projection.
        conv_weight: BF16 ``[conv_dim, kernel]``, with the newest sample last.
        conv_history: BF16 ``[conv_dim, kernel - 1]``, the inputs before the chunk, zero at a sequence's start.
        decay: float32 ``[value_heads, tokens]``, the log decay summed from the start of each piece.
        beta: float32 ``[value_heads, tokens]``. Both are 0 for padding, which then leaves the state as it was.
        state: float32 ``[value_heads, k_head, v_head]`` before the chunk.
        masks: float32 ``[10, 128, 128]``: the 8-block diagonal, then for blocks of 8, 16, 32 and 64 the places
            where a lower block meets the upper block beside it, then the same five transposed.
        norm_weight: BF16 ``[128, v_head]``, the gated norm's weight on every partition.
        key_heads: query and key heads. Each serves ``value_heads // key_heads`` value heads.
        eps: the norms' epsilon.

    Returns:
        (BF16 ``[tokens, value_heads * v_head]`` for the output projection, float32 state after the chunk).
    """
    tokens, conv_dim = qkv.shape
    value_heads, k_head, v_head = state.shape
    kernel = conv_weight.shape[1]
    repeat = value_heads // key_heads
    channel_tiles = conv_dim // PIECE
    pieces = tokens // PIECE
    gated = nl.ndarray((tokens, value_heads * v_head), dtype=z.dtype, buffer=nl.shared_hbm)
    final = nl.ndarray((value_heads, k_head, v_head), dtype=nl.float32, buffer=nl.shared_hbm)
    i_p = nl.arange(PIECE)[:, None]
    i_f = nl.arange(PIECE)[None, :]
    ones = nl.ones((1, PIECE), dtype=nl.float32, buffer=nl.sbuf)
    ones_column = nl.ones((PIECE, 1), dtype=nl.float32, buffer=nl.sbuf)
    eye = nisa.affine_select(i_p - i_f == 0, nl.ones((nl.par_dim(PIECE), PIECE), dtype=nl.float32, buffer=nl.sbuf), 0.0)
    mask = [nl.load(masks[m]) for m in list(range(10))]
    epsilon = nl.full((1, 1), eps, dtype=nl.float32, buffer=nl.sbuf)
    epsilon_column = nl.full((PIECE, 1), eps, dtype=nl.float32, buffer=nl.sbuf)
    weight_row = nl.load(norm_weight)

    # Conv, SiLU and norms with channels on partitions. Queries and keys stay that way, ``[k_head, tokens]``, as the
    # rule's matmuls want them. Values go back to tokens on partitions, one ``[128, v_head]`` tile per head and piece.
    q_t, k_t, v_p = [], [], []
    for c in list(range(channel_tiles)):
        window = nl.ndarray((nl.par_dim(PIECE), kernel - 1 + tokens), dtype=nl.float32, buffer=nl.sbuf)
        window[:, 0 : kernel - 1] = nl.load(conv_history[c * PIECE : (c + 1) * PIECE, :], dtype=nl.float32)
        for t in list(range(pieces)):
            rows = nl.load(qkv[t * PIECE : (t + 1) * PIECE, c * PIECE : (c + 1) * PIECE])
            window[:, kernel - 1 + t * PIECE : kernel - 1 + (t + 1) * PIECE] = nisa.nc_transpose(rows)
        w = nl.load(conv_weight[c * PIECE : (c + 1) * PIECE, :], dtype=nl.float32)
        mixed = nisa.tensor_scalar(window[:, 0:tokens], nl.multiply, w[:, 0:1])
        for j in list(range(1, kernel)):
            mixed = nisa.scalar_tensor_tensor(data=window[:, j : j + tokens], op0=nl.multiply, operand0=w[:, j : j + 1], op1=nl.add, operand1=mixed)
        mixed = nisa.tensor_tensor(mixed, nisa.activation(nl.sigmoid, mixed), op=nl.multiply)
        if c < 2 * key_heads:
            # L2 norm over the head's 128 channels, which are the partitions: a ones-vector matmul sums them.
            squares = nisa.tensor_tensor(mixed, mixed, op=nl.multiply)
            inverse = nisa.activation(nl.rsqrt, _matmul(ones_column, squares, 1, tokens), bias=epsilon)
            if c < key_heads:
                inverse = nisa.tensor_scalar(inverse, nl.multiply, k_head**-0.5)
            spread = _matmul(ones, inverse, PIECE, tokens)
            (q_t if c < key_heads else k_t).append(nisa.tensor_tensor(mixed, spread, op=nl.multiply))
        else:
            for t in list(range(pieces)):
                flipped = nl.ndarray((nl.par_dim(PIECE), v_head), dtype=nl.float32, buffer=nl.sbuf)
                flipped[...] = nisa.nc_transpose(mixed[:, t * PIECE : (t + 1) * PIECE])
                v_p.append(flipped)

    # Groups of heads, each head's steps emitted in turn with the others' so their chains overlap.
    c = (state, decay, beta, z, gated, final, q_t, k_t, v_p, mask, eye, ones, epsilon_column, weight_row, i_p, i_f, repeat, pieces, k_head, v_head)
    for g0 in list(range(0, value_heads, DELTA_HEAD_GROUP)):
        _interleave([_delta_head(h, c) for h in list(range(g0, min(g0 + DELTA_HEAD_GROUP, value_heads)))])
    return gated, final


fp8_linear = nki.jit(fp8_linear_body)
delta_prefill = nki.jit(delta_prefill_body)
# Loops with a trip count read at run time need these, as the SDK's own dynamic-loop kernels use.
DYNAMIC_LOOP_FLAGS = "experimental-native-scalar-support, experimental-local-tensor-parent"
prefill_attention = nki.compiler.skip_middle_end_transformations(nki.jit(prefill_attention_body, experimental_flags=DYNAMIC_LOOP_FLAGS))
lm_head_group_maxima = nki.jit(lm_head_group_maxima_body)
