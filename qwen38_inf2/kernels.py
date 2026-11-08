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
# Intermediate columns per iteration of `mlp_up_body`: two of its 128-column halves each for gate and up keep 4
# of the 8 PSUM banks busy.
MLP_UP_TILE = 256
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


def mlp_up_body(x: nl.ndarray, gate: nl.ndarray, up: nl.ndarray) -> nl.ndarray:
    """``silu(x.T @ gate) * (x.T @ up)``, transposed: ``[intermediate, tokens]``, the layout `fp8_linear_body` takes
    for the down projection, so it needs no transpose and the gate and up outputs never go through HBM.

    Each product, the sigmoid, the SiLU and its product with up are rounded to BF16, as the model's separate BF16
    ops round them. Weight tiles are stationary and the tokens move, which costs the Tensor Engine the same as the
    other way round.

    Args:
        x: BF16 ``[hidden, tokens]``, at most 512 tokens.
        gate, up: uint8 ``[hidden, intermediate]``, FP8 e5m2 bytes.

    Returns:
        BF16 ``[intermediate, tokens]``.
    """
    in_features, tokens = x.shape
    _, out_features = gate.shape
    assert tokens <= OUTPUT_TILE and in_features % CONTRACT_TILE == 0 and out_features % MLP_UP_TILE == 0
    out = nl.ndarray((out_features, tokens), dtype=x.dtype, buffer=nl.shared_hbm)
    contract = in_features // CONTRACT_TILE
    sub = MLP_UP_TILE // STATIONARY_TILE
    x_sb = nl.ndarray((nl.par_dim(CONTRACT_TILE), contract, tokens), dtype=nl.float8_e5m2, buffer=nl.sbuf)
    for k in nl.affine_range(contract):
        x_sb[:, k, :] = nisa.tensor_copy(nl.load(x[k * CONTRACT_TILE : (k + 1) * CONTRACT_TILE, :]), dtype=nl.float8_e5m2)
    for n in nl.affine_range(out_features // MLP_UP_TILE):
        gates = [nl.zeros((nl.par_dim(STATIONARY_TILE), tokens), nl.float32, buffer=nl.psum) for _ in list(range(sub))]
        ups = [nl.zeros((nl.par_dim(STATIONARY_TILE), tokens), nl.float32, buffer=nl.psum) for _ in list(range(sub))]
        for k in nl.affine_range(contract):
            rows = slice(k * CONTRACT_TILE, (k + 1) * CONTRACT_TILE)
            cols = slice(n * MLP_UP_TILE, (n + 1) * MLP_UP_TILE)
            g_tile = nl.load(gate[rows, cols]).view(nl.float8_e5m2)
            u_tile = nl.load(up[rows, cols]).view(nl.float8_e5m2)
            for j in list(range(sub)):
                part = slice(j * STATIONARY_TILE, (j + 1) * STATIONARY_TILE)
                gates[j] += nisa.nc_matmul(g_tile[:, part], x_sb[:, k, :])
                ups[j] += nisa.nc_matmul(u_tile[:, part], x_sb[:, k, :])
        for j in list(range(sub)):
            g = nisa.tensor_copy(gates[j], dtype=nl.bfloat16)
            u = nisa.tensor_copy(ups[j], dtype=nl.bfloat16, engine=nisa.vector_engine)
            # Rounded to BF16 at each step, as the model's BF16 ops round: a float32 product here was more precise
            # than the reference and flipped a greedy choice in the agentic benchmark's longest request.
            silu = nisa.tensor_tensor(g, nisa.activation(nl.sigmoid, g, dtype=nl.bfloat16), op=nl.multiply, dtype=nl.bfloat16)
            start = n * MLP_UP_TILE + j * STATIONARY_TILE
            nl.store(out[start : start + STATIONARY_TILE, :], nisa.tensor_tensor(silu, u, op=nl.multiply, dtype=x.dtype))
    return out


# The FP8 type lm_head's rows take against its FP8 e5m2 weights, and the weight columns per load.
HEAD_X_DTYPE = nl.float8_e5m2
HEAD_LOAD_WIDTH = 2048


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

    # With FP8 weights the rows go to FP8 too: a BF16 stationary made the Vector Engine widen every weight tile
    # to BF16 first, 6.8 ms of casts per call against 3.7 ms of matmuls.
    fp8 = weight.dtype != nl.bfloat16
    x_tiles = []
    for start, size in contract_tiles:
        tile = nl.load(x[start : start + size, :])
        x_tiles.append(nisa.tensor_copy(tile, dtype=HEAD_X_DTYPE) if fp8 else tile)

    # A multiple of group_rows, so no group straddles two tiles. Each load takes HEAD_LOAD_WIDTH columns, several
    # matmuls' worth, since short rows of a load left the DMA queues at a quarter of their bandwidth.
    tile_width = OUTPUT_TILE - OUTPUT_TILE % group_rows
    for load_start, load_size in _tiles(tokens, HEAD_LOAD_WIDTH - HEAD_LOAD_WIDTH % tile_width):
        outputs = _tiles(load_size, tile_width)
        psums = [nl.zeros((nl.par_dim(batch), size), nl.float32, buffer=nl.psum) for _, size in outputs]
        for x_tile, (start, size) in zip(x_tiles, contract_tiles):
            moving = nl.load(weight[start : start + size, load_start : load_start + load_size])
            if fp8:
                moving = moving.view(nl.float8_e5m2)
            for psum, (out_start, out_size) in zip(psums, outputs):
                psum += nisa.nc_matmul(x_tile, moving[:, out_start : out_start + out_size])
        for psum, (out_start, out_size) in zip(psums, outputs):
            logits = nl.ndarray((nl.par_dim(batch), out_size), dtype=nl.float32, buffer=nl.sbuf)
            logits[:, :] = nisa.tensor_copy(psum, engine=nisa.vector_engine)
            groups = _tiles(out_size, group_rows)
            maxima = nl.ndarray((nl.par_dim(batch), len(groups)), dtype=nl.float32, buffer=nl.sbuf)
            for index, (start, size) in enumerate(groups):
                maxima[:, index : index + 1] = nisa.tensor_reduce(nl.maximum, logits[:, start : start + size], axis=[1], keepdims=True)
            first = (load_start + out_start) // group_rows
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


# The largest finite FP8 e4m3 value on NeuronCore-v2, whose e4m3 has infinities, unlike OCP e4m3fn's 448.
FP8_E4M3_MAX = 240.0


def fp8_encode_body(x: nl.ndarray, inverse_scale: nl.ndarray):
    """FP8 e4m3 bytes of ``x * inverse_scale`` for the KV cache, clamped to the largest finite value, and the values
    those bytes hold, as BF16, so a step can use its own key exactly as later steps will read it.

    Args:
        x: BF16 ``[rows, head]``.
        inverse_scale: float32 ``[128, 1]``, 1 / the layer's scale on every partition.

    Returns:
        (uint8 ``[rows, head]``, BF16 ``[rows, head]``).
    """
    rows, head = x.shape
    coded = nl.ndarray((rows, head), dtype=nl.uint8, buffer=nl.shared_hbm)
    held = nl.ndarray((rows, head), dtype=x.dtype, buffer=nl.shared_hbm)
    inverse = nl.load(inverse_scale)
    for r0, rs in _tiles(rows, STATIONARY_TILE):
        scaled = nisa.tensor_scalar(nl.load(x[r0 : r0 + rs, :]), nl.multiply, inverse[0:rs, :], dtype=nl.float32)
        fp8 = nisa.tensor_scalar(scaled, nl.minimum, FP8_E4M3_MAX, op1=nl.maximum, operand1=-FP8_E4M3_MAX, dtype=nl.float8_e4m3)
        nl.store(coded[r0 : r0 + rs, :], fp8.view(nl.uint8))
        nl.store(held[r0 : r0 + rs, :], nisa.tensor_copy(fp8, dtype=x.dtype))
    return coded, held


def _transposed(tile, rows: int, cols: int, dtype, identity):
    """``tile.T``, ``[rows, cols]``, in SBUF, as `nc_transpose` does it: `tile` stationary against an identity.

    In a loop with a run-time trip count `nc_transpose` fails to compile, because it reloads its identity from a
    constant in HBM every iteration. So the caller makes `identity`, ``[cols, cols]`` of `tile`'s dtype, once in
    SBUF. With middle-end transformations skipped, the copy out of PSUM is explicit too.
    """
    flipped = nl.zeros((nl.par_dim(rows), cols), nl.float32, buffer=nl.psum)
    flipped += nisa.nc_matmul(tile, identity)
    return nisa.tensor_copy(flipped, dtype=dtype)


# Prefill attention's exp weights are at most e^SCORE_HEADROOM where the bound on its scores allows, and every
# query's weight on its own key is at least e^-SELF_FLOOR (`prefill_attention_body`).
SCORE_HEADROOM = 70.0
SELF_FLOOR = 80.0
# Query heads' chunks whose steps the prefill attention kernel emits in turn.
PREFILL_HEAD_GROUP = 2


def _self_floor(q_sb, k_own, d_tiles, groups: int, chunk: int, ones_col, ones_row):
    """``[128, 1]``, the same on every partition: the lowest score of any of the chunk's queries with its own key.
    Each query head's chunk is a row of products summed over head dims by a matmul with ones."""
    lows = nl.ndarray((1, groups), dtype=nl.float32, buffer=nl.sbuf)
    for g in list(range(groups)):
        own = nl.zeros((1, chunk), nl.float32, buffer=nl.psum)
        for d, (d0, ds) in enumerate(d_tiles):
            products = nisa.tensor_tensor(q_sb[d][:, g * chunk : (g + 1) * chunk], k_own[d], op=nl.multiply, dtype=nl.float32)
            own += nisa.nc_matmul(ones_col[0:ds, :], products)
        lows[:, g : g + 1] = nisa.tensor_reduce(nl.minimum, own, axis=[1], keepdims=True)
    low = nisa.tensor_reduce(nl.minimum, lows, axis=[1], keepdims=True)
    spread = nl.zeros((nl.par_dim(STATIONARY_TILE), 1), nl.float32, buffer=nl.psum)
    spread += nisa.nc_matmul(ones_row, low)
    return nisa.tensor_copy(spread, dtype=nl.float32)


def _attend_chunk(q_sb, c: int, chunk: int, keys, values, biases, masks, acc, d_tiles, head: int):
    """Query head c's chunk of queries against a block of key tiles for `_prefill_attention`: scores with keys on
    partitions, which the exp turns into weights already laid out for the values' matmul, then each query tile's
    weighted values and, from the values' column of ones, its weights' sum. A generator, for `_interleave`."""
    weights = []
    for kt in list(range(len(values))):
        scores = nl.zeros((nl.par_dim(STATIONARY_TILE), chunk), nl.float32, buffer=nl.psum)
        for d, (d0, ds) in enumerate(d_tiles):
            scores += nisa.nc_matmul(keys[d][:, kt * STATIONARY_TILE : (kt + 1) * STATIONARY_TILE], q_sb[d][:, c * chunk : (c + 1) * chunk])
        yield
        if masks is not None:
            scores = nisa.tensor_tensor(scores, masks[kt], op=nl.add, dtype=nl.float32)
        weights.append(nisa.activation(nl.exp, scores, bias=biases[kt], dtype=nl.bfloat16))
        yield
    per_chunk = chunk // STATIONARY_TILE
    for qt in list(range(per_chunk)):
        mixed = nl.zeros((nl.par_dim(STATIONARY_TILE), head + 1), nl.float32, buffer=nl.psum)
        for kt in list(range(len(values))):
            mixed += nisa.nc_matmul(weights[kt][:, qt * STATIONARY_TILE : (qt + 1) * STATIONARY_TILE], values[kt])
        t = c * per_chunk + qt
        acc[t][...] = nisa.tensor_tensor(acc[t], mixed, op=nl.add)
        yield


def prefill_attention_body(
    q: nl.ndarray, k_cache: nl.ndarray, v_cache: nl.ndarray, k_new: nl.ndarray, v_new: nl.ndarray, start: nl.ndarray,
    first_block: nl.ndarray, blocks: nl.ndarray, ceiling: nl.ndarray, positions: int,
) -> nl.ndarray:
    """Flash attention for a prefill chunk of one row: its queries against the row's cache and against itself.

    Scores stay in SBUF and PSUM, so nothing grows with the cache length but time. The loop over cache blocks has a
    trip count read at run time, `blocks`, so a chunk only visits the blocks holding its row's earlier keys: early
    chunks of a long prompt cost what they would with a short cache. Cached positions before `start` are all
    visible and the rest are not, a mask per key that the exp takes as its bias. The chunk's own keys are visible
    causally.

    Scores are computed transposed, keys on partitions, so the exp lays the softmax weights out as the values'
    matmul needs them, with no transposes. That leaves no cheap per-query maximum, so the softmax subtracts one
    constant per KV head instead, since any constant the same for all of a query's keys cancels in its
    normalization: min(`ceiling`, the lowest score of a query with its own key + `SELF_FLOOR`). Every query then
    keeps a weight of at least e^-SELF_FLOOR on its own key, so no sum underflows, and `ceiling`, a bound on the
    layer's scores less `SCORE_HEADROOM`, keeps every weight under e^SCORE_HEADROOM where it applies. Each query's
    weights' sum comes from a column of ones beside its values. This takes 28% less time than transposing the
    weights of a per-query online softmax did, at any cache length.

    Args:
        q: BF16 ``[kv_heads, head, queries]``, already scaled. A KV head's query heads are stacked, so query
            ``g * chunk + i`` is head g's token i.
        k_cache, v_cache: ``[rows * kv_heads * positions, head]``, every row's whole cache, BF16 or else uint8
            holding FP8 e4m3 bytes from `fp8_encode`, in units of the layer's scales: then `q` must be multiplied by
            the key scale and the result by the value scale, and `k_new` and `v_new` must be in the same units. The
            chunk's row is read in place, a block at a time with an index held on the chip, and key blocks are
            transposed on the Tensor Engine, once per block for all query tiles.
        k_new: BF16 ``[kv_heads, head, chunk]``. v_new: BF16 ``[kv_heads, chunk, head]``. The chunk's own keys and values.
        start: float32 ``[128, 1]``, the chunk's first position on every partition.
        first_block: int32 ``[1, 1]``, ``slot * kv_heads * positions / KEY_BLOCK``, the chunk's row's first block.
        blocks: int32 ``[1, 1]``, ``ceil(start / KEY_BLOCK)``, the cache blocks to visit.
        ceiling: float32 ``[128, 1]``, the same on every partition: the most to subtract from scores, the bound on
            the layer's scores less `SCORE_HEADROOM`.
        positions: cache positions per row and KV head.

    Returns:
        BF16 ``[kv_heads, queries, head]``.
    """
    out = nl.ndarray((q.shape[0], q.shape[2], q.shape[1]), dtype=q.dtype, buffer=nl.shared_hbm)
    count = nisa.tensor_reduce(nl.add, nl.load(blocks), axis=[1])
    _prefill_attention(q, k_cache, v_cache, k_new, v_new, start, first_block, count, ceiling, positions, out)
    return out


def _prefill_attention(q, k_cache, v_cache, k_new, v_new, start, first_block, count, ceiling, positions: int, out) -> None:
    """`prefill_attention_body` with the block count as an SBUF scalar, `count`, writing into `out`."""
    kv_heads, head, queries = q.shape
    chunk = k_new.shape[2]
    groups = queries // chunk
    assert positions % KEY_BLOCK == 0, f"the cache's {positions} positions must be a multiple of {KEY_BLOCK}"
    assert chunk % STATIONARY_TILE == 0 and chunk <= OUTPUT_TILE, f"a chunk of {chunk} must be whole tiles of one matmul's output"
    d_tiles = _tiles(head, CONTRACT_TILE)
    q_tiles = _tiles(queries, STATIONARY_TILE)
    start_tile = nl.load(start)
    ceiling_tile = nl.load(ceiling)
    # Each block of the cache comes in one load: partition p takes the block's positions per_partition * p on,
    # contiguous, so a load's rows are per_partition keys long, where 128-position tiles made them one key long and
    # left the DMA queues waiting. Key tile j then holds positions per_partition * p + j.
    per_partition = KEY_BLOCK // STATIONARY_TILE
    k_blocks = k_cache.reshape((k_cache.shape[0] // KEY_BLOCK, STATIONARY_TILE, per_partition * head))
    v_blocks = v_cache.reshape((v_cache.shape[0] // KEY_BLOCK, STATIONARY_TILE, per_partition * head))
    i_w = nl.arange(per_partition * head)[None, :]
    i_p = nl.arange(STATIONARY_TILE)[:, None]
    i_s = nl.arange(STATIONARY_TILE)[None, :]
    i_c = nl.arange(chunk)[None, :]
    identity = nisa.affine_select(i_p - i_s == 0, nl.ones((nl.par_dim(STATIONARY_TILE), STATIONARY_TILE), dtype=nl.bfloat16, buffer=nl.sbuf), 0.0)
    fp8 = k_cache.dtype == nl.uint8
    identity_fp8 = nisa.tensor_copy(identity, dtype=nl.float8_e4m3) if fp8 else identity
    ones_col = nl.ones((nl.par_dim(CONTRACT_TILE), 1), dtype=nl.float32, buffer=nl.sbuf)
    ones_row = nl.ones((1, STATIONARY_TILE), dtype=nl.float32, buffer=nl.sbuf)
    ones_v = nl.ones((nl.par_dim(STATIONARY_TILE), 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    zero = nl.zeros((nl.par_dim(STATIONARY_TILE), chunk), dtype=nl.float32, buffer=nl.sbuf)
    # Own key tile kt holds tokens 128 kt + p, which query column i (token i of its head's chunk) sees from i on.
    own_masks = [nisa.affine_select(i_c - i_p - kt * STATIONARY_TILE >= 0, zero, MASKED) for kt in list(range(chunk // STATIONARY_TILE))]
    tile_positions = nisa.iota(per_partition * i_p + 0 * nl.arange(1)[None, :], dtype=nl.float32)

    def with_ones(values_tile, convert: bool):
        tile = nl.ndarray((nl.par_dim(STATIONARY_TILE), head + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        if convert:
            tile[:, 0:head] = nisa.tensor_copy(values_tile.view(nl.float8_e4m3), dtype=nl.bfloat16, engine=nisa.gpsimd_engine)
        else:
            tile[:, 0:head] = nisa.tensor_copy(values_tile, dtype=nl.bfloat16)
        tile[:, head : head + 1] = nisa.tensor_copy(ones_v)
        return tile

    for h in list(range(kv_heads)):
        q_sb = [nl.load(q[h, d0 : d0 + ds, :]) for d0, ds in d_tiles]
        own_keys = [nl.load(k_new[h, d0 : d0 + ds, :]) for d0, ds in d_tiles]
        floor = _self_floor(q_sb, own_keys, d_tiles, groups, chunk, ones_col, ones_row)
        constant = nisa.tensor_scalar(floor, nl.add, SELF_FLOOR, op1=nl.minimum, operand1=ceiling_tile)
        minus = nisa.tensor_scalar(constant, nl.multiply, -1.0)
        acc = [nl.zeros((nl.par_dim(STATIONARY_TILE), head + 1), dtype=nl.float32, buffer=nl.sbuf) for _ in q_tiles]

        def attend(keys, values, biases, masks):
            for g0 in list(range(0, groups, PREFILL_HEAD_GROUP)):
                _interleave(
                    [_attend_chunk(q_sb, c, chunk, keys, values, biases, masks, acc, d_tiles, head) for c in list(range(g0, min(g0 + PREFILL_HEAD_GROUP, groups)))]
                )

        at = nisa.tensor_scalar(nl.load(first_block), nl.add, h * (positions // KEY_BLOCK))
        base = nisa.tensor_copy(tile_positions)
        for _ in range(scalar(count)):
            keys = [nl.ndarray((nl.par_dim(ds), KEY_BLOCK), dtype=q.dtype, buffer=nl.sbuf) for _, ds in d_tiles]
            values, biases = [], []
            block_keys = nl.load(k_blocks[at, i_p, i_w])
            block_values = nl.load(v_blocks[at, i_p, i_w])
            for j in list(range(per_partition)):
                k0 = j * STATIONARY_TILE
                rows = block_keys[:, j * head : (j + 1) * head]
                values.append(with_ones(block_values[:, j * head : (j + 1) * head], fp8))
                if fp8:
                    rows = rows.view(nl.float8_e4m3)
                for d, (d0, ds) in enumerate(d_tiles):
                    keys[d][:, k0 : k0 + STATIONARY_TILE] = _transposed(rows[:, d0 : d0 + ds], ds, STATIONARY_TILE, q.dtype, identity_fp8)
                # Positions from `start` on hold no keys for this chunk yet.
                hidden = nisa.tensor_scalar(base, nl.add, float(j), op1=nl.greater_equal, operand1=start_tile)
                biases.append(nisa.tensor_scalar(hidden, nl.multiply, MASKED, op1=nl.add, operand1=minus))
            attend(keys, values, biases, None)
            at[...] = nisa.tensor_scalar(at, nl.add, 1)
            base[...] = nisa.tensor_scalar(base, nl.add, float(KEY_BLOCK))

        own_values = [with_ones(nl.load(v_new[h, k0 : k0 + STATIONARY_TILE, :]), False) for k0 in list(range(0, chunk, STATIONARY_TILE))]
        attend(own_keys, own_values, [minus] * len(own_values), own_masks)

        for t, (q0, qs) in enumerate(q_tiles):
            inverse = nisa.reciprocal(acc[t][:, head : head + 1])
            result = nisa.tensor_scalar(acc[t][:, 0:head], nl.multiply, inverse, dtype=q.dtype)
            nl.store(out[h, q0 : q0 + qs, :], result)


# Cache positions per iteration of the decode attention kernel's device loop. Every pair's keys and values for a
# block are in SBUF at once, so this is smaller than prefill's.
DECODE_BLOCK = 1024
# Blocks per iteration of that loop, so the model asks for ceil(positions / DECODE_STEP) iterations.
DECODE_SPLIT = 2
DECODE_STEP = DECODE_SPLIT * DECODE_BLOCK
DECODE_CONVERT_ENGINE = nisa.scalar_engine
# Whether the decode attention kernel loads each block of the cache in one DMA per pair (`_decode_attention`).
DECODE_PACKED = True


def decode_attention_body(
    q: nl.ndarray, k_cache: nl.ndarray, v_cache: nl.ndarray, k_own: nl.ndarray, v_own: nl.ndarray, limit: nl.ndarray,
    blocks: nl.ndarray, groups: int, positions: int,
) -> nl.ndarray:
    """Attention for one token per row: its queries against its cache as it was, then against its own key.

    The query heads of every (row, KV head) pair are stacked on partitions, so the softmax runs once for all of
    them. Each pair's keys meet a copy of the queries with every other pair's columns zeroed, so its scores land
    only in its own rows, and its values are taken only by its own rows the same way. The loop over cache blocks
    has a trip count read at run time, `blocks`, so it stops at the furthest row's position.

    Args:
        q: BF16 ``[queries, head]``, already scaled. Row ``p * groups + g`` is pair p's query head g, and pair
            ``b * kv_heads + h`` is row b's KV head h.
        k_cache, v_cache: ``[pairs * positions, head]``, every pair's cache, BF16 or uint8 FP8 bytes as for
            `prefill_attention_body`.
        k_own, v_own: BF16 ``[queries, head]``, the step's own key and value of each query row's pair.
        limit: float32 ``[queries, 1]``, each query row's position: it sees cache positions before it.
        blocks: int32 ``[1, 1]``, how many steps of `DECODE_STEP` positions hold any row's keys.
        groups: query heads per pair.
        positions: cache positions per pair.

    Returns:
        BF16 ``[queries, head]``.
    """
    out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    count = nisa.tensor_reduce(nl.add, nl.load(blocks), axis=[1])
    _decode_attention(q, k_cache, v_cache, k_own, v_own, limit, count, groups, positions, out)
    return out


def verify_attention_body(
    q: nl.ndarray, k_cache: nl.ndarray, v_cache: nl.ndarray, k_own: nl.ndarray, v_own: nl.ndarray, own_mask: nl.ndarray,
    limit: nl.ndarray, blocks: nl.ndarray, groups: int, positions: int, pair_offset: int,
) -> nl.ndarray:
    """`decode_attention_body` for a verify step's K tokens per row, for the pairs from `pair_offset` on.

    Each token's queries are query heads of its pair, all against the pair's cache before the row's position, so
    `groups` is query heads times K. Then every query meets every one of the step's own keys, through `own_mask`.

    Args:
        q: BF16 ``[queries, head]``, row ``p * groups + g * K + t`` the query head g of token t of pair
            ``pair_offset + p``.
        k_own, v_own: BF16 ``[pairs * K, head]``, the pairs' own keys and values, row ``p * K + s``.
        own_mask: float32 ``[queries, pairs * K]``: 0 where a query sees an own key, its pair's tokens up to its
            own, and MASKED elsewhere.
        The rest as for `decode_attention_body`.
    """
    out = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    count = nisa.tensor_reduce(nl.add, nl.load(blocks), axis=[1])
    _decode_attention(q, k_cache, v_cache, k_own, v_own, limit, count, groups, positions, out, own_mask, pair_offset)
    return out


def _decode_attention(q, k_cache, v_cache, k_own, v_own, limit, count, groups: int, positions: int, out, own_mask=None, pair_offset: int = 0) -> None:
    """`decode_attention_body` with the block count as an SBUF scalar, `count`, writing into `out`."""
    queries, head = q.shape
    pairs = queries // groups
    assert positions % DECODE_STEP == 0, f"the cache's {positions} positions must be a multiple of {DECODE_STEP}"
    fp8 = k_cache.dtype == nl.uint8
    d_tiles = _tiles(head, CONTRACT_TILE)
    i_p = nl.arange(STATIONARY_TILE)[:, None]
    i_s = nl.arange(STATIONARY_TILE)[None, :]
    i_q = nl.arange(queries)[None, :]
    i_h = nl.arange(head)[None, :]
    i_r = nl.arange(queries)[:, None]
    limits = nl.load(limit)
    identity = nisa.affine_select(i_p - i_s == 0, nl.ones((nl.par_dim(STATIONARY_TILE), STATIONARY_TILE), dtype=nl.bfloat16, buffer=nl.sbuf), 0.0)
    identity_fp8 = nisa.tensor_copy(identity, dtype=nl.float8_e4m3) if fp8 else identity

    # Pair p's columns of a ``[128, queries]`` tile.
    ones = nl.full((nl.par_dim(STATIONARY_TILE), queries), 1.0, dtype=nl.bfloat16, buffer=nl.sbuf)
    columns = []
    for p in list(range(pairs)):
        mine = nisa.affine_select(i_q - p * groups + 0 * i_p >= 0, ones, 0.0)
        columns.append(nisa.affine_select((p + 1) * groups - 1 - i_q + 0 * i_p >= 0, mine, 0.0))

    # Queries with head dims on partitions, as the Tensor Engine contracts, and per pair a copy of only its columns.
    q_rows = nl.load(q)
    q_pair = [[] for _ in list(range(pairs))]
    q_t = []
    for d0, ds in d_tiles:
        flipped = _transposed(q_rows[:, d0 : d0 + ds], ds, queries, q.dtype, identity[0:queries, 0:queries])
        q_t.append(flipped)
        for p in list(range(pairs)):
            q_pair[p].append(nisa.tensor_tensor(flipped, columns[p][0:ds, :], op=nl.multiply, dtype=q.dtype))

    best = nl.full((nl.par_dim(queries), 1), MASKED, dtype=nl.float32, buffer=nl.sbuf)
    total = nl.zeros((nl.par_dim(queries), 1), dtype=nl.float32, buffer=nl.sbuf)
    acc = nl.zeros((nl.par_dim(queries), head), dtype=nl.float32, buffer=nl.sbuf)

    # Caches by pair and step of DECODE_SPLIT blocks, so the loop variable indexes steps directly and every load is
    # one contiguous DMA. A step's blocks have their own buffers and are taken stage by stage, so one block's loads
    # and transposes overlap the other's softmax. The step's positions are carried from step to step.
    step = DECODE_SPLIT * DECODE_BLOCK
    cached_pairs = k_cache.shape[0] // positions
    k_steps = k_cache.reshape((cached_pairs, positions // step, DECODE_SPLIT, DECODE_BLOCK, head))
    v_steps = v_cache.reshape((cached_pairs, positions // step, DECODE_SPLIT, DECODE_BLOCK, head))
    halves = list(range(DECODE_SPLIT))
    per_partition = DECODE_BLOCK // STATIONARY_TILE
    if DECODE_PACKED:
        # Each block in one load per pair: partition p takes the block's positions per_partition * p on, contiguous,
        # so every row of the load is per_partition keys long. Tile j of a block then holds positions
        # per_partition * p + j, and its columns of the scores and the mask follow that order.
        k_packed = k_cache.reshape((cached_pairs, positions // step, DECODE_SPLIT, STATIONARY_TILE, per_partition * head))
        v_packed = v_cache.reshape((cached_pairs, positions // step, DECODE_SPLIT, STATIONARY_TILE, per_partition * head))
        i_w = nl.arange(per_partition * head)[None, :]
        later = nl.ndarray((nl.par_dim(queries), step), dtype=nl.float32, buffer=nl.sbuf)
        for half in halves:
            for j in list(range(per_partition)):
                at = half * DECODE_BLOCK + j * STATIONARY_TILE
                later[:, at : at + STATIONARY_TILE] = nisa.iota(half * DECODE_BLOCK + j + per_partition * i_s + 0 * i_r, dtype=nl.float32)
    else:
        later = nisa.iota(nl.arange(step)[None, :] + 0 * i_r, dtype=nl.float32)
    for b in range(scalar(count)):
        raw = [[[] for _ in list(range(pairs))] for _ in halves]
        for half in halves:
            for p in list(range(pairs)):
                # NKI's tracer doesn't honor `continue`, so the two ways of loading are branches.
                if DECODE_PACKED:
                    block_keys = nl.load(k_packed[pair_offset + p, b, half, i_p, i_w])
                    block_values = nl.load(v_packed[pair_offset + p, b, half, i_p, i_w])
                    for j in list(range(per_partition)):
                        raw[half][p].append((block_keys[:, j * head : (j + 1) * head], block_values[:, j * head : (j + 1) * head]))
                else:
                    for k0 in list(range(0, DECODE_BLOCK, STATIONARY_TILE)):
                        rows = nl.load(k_steps[pair_offset + p, b, half, k0 + i_p, i_h])
                        cached_values = nl.load(v_steps[pair_offset + p, b, half, k0 + i_p, i_h])
                        raw[half][p].append((rows, cached_values))
        keys = [[] for _ in halves]
        values = [[] for _ in halves]
        masked = []
        for half in halves:
            for p in list(range(pairs)):
                pair_keys = [nl.ndarray((nl.par_dim(ds), DECODE_BLOCK), dtype=q.dtype, buffer=nl.sbuf) for _, ds in d_tiles]
                pair_values = []
                for k, (rows, cached_values) in enumerate(raw[half][p]):
                    if fp8:
                        # FP8 keys stay FP8 for the transpose, which writes BF16. Values become BF16 on an engine
                        # this kernel leaves mostly idle.
                        rows = rows.view(nl.float8_e4m3)
                        cached_values = nisa.tensor_copy(cached_values.view(nl.float8_e4m3), dtype=nl.bfloat16, engine=DECODE_CONVERT_ENGINE)
                    pair_values.append(cached_values)
                    for d, (d0, ds) in enumerate(d_tiles):
                        pair_keys[d][:, k * STATIONARY_TILE : (k + 1) * STATIONARY_TILE] = _transposed(rows[:, d0 : d0 + ds], ds, STATIONARY_TILE, q.dtype, identity_fp8)
                keys[half].append(pair_keys)
                values[half].append(pair_values)
            tile = nl.ndarray((nl.par_dim(queries), DECODE_BLOCK), dtype=nl.float32, buffer=nl.sbuf)
            for j0 in list(range(0, DECODE_BLOCK, OUTPUT_TILE)):
                scores = nl.zeros((nl.par_dim(queries), OUTPUT_TILE), nl.float32, buffer=nl.psum)
                for p in list(range(pairs)):
                    for d, _ in enumerate(d_tiles):
                        scores += nisa.nc_matmul(q_pair[p][d], keys[half][p][d][:, j0 : j0 + OUTPUT_TILE])
                # A row sees the cache positions before its own.
                at = half * DECODE_BLOCK + j0
                bias = nisa.tensor_scalar(later[:, at : at + OUTPUT_TILE], nl.greater_equal, limits, op1=nl.multiply, operand1=MASKED)
                tile[:, j0 : j0 + OUTPUT_TILE] = nisa.tensor_tensor(scores, bias, op=nl.add, dtype=nl.float32)
            masked.append(tile)

        for half in halves:
            new_best = nisa.tensor_tensor(best, nisa.tensor_reduce(nl.maximum, masked[half], axis=[1], keepdims=True), op=nl.maximum)
            negative = nisa.tensor_scalar(new_best, nl.multiply, -1.0)
            row_sum = nl.ndarray((nl.par_dim(queries), 1), dtype=nl.float32, buffer=nl.sbuf)
            weights = nisa.activation_reduce(nl.exp, masked[half], reduce_op=nl.add, reduce_res=row_sum, bias=negative, dtype=nl.bfloat16)
            rescale = nisa.activation(nl.exp, best, bias=negative)
            total[...] = nisa.scalar_tensor_tensor(data=total, op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=row_sum)
            mixed = nl.zeros((nl.par_dim(queries), head), nl.float32, buffer=nl.psum)
            for k, k0 in enumerate(list(range(0, DECODE_BLOCK, STATIONARY_TILE))):
                flipped = _transposed(weights[:, k0 : k0 + STATIONARY_TILE], STATIONARY_TILE, queries, nl.bfloat16, identity[0:queries, 0:queries])
                for p in list(range(pairs)):
                    mine = nisa.tensor_tensor(flipped, columns[p], op=nl.multiply, dtype=nl.bfloat16)
                    mixed += nisa.nc_matmul(mine, values[half][p][k])
            acc[...] = nisa.scalar_tensor_tensor(data=acc, op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=mixed)
            best[...] = new_best
        later[...] = nisa.tensor_scalar(later, nl.add, float(step))

    if own_mask is None:
        # The step's own key and value, one per query row.
        own_key = nl.load(k_own)
        own_value = nl.load(v_own)
        own = nisa.tensor_reduce(nl.add, nisa.tensor_tensor(q_rows, own_key, op=nl.multiply, dtype=nl.float32), axis=[1], keepdims=True)
        new_best = nisa.tensor_tensor(best, own, op=nl.maximum)
        negative = nisa.tensor_scalar(new_best, nl.multiply, -1.0)
        weight = nisa.activation(nl.exp, own, bias=negative)
        rescale = nisa.activation(nl.exp, best, bias=negative)
        total = nisa.scalar_tensor_tensor(data=total, op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=weight)
        taken = nisa.tensor_scalar(own_value, nl.multiply, weight, dtype=nl.float32)
        acc = nisa.scalar_tensor_tensor(data=acc, op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=taken)
    else:
        # Every query against every own key of the step, through the mask: a verify step's own keys.
        own_count = k_own.shape[0]
        own_keys = nl.load(k_own)
        own_values = nl.load(v_own)
        scores = nl.zeros((nl.par_dim(queries), own_count), nl.float32, buffer=nl.psum)
        for d, (d0, ds) in enumerate(d_tiles):
            keys_t = _transposed(own_keys[:, d0 : d0 + ds], ds, own_count, q.dtype, identity[0:own_count, 0:own_count])
            scores += nisa.nc_matmul(q_t[d], keys_t)
        masked = nisa.tensor_tensor(scores, nl.load(own_mask), op=nl.add, dtype=nl.float32)
        new_best = nisa.tensor_tensor(best, nisa.tensor_reduce(nl.maximum, masked, axis=[1], keepdims=True), op=nl.maximum)
        negative = nisa.tensor_scalar(new_best, nl.multiply, -1.0)
        row_sum = nl.ndarray((nl.par_dim(queries), 1), dtype=nl.float32, buffer=nl.sbuf)
        weights = nisa.activation_reduce(nl.exp, masked, reduce_op=nl.add, reduce_res=row_sum, bias=negative, dtype=nl.bfloat16)
        rescale = nisa.activation(nl.exp, best, bias=negative)
        total = nisa.scalar_tensor_tensor(data=total, op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=row_sum)
        mixed = nl.zeros((nl.par_dim(queries), head), nl.float32, buffer=nl.psum)
        mixed += nisa.nc_matmul(_transposed(weights, own_count, queries, nl.bfloat16, identity[0:queries, 0:queries]), own_values)
        acc = nisa.scalar_tensor_tensor(data=acc, op0=nl.multiply, operand0=rescale, op1=nl.add, operand1=mixed)
    nl.store(out, nisa.tensor_scalar(acc, nl.multiply, nisa.reciprocal(total), dtype=q.dtype))


PIECE = 128


# The gated delta kernel's matmul inputs. float32 takes the Tensor Engine 4 passes, tfloat32 one, with 10 mantissa
# bits against BF16's 7; products still accumulate in float32. A tfloat32 matmul's inputs must be written as
# tfloat32 by the instructions that make them, so every producer of one says so.
DELTA_MATMUL_DTYPE = nl.tfloat32


def _matmul(stationary, moving, rows: int, cols: int):
    """``stationary.T @ moving`` into a fresh float32 PSUM tile."""
    product = nl.zeros((nl.par_dim(rows), cols), nl.float32, buffer=nl.psum)
    product += nisa.nc_matmul(stationary, moving)
    return product


def _to_sbuf(tile, rows: int, cols: int, dtype=nl.float32):
    copied = nl.ndarray((nl.par_dim(rows), cols), dtype=dtype, buffer=nl.sbuf)
    copied[...] = nisa.tensor_copy(tile)
    return copied


def _conv_tile(c, conv):
    """Conv, SiLU and norms for `delta_prefill_body`'s 128 channels from `c * 128`, with channels on partitions.

    Queries and keys stay that way, ``[k_head, tokens]``, as the rule's matmuls want them. Values go back to tokens
    on partitions, one ``[128, v_head]`` tile per head and piece. A generator yielding between dependent steps, so
    `_interleave` can overlap several tiles' chains.
    """
    qkv, conv_weight, conv_history, q_t, k_t, v_p, ones, ones_column, epsilon, key_heads, kernel, tokens, pieces, k_head, v_head = conv
    mm = DELTA_MATMUL_DTYPE
    window = nl.ndarray((nl.par_dim(PIECE), kernel - 1 + tokens), dtype=nl.float32, buffer=nl.sbuf)
    window[:, 0 : kernel - 1] = nl.load(conv_history[c * PIECE : (c + 1) * PIECE, :], dtype=nl.float32)
    for t in list(range(pieces)):
        rows = nl.load(qkv[t * PIECE : (t + 1) * PIECE, c * PIECE : (c + 1) * PIECE])
        window[:, kernel - 1 + t * PIECE : kernel - 1 + (t + 1) * PIECE] = nisa.nc_transpose(rows)
    w = nl.load(conv_weight[c * PIECE : (c + 1) * PIECE, :], dtype=nl.float32)
    yield
    mixed = nisa.tensor_scalar(window[:, 0:tokens], nl.multiply, w[:, 0:1])
    for j in list(range(1, kernel)):
        mixed = nisa.scalar_tensor_tensor(data=window[:, j : j + tokens], op0=nl.multiply, operand0=w[:, j : j + 1], op1=nl.add, operand1=mixed)
    yield
    mixed = nisa.tensor_tensor(mixed, nisa.activation(nl.sigmoid, mixed), op=nl.multiply)
    yield
    if c < 2 * key_heads:
        # L2 norm over the head's 128 channels, which are the partitions: a ones-vector matmul sums them.
        squares = nisa.tensor_tensor(mixed, mixed, op=nl.multiply, dtype=mm)
        inverse = nisa.activation(nl.rsqrt, _matmul(ones_column, squares, 1, tokens), bias=epsilon, dtype=mm)
        if c < key_heads:
            inverse = nisa.tensor_scalar(inverse, nl.multiply, k_head**-0.5, dtype=mm)
        yield
        spread = _matmul(ones, inverse, PIECE, tokens)
        normed = nisa.tensor_tensor(mixed, spread, op=nl.multiply, dtype=mm)
        if c < key_heads:
            q_t[c] = normed
        else:
            k_t[c - key_heads] = normed
    else:
        for t in list(range(pieces)):
            flipped = nl.ndarray((nl.par_dim(PIECE), v_head), dtype=nl.float32, buffer=nl.sbuf)
            flipped[...] = nisa.nc_transpose(mixed[:, t * PIECE : (t + 1) * PIECE])
            v_p[(c - 2 * key_heads) * pieces + t] = flipped
    yield


def _delta_head(h, c):
    """One value head's chunked gated delta rule for `delta_prefill_body`, piece by piece from its state.

    A generator that yields between dependent steps, so `_interleave` can emit several heads' steps in turn: the
    engines take instructions in order, and one head alone kept each engine waiting on the others at every step.
    `c` holds the kernel's inputs, constants and conv outputs. A piece's key transposes and its keys' products with
    keys and queries depend only on the key head, so the first of its value heads to need them makes them for all.
    """
    (state, decay, beta, z, gated, final, q_t, k_t, v_p, mask, eye, ones, epsilon_column, weight_row, i_p, i_f,
     repeat, pieces, k_head, v_head, shared) = c
    mm = DELTA_MATMUL_DTYPE
    kh = h // repeat
    s = nisa.tensor_copy(nl.load(state[h]), dtype=mm)
    for t in list(range(pieces)):
        cols = slice(t * PIECE, (t + 1) * PIECE)
        q_tp = q_t[kh][:, cols]
        k_tp = k_t[kh][:, cols]
        if (kh, t) not in shared:
            k_p = nl.ndarray((nl.par_dim(PIECE), k_head), dtype=nl.float32, buffer=nl.sbuf)
            k_p[...] = nisa.nc_transpose(k_tp)
            shared[(kh, t)] = (k_p, _to_sbuf(_matmul(k_tp, k_tp, PIECE, PIECE), PIECE, PIECE), _to_sbuf(_matmul(k_tp, q_tp, PIECE, PIECE), PIECE, PIECE))
        k_p, kk, qk = shared[(kh, t)]
        v_now = v_p[(h * pieces) + t]
        # The piece's log decays and betas, rows broadcast to every partition by one float32 matmul so they match
        # the columns loaded straight from HBM exactly, one value per token on each partition. Splitting them into
        # tfloat32 parts for a one-pass broadcast was slower, from the adds it puts at the start of every piece.
        rows_in = nl.ndarray((1, 2 * PIECE), dtype=nl.float32, buffer=nl.sbuf)
        rows_in[:, 0:PIECE] = nl.load(decay[h : h + 1, cols])
        rows_in[:, PIECE : 2 * PIECE] = nl.load(beta[h : h + 1, cols])
        rows = _matmul(ones, rows_in, PIECE, 2 * PIECE)
        g_row, b_row = _to_sbuf(rows[:, 0:PIECE], PIECE, PIECE), _to_sbuf(rows[:, PIECE : 2 * PIECE], PIECE, PIECE)
        g_col = nl.load(decay[h, t * PIECE + i_p])
        b_col = nl.load(beta[h, t * PIECE + i_p])
        g_last = g_row[:, PIECE - 1 : PIECE]
        yield

        # exp(g_i - g_j), selected before it can overflow: rows i, and transposed with rows j.
        between = nisa.activation(nl.exp, g_row, bias=g_col, scale=-1.0)
        between_t = nisa.activation(nl.exp, g_row, bias=nisa.tensor_scalar(g_col, nl.multiply, -1.0))
        below = nisa.affine_select(i_p - i_f > 0, between, 0.0)
        above_t = nisa.affine_select(i_f - i_p > 0, between_t, 0.0)
        causal_t = nisa.affine_select(i_f - i_p >= 0, between_t, 0.0)
        yield

        negative_beta = nisa.tensor_scalar(b_col, nl.multiply, -1.0)
        lower = nisa.scalar_tensor_tensor(data=kk, op0=nl.multiply, operand0=negative_beta, op1=nl.multiply, operand1=below)
        lower_t = nisa.tensor_tensor(nisa.tensor_tensor(kk, above_t, op=nl.multiply), b_row, op=nl.multiply)
        lower_t = nisa.tensor_scalar(lower_t, nl.multiply, -1.0)
        yield

        # (I - lower)^-1: 8x8 diagonal blocks by doubling, then neighbouring blocks merged, kept with its transpose
        # because each product needs its left factor transposed. A merge's update of the transpose reuses the
        # product it made for the inverse, and the last merge and doubling make only what is used after them.
        power = nisa.tensor_tensor(lower, mask[0], op=nl.multiply, dtype=mm)
        power_t = nisa.tensor_tensor(lower_t, mask[5], op=nl.multiply, dtype=mm)
        inverse = nisa.tensor_tensor(eye, power, op=nl.add, dtype=mm)
        inverse_t = nisa.tensor_tensor(eye, power_t, op=nl.add, dtype=mm)
        for step in list(range(2)):
            squared = _to_sbuf(_matmul(power_t, power, PIECE, PIECE), PIECE, PIECE, mm)
            if step == 0:
                squared_t = _to_sbuf(_matmul(power, power_t, PIECE, PIECE), PIECE, PIECE, mm)
            yield
            inverse = nisa.tensor_tensor(inverse, _matmul(inverse_t, squared, PIECE, PIECE), op=nl.add, dtype=mm)
            inverse_t = nisa.tensor_tensor(inverse_t, _matmul(squared, inverse_t, PIECE, PIECE), op=nl.add, dtype=mm)
            yield
            if step == 0:
                power, power_t = squared, squared_t
        for level in list(range(4)):
            # inverse + inverse @ beside @ inverse, from right = beside @ inverse; its transpose from right too.
            beside_t = nisa.tensor_tensor(lower_t, mask[6 + level], op=nl.multiply, dtype=mm)
            right = _to_sbuf(_matmul(beside_t, inverse, PIECE, PIECE), PIECE, PIECE, mm)
            yield
            if level < 3:
                merged = nisa.tensor_tensor(inverse, _matmul(inverse_t, right, PIECE, PIECE), op=nl.add, dtype=mm)
            inverse_t = nisa.tensor_tensor(inverse_t, _matmul(right, inverse_t, PIECE, PIECE), op=nl.add, dtype=mm)
            if level < 3:
                inverse = merged
            yield

        v_beta = nisa.tensor_scalar(v_now, nl.multiply, b_col, dtype=mm)
        new_values = _to_sbuf(_matmul(inverse_t, v_beta, PIECE, v_head), PIECE, v_head)
        yield
        k_scaled = nisa.tensor_scalar(k_p, nl.multiply, b_col, op1=nl.multiply, operand1=nisa.activation(nl.exp, g_col), dtype=mm)
        cumulative_t = _to_sbuf(_matmul(k_scaled, inverse_t, k_head, PIECE), k_head, PIECE, mm)
        yield

        # From here on the piece depends on the state.
        v_new = nisa.tensor_tensor(new_values, _matmul(cumulative_t, s, PIECE, v_head), op=nl.subtract, dtype=mm)
        yield
        q_scaled_t = nisa.tensor_tensor(q_tp, nisa.activation(nl.exp, g_row[0:k_head, :]), op=nl.multiply, dtype=mm)
        attn_t = nisa.tensor_tensor(qk, causal_t, op=nl.multiply, dtype=mm)
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

        k_decayed = nisa.tensor_scalar(k_p, nl.multiply, nisa.activation(nl.exp, g_col, bias=g_last, scale=-1.0), dtype=mm)
        s = nisa.scalar_tensor_tensor(
            data=s, op0=nl.multiply, operand0=nisa.activation(nl.exp, g_last), op1=nl.add, operand1=_matmul(k_decayed, v_new, k_head, v_head), dtype=mm
        )
        yield
    nl.store(final[h], nisa.tensor_copy(s, dtype=nl.float32))


def _interleave(steps) -> None:
    """Run generators round-robin, one step each, until all finish."""
    active = list(steps)
    while active:
        for g in list(active):
            try:
                next(g)
            except StopIteration:
                active.remove(g)


# Value heads, and 128-channel tiles of the conv, whose steps the delta prefill kernel interleaves.
DELTA_HEAD_GROUP = 6
DELTA_CONV_GROUP = 4


def delta_prefill_body(
    qkv: nl.ndarray, z: nl.ndarray, conv_weight: nl.ndarray, conv_history: nl.ndarray, decay: nl.ndarray,
    beta: nl.ndarray, state: nl.ndarray, masks: nl.ndarray, norm_weight: nl.ndarray, key_heads: int, eps: float,
):
    """A gated delta layer's prefill between its input and output projections, for a chunk of one row.

    The causal conv, SiLU, query and key L2 norms, the chunked gated delta rule as `model.chunked_gated_delta`
    computes it in 128-token pieces, and the gated RMS norm all run here in SBUF, so none of it makes a round trip
    through HBM. Each piece's triangular solve is a blocked inverse, and the state stays in SBUF from
    piece to piece. Every product is arranged so its left operand is already transposed, which the Tensor Engine
    needs. The rule's matmuls take tfloat32 inputs.

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
    ones = nl.ones((1, PIECE), dtype=DELTA_MATMUL_DTYPE, buffer=nl.sbuf)
    ones_column = nl.ones((PIECE, 1), dtype=DELTA_MATMUL_DTYPE, buffer=nl.sbuf)
    eye = nisa.affine_select(i_p - i_f == 0, nl.ones((nl.par_dim(PIECE), PIECE), dtype=nl.float32, buffer=nl.sbuf), 0.0)
    mask = [nl.load(masks[m]) for m in list(range(10))]
    epsilon = nl.full((1, 1), eps, dtype=nl.float32, buffer=nl.sbuf)
    epsilon_column = nl.full((PIECE, 1), eps, dtype=nl.float32, buffer=nl.sbuf)
    weight_row = nl.load(norm_weight)

    # Conv, SiLU and norms, 128 channels at a time, several tiles' steps in turn.
    q_t, k_t, v_p = [None] * key_heads, [None] * key_heads, [None] * (value_heads * pieces)
    conv = (qkv, conv_weight, conv_history, q_t, k_t, v_p, ones, ones_column, epsilon, key_heads, kernel, tokens, pieces, k_head, v_head)
    for c0 in list(range(0, channel_tiles, DELTA_CONV_GROUP)):
        _interleave([_conv_tile(c, conv) for c in list(range(c0, min(c0 + DELTA_CONV_GROUP, channel_tiles)))])

    # Groups of heads, each head's steps emitted in turn with the others' so their chains overlap.
    rows_ones = nl.ones((1, PIECE), dtype=nl.float32, buffer=nl.sbuf)
    c = (state, decay, beta, z, gated, final, q_t, k_t, v_p, mask, eye, rows_ones, epsilon_column, weight_row, i_p, i_f, repeat, pieces, k_head, v_head, {})
    for g0 in list(range(0, value_heads, DELTA_HEAD_GROUP)):
        _interleave([_delta_head(h, c) for h in list(range(g0, min(g0 + DELTA_HEAD_GROUP, value_heads)))])
    return gated, final


# Pairs of a row and value head whose steps the delta verify kernel interleaves.
DELTA_VERIFY_GROUP = 4
# Its matmuls' inputs: float32 cost nothing measurable over tfloat32 here, matmuls this small being cheap.
DELTA_VERIFY_MATMUL_DTYPE = nl.float32


def _verify_pair(p, c):
    """`delta_verify_state_body` for pair `p`. A generator yielding between dependent steps, so `_interleave` can
    overlap several pairs' chains."""
    state, scale, weighted, deltas, reads, committed, products = c
    mm = DELTA_VERIFY_MATMUL_DTYPE
    k_head, v_head = state.shape[1:]
    width = reads.shape[2]
    s = nl.load(state[p], dtype=nl.float32)
    factor = nl.load(scale[p])
    w = nisa.tensor_copy(nl.load(weighted[p]), dtype=mm)
    d = nisa.tensor_copy(nl.load(deltas[p]), dtype=mm)
    r = nisa.tensor_copy(nl.load(reads[p]), dtype=mm)
    yield
    update = _matmul(w, d, k_head, v_head)
    yield
    kept = nisa.scalar_tensor_tensor(data=s, op0=nl.multiply, operand0=factor, op1=nl.add, operand1=update, dtype=state.dtype)
    nl.store(committed[p], kept)
    yield
    read = _matmul(r, nisa.tensor_copy(kept, dtype=mm), width, v_head)
    yield
    nl.store(products[p], _to_sbuf(read, width, v_head))


def delta_verify_state_body(state: nl.ndarray, scale: nl.ndarray, weighted: nl.ndarray, deltas: nl.ndarray, reads: nl.ndarray):
    """A gated delta layer's verify step against its state, for every pair of a row and value head: the last step's
    accepted tokens committed, then the new tokens' keys and queries read against the result.

    Also returns a zero, for the caller to add to what the step keeps for the next one, so that is written only
    after this runs: NxD writes a state in place, and a new value that doesn't depend on the old one could be
    written first. The commit read this step's log decays instead of the last step's. Passing the values themselves
    through instead put the layer's other ops in layouts that made the verify graph 3.4 times slower.

    The commit is ``state * scale + weighted.T @ deltas``, so the masks for inactive rows and a sequence's first
    token come in `scale` and `weighted`. The state stays in SBUF between the two, and the matmuls are float32. As
    XLA ops, the same steps made the 16-layer verify graph 2.8 times slower.

    Args:
        state: ``[pairs, k_head, v_head]``, as stored.
        scale: float32 ``[pairs, k_head, 1]``, the decay over the committed tokens, on every partition.
        weighted: float32 ``[pairs, K, k_head]``, each pending key times the decay after it, 0 where not committed.
        deltas: float32 ``[pairs, K, v_head]``, the pending deltas.
        reads: float32 ``[pairs, k_head, R]``, the new tokens' keys then queries, transposed.

    Returns:
        (the committed state ``[pairs, k_head, v_head]`` in the state's dtype, float32 ``[pairs, R, v_head]``: each
        key and query times that state, and a float32 ``[1, 1]`` zero).
    """
    pairs, k_head, v_head = state.shape
    committed = nl.ndarray((pairs, k_head, v_head), dtype=state.dtype, buffer=nl.shared_hbm)
    products = nl.ndarray((pairs, reads.shape[2], v_head), dtype=nl.float32, buffer=nl.shared_hbm)
    zero = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nl.store(zero, nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf))
    c = (state, scale, weighted, deltas, reads, committed, products)
    for p0 in list(range(0, pairs, DELTA_VERIFY_GROUP)):
        _interleave([_verify_pair(p, c) for p in list(range(p0, min(p0 + DELTA_VERIFY_GROUP, pairs)))])
    return committed, products, zero


fp8_linear = nki.jit(fp8_linear_body)
mlp_up = nki.jit(mlp_up_body)
fp8_encode = nki.jit(fp8_encode_body)
delta_prefill = nki.jit(delta_prefill_body)
delta_verify_state = nki.jit(delta_verify_state_body)
# Loops with a trip count read at run time need these, as the SDK's own dynamic-loop kernels use.
DYNAMIC_LOOP_FLAGS = "experimental-native-scalar-support, experimental-local-tensor-parent"
prefill_attention = nki.compiler.skip_middle_end_transformations(nki.jit(prefill_attention_body, experimental_flags=DYNAMIC_LOOP_FLAGS))
decode_attention = nki.compiler.skip_middle_end_transformations(nki.jit(decode_attention_body, experimental_flags=DYNAMIC_LOOP_FLAGS))
verify_attention = nki.compiler.skip_middle_end_transformations(nki.jit(verify_attention_body, experimental_flags=DYNAMIC_LOOP_FLAGS))
lm_head_group_maxima = nki.jit(lm_head_group_maxima_body)
