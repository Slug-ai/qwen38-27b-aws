"""Hand-written Neuron kernels. Requires the Neuron SDK, so import this only in that environment."""

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

# The Tensor Engine contracts at most 128 elements per instruction and produces at most 512 output columns.
CONTRACT_TILE = 128
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
        x: BF16 ``[in_features, batch]``. Stationary in the Tensor Engine, so batch can be at most 128.
        weight: uint8 ``[in_features, out_features]``, holding FP8 e5m2 bytes from `fp4.to_fp8`.

    Returns:
        BF16 ``[batch, out_features]``.
    """
    in_features, batch = x.shape
    _, out_features = weight.shape
    out = nl.ndarray((batch, out_features), dtype=x.dtype, buffer=nl.shared_hbm)
    contract_tiles = _tiles(in_features, CONTRACT_TILE)

    x_tiles = []
    for start, size in contract_tiles:
        x_tiles.append(nisa.tensor_copy(nl.load(x[start : start + size, :]), dtype=nl.float8_e5m2))

    for load_start, load_size in _tiles(out_features, LOAD_TILE):
        outputs = _tiles(load_size, OUTPUT_TILE)
        psums = [nl.zeros((nl.par_dim(batch), size), nl.float32, buffer=nl.psum) for _, size in outputs]
        for x_tile, (start, size) in zip(x_tiles, contract_tiles):
            moving = nl.load(weight[start : start + size, load_start : load_start + load_size]).view(nl.float8_e5m2)
            for psum, (out_start, out_size) in zip(psums, outputs):
                psum += nisa.nc_matmul(x_tile, moving[:, out_start : out_start + out_size])
        for psum, (out_start, out_size) in zip(psums, outputs):
            result = nl.ndarray((nl.par_dim(batch), out_size), dtype=x.dtype, buffer=nl.sbuf)
            result[:, :] = nisa.tensor_copy(psum, engine=nisa.vector_engine)
            nl.store(out[:, load_start + out_start : load_start + out_start + out_size], result)
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


fp8_linear = nki.jit(fp8_linear_body)
lm_head_group_maxima = nki.jit(lm_head_group_maxima_body)
