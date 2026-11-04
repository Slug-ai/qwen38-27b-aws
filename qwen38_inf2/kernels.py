"""Hand-written Neuron kernels. Requires the Neuron SDK, so import this only in that environment."""

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

# The Tensor Engine contracts at most 128 elements per instruction and produces at most 512 output columns.
CONTRACT_TILE = 128
OUTPUT_TILE = 512


def _tiles(total: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(size, total - start)) for start in range(0, total, size)]


def fp8_linear_body(x: nl.ndarray, weight: nl.ndarray) -> nl.ndarray:
    """``x.T @ weight``, with weights read straight from memory as FP8, so nothing is decoded on the chip.

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
        x_tiles.append(nl.load(x[start : start + size, :]))

    for out_start, out_size in _tiles(out_features, OUTPUT_TILE):
        psum = nl.zeros((nl.par_dim(batch), out_size), nl.float32, buffer=nl.psum)
        for x_tile, (start, size) in zip(x_tiles, contract_tiles):
            moving = nl.load(weight[start : start + size, out_start : out_start + out_size]).view(nl.float8_e5m2)
            psum += nisa.nc_matmul(x_tile, moving)
        result = nl.ndarray((nl.par_dim(batch), out_size), dtype=x.dtype, buffer=nl.sbuf)
        result[:, :] = nisa.tensor_copy(psum, engine=nisa.vector_engine)
        nl.store(out[:, out_start : out_start + out_size], result)
    return out


fp8_linear = nki.jit(fp8_linear_body)
