"""Read the checkpoint and cut it into per-rank weights for `Qwen38Step`.

Only text tensors are read. `model.visual.*`, `mtp*` and the KV-cache FP8 scales never reach the
device. `TensorSource` reads a local file through mmap or a remote one through HTTP range requests,
so tensors can be inspected or streamed without downloading the 19 GB file.
"""

import json
import mmap
import re
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from . import fp4
from .config import FULL, LINEAR, TextConfig

BUILD_INFO_FILE = "build_info.json"
TEXT_PREFIX = "model.language_model."
LM_HEAD = "lm_head.weight"
EMBED_TOKENS = TEXT_PREFIX + "embed_tokens.weight"
FINAL_NORM = TEXT_PREFIX + "norm.weight"
# Static FP8 calibration scales for the KV cache. The cache here is BF16, so they are unused.
UNUSED_LAYER_SUFFIXES = ("self_attn.k_scale", "self_attn.v_scale")

_DTYPES = {
    "BF16": (torch.bfloat16, 2),
    "U8": (torch.uint8, 1),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
}
_NUMPY_DTYPES = {"BF16": np.uint16, "U8": np.uint8, "F16": np.float16, "F32": np.float32}


class TensorSource:
    """Lazy access to tensors in one safetensors file, by local path or by URL."""

    def __init__(self, location: str | Path):
        self.location = str(location)
        self.is_remote = self.location.startswith("http")
        self._mmap = None
        if not self.is_remote:
            with open(self.location, "rb") as f:
                self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack("<Q", self._read(0, 8))[0]
        self.header = json.loads(bytes(self._read(8, header_len)))
        self.header.pop("__metadata__", None)
        self._base = 8 + header_len

    def _read(self, offset: int, size: int) -> bytes | memoryview:
        if not self.is_remote:
            return memoryview(self._mmap)[offset : offset + size]
        headers = {"Range": f"bytes={offset}-{offset + size - 1}"}
        with urllib.request.urlopen(urllib.request.Request(self.location, headers=headers)) as response:
            return response.read()

    def names(self) -> list[str]:
        return list(self.header)

    def shape(self, name: str) -> list[int]:
        return self.header[name]["shape"]

    def nbytes(self, name: str) -> int:
        start, end = self.header[name]["data_offsets"]
        return end - start

    def get(self, name: str) -> torch.Tensor:
        entry = self.header[name]
        start, end = entry["data_offsets"]
        return self._decode(self._read(self._base + start, end - start), entry["dtype"], entry["shape"])

    def get_rows(self, name: str, row_ids: list[int] | torch.Tensor) -> torch.Tensor:
        """Read selected rows of a matrix. Lets a test use a few embeddings without 2.5 GB of table."""
        entry = self.header[name]
        dtype = entry["dtype"]
        rows, cols = entry["shape"]
        row_bytes = cols * _DTYPES[dtype][1]
        start = entry["data_offsets"][0]
        parts = [self._read(self._base + start + int(r) * row_bytes, row_bytes) for r in row_ids]
        return self._decode(b"".join(parts), dtype, [len(parts), cols])

    def get_row_range(self, name: str, first: int, count: int) -> torch.Tensor:
        """Read contiguous rows of a matrix, so large tables can be processed in pieces."""
        entry = self.header[name]
        cols = entry["shape"][1]
        row_bytes = cols * _DTYPES[entry["dtype"]][1]
        offset = self._base + entry["data_offsets"][0] + first * row_bytes
        return self._decode(self._read(offset, count * row_bytes), entry["dtype"], [count, cols])

    @staticmethod
    def _decode(data: bytes | memoryview, dtype: str, shape: list[int]) -> torch.Tensor:
        array = np.frombuffer(data, dtype=_NUMPY_DTYPES[dtype]).copy()
        tensor = torch.from_numpy(array)
        return (tensor.view(torch.bfloat16) if dtype == "BF16" else tensor).reshape(shape)


class DictSource:
    """In-memory stand-in for `TensorSource`, used by tests."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def names(self) -> list[str]:
        return list(self.tensors)

    def shape(self, name: str) -> list[int]:
        return list(self.tensors[name].shape)

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def get_rows(self, name: str, row_ids) -> torch.Tensor:
        return self.tensors[name][torch.as_tensor(row_ids, dtype=torch.long)]

    def get_row_range(self, name: str, first: int, count: int) -> torch.Tensor:
        return self.tensors[name][first : first + count]


def _slice_range(total: int, rank: int, tp_degree: int) -> tuple[int, int]:
    size = total // tp_degree
    return rank * size, (rank + 1) * size


def _rows(t: torch.Tensor, rank: int, tp_degree: int) -> torch.Tensor:
    start, end = _slice_range(t.shape[0], rank, tp_degree)
    return t[start:end]


def _cols(t: torch.Tensor, rank: int, tp_degree: int) -> torch.Tensor:
    start, end = _slice_range(t.shape[1], rank, tp_degree)
    return t[:, start:end]


def _qkv_rows(t: torch.Tensor, cfg: TextConfig, rank: int, tp_degree: int) -> torch.Tensor:
    """Rows are laid out [q | k | v], so a rank takes its head slice from each of the three blocks."""
    key_dim, value_dim = cfg.linear_key_dim, cfg.linear_value_dim
    return torch.cat(
        (
            _rows(t[:key_dim], rank, tp_degree),
            _rows(t[key_dim : 2 * key_dim], rank, tp_degree),
            _rows(t[2 * key_dim : 2 * key_dim + value_dim], rank, tp_degree),
        )
    )


def _fp4_weights(source, hf_name: str, name: str, cut, weight_format: str) -> dict[str, torch.Tensor]:
    """Cut one FP4 matrix. Scale bytes become BF16 here so the device never computes 2**e."""
    packed, scale = cut(source.get(f"{hf_name}.weight_packed")), cut(source.get(f"{hf_name}.weight_scale"))
    if weight_format == "fp8":
        return {f"{name}.weight_fp8": fp4.to_fp8(packed, scale).transpose(0, 1).contiguous()}
    return {f"{name}.weight_packed": fp4.to_sign_free(packed), f"{name}.weight_scale": fp4.scale_to_float(scale)}


def layer_weights(
    source, cfg: TextConfig, index: int, rank: int, tp_degree: int, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """Weights for one decoder layer on one rank, with `Qwen38Step` parameter names."""
    hf = f"{TEXT_PREFIX}layers.{index}"
    ours = f"layers.{index}"
    by_rows = lambda t: _rows(t, rank, tp_degree)  # noqa: E731
    by_cols = lambda t: _cols(t, rank, tp_degree)  # noqa: E731
    by_qkv = lambda t: _qkv_rows(t, cfg, rank, tp_degree)  # noqa: E731
    matrix = lambda hf_name, name, cut: _fp4_weights(source, hf_name, name, cut, cfg.weight_format)  # noqa: E731
    out: dict[str, torch.Tensor] = {}

    out[f"{ours}.input_layernorm_weight"] = source.get(f"{hf}.input_layernorm.weight").to(dtype)
    out[f"{ours}.post_attention_layernorm_weight"] = source.get(f"{hf}.post_attention_layernorm.weight").to(dtype)
    out |= matrix(f"{hf}.mlp.gate_proj", f"{ours}.mlp.gate_proj", by_rows)
    out |= matrix(f"{hf}.mlp.up_proj", f"{ours}.mlp.up_proj", by_rows)
    out |= matrix(f"{hf}.mlp.down_proj", f"{ours}.mlp.down_proj", by_cols)

    if cfg.layer_types[index] == LINEAR:
        la, hl = f"{ours}.linear_attn", f"{hf}.linear_attn"
        out |= matrix(f"{hl}.in_proj_qkv", f"{la}.in_proj_qkv", by_qkv)
        out |= matrix(f"{hl}.in_proj_z", f"{la}.in_proj_z", by_rows)
        out |= matrix(f"{hl}.out_proj", f"{la}.out_proj", by_cols)
        out[f"{la}.in_proj_b"] = by_rows(source.get(f"{hl}.in_proj_b.weight")).to(dtype)
        out[f"{la}.in_proj_a"] = by_rows(source.get(f"{hl}.in_proj_a.weight")).to(dtype)
        out[f"{la}.conv_weight"] = by_qkv(source.get(f"{hl}.conv1d.weight").squeeze(1)).to(dtype)
        out[f"{la}.A_log"] = by_rows(source.get(f"{hl}.A_log")).to(dtype)
        out[f"{la}.dt_bias"] = by_rows(source.get(f"{hl}.dt_bias")).to(dtype)
        out[f"{la}.norm_weight"] = source.get(f"{hl}.norm.weight").to(dtype)
    elif cfg.layer_types[index] == FULL:
        sa, hs = f"{ours}.self_attn", f"{hf}.self_attn"
        out |= matrix(f"{hs}.q_proj", f"{sa}.q_proj", by_rows)
        out |= matrix(f"{hs}.k_proj", f"{sa}.k_proj", by_rows)
        out |= matrix(f"{hs}.v_proj", f"{sa}.v_proj", by_rows)
        out |= matrix(f"{hs}.o_proj", f"{sa}.o_proj", by_cols)
        out[f"{sa}.q_norm_weight"] = source.get(f"{hs}.q_norm.weight").to(dtype)
        out[f"{sa}.k_norm_weight"] = source.get(f"{hs}.k_norm.weight").to(dtype)
    return out


def rank_weights(source, cfg: TextConfig, rank: int, tp_degree: int, dtype: torch.dtype = torch.bfloat16):
    """All on-device weights for one rank, when the whole model is one graph."""
    return block_weights(source, cfg, 0, cfg.num_hidden_layers, rank, tp_degree, dtype)


def block_weights(
    source, cfg: TextConfig, block: int, block_layers: int, rank: int, tp_degree: int, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """Weights for one repeat of a compiled block, renamed to that block's own layer numbers."""
    first = block * block_layers
    if cfg.layer_types[first : first + block_layers] != cfg.layer_types[:block_layers]:
        raise ValueError(f"layers {first}..{first + block_layers - 1} do not repeat the compiled block")
    out = {}
    for offset in range(block_layers):
        for name, tensor in layer_weights(source, cfg, first + offset, rank, tp_degree, dtype).items():
            out[f"layers.{offset}." + name.split(".", 2)[2]] = tensor
    return out


def expected_text_tensors(names: list[str]) -> list[str]:
    """Checkpoint tensors the text model must consume: everything except vision, MTP and FP8 scales."""
    keep = []
    for name in names:
        if name.startswith(("model.visual.", "mtp")) or re.match(r"^mtp", name):
            continue
        if name.endswith(UNUSED_LAYER_SUFFIXES):
            continue
        keep.append(name)
    return keep


class RecordingSource:
    """Wraps a source and records which tensors were read, to prove none were missed."""

    def __init__(self, source):
        self.source = source
        self.read: set[str] = set()

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get(self, name: str) -> torch.Tensor:
        self.read.add(name)
        return self.source.get(name)


def shard_path(weights_dir: str | Path, block: int, rank: int) -> Path:
    return Path(weights_dir) / f"block{block:02d}_tp{rank}.safetensors"


def write_rank_weights(source, cfg: TextConfig, weights_dir: str | Path, block_layers: int, tp_degree: int = 2) -> None:
    """Write one file per block per NeuronCore. Vision and MTP tensors are never read."""
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    for block in range(cfg.num_hidden_layers // block_layers):
        for rank in range(tp_degree):
            weights = block_weights(source, cfg, block, block_layers, rank, tp_degree)
            save_file({k: v.contiguous() for k, v in weights.items()}, shard_path(weights_dir, block, rank))


def head_weights(source, rank: int, tp_degree: int) -> dict[str, torch.Tensor]:
    """Final norm and this core's slice of lm_head, ``[hidden, tokens]`` as the Tensor Engine contracts it."""
    start, end = _slice_range(source.shape(LM_HEAD)[0], rank, tp_degree)
    gate = torch.zeros(tp_degree, dtype=torch.float32)
    gate[rank] = 1.0
    return {
        "norm_weight": source.get(FINAL_NORM).to(torch.bfloat16),
        "lm_head": source.get_row_range(LM_HEAD, start, end - start).to(torch.bfloat16).transpose(0, 1).contiguous(),
        "rank_gate": gate,
    }


def head_path(weights_dir: str | Path, rank: int) -> Path:
    return Path(weights_dir) / f"head_tp{rank}.safetensors"


def write_head_weights(source, weights_dir: str | Path, tp_degree: int = 2) -> None:
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    for rank in range(tp_degree):
        save_file(head_weights(source, rank, tp_degree), head_path(weights_dir, rank))


def load_head_weights(weights_dir: str | Path, rank: int) -> dict[str, torch.Tensor]:
    return load_file(head_path(weights_dir, rank))


def load_rank_block(weights_dir: str | Path, block: int, rank: int, layers: int | None = None) -> dict[str, torch.Tensor]:
    """One block's weights for one core, optionally keeping only its first `layers` layers."""
    weights = load_file(shard_path(weights_dir, block, rank))
    if layers is None:
        return weights
    return {name: tensor for name, tensor in weights.items() if int(name.split(".")[1]) < layers}


def rank_block_header(weights_dir: str | Path, block: int, rank: int, layers: int | None = None) -> dict:
    """Names, shapes and dtypes of one block's weights, read without loading them."""
    header = TensorSource(shard_path(weights_dir, block, rank)).header
    if layers is None:
        return header
    return {name: entry for name, entry in header.items() if int(name.split(".")[1]) < layers}


def read_build_info(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / BUILD_INFO_FILE).read_text())


class ShapeSource:
    """Serves meta tensors from a header, so the full model's shapes can be checked without weights."""

    def __init__(self, header: dict):
        self.header = header

    def names(self) -> list[str]:
        return list(self.header)

    def shape(self, name: str) -> list[int]:
        return self.header[name]["shape"]

    def get(self, name: str) -> torch.Tensor:
        entry = self.header[name]
        return torch.empty(entry["shape"], dtype=_DTYPES[entry["dtype"]][0], device="meta")


def device_bytes_per_rank(weights: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in weights.values())
