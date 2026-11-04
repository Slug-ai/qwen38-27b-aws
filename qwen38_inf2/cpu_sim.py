"""Run the model on CPU as if on NeuronCores, from the same per-core weight files.

Each core is a thread and the collective is a real sum across threads, so this exercises the weight
files, block chaining, state handling and the greedy loop end to end. Only the compiled graph itself
is replaced, by the PyTorch source it was traced from.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import torch
from safetensors.torch import load_file

from . import checkpoint
from .config import TextConfig
from .generate import BlockChain
from .model import Qwen38Step


class ThreadedAllReduce:
    """A sum across ranks running in lock-step threads, like the real collective."""

    def __init__(self, world_size: int):
        self.values: dict[int, torch.Tensor] = {}
        self.result = None
        self.count = 0
        self.barrier = threading.Barrier(world_size, action=self._finish)

    def _finish(self) -> None:
        self.result = sum(self.values.values())
        self.values = {}
        self.count += 1

    def bind(self, rank: int):
        def all_reduce(x: torch.Tensor) -> torch.Tensor:
            self.values[rank] = x
            self.barrier.wait()
            return self.result

        return all_reduce


class SimulatedCoresStep:
    """Same call signature as `neuron.NeuronStep`, with one thread per NeuronCore."""

    def __init__(self, weights_dir, cfg: TextConfig, build_info: dict, dtype: torch.dtype = torch.bfloat16):
        block_layers, tp_degree = build_info["layers"], build_info["tp_degree"]
        block_cfg = replace(cfg, layer_types=cfg.layer_types[:block_layers])
        self.reducer = ThreadedAllReduce(tp_degree)
        self.ranks = []
        for rank in range(tp_degree):
            blocks = []
            for block in range(cfg.num_hidden_layers // block_layers):
                model = Qwen38Step(
                    block_cfg, build_info["max_seq_len"], tp_degree, dtype, self.reducer.bind(rank)
                ).eval()
                weights = load_file(checkpoint.shard_path(weights_dir, block, rank))
                # Packed FP4 and its scales stay as stored. The rest takes the model's dtype.
                weights = {
                    name: t if name.endswith(("weight_packed", "weight_scale")) else t.to(dtype)
                    for name, t in weights.items()
                }
                missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
                if unexpected or not all(name.endswith(("_cache", "_state")) for name in missing):
                    raise ValueError(f"weights do not fit the model: {unexpected or missing}")
                blocks.append(model)
            self.ranks.append(BlockChain(blocks))
        self.pool = ThreadPoolExecutor(tp_degree)

    def _run_rank(self, rank: int, inputs: dict) -> torch.Tensor:
        with torch.no_grad():
            return self.ranks[rank](**inputs)

    def __call__(self, **inputs: torch.Tensor) -> torch.Tensor:
        futures = [self.pool.submit(self._run_rank, rank, inputs) for rank in range(len(self.ranks))]
        return [future.result() for future in futures][0]
