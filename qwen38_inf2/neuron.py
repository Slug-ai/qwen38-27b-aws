"""Compile and run `Qwen38Step` on Neuron. Requires the Neuron SDK, so import this only in that environment."""

from pathlib import Path

import torch
from neuronx_distributed import ModelBuilder, NxDParallelState
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
from neuronx_distributed.trace.nxd_model import NxDModel
from safetensors.torch import load_file, save_file

from . import checkpoint
from .config import TextConfig
from .generate import step_inputs
from .model import Qwen38Step

DTYPE = torch.bfloat16
MODEL_FILE = "nxd_model.pt"
# NxD's defaults plus an explicit target. Inf2 shares NeuronCore-v2 codegen with Trn1.
DEFAULT_COMPILER_ARGS = "--enable-saturate-infinity --auto-cast=none --model-type=transformer -O1"


def example_inputs(cfg: TextConfig, max_seq_len: int) -> dict[str, torch.Tensor]:
    return {
        "inputs_embeds": torch.zeros(1, 1, cfg.hidden_size, dtype=DTYPE),
        **step_inputs(cfg, 0, max_seq_len, DTYPE),
    }


def build(
    cfg: TextConfig,
    max_seq_len: int,
    output_dir: str | Path,
    tp_degree: int = 2,
    target: str = "inf2",
    extra_compiler_args: str = "",
    workdir: str | Path | None = None,
) -> Path:
    """Trace and compile the single-token graph for `tp_degree` NeuronCores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler_args = f"{DEFAULT_COMPILER_ARGS} --target {target} {extra_compiler_args}".strip()
    # The compiler changes into this directory before opening its log, so it must be absolute.
    workdir = str(Path(workdir).resolve()) if workdir else None

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        model = Qwen38Step(
            cfg, max_seq_len, tp_degree=tp_degree, dtype=DTYPE, all_reduce=reduce_from_tensor_model_parallel_region
        )
        builder = ModelBuilder(model)
        builder.trace(kwargs=example_inputs(cfg, max_seq_len), tag="decode")
        nxd_model = builder.compile(compiler_args=compiler_args, compiler_workdir=workdir)

    path = output_dir / MODEL_FILE
    nxd_model.save(str(path))
    return path


def shard_path(weights_dir: str | Path, rank: int) -> Path:
    return Path(weights_dir) / f"tp{rank}.safetensors"


def write_rank_weights(source, cfg: TextConfig, weights_dir: str | Path, tp_degree: int = 2) -> None:
    """Cut the checkpoint into one file per NeuronCore. Vision and MTP tensors are never read."""
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    for rank in range(tp_degree):
        weights = checkpoint.rank_weights(source, cfg, rank, tp_degree, DTYPE)
        save_file({k: v.contiguous() for k, v in weights.items()}, shard_path(weights_dir, rank))
        del weights


class NeuronStep:
    """The compiled graph, with the same call signature as `Qwen38Step` for `greedy_generate`."""

    def __init__(self, model_dir: str | Path, weights_dir: str | Path, tp_degree: int = 2):
        self.model = NxDModel.load(str(Path(model_dir) / MODEL_FILE))
        self.model.set_weights([load_file(shard_path(weights_dir, rank)) for rank in range(tp_degree)])
        self.model.to_neuron()

    def __call__(self, **inputs: torch.Tensor) -> torch.Tensor:
        return self.model(**inputs)
