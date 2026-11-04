"""Compile and run `Qwen38Step` on Neuron. Requires the Neuron SDK, so import this only in that environment."""

import json
from pathlib import Path

import torch
from neuronx_distributed import ModelBuilder, NxDParallelState
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
from neuronx_distributed.trace.nxd_model import NxDModel
from safetensors.torch import load_file

from . import checkpoint
from .config import TextConfig
from .generate import BlockChain, step_inputs
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
    optimize_layout: bool = False,
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
        nxd_model = builder.compile(
            priority_model_key="decode" if optimize_layout else None,
            compiler_args=compiler_args,
            compiler_workdir=workdir,
        )

    path = output_dir / MODEL_FILE
    nxd_model.save(str(path))
    info = {"max_seq_len": max_seq_len, "tp_degree": tp_degree, "target": target, "layers": cfg.num_hidden_layers}
    (output_dir / checkpoint.BUILD_INFO_FILE).write_text(json.dumps(info, indent=2))
    return path


class NeuronStep(BlockChain):
    """The compiled block, loaded once per repeat with that repeat's weights and its own caches."""

    def __init__(self, model_dir: str | Path, weights_dir: str | Path, cfg: TextConfig):
        info = checkpoint.read_build_info(model_dir)
        blocks = []
        for block in range(cfg.num_hidden_layers // info["layers"]):
            model = NxDModel.load(str(Path(model_dir) / MODEL_FILE))
            model.set_weights([load_file(checkpoint.shard_path(weights_dir, block, rank)) for rank in range(info["tp_degree"])])
            model.to_neuron()
            blocks.append(_named_call(model))
        super().__init__(blocks)


def _named_call(model):
    return lambda **inputs: model(**inputs)
