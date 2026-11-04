"""Compile and run `Qwen38Step` on Neuron. Requires the Neuron SDK, so import this only in that environment."""

import json
from pathlib import Path

import torch
from neuronx_distributed import ModelBuilder, NxDParallelState
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
from neuronx_distributed.trace.nxd_model import NxDModel

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


def compile_graph(
    make_model,
    inputs: dict[str, torch.Tensor],
    output_dir: str | Path,
    info: dict,
    tp_degree: int = 2,
    target: str = "inf2",
    extra_compiler_args: str = "",
    workdir: str | Path | None = None,
    optimize_layout: bool = False,
) -> Path:
    """Trace and compile one module for `tp_degree` NeuronCores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler_args = f"{DEFAULT_COMPILER_ARGS} --target {target} {extra_compiler_args}".strip()
    # The compiler changes into this directory before opening its log, so it must be absolute.
    workdir = str(Path(workdir).resolve()) if workdir else None

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        builder = ModelBuilder(make_model())
        builder.trace(kwargs=inputs, tag="decode")
        nxd_model = builder.compile(
            priority_model_key="decode" if optimize_layout else None,
            compiler_args=compiler_args,
            compiler_workdir=workdir,
        )

    path = output_dir / MODEL_FILE
    nxd_model.save(str(path))
    info = {"tp_degree": tp_degree, "target": target, **info}
    (output_dir / checkpoint.BUILD_INFO_FILE).write_text(json.dumps(info, indent=2))
    return path


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
    """Compile the single-token graph for the layers in `cfg`."""
    return compile_graph(
        lambda: Qwen38Step(
            cfg, max_seq_len, tp_degree=tp_degree, dtype=DTYPE, all_reduce=reduce_from_tensor_model_parallel_region
        ),
        example_inputs(cfg, max_seq_len),
        output_dir,
        {"max_seq_len": max_seq_len, "layers": cfg.num_hidden_layers},
        tp_degree,
        target,
        extra_compiler_args,
        workdir,
        optimize_layout,
    )


def load_graph(model_dir: str | Path, rank_weights: list[dict[str, torch.Tensor]]):
    """Load a compiled graph onto the chip and return a function that runs it."""
    model = NxDModel.load(str(Path(model_dir) / MODEL_FILE))
    model.set_weights(rank_weights)
    model.to_neuron()
    return lambda **inputs: model(**inputs)


class NeuronStep(BlockChain):
    """The compiled block, loaded once per repeat with that repeat's weights and its own caches."""

    def __init__(self, model_dir: str | Path, weights_dir: str | Path, cfg: TextConfig, num_blocks: int | None = None):
        info = checkpoint.read_build_info(model_dir)
        num_blocks = num_blocks or cfg.num_hidden_layers // info["layers"]
        super().__init__(
            [
                load_graph(
                    model_dir,
                    [checkpoint.load_rank_block(weights_dir, block, rank, info["layers"]) for rank in range(info["tp_degree"])],
                )
                for block in range(num_blocks)
            ]
        )
