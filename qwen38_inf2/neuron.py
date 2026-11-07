"""Compile and run `Qwen38Step` on Neuron. Requires the Neuron SDK, so import this only in that environment."""

import json
from pathlib import Path

import torch
from neuronx_distributed import ModelBuilder, NxDParallelState
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
from neuronx_distributed.trace.nxd_model import NxDModel

from . import checkpoint
from .config import TextConfig
from .generate import BlockChain, batch_step_inputs, prefill_inputs, state_copy_inputs, verify_inputs
from .model import HeadStep, Qwen38Step

DTYPE = torch.bfloat16
MODEL_FILE = "nxd_model.pt"
# NxD's defaults plus an explicit target. Inf2 shares NeuronCore-v2 codegen with Trn1.
DEFAULT_COMPILER_ARGS = "--enable-saturate-infinity --auto-cast=none --model-type=transformer -O1"


def example_inputs(cfg: TextConfig, max_seq_len: int, batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "inputs_embeds": torch.zeros(batch_size, 1, cfg.hidden_size, dtype=DTYPE),
        **batch_step_inputs(cfg, [0] * batch_size, max_seq_len, DTYPE),
    }


def verify_example_inputs(cfg: TextConfig, max_seq_len: int, batch_size: int, tokens: int) -> dict[str, torch.Tensor]:
    return {
        "inputs_embeds": torch.zeros(batch_size, tokens, cfg.hidden_size, dtype=DTYPE),
        **verify_inputs(cfg, [0] * batch_size, tokens, max_seq_len, DTYPE, [True] * batch_size, [0] * batch_size),
    }


def prefill_example_inputs(cfg: TextConfig, chunk: int) -> dict[str, torch.Tensor]:
    return {
        "inputs_embeds": torch.zeros(1, chunk, cfg.hidden_size, dtype=DTYPE),
        **prefill_inputs(cfg, 0, 0, chunk, chunk, DTYPE),
    }


def compile_graph(
    make_model,
    inputs: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]],
    output_dir: str | Path,
    info: dict,
    tp_degree: int = 2,
    target: str = "inf2",
    extra_compiler_args: str = "",
    workdir: str | Path | None = None,
    optimize_layout: bool = False,
) -> Path:
    """Trace and compile one module for `tp_degree` NeuronCores.

    `inputs` is one set of example inputs, traced as "decode", or a dict of them by tag. Every tag becomes its
    own graph in the same model, sharing its weights and state buffers, and each call runs the graph whose
    input names and shapes it matches.
    """
    output_dir = Path(output_dir)
    traces = inputs if all(isinstance(v, dict) for v in inputs.values()) else {"decode": inputs}
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler_args = f"{DEFAULT_COMPILER_ARGS} --target {target} {extra_compiler_args}".strip()
    # The compiler changes into this directory before opening its log, so it must be absolute.
    workdir = str(Path(workdir).resolve()) if workdir else None

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        builder = ModelBuilder(make_model())
        for tag, example in traces.items():
            builder.trace(kwargs=example, tag=tag)
        nxd_model = builder.compile(
            priority_model_key="decode" if optimize_layout else None,
            compiler_args=compiler_args,
            compiler_workdir=workdir,
            # One graph at a time: decode and prefill compiled together ran a 15 GB inf2.xlarge out of memory.
            max_workers=1,
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
    batch_size: int = 1,
    chunk: int | None = None,
    verify_tokens: int = 0,
) -> Path:
    """Compile the one-token-per-row graph for the layers in `cfg`, plus a prefill graph of `chunk` tokens if given,
    and a verify graph of `verify_tokens` tokens per row for speculative decoding if given."""
    # NxD allocates the state buffers the first graph uses, so the graph using the most of them goes first.
    traces = {"verify": verify_example_inputs(cfg, max_seq_len, batch_size, verify_tokens)} if verify_tokens else {}
    traces["decode"] = example_inputs(cfg, max_seq_len, batch_size)
    if chunk:
        from .kernels import KEY_BLOCK

        if max_seq_len % chunk or max_seq_len % KEY_BLOCK:
            raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of the chunk {chunk} and of {KEY_BLOCK}")
        traces["prefill"] = prefill_example_inputs(cfg, chunk)
        # Saving and restoring a row's recurrent states, so a prompt extending an earlier one skips what they share.
        traces["save"] = state_copy_inputs(cfg, 0, True, DTYPE)
        traces["restore"] = state_copy_inputs(cfg, 0, False, DTYPE)
    return compile_graph(
        lambda: Qwen38Step(
            cfg,
            max_seq_len,
            tp_degree=tp_degree,
            dtype=DTYPE,
            all_reduce=reduce_from_tensor_model_parallel_region,
            batch_size=batch_size,
            verify_tokens=verify_tokens,
            saved_states=bool(chunk),
        ),
        traces,
        output_dir,
        {
            "max_seq_len": max_seq_len,
            "layers": cfg.num_hidden_layers,
            "batch_size": batch_size,
            "weight_format": cfg.weight_format,
            "recurrent_state_format": cfg.recurrent_state_format,
            "kv_cache_format": cfg.kv_cache_format,
            **({"chunk": chunk} if chunk else {}),
            **({"verify_tokens": verify_tokens} if verify_tokens else {}),
        },
        tp_degree,
        target,
        extra_compiler_args,
        workdir,
        optimize_layout,
    )


def build_head(
    cfg: TextConfig, batch_size: int, output_dir: str | Path, tp_degree: int = 2, target: str = "inf2", workdir=None, head_format: str = "bf16"
) -> Path:
    """Compile lm_head for the chip. It is one graph, loaded once, separate from the blocks."""
    return compile_graph(
        lambda: HeadStep(cfg, tp_degree, batch_size, reduce_from_tensor_model_parallel_region, head_format=head_format),
        {"hidden": torch.zeros(batch_size, 1, cfg.hidden_size, dtype=DTYPE)},
        output_dir,
        {"batch_size": batch_size, "group_rows": 32, "head_format": head_format},
        tp_degree,
        target,
        "",
        workdir,
    )


def load_head_graph(model_dir: str | Path, weights_dir: str | Path):
    tp_degree = checkpoint.read_build_info(model_dir)["tp_degree"]
    return load_graph(model_dir, [checkpoint.load_head_weights(weights_dir, rank) for rank in range(tp_degree)])


def preflight(model_dir: str | Path, inputs: dict[str, torch.Tensor], weight_headers: list[dict] = ()) -> list[str]:
    """Run everything NxD does before touching a device, and compare weight files with the graph.

    Needs no Neuron hardware. `weight_headers` are safetensors headers, one per core, so no weights load.
    Returns a list of problems. A problem found here would otherwise only show up on a chip.
    """
    from torch_neuronx.proto import metaneff_pb2

    model = NxDModel.load(str(Path(model_dir) / MODEL_FILE))
    ordered, names = model.convert_dict_to_ordered_list(inputs, 0)
    key = model.router(ordered, names)[0]
    flattened = model.flattener_map[key](ordered)
    meta = model.get_metaneff(key)
    tensor_type = metaneff_pb2.MetaTensor.Type.Name

    problems = []
    wanted = [t for t in meta.input_tensors if tensor_type(t.type) == "USER_INPUT"]
    if len(flattened) != len(wanted):
        problems.append(f"call passes {len(flattened)} tensors but the graph wants {len(wanted)}")
    for got, tensor in zip(flattened, wanted):
        if list(got.shape) != list(tensor.shape):
            problems.append(f"{tensor.user_input_key.decode()}: shape {list(got.shape)}, graph wants {list(tensor.shape)}")

    graph_weights = {t.checkpoint_key.decode(): t for t in meta.input_tensors if tensor_type(t.type) == "INPUT_WEIGHT"}
    data_type = metaneff_pb2.MetaTensor.DataType.Name
    dtypes = {"BF16": "BFLOAT16", "U8": "UINT8"}
    for rank, header in enumerate(weight_headers):
        if set(header) != set(graph_weights):
            problems.append(f"core {rank} weight names differ: {sorted(set(header) ^ set(graph_weights))[:4]}")
        for name in set(header) & set(graph_weights):
            tensor = graph_weights[name]
            if header[name]["shape"] != list(tensor.shape) or dtypes[header[name]["dtype"]] != data_type(tensor.data_type):
                problems.append(f"core {rank} {name}: file {header[name]['shape']}, graph {list(tensor.shape)}")
    return problems


def load_graph(model_dir: str | Path, rank_weights: list[dict[str, torch.Tensor]]):
    """Load a compiled graph onto the chip and return a function that runs it."""
    model = NxDModel.load(str(Path(model_dir) / MODEL_FILE))
    model.set_weights(rank_weights)
    model.to_neuron()
    return lambda **inputs: model(**inputs)


class NeuronStep(BlockChain):
    """The compiled block, loaded once per repeat with that repeat's weights and its own caches.

    A build with a prefill graph takes prefill inputs through the same call, since each loaded block holds both.
    """

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
