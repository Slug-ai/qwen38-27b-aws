"""Load the compiled model once and run prompts through it. Shared by scripts/demo.py and scripts/benchmark.py."""

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from . import checkpoint
from .config import TextConfig
from .generate import EOS_TOKEN_IDS, ChipHead, HostHead, batched_greedy_generate, chat_prompt


def _log(message: str) -> None:
    print(message, flush=True)


@dataclass
class Loaded:
    cfg: TextConfig
    tokenizer: Tokenizer
    head: ChipHead | HostHead
    step: Callable[..., torch.Tensor]
    batch_size: int
    max_seq_len: int
    load_seconds: float


def load(model_dir, compiled, weights, head_dir=None, simulate: bool = False, log: Callable[[str], None] = _log) -> Loaded:
    """Load the graphs and weights, about 3 minutes on the chip, then read lm_head and the embeddings into the disk cache."""
    # The runtime reads this when it starts. The default page size reserves far more scratchpad than these graphs need.
    os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "64")
    cfg = TextConfig.from_file(f"{model_dir}/config.json")
    info = checkpoint.read_build_info(compiled)
    batch_size = info.get("batch_size", 1)
    tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    source = checkpoint.TensorSource(f"{model_dir}/model.safetensors")
    if head_dir:
        from .neuron import load_head_graph

        head = ChipHead(source, cfg.rms_norm_eps, load_head_graph(head_dir, weights), batch_size)
    else:
        head = HostHead(source, cfg.rms_norm_eps)

    log(f"loading the {batch_size}-row graph, which takes about 3 minutes on the chip")
    started = time.perf_counter()
    if simulate:
        from .cpu_sim import SimulatedCoresStep

        step = SimulatedCoresStep(weights, cfg, info, torch.bfloat16)
    else:
        from .neuron import NeuronStep

        step = NeuronStep(compiled, weights, cfg)
    load_seconds = time.perf_counter() - started
    log(f"loaded in {load_seconds:.1f}s")
    if head_dir:
        # Loading the weights pushes these tables out of the disk cache, and each step reads scattered rows of them.
        started = time.perf_counter()
        source.touch(checkpoint.LM_HEAD)
        source.touch(checkpoint.EMBED_TOKENS)
        log(f"read lm_head and embeddings into the disk cache in {time.perf_counter() - started:.1f}s")
    return Loaded(cfg, tokenizer, head, step, batch_size, info["max_seq_len"], load_seconds)


def encode(tokenizer, lines: Sequence[str]) -> list[list[int]]:
    """Token ids for each user message, in the checkpoint's chat template."""
    return [tokenizer.encode(chat_prompt(line), add_special_tokens=False).ids for line in lines]


def generate(step, head, cfg, prompts: Sequence[Sequence[int]], batch_size: int, max_new_tokens: int, max_seq_len: int, on_token=None):
    """Run token-id prompts through the loaded batch, and return each one's new tokens and the wall time.

    Each prompt starts its row at position 0, which clears that row's state, so nothing carries over from
    an earlier call, including one that was stopped partway.
    """
    outputs = [[] for _ in prompts]
    started = time.perf_counter()
    for index, token_id in batched_greedy_generate(step, head, cfg, prompts, batch_size, max_new_tokens, max_seq_len, EOS_TOKEN_IDS):
        outputs[index].append(token_id)
        if on_token:
            on_token(index, token_id)
    return outputs, time.perf_counter() - started
