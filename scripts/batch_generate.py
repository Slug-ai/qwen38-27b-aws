#!/usr/bin/env python3
"""The throughput benchmark: real distinct prompts through a real full batch, on Inf2 or with --simulate on CPU.

The headline is tokens actually produced divided by wall time, including prompt steps and any idle rows.
Nothing is projected from a single step. Row 0 runs the saved greedy prompt, and a saved 38-token passage runs
through every row first, so a wrong graph can't look fast.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from qwen38_inf2 import checkpoint
from qwen38_inf2.chip_check import NEAR_TIE_LOGITS
from qwen38_inf2.config import TextConfig
from qwen38_inf2.generate import EOS_TOKEN_IDS, ChipHead, HostHead, batch_step_inputs, batched_greedy_generate, chat_prompt

TOPICS = (
    "lighthouses, volcanoes, honeybees, glaciers, compilers, bicycles, coral reefs, origami, telescopes, chess, "
    "sourdough bread, railways, the Roman Empire, tides, fireflies, the printing press, octopuses, jazz, windmills, "
    "the Silk Road, photosynthesis, submarines, cartography, penguins, the steam engine, mushrooms, suspension bridges, "
    "the Moon, quilting, black holes, river deltas, paper money, falconry, the immune system, stained glass, monsoons, "
    "typewriters, tectonic plates, beekeeping, sundials"
).split(", ")
TEMPLATES = [
    "Write two sentences about {}.",
    "What is one surprising fact about {}?",
    "Explain {} to a ten year old in one short paragraph.",
    "Give three bullet points on the history of {}.",
    "Write a short poem about {}.",
]


def benchmark_prompts() -> list[str]:
    """A fixed list of distinct prompts, so every change is measured on the same work."""
    return [template.format(topic) for topic in TOPICS for template in TEMPLATES]


class TimedStep:
    """Add up time on the chip, so the rest of the wall time can be put down to the host."""

    def __init__(self, step):
        self.step = step
        self.seconds = 0.0
        self.count = 0
        self.starts = []

    def __call__(self, **inputs):
        started = time.perf_counter()
        self.starts.append(started)
        hidden = self.step(**inputs)
        self.seconds += time.perf_counter() - started
        self.count += 1
        return hidden


def check_saved_tokens(step, head, cfg, golden: dict, batch_size: int, max_seq_len: int) -> dict:
    """Run the saved test passage through every row and compare the top token at each position with the CPU's.

    Every row must agree, and a top token may only differ where the CPU's top two logits were nearly tied.
    """
    wrong, near_ties, rows_disagree = 0, 0, 0
    for position, token_id in enumerate(golden["tokens"]):
        inputs = batch_step_inputs(cfg, [position] * batch_size, max_seq_len, torch.bfloat16)
        tops = head.next_tokens(step(inputs_embeds=head.embed_batch([token_id] * batch_size, torch.bfloat16), **inputs))
        rows_disagree += len(set(tops)) > 1
        if tops[0] != golden["top"][position]:
            if golden["margin"][position] < NEAR_TIE_LOGITS:
                near_ties += 1
            else:
                wrong += 1
    return {"positions": len(golden["tokens"]), "wrong": wrong, "near_tie_flips": near_ties, "positions_where_rows_disagree": rows_disagree}


def compare_tokens(outputs: dict, reference: dict) -> dict:
    """How far each sequence stays identical to a reference build. One near tie changes every later token."""
    common = [index for index in outputs if str(index) in reference]
    prefixes = []
    for index in common:
        want = reference[str(index)]
        matching = next((i for i, (a, b) in enumerate(zip(outputs[index], want)) if a != b), min(len(outputs[index]), len(want)))
        prefixes.append(matching)
    total = sum(len(reference[str(index)]) for index in common)
    return {
        "sequences": len(common),
        "identical_sequences": sum(outputs[index] == reference[str(index)] for index in common),
        "tokens_before_first_difference": sum(prefixes),
        "tokens_compared": total,
    }


def screen(step, cfg, batch_size: int, max_seq_len: int, blocks: int, steps: int = 12) -> None:
    """A quick check that a new graph runs. Its output means nothing, and its timing isn't reliable enough to compare builds."""
    embeds = torch.zeros(batch_size, 1, cfg.hidden_size, dtype=torch.bfloat16)
    times = []
    for position in range(steps):
        inputs = batch_step_inputs(cfg, [position] * batch_size, max_seq_len, torch.bfloat16)
        started = time.perf_counter()
        step(inputs_embeds=embeds, **inputs)
        times.append(time.perf_counter() - started)
    median = sorted(times)[steps // 2] / blocks
    print(f"screening only: {median:.3f}s per block at batch {batch_size}, so about {16 * median:.2f}s on the chip per full step")


def main():
    # The runtime reads this when it starts. The default page size reserves far more scratchpad than these graphs need.
    os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "64")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--head", help="Compiled lm_head for the chip. Its weights go in --weights too. Omit to run lm_head on the host.")
    parser.add_argument("--prompts-file", type=Path, help="One prompt per line. Defaults to the fixed benchmark set.")
    parser.add_argument("--sequences", type=int, help="Total sequences to run. Defaults to two full batches.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--golden", type=Path, default=Path("tests/golden/model.json"), help="Saved greedy result, checked in row 0.")
    parser.add_argument("--output", type=Path, help="Write the results as JSON here.")
    parser.add_argument("--save-tokens", type=Path, help="Write every sequence's generated tokens as JSON, to compare builds.")
    parser.add_argument("--compare-tokens", type=Path, help="Compare generated tokens with a file from --save-tokens.")
    parser.add_argument("--show", type=int, default=8, help="How many outputs to print.")
    parser.add_argument("--simulate", action="store_true", help="Run on CPU instead of a chip.")
    parser.add_argument("--screen-blocks", type=int, help="Time this many real blocks at the real batch size, then stop. Only shows a graph runs, because its timing varies by nearly 2x between runs.")
    args = parser.parse_args()

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    info = checkpoint.read_build_info(args.compiled)
    batch_size = info.get("batch_size", 1)
    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    lines = args.prompts_file.read_text().splitlines() if args.prompts_file else benchmark_prompts()
    lines = [line for line in lines if line.strip()]
    sequences = args.sequences or 2 * batch_size
    if sequences > len(lines) + 1:
        parser.error(f"only {len(lines)} distinct prompts for {sequences} sequences")
    golden = json.loads(args.golden.read_text())
    prompts = [golden["prompt"]] + [
        tokenizer.encode(chat_prompt(line), add_special_tokens=False).ids for line in lines[: sequences - 1]
    ]

    source = checkpoint.TensorSource(f"{args.model_dir}/model.safetensors")
    if args.head:
        from qwen38_inf2.neuron import load_head_graph

        head = ChipHead(source, cfg.rms_norm_eps, load_head_graph(args.head, args.weights), batch_size)
    else:
        head = HostHead(source, cfg.rms_norm_eps)
    started = time.perf_counter()
    if args.simulate:
        from qwen38_inf2.cpu_sim import SimulatedCoresStep

        inner = SimulatedCoresStep(args.weights, cfg, info, torch.bfloat16)
    else:
        from qwen38_inf2.neuron import NeuronStep

        inner = NeuronStep(args.compiled, args.weights, cfg, args.screen_blocks)
    load_seconds = time.perf_counter() - started
    print(f"loaded {batch_size}-row graph in {load_seconds:.1f}s", flush=True)

    if args.screen_blocks:
        screen(inner, cfg, batch_size, info["max_seq_len"], args.screen_blocks)
        return

    if args.head:
        # Loading the weights pushes these tables out of the disk cache, and each step reads scattered rows of them.
        started = time.perf_counter()
        source.touch(checkpoint.LM_HEAD)
        source.touch(checkpoint.EMBED_TOKENS)
        print(f"read lm_head and embeddings into the disk cache in {time.perf_counter() - started:.1f}s", flush=True)

    saved_tokens = check_saved_tokens(inner, head, cfg, golden, batch_size, info["max_seq_len"])
    print(f"saved test passage: {saved_tokens}", flush=True)

    step = TimedStep(inner)
    outputs = defaultdict(list)
    tokens_at_step = defaultdict(int)
    started = time.perf_counter()
    for index, token_id in batched_greedy_generate(
        step, head, cfg, prompts, batch_size, args.max_new_tokens, info["max_seq_len"], EOS_TOKEN_IDS
    ):
        outputs[index].append(token_id)
        tokens_at_step[step.count - 1] += 1
    finished = time.perf_counter()
    wall = finished - started

    want = golden["output"]
    golden_ok = outputs[0][: len(want)] == want
    for index in range(1, min(args.show + 1, sequences)):
        print(f"\n[{index}] input:  {lines[index - 1]}")
        print(f"[{index}] output: {tokenizer.decode(outputs[index], skip_special_tokens=True)!r}")

    # A stretch where every row produced a token is measured from those steps' real tokens and real time.
    ends = step.starts[1:] + [finished]
    full_steps = [i for i in range(step.count) if tokens_at_step[i] == batch_size]
    full_seconds = sum(ends[i] - step.starts[i] for i in full_steps)

    generated = sum(len(tokens) for tokens in outputs.values())
    fed = sum(len(p) for p in prompts)
    results = {
        "batch_size": batch_size,
        "max_seq_len": info["max_seq_len"],
        "sequences": sequences,
        "distinct_prompts": len({tuple(p) for p in prompts}),
        "prompt_tokens": fed,
        "generated_tokens": generated,
        "steps": step.count,
        "chip_seconds_per_step": step.seconds / step.count,
        "host_seconds_per_step": (wall - step.seconds) / step.count,
        "wall_seconds": wall,
        "generated_tokens_per_second": generated / wall,
        "all_tokens_per_second": (generated + fed) / wall,
        "steps_with_every_row_generating": len(full_steps),
        "tokens_per_second_when_every_row_generated": batch_size * len(full_steps) / full_seconds if full_steps else None,
        "golden_row_matches": golden_ok,
        "golden_row_tokens": outputs[0][: len(want)],
        "saved_test_passage": saved_tokens,
        "load_seconds": load_seconds,
    }
    if args.save_tokens:
        args.save_tokens.write_text(json.dumps({str(index): tokens for index, tokens in outputs.items()}))
    if args.compare_tokens:
        results["compared_with_reference"] = compare_tokens(outputs, json.loads(args.compare_tokens.read_text()))
    print()
    print(json.dumps(results, indent=1))
    if args.output:
        args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
