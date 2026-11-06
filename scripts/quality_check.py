#!/usr/bin/env python3
"""Measure a build's output quality: how often its top token is the real next token of a fixed English text.

Greedy text can split at a near tie without being worse, so this compares builds on the same known text. Results
go to a file, and passing an earlier build's file also reports how often the two builds pick the same token.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from qwen38_inf2 import checkpoint
from qwen38_inf2.config import TextConfig
from qwen38_inf2.generate import ChipHead, batch_step_inputs, prefill_inputs
from qwen38_inf2.neuron import NeuronStep, load_head_graph

# The opening of the US Declaration of Independence and the Gettysburg Address, both public domain.
TEXT = """When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.

We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness.

Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this."""


def main():
    os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "128")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="Results file from another build.")
    args = parser.parse_args()

    cfg = TextConfig.from_file(f"{args.model_dir}/config.json")
    info = checkpoint.read_build_info(args.compiled)
    batch = info.get("batch_size", 1)
    tokens = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json").encode(TEXT, add_special_tokens=False).ids
    tokens = tokens[: info["max_seq_len"] - info.get("chunk", 1)]

    source = checkpoint.TensorSource(f"{args.model_dir}/model.safetensors")
    head = ChipHead(source, cfg.rms_norm_eps, load_head_graph(args.head, args.weights), batch)
    step = NeuronStep(args.compiled, args.weights, cfg)

    top = []
    rows_disagree = 0
    for position, token_id in enumerate(tokens):
        inputs = batch_step_inputs(cfg, [position] * batch, info["max_seq_len"], torch.bfloat16)
        tops = head.next_tokens(step(inputs_embeds=head.embed_batch([token_id] * batch, torch.bfloat16), **inputs))
        rows_disagree += len(set(tops)) > 1
        top.append(tops[0])

    correct = sum(a == b for a, b in zip(top[:-1], tokens[1:]))
    results = {"build": args.compiled}
    chunk = info.get("chunk")
    if chunk:
        # The same text through the prefill graph, a chunk at a time into row 0, which must agree with decode.
        prefilled = []
        for start in range(0, len(tokens), chunk):
            ids = tokens[start : start + chunk]
            embeds = head.embed_batch(ids + [0] * (chunk - len(ids)), torch.bfloat16).reshape(1, chunk, -1)
            hidden = step(inputs_embeds=embeds, **prefill_inputs(cfg, 0, start, len(ids), chunk, torch.bfloat16))[0]
            for first in range(0, len(ids), batch):
                prefilled += head.next_tokens(hidden[first : min(first + batch, len(ids))].unsqueeze(1))
        results |= {
            "prefill_top_token_is_next_token": sum(a == b for a, b in zip(prefilled[:-1], tokens[1:])),
            "prefill_same_top_token_as_decode": sum(a == b for a, b in zip(prefilled, top)),
        }
    results |= {
        "positions": len(tokens) - 1,
        "top_token_is_next_token": correct,
        "accuracy": correct / (len(tokens) - 1),
        "positions_where_rows_disagree": rows_disagree,
    }
    if args.compare:
        other = json.loads(args.compare.read_text())["top"]
        results["same_top_token_as_other_build"] = sum(a == b for a, b in zip(top, other))
    print(json.dumps(results, indent=1))
    args.output.write_text(json.dumps({**results, "top": top}))


if __name__ == "__main__":
    main()
