#!/usr/bin/env python3
"""Check a compiled graph's structure without Neuron hardware. Needs the Neuron SDK.

A wrong graph can compile cleanly and then produce garbage on a chip, so this checks what the compiler
was actually given: every cache is on-device state that persists between tokens, each layer has its
two collectives, and every FP4 matrix is decoded in the graph. Exits non-zero on any mismatch.
"""

import argparse
import collections
import json
import sys
from dataclasses import replace
from pathlib import Path

from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import hlo_pb2

from qwen38_inf2.config import LINEAR, TextConfig, tiny_config

STATE_SUFFIXES = (".conv_state", ".recurrent_state", ".k_cache", ".v_cache")
USER_INPUTS = {"inputs_embeds", "cos", "sin", "write_mask", "attn_mask", "keep_state"}
# Ops that would mean the graph is not the one the model intends, or that lower poorly on this target.
FORBIDDEN_OPS = {"gather", "scatter", "convolution", "while", "triangular-solve", "custom-call"}


def expected_counts(cfg: TextConfig) -> dict[str, int]:
    linear = sum(t == LINEAR for t in cfg.layer_types)
    full = cfg.num_hidden_layers - linear
    mlp_matrices = 3 * cfg.num_hidden_layers
    return {
        "states": 2 * linear + 2 * full,  # conv + recurrent, or k + v cache
        "all_reduce": 2 * cfg.num_hidden_layers,  # one for the mixer, one for the MLP
        "fp4_matrices": 3 * linear + 4 * full + mlp_matrices,
    }


def inspect(workdir: Path, cfg: TextConfig) -> tuple[dict, list[str]]:
    module = hlo_pb2.HloModuleProto()
    module.ParseFromString((workdir / "model" / "graph.hlo").read_bytes())
    meta = metaneff_pb2.MetaNeff()
    meta.ParseFromString((workdir / "metaneff.pb").read_bytes())

    entry = next(c for c in module.computations if c.id == module.entry_computation_id)
    ops = collections.Counter(i.opcode for i in entry.instructions)
    type_names = metaneff_pb2.MetaTensor.Type.Name
    dtype_names = metaneff_pb2.MetaTensor.DataType.Name

    by_role = collections.defaultdict(list)
    for tensor in meta.input_tensors:
        by_role[type_names(tensor.type)].append(tensor)
    state_keys = {t.checkpoint_key.decode() for t in by_role.get("INPUT_STATE", [])}
    user_keys = {t.user_input_key.decode() for t in meta.input_tensors if t.user_input_key}
    aliased = {meta.input_tensors[i].checkpoint_key.decode() for i in meta.output_aliases_to.values()}

    weight_dtypes = collections.Counter()
    weight_bytes = 0
    for tensor in by_role.get("INPUT_WEIGHT", []):
        count = 1
        for dim in tensor.shape:
            count *= dim
        dtype = dtype_names(tensor.data_type)
        weight_dtypes[dtype] += 1
        weight_bytes += count * {"UINT8": 1, "BFLOAT16": 2, "FLOAT": 4}[dtype]

    want = expected_counts(cfg)
    got = {
        "states": len(aliased),
        "all_reduce": ops.get("all-reduce", 0),
        "fp4_matrices": ops.get("floor", 0),
    }
    problems = [f"{name}: expected {want[name]}, found {got[name]}" for name in want if want[name] != got[name]]
    not_cache = sorted(k for k in aliased if not k.endswith(STATE_SUFFIXES))
    if state_keys != aliased:
        problems.append(f"state tensors not written back each token: {sorted(state_keys ^ aliased)[:5]}")
    if not_cache:
        problems.append(f"non-cache tensors are linked as state: {not_cache[:5]}")
    if weight_dtypes["UINT8"] != want["fp4_matrices"]:
        problems.append(f"expected {want['fp4_matrices']} packed uint8 weights, found {weight_dtypes['UINT8']}")
    if user_keys != USER_INPUTS:
        problems.append(f"user inputs differ from the graph contract: {sorted(user_keys ^ USER_INPUTS)}")
    bad_ops = sorted(FORBIDDEN_OPS & set(ops))
    if bad_ops:
        problems.append(f"unexpected ops: {bad_ops}")

    report = {
        "counts": got,
        "user_inputs": sorted(user_keys),
        "weight_dtypes": dict(weight_dtypes),
        "weight_gb_per_core": round(weight_bytes / 1e9, 3),
        "opcodes": dict(ops.most_common()),
    }
    return report, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path, help="The timestamped directory under compiler-work/decode.")
    parser.add_argument("--config", help="Model config.json. Omit for the tiny model.")
    parser.add_argument("--layers", type=int, help="The --layers value used when compiling.")
    args = parser.parse_args()

    cfg = TextConfig.from_file(args.config) if args.config else tiny_config()
    if args.layers:
        cfg = replace(cfg, layer_types=cfg.layer_types[: args.layers])

    report, problems = inspect(args.workdir, cfg)
    print(json.dumps(report, indent=1))
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("graph check", "FAILED" if problems else "passed")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
