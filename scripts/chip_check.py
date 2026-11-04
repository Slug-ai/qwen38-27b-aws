#!/usr/bin/env python3
"""Check a chip against saved CPU results, one level at a time, stopping at the first failure.

    decode   FP4 decode alone must be exact
    matrix   one real matrix, one core
    layer    one gated delta layer, two cores, several tokens
    block    the compiled 4-layer block, adding full attention
    model    all 16 blocks, top tokens on a test passage
    greedy   real generation
    speed    real per-token time, split between chip and host

`--backend cpu` runs the same levels on the simulated cores, to test this script without a chip.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from qwen38_inf2 import checkpoint, chip_check, probes
from qwen38_inf2.config import TextConfig
from qwen38_inf2.generate import HostHead, greedy_generate, step_inputs

LEVELS = ["decode", "matrix", "layer", "block", "model", "greedy", "speed"]
SPEED_TOKENS = 16


class CpuBackend:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg

    def decode(self, weights):
        probe = probes.DecodeProbe()
        probe.load_state_dict(weights)
        return probe

    def matrix(self, weights):
        probe = probes.matrix_probe()
        probe.load_state_dict(weights, assign=True)
        return probe

    def blocks(self, compiled, layers, num_blocks):
        from qwen38_inf2.cpu_sim import SimulatedCoresStep

        info = {"layers": layers, "tp_degree": 2, "max_seq_len": self.args.max_seq_len}
        return SimulatedCoresStep(self.args.weights, self.cfg, info, chip_check.DTYPE, num_blocks)


class NeuronBackend:
    def __init__(self, args, cfg):
        import qwen38_inf2.neuron as neuron

        self.neuron, self.args, self.cfg = neuron, args, cfg

    def decode(self, weights):
        return self.neuron.load_graph(self.args.levels_dir / "decode", [weights])

    def matrix(self, weights):
        return self.neuron.load_graph(self.args.levels_dir / "matrix", [weights])

    def blocks(self, compiled, layers, num_blocks):
        return self.neuron.NeuronStep(compiled, self.args.weights, self.cfg, num_blocks)


def run_level(level, args, cfg, backend, state):
    golden_dir = args.golden
    if level == "decode":
        golden = load_file(golden_dir / "decode.safetensors")
        expected = {"expected": golden["expected"]}
        weights = {name: golden[name] for name in ("weight_packed", "weight_scale")}
        return chip_check.check_decode(backend.decode(weights), expected)
    if level == "matrix":
        golden = load_file(golden_dir / "matrix.safetensors")
        return chip_check.check_matrix(backend.matrix(probes.matrix_probe_weights(args.weights)), golden)
    if level == "layer":
        golden = load_file(golden_dir / "layer.safetensors")
        return chip_check.check_tokens(backend.blocks(args.levels_dir / "layer", 1, 1), cfg, golden, args.max_seq_len, level)
    if level == "block":
        golden = load_file(golden_dir / "block.safetensors")
        step = backend.blocks(args.compiled, args.block_layers, 1)
        return chip_check.check_tokens(step, cfg, golden, args.max_seq_len, level)

    # The remaining levels share one loaded model, since loading all 16 blocks is slow.
    golden = json.loads((golden_dir / "model.json").read_text())
    if "step" not in state:
        source = checkpoint.TensorSource(args.model_dir / "model.safetensors")
        state["head"] = HostHead(source, cfg.rms_norm_eps)
        state["step"] = backend.blocks(args.compiled, args.block_layers, cfg.num_hidden_layers // args.block_layers)
    step, head = state["step"], state["head"]
    if level == "model":
        return chip_check.check_model(step, head, cfg, golden, args.max_seq_len)
    if level == "greedy":
        return chip_check.check_greedy(step, head, cfg, golden, args.max_seq_len, greedy_generate)
    return chip_check.measure_speed(step, head, cfg, golden["prompt"], SPEED_TOKENS, args.max_seq_len)


def run_preflight(args, cfg):
    """Check every graph against its real call and every weight file, with no device. Returns True if clean."""
    from qwen38_inf2 import neuron

    dtype = chip_check.DTYPE
    tokens = {"inputs_embeds": torch.zeros(1, 1, cfg.hidden_size, dtype=dtype), **step_inputs(cfg, 0, args.max_seq_len, dtype)}
    num_blocks = cfg.num_hidden_layers // args.block_layers
    block_files = [
        [checkpoint.rank_block_header(args.weights, block, rank, args.block_layers) for rank in range(2)]
        for block in range(num_blocks)
    ]
    checks = [
        ("decode", args.levels_dir / "decode", {"gain": torch.ones(1, dtype=dtype)}, [[]]),
        ("matrix", args.levels_dir / "matrix", {"x": torch.zeros(1, 1, probes.MATRIX_IN_FEATURES, dtype=dtype)}, [[]]),
        ("layer", args.levels_dir / "layer", tokens, [[checkpoint.rank_block_header(args.weights, 0, r, 1) for r in range(2)]]),
        ("block", args.compiled, tokens, block_files),
    ]
    clean = True
    for name, model_dir, inputs, weight_sets in checks:
        problems = sorted({p for headers in weight_sets for p in neuron.preflight(model_dir, inputs, headers)})
        print(f"{name}: {'ok' if not problems else 'PROBLEMS'} ({len(weight_sets)} weight set(s) compared)", flush=True)
        for problem in problems[:8]:
            print(f"  {problem}")
        clean = clean and not problems
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("neuron", "cpu"), default="neuron")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True, help="The compiled block.")
    parser.add_argument("--levels-dir", type=Path, default=Path("artifacts/levels"), help="Output of compile_levels.py.")
    parser.add_argument("--golden", type=Path, default=Path("tests/golden"))
    parser.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--block-layers", type=int, default=4)
    parser.add_argument("--keep-going", action="store_true", help="Continue past a failed level.")
    parser.add_argument("--preflight", action="store_true", help="Check graphs and weight files with no device, then stop.")
    parser.add_argument("--output", type=Path, help="Write results as JSON here.")
    args = parser.parse_args()

    cfg = TextConfig.from_file(args.model_dir / "config.json")
    if args.preflight:
        clean = run_preflight(args, cfg)
        print("preflight", "passed" if clean else "FAILED")
        sys.exit(0 if clean else 1)
    backend = (CpuBackend if args.backend == "cpu" else NeuronBackend)(args, cfg)

    results, state, failed = [], {}, False
    with torch.no_grad():
        for level in args.levels:
            print(f"--- {level}", flush=True)
            result = run_level(level, args, cfg, backend, state)
            results.append(vars(result))
            print("PASS" if result.passed else "FAIL", json.dumps(result.details)[:600], flush=True)
            if result.hint:
                print(f"hint: {result.hint}", flush=True)
            failed = failed or not result.passed
            if failed and not args.keep_going:
                print("Stopping at the first failure. Pass --keep-going to continue anyway.")
                break

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
