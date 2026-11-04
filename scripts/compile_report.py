#!/usr/bin/env python3
"""Summarize what a compiled block costs, using only the compiler's own outputs.

No Neuron hardware or SDK needed. These are static figures, useful for comparing one compiled design with
another, not speed measurements. The fields come from undocumented compiler files, so treat the
memory figures as estimates.
"""

import argparse
import io
import json
import tarfile
from pathlib import Path

GB = 1e9
BLOCKS = 16


def read_neff_members(path: Path) -> tarfile.TarFile:
    raw = path.read_bytes()
    return tarfile.open(fileobj=io.BytesIO(raw[raw.find(b"\x1f\x8b\x08") :]), mode="r:gz")


def report(workdir: Path) -> dict:
    tensorizer = json.loads((workdir / "global_metric_store.json").read_text())["Sum"]["tensorizer"]
    neff = read_neff_members(workdir / "graph.neff")
    variables = json.loads(neff.extractfile("sg00/def.json").read())["var"]
    virtual = [v for v in variables.values() if v["type"] == "virtual"]
    copies = [v["size"] for name, v in variables.items() if "d2dtranspose" in name]

    per_block = {
        "ddr_transfer_gb": tensorizer["StaticProfiler::DDRTransferBytes"] / GB,
        "internal_transfer_gb": tensorizer["StaticProfiler::InternalTransferBytes"] / GB,
        "instructions_executed": tensorizer["StaticProfiler::TotalDynamicInstancesCount"],
        "weight_transpose_copies": len(copies),
        "weight_transpose_gb": sum(copies) / GB,
        "input_gb": sum(v["size"] for v in variables.values() if v["type"] == "input") / GB,
        "scratch_gb": max((v["backing_variable_off"] + v["size"] for v in virtual), default=0) / GB,
        "program_gb": sum(m.size for m in neff.getmembers() if m.name.endswith(".bin")) / GB,
    }
    return {
        "per_block_per_core": per_block,
        "whole_model_per_core_gb": {
            "weights_and_caches": BLOCKS * per_block["input_gb"],
            "scratch": BLOCKS * per_block["scratch_gb"],
            "program": BLOCKS * per_block["program_gb"],
            "total": BLOCKS * (per_block["input_gb"] + per_block["scratch_gb"] + per_block["program_gb"]),
        },
        "ddr_transfer_per_token_whole_chip_gb": BLOCKS * 2 * per_block["ddr_transfer_gb"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path, help="The timestamped directory under compiler-work/decode.")
    print(json.dumps(report(parser.parse_args().workdir), indent=1))


if __name__ == "__main__":
    main()
