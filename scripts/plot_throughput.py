#!/usr/bin/env python3
"""Draw the throughput log as a black and white step chart, from the measured results in docs/throughput.

Needs matplotlib, which nothing else in this repo uses: uv run --with matplotlib scripts/plot_throughput.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "throughput"
# In the order of the files in docs/throughput, one label per measured change.
LABELS = [
    "Baseline",
    "Batching",
    "Faster FP4 decode",
    "FP8 weights",
    "Max length 512",
    "lm_head on chip",
    "BF16 gated delta state",
    "16-layer blocks",
    "Largest batch that fits",
    "FP8 lm_head",
    "128 MB scratchpad",
    "64 MB scratchpad",
    "Warm disk cache",
    "Batch 63",
    "Faster row inputs",
    "FP8 x FP8 matmuls",
]
INK, MUTED, GRID = "#000000", "#555555", "#dddddd"


def main():
    results = [json.loads(path.read_text()) for path in sorted(RESULTS.glob("*.json"))]
    if len(results) != len(LABELS):
        raise ValueError(f"{len(results)} results but {len(LABELS)} labels")
    rate = [r["generated_tokens_per_second"] for r in results]
    x = list(range(1, len(rate) + 1))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=200, facecolor="white")
    ax.set_facecolor("white")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 400)
    ax.set_xlim(0.4, len(rate) + 0.75)

    ax.step(x + [len(rate) + 0.6], rate + [rate[-1]], where="post", color=INK, linewidth=2, zorder=2)
    ax.scatter(x, rate, s=64, color=INK, edgecolor="white", linewidth=1.5, zorder=3)
    for position, value in zip(x, rate):
        ax.annotate(
            f"{value:.2f}" if value < 10 else f"{value:.1f}",
            (position, value),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            zorder=4,
        )

    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Generated tokens per second (log scale)", color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\n(batch {r['batch_size']})" for label, r in zip(LABELS, results)],
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
    )
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=0, labelcolor=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    fig.suptitle(
        f"Qwen3.8-27B on one Inf2 chip: {rate[0]:.2f} to {rate[-1]:.1f} tok/s, about {rate[-1] / rate[0]:.0f}x",
        x=0.065,
        y=0.97,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.895,
        "Each step is one change, measured on the real chip with distinct prompts through full batches, counting prompt steps and idle rows",
        ha="left",
        fontsize=11,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.87, bottom=0.235)
    output = RESULTS.parent / "throughput.png"
    fig.savefig(output, facecolor="white")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
