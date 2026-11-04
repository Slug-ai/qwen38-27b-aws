# Throughput log

**The goal is maximum throughput, not latency.** Time per token doesn't matter, so any trade that raises total tokens per second is worth making.

One change per row, each measured on the real chip with `scripts/batch_generate.py`.

**The benchmark:** `inf2.xlarge`, two full batches of distinct prompts (about 21 prompt tokens each), 32 new tokens per sequence, greedy. Row 0 runs the saved greedy prompt, and its output has to match, so a wrong graph can't look fast. From row 6, a saved 38-token passage also runs through every row first, and the top token must match the CPU's at each position.

**Generated tok/s** is the headline: tokens actually produced divided by wall time, including prompt steps and idle rows. **All tok/s** also counts prompt tokens processed. Nothing here is projected from a single step. Load time isn't included.

| # | Change | Batch | Chip s/step | Host s/step | Generated tok/s | All tok/s | Correct |
|---|---|---|---|---|---|---|---|
| 1 | Layout-optimized block, no batching | 1 | 3.75 | 0.13 | 0.19 | 0.26 | yes |
| 2 | Batching: all rows share one FP4 decode | 32 | 4.10 | 0.79 | 3.61 | 6.04 | yes |
| 3 | Faster decode: sign-free codes, integer mask instead of `floor`, no interleave | 32 | 2.25 | 0.81 | 5.78 | 9.66 | yes |
| 4 | No decode: FP8 weights read directly by a kernel (1.8x the weight memory, so fewer rows) | 16 | 0.21 | 0.68 | 10.43 | 17.29 | yes |
| 5 | Max sequence length 2048 cut to 512, so more rows fit | 32 | 0.24 | 0.80 | 17.03 | 28.49 | yes |
| 6 | `lm_head` on the chip, using 1.27 GB per core, so fewer rows | 20 | 0.18 | 0.04 | 51.48 | 85.74 | yes, 38 of 38 |
| 7 | Gated delta state in BF16, which halves the largest cost per row | 32 | 0.23 | 0.03 | 67.00 | 112.04 | yes, 38 of 38, see below |

Rows 1 to 4 allow 2048 tokens of prompt plus output, and from row 5 it's 512. Batch 24 at 2048 ran out of chip memory.

Chip s/step counts the 16 blocks, and the chip part of `lm_head` counts as host time. In row 6, with every row generating, the run reached 81.2 tok/s over 13 steps.

Row 7 changes numbers, not just speed. Compared with row 6's build on the same prompts, 36 of 40 sequences are identical, and 1219 of 1280 tokens come before a sequence's first difference. Greedy text can split at a near tie, and the two runs also differ in batch size. Whether output quality changes hasn't been measured. With every row generating, row 7 reached 110.0 tok/s over 14 steps.

Raw results are in [throughput/](throughput/). `git log docs/throughput.md` gives the commit for each row.

## Where the time goes

`neuron-profile` on one block at batch 32, before change 3 (0.268 s, core 0). Engines overlap, so these add to more than 100%:

| Engine | Busy |
|---|---|
| Vector | 74% |
| Scalar | 39% |
| GpSimd | 16% |
| Tensor (matmul) | 9.6% |
| DMA | 9.6% |
| Collectives | 1% |

To capture one, run with `NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=1 NEURON_RT_INSPECT_OUTPUT_DIR=dir`, then `neuron-profile view -n dir/*/neff_<id>.neff -s dir/*/<id>_vnc_0.ntff --output-format summary-text`. The larger NEFF is the model, and the smaller one is the layout transformer.
