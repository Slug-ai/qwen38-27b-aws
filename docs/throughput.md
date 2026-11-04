# Throughput log

**The goal is maximum throughput, not latency.** Time per token doesn't matter, so any trade that raises total tokens per second is worth making.

One change per row, each measured on the real chip with `scripts/batch_generate.py`.

**The benchmark:** `inf2.xlarge`, two full batches of distinct prompts (about 21 prompt tokens each), 32 new tokens per sequence, greedy. Row 0 runs the saved greedy prompt, and its output has to match, so a wrong graph can't look fast.

**Generated tok/s** is the headline: tokens actually produced divided by wall time, including prompt steps and idle rows. **All tok/s** also counts prompt tokens processed. Nothing here is projected from a single step. Load time isn't included.

| # | Change | Batch | Chip s/step | Host s/step | Generated tok/s | All tok/s | Correct |
|---|---|---|---|---|---|---|---|
| 1 | Layout-optimized block, no batching | 1 | 3.75 | 0.13 | 0.19 | 0.26 | yes |
| 2 | Batching: all rows share one FP4 decode | 32 | 4.10 | 0.79 | 3.61 | 6.04 | yes |
| 3 | Faster decode: sign-free codes, integer mask instead of `floor`, no interleave | 32 | 2.25 | 0.81 | 5.78 | 9.66 | yes |

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
