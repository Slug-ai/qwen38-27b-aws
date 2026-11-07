# Throughput log

![Throughput by change](throughput.png)

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
| 8 | 16-layer blocks, so 4 loaded graphs instead of 16, which frees memory for rows | 40 | 0.24 | 0.05 | 74.29 | 124.58 | yes, 38 of 38 |
| 9 | Largest batch that fits this build. Batch 48 runs out of memory | 44 | 0.26 | 0.06 | 77.42 | 129.82 | yes, 38 of 38, see below |
| 10 | `lm_head` stored as FP8, which halves it, with the host rechecking the top 4 groups exactly | 56 | 0.29 | 0.07 | 88.72 | 148.92 | yes, 38 of 38, see below |
| 11 | Scratchpad pages of 128 MB instead of the runtime's default, which frees 256 MB per core | 60 | 0.30 | 0.03 | 99.64 | 167.39 | yes, 38 of 38, see below |
| 12 | Scratchpad pages of 64 MB, which fits two more rows. Not an improvement, because host time doubled | 62 | 0.29 | 0.06 | 97.29 | 163.49 | yes, 38 of 38 |
| 13 | Read lm_head and the embeddings into the disk cache after loading | 62 | 0.29 | 0.03 | 108.32 | 182.03 | yes, 38 of 38 |
| 14 | One more row, and the compiler's plan for this batch size is faster | 63 | 0.26 | 0.03 | 120.65 | 202.78 | yes, 38 of 38, see below |
| 15 | Build each row's inputs in one vectorized call | 63 | 0.26 | 0.02 | 122.67 | 206.18 | yes, 38 of 38, see below |
| 16 | Matmul inputs rounded to FP8 e5m2, so the Tensor Engine multiplies FP8 by FP8 with no weight cast, and weights read 2048 columns at a time | 63 | 0.20 | 0.02 | 156.20 | 262.54 | yes, 37 of 38 and 1 near tie, see below |

Rows 1 to 4 allow 2048 tokens of prompt plus output, and from row 5 it's 512. Batch 24 at 2048 ran out of chip memory.

Chip s/step counts the 16 blocks, and the chip part of `lm_head` counts as host time. In row 6, with every row generating, the run reached 81.2 tok/s over 13 steps.

Row 7 changes numbers, not just speed. Compared with row 6's build on the same prompts, 36 of 40 sequences are identical, and 1219 of 1280 tokens come before a sequence's first difference. Row 8 gives exactly the same comparison, but row 9 falls to 33 of 40 sequences and 1178 tokens, so batch size affects results as well as the BF16 state. Row 10 gives the same comparison as row 9, so FP8 lm_head with the host's recheck changed no token choices in those sequences. Greedy text can split at a near tie. On quality, `scripts/quality_check.py` runs 324 tokens of well-known English text through a build: BF16 state and float32 state both get the real next token at 315 of 323 positions (97.5%), and pick the same token at 323 of 324. Famous text is an easy test, so this rules out large damage rather than subtle damage. With every row generating, row 7 reached 110.0 tok/s over 14 steps.

Row 11's host time fell with no host code change, probably because the checkpoint's lm_head rows were already in the disk cache from earlier runs, so part of its gain isn't from the scratchpad. A long-running server would also see the warm number. With every row generating, row 11 reached 172.1 tok/s over 12 steps.

Row 13 reached 192.1 tok/s with every row generating, over 13 steps. A run at batch 60 with the same warm cache gave 101.01 generated tok/s and 0.30 s on the chip, so batch 62 really is faster per step there.

Row 15 reached 214.5 tok/s with every row generating, over 19 steps. At batch 63, the token comparison with the float32-state build is 31 of 40 identical sequences and 1135 of 1280 tokens before a first difference. The quality check there still gives 315 of 323, and the same top token as the float32-state build at all 324 positions.

Row 16 changes numbers: every matmul's input is rounded to FP8 e5m2, 2 mantissa bits with an exponent per value. The checkpoint was calibrated for MXFP4 activations, 1 mantissa bit with one exponent per 32 values, so this is still finer than its intended serving, and on a tiny model it gives under half the logit error of MXFP4 activations (`tests/test_model_parity.py`). The quality check gives 316 of 323 against 315 for row 15's build, with the same top token at 322 of 324 positions. Greedy text drifts more, because each near tie can flip: 30 of 126 sequences are identical to row 15's, and 2110 of 4032 tokens come before a first difference. Row 15's build, rerun in the same session on this instance, gave 123.23 generated tok/s, so the gain is 27%. With every row generating, row 16 reached 269.3 tok/s over 19 steps.

Raw results are in [throughput/](throughput/). `git log docs/throughput.md` gives the commit for each row.

## Agentic workload

The table above measures 21-token prompts with prompt plus output capped at 512 tokens, which no coding agent's request fits. From here the headline is `scripts/agentic_benchmark.py`: requests shaped like Claude Code turns, a system prompt with tools, a task, standard library files read back as tool results and a question about them. By default 16 requests from 8K to 30K prompt tokens run, plus the saved greedy prompt, with up to 256 new tokens each, greedy. It reports input, output and total tokens per second, all measured over the whole run. Two checks must pass: the saved greedy prompt gives its saved tokens, and the longest request recalls a key planted in its first file.

| # | Change | Rows x max length | Workload | Input tok/s | Output tok/s | Total tok/s | Correct |
|---|---|---|---|---|---|---|---|
| A1 | Prompts fed one token per decode step, as in row 16 | 2 x 32768 | 2 prompts of 4K | 24.03 | 0.41 | 24.45 | yes |
| A2 | Prefill graph: 512 prompt tokens of one row per call, with NKI kernels for the chunked gated delta rule and for flash attention over the cache | 2 x 32768 | 2 prompts of 4K | 400.65 | 6.90 | 407.55 | yes |
| A2 | Same build | 2 x 32768 | 16 prompts of 8K to 30K | 568.58 | 2.42 | 571.00 | yes, 114 requests per hour |
| A3 | Gated delta layers' conv, norms and gating fused into the prefill kernel | 2 x 32768 | 16 prompts of 8K to 30K | 711.59 | 2.54 | 714.12 | yes, 143 requests per hour |
| A3 | Same build | 2 x 32768 | 2 prompts of 4K | 458.68 | 6.54 | 465.22 | yes |
| A4 | The attention kernel takes 2048 cache positions per loop iteration instead of 512 | 2 x 32768 | 16 prompts of 8K to 30K | 781.05 | 3.87 | 784.93 | yes, 157 requests per hour |
| A4 | Same build | 2 x 32768 | 2 prompts of 4K | 498.07 | 8.57 | 506.64 | yes |
| A5 | The attention kernel's loop over cache blocks runs only as far as the chunk's start, a trip count read at run time | 2 x 32768 | 16 prompts of 8K to 30K | 879.80 | 4.36 | 884.10 | yes, 177 requests per hour |
| A5 | Same build | 2 x 32768 | 2 prompts of 4K | 570.50 | 9.82 | 580.32 | yes |
| A6 | The gated delta prefill kernel emits 4 value heads' dependent steps in turn | 2 x 32768 | 16 prompts of 8K to 30K | 902.10 | 4.48 | 906.58 | yes, 181 requests per hour |
| A6 | Same build | 2 x 32768 | 2 prompts of 4K | 577.82 | 9.94 | 587.76 | yes |
| A7 | The same kernel's conv, SiLU and norms emit 4 channel tiles' steps in turn | 2 x 32768 | 16 prompts of 8K to 30K | 907.93 | 4.50 | 912.43 | yes, 182 requests per hour |
| A7 | Same build | 2 x 32768 | 2 prompts of 4K | 586.79 | 10.10 | 596.89 | yes |
| A8 | KV cache in FP8 e4m3 with the checkpoint's scales, read by NKI attention kernels, so 4 rows fit where 2 did | 4 x 32768 | 16 prompts of 8K to 30K | 983.62 | 3.68 | 987.30 | yes, 197 requests per hour |
| A8 | Same build | 4 x 32768 | 2 prompts of 4K | 609.82 | 10.49 | 620.32 | yes |
| A9 | The prefill attention kernel emits 8 query tiles' steps in turn | 4 x 32768 | 16 prompts of 8K to 30K | 1003.86 | 3.75 | 1007.61 | yes, 201 requests per hour |
| A9 | Same build | 4 x 32768 | 2 prompts of 4K | 613.27 | 10.55 | 623.83 | yes |
| A10 | The MLP's gate, up and SiLU in one kernel whose output is the down projection's input layout | 4 x 32768 | 16 prompts of 8K to 30K | 1036.91 | 3.65 | 1040.56 | golden row yes; planted key no, see below; recall check as A9's |
| A10 | Same build | 4 x 32768 | 2 prompts of 4K | 623.67 | 10.73 | 634.41 | yes |
| A11 | The gated delta kernel's matmuls take tfloat32 inputs, one pass on the Tensor Engine, instead of float32's four | 4 x 32768 | 16 prompts of 8K to 30K | 953.12 | 6.10 | 959.22 | golden row yes; planted key no; recall check as A9's, see below |
| A11 | Same build | 4 x 32768 | 2 prompts of 4K | 630.50 | 10.85 | 641.36 | yes |

A1 and A2 are the same build, A1 run with `--no-prefill`. A1 on the full workload would take about 3.5 hours, so both ran the small one. Only 2 rows fit: a BF16 KV cache of 32K positions takes 1.07 GB per row on each core. The FP8 cache the checkpoint was calibrated for would fit about twice as many, but the compiler accepts FP8 only inside NKI kernels.

In A2 the full workload made 604 prefill calls and 760 decode steps in 536 s, so a 512-token chunk took about 0.79 s across the 4 blocks. One layer's prefill, timed alone on one core: gated delta 7.8 ms (21.9 ms as XLA ops), attention against a 32K cache 17.6 ms (36.8 ms with XLA scatters and a copy of the row), MLP 3.7 ms. The quality check gives 311 of 323 through decode and 313 of 323 through prefill, and the two paths pick the same top token at 319 of 324 positions.

In A3 everything in a gated delta layer's prefill between its input and output projections is one NKI kernel, so the causal conv, SiLU, query and key norms and the gated RMS norm no longer make round trips through HBM as XLA ops. One layer's prefill fell from 7.76 ms to 4.57 ms for 512 tokens. The quality check gives 315 of 323 through prefill, and 311 through decode.

In A4 the prefill attention kernel waits for each block's loads before computing it, so fewer, larger blocks hide more of that latency: one layer against a 32K cache takes 8.2 ms with 2048-position blocks, against 10.4 ms with 512 and 9.0 ms with 4096. The quality check is unchanged, 315 of 323 through prefill.

In A5 a chunk's attention visits only the cache blocks before its start: `range(scalar(n))` in NKI, with the SDK's `experimental-native-scalar-support` flag and middle-end transformations skipped, gives a device loop whose trip count is read from a tensor at run time. Before, every chunk scanned all 32K positions, so the first chunk of a prompt cost as much attention as the last. Alone on one core, the kernel takes 0.21 ms plus 0.59 ms per 2048 cached positions, against 8.2 ms for the full scan, so a 2048-position block costs about 19% more than in the static loop, and the full 32K scan would be 9.7 ms. On this workload a chunk visits about a third of the cache on average. The same 604 prefill calls and 797 decode steps took 346.6 s against 390.5 s. The quality check is unchanged, 315 of 323 through prefill and 311 through decode.

In A6 the gated delta prefill kernel interleaves value heads. Each head's piece is a chain of about a dozen dependent steps, matmul then copy out of PSUM, and every engine takes its instructions in order, so one head at a time left each engine waiting on the others: the kernel's profile had the Tensor Engine busy 25% of the time, the Vector Engine 29% and the Scalar Engine 18%. Now each head's work is a Python generator that yields between dependent steps, and 4 heads' steps are emitted in turn. The kernel alone fell from 2.10 ms to 1.66 ms per layer, with bit-identical outputs; groups of 2, 6 and 8 heads gave 1.75, 1.68 and 1.77 ms. The quality check is unchanged, 315 of 323 through prefill.

A7 does the same for the kernel's first stage, the conv, SiLU and norms over 40 tiles of 128 channels: the kernel alone fell from 1.66 ms to 1.60 ms, again bit-identical, and the full workload took 335.9 s against 338.1 s.

In A8 the KV cache is FP8 e4m3, the format the checkpoint was calibrated for, with its static per-layer scales, at half the memory, so 4 rows of 32K positions fit where 2 did. The compiler takes FP8 only inside NKI kernels, so the cache holds the bytes as uint8: a small kernel encodes each step's keys and values, both attention kernels read the bytes, and decode attention moved into a kernel too, since XLA can't read them. The chip's e4m3 stops at 240 where the checkpoint's e4m3fn stops at 448, so the scales are stretched by 448 / 240. The decode kernel stacks every row's query heads on partitions, loops over the cache with a run-time trip count up to the furthest row, and takes 2048 positions per iteration in two halves so one half's loads overlap the other's softmax: 1.95 ms per layer for 4 rows at 20K positions, alone on one core. The quality check gives 314 of 323 through prefill and 313 through decode, against 315 and 311 with the BF16 cache.

The FP8 cache changes where some greedy answers stop: 1140 tokens were generated against 1513, mostly because two requests that ran to 256 and 232 tokens with the BF16 cache stopped at 27 and 29. So part of A8's gain is less decoding. Prefill took 261.2 s over the same 604 calls, against 261.6 s for A7's build rerun with the same timing ([its JSON](throughput/agentic/07_interleaved_conv_full_rerun.json), 916.2 tok/s), and decode took 48.9 s over 417 steps of 2.69 rows on average against 72.9 s over 797 steps of 1.88 rows. At A8's 42.9 ms of decode per generated token, A7's 1513 tokens would have taken about 65 s, which puts A8 at about 940 tok/s like for like, 3% over A7. Steps carry 2.69 rows rather than 4 because the scheduler prefills each new request alone, and the last long answers run with few rows.

In A9 the prefill attention kernel takes a block of keys for 8 query tiles at a time, stage by stage: scores, the softmax update, the transposes of the weights, then the weighted values, each for all 8 before the next. Alone on one core, 5 blocks of the cache fell from 2.98 ms to 2.30 ms per layer, with groups of 4 and 6 at 2.40 ms. Only the order of instructions changed, and every request's output is the same as in A8. Prefill took 255.0 s against 261.2 s.

In A10 the MLP's gate and up projections run in one kernel, with weight tiles stationary and the tokens moving, which costs the Tensor Engine the same, and the SiLU and product are applied on the chip with each step rounded to BF16 as the separate ops round. Its output is already the down projection's input layout, so a transpose and two round trips of the 8704-wide intermediate through HBM go. Prefill took 247.2 s against 255.0 s. The 30K request answered with a tool call, reading another file, instead of the planted key: from A10 on, rounding changes are judged by the recall check below rather than by that one greedy answer, as agreed, and it finds recall unchanged.

A11's total is lower than A10's only because its greedy answers ran longer: 1951 tokens generated against 1229, four requests reaching the 256-token cap, so decode took 78.1 s over 632 steps. On the same prompts prefill took 241.9 s against 247.2 s, 2.1% less, and decode 40.0 ms per generated token against 42.2. The kernel's inputs keep 10 mantissa bits instead of 23, still more than the BF16 the recurrent state is stored in between chunks.

### How sure the planted key is

The agentic benchmark's planted key is one greedy answer, and a change that only rounds differently can flip it. A gate, up and SiLU kernel did, both ways it rounded, though on one real layer its outputs differ from the separate ops' by 0.6% RMS, the noise of the down projection's FP8 input: on the 30K request the current build's first token is the key's, logit 22.24 against 20.96 for `<tool_call>`, and with that kernel `<tool_call>` leads, 22.25 against 21.38.

`scripts/long_context_check.py` measures recall more steadily. It plants a different key in each of 8 prompts of 8K to 30K tokens, in the first file read or about halfway, asks for it, and scores each of the key's tokens fed back one at a time by its exact log-probability over the whole vocabulary:

| Build | Mean log p(key) | Key's first token on top | Mean p(first token) | Mean log p(rest of key) |
|---|---|---|---|---|
| A9 | -0.954 | 5 of 8 | 0.414 | -0.0135 |
| A10 | -1.202 | 3 of 8 | 0.335 | -0.0132 |
| A11 | -1.017 | 5 of 8 | 0.399 | -0.0079 |

Recall itself is the same: once the key starts, both finish it with probability 0.987. Nearly all the uncertainty is in the first token, whether to answer or to read another file first, at a probability of 0.25 to 0.6 per request, and it moves by up to 0.25 per request between the two builds in both directions, a mean difference of about 1.4 standard errors. Raw results are in [throughput/long_context/](throughput/long_context/).

From here builds are also compared with every request generating exactly 64 tokens (`--ignore-eos --max-new-tokens 64`), so two builds do the same work wherever their greedy answers would stop: the build after A11, with saved state rows and no other change, gives 1094.0 tok/s that way, prefill 242.5 s and decode 37.2 s over 315 steps of 3.40 rows.

Output tokens are few in this workload: requests stop at the end of the answer, about 64 tokens each, and only 2 rows fit at 32K positions, so most of the time is prefill.

Raw results are in [throughput/agentic/](throughput/agentic/).

## Multi-turn agentic workload

A coding agent sends its whole conversation again every turn, so most of each prompt is the last one. `scripts/multi_turn_benchmark.py` runs 8 such sessions of up to 12 turns and 30K tokens: turn 1 is the system prompt and task, and each later turn adds the agent's read of the next standard library file and the file itself. The reads are scripted, so every build sees the same prompts, while the model's own replies, up to 128 tokens, are generated and counted. Session 0 plants a key in its first file, and its last turn asks for it.

A session's turns run in one slot. After each prompt's last whole chunk, the slot's recurrent and conv states are saved, into spare rows of the same state buffers; the next turn restores them and prefills from that boundary, since the KV cache still holds those positions. The saved rows cost 151 MB per core and leave single-turn prefill as it was: the same build ran the single-turn agentic benchmark's prefill in 242.6 s against 241.9 s for A11, with the same outputs.

69% of the prompt tokens sent are reused, and the run takes 2.49 times less time. Because a restored state is exactly what the shared chunks gave, every turn generates the same tokens either way.

| # | Change | Rows x max length | Workload | Prompt tokens sent | Prefilled | Generated | Time | Total tok/s | Correct |
|---|---|---|---|---|---|---|---|---|---|
| M1 | Every turn prefilled whole | 4 x 32768 | 8 sessions, 53 turns | 654969 | 654969 | 1657 | 573.7 s | 1144.6 | key recalled |
| M1 | Turns continue from the slot's saved states | 4 x 32768 | 8 sessions, 53 turns | 654969 | 204409 | 1657 | 230.3 s | 2851.1 | key recalled; every turn's tokens the same as without |

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
