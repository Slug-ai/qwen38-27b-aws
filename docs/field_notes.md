# Field notes: bringing this up on a real Inf2 instance

Written as we go, so the next instance is faster. Newest learnings are added to the section they belong to.

## Instance used

- `inf2.xlarge`, us-east-2, AMI "Deep Learning AMI Neuron (Amazon Linux 2023) 20260115".
- AMD EPYC 7R13, 4 vCPUs, 15 GB RAM, 200 GB disk, `/tmp` is a 7.7 GB tmpfs.
- `neuron-ls` shows 1 device, 2 cores, 32 GB.

## Logging in

- The login user on this AMI is `ec2-user`, not `ubuntu`. The README's install steps assume Ubuntu and don't apply here.
- SSH refuses connections for about 2 minutes after launch, then opens.
- A key pair's private key can only be downloaded once. AWS names the file after the key pair, so `keypair.pem` doesn't fit an instance launched with `pair1`.
- Lost key fix, no relaunch: connect with EC2 Instance Connect in the browser and append the public key to `~/.ssh/authorized_keys`. Get the public key with `ssh-keygen -y -f key.pem`. The browser session may log in as root, in which case the key only works for root.
- `ssh` needs `chmod 600` on the key.

## SDK versions

The AMI ships an older SDK than the one our graphs were compiled with:

| | AMI 20260115 | Pinned (2.28.1) |
|---|---|---|
| neuronx-cc | 2.22.12471.0 | 2.23.6484.0 |
| torch-neuronx | 2.9.0.2.11.19912 | 2.9.0.2.12.22436 |
| neuronx-distributed | 0.16.25997 | 0.17.26814 |
| runtime-lib | 2.29.40.0 | 2.30.51.0 |
| collectives | 2.29.41.0 | 2.30.59.0 |
| tools | 2.27.33.0 | 2.28.23.0 |
| dkms driver | 2.25.4.0 | 2.26.10.0 |

## Installing packages on Amazon Linux 2023

- The AMI locks package versions with the `versionlock` dnf plugin, and `dnf install` then says "Unable to find a match" even for versions `dnf list` shows. Remove the lock first:
  ```bash
  dnf versionlock delete 'aws-neuronx-collectives*' 'aws-neuronx-runtime-lib*' 'aws-neuronx-tools*' 'aws-neuronx-dkms*'
  dnf install -y aws-neuronx-runtime-lib-2.30.51.0_faafe26f0-1 aws-neuronx-collectives-2.30.59.0_f5cdefb39-1 aws-neuronx-tools-2.28.23.0_f1c114a9d-1
  ```
- Version names need the full suffix, such as `2.30.51.0_faafe26f0-1`.
- The Python 3.11 venv from `uv` also needs `libpython3.11` on `LD_LIBRARY_PATH`. The AMI's own venvs use Python 3.12.

## Python on Amazon Linux 2023

- torch-xla 2.9.0 is tagged `manylinux_2_28` but binds `hypot` and `hypotf` to GLIBC_2.35, and this AMI has glibc 2.34. `import torch_xla` fails with ``version `GLIBC_2.35' not found``, including in the AMI's own `/opt/aws_neuronx_venv_pytorch_2_9`.
- Fix: `python scripts/patch_xla_glibc.py .venv/lib/python3.12/site-packages/_XLAC*.so`. It also marks the version requirement weak, because from how the loader works `patchelf --clear-symbol-version` alone shouldn't be enough. That wasn't tried.
- First imports are slow (27 s here, and much longer for very large files) because the AMI's disk loads from its snapshot on first read.
- Environment that imports cleanly, using the pins in `docker/requirements-neuron.txt`:
  ```bash
  uv venv --python /usr/bin/python3.12 .venv
  uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pip.repos.neuron.amazonaws.com -r docker/requirements-neuron.txt
  ```
- The checkpoint downloads from Hugging Face at about 20 MB/s here, so 19 GB takes about 15 minutes. Start it first, in the background.

## First results on the chip

- `decode` passes: 512 of 512 values exact, 15 s including imports. Graphs compiled with the SDK 2.28.1 compiler load and run with the 2.28.1 runtime on the AMI's older 2.25.4 driver.
- `kernel-devel` for the AMI's running kernel isn't in the repos, so the driver wasn't upgraded. So far it hasn't mattered.
- `matrix` passes: error 0.00197, identical to the CPU's BF16 error.
- `layer` and `block` pass: errors 0.03% to 0.68% against a 2% limit, on 6 tokens.
- A process can't run one-core and two-core graphs together. The runtime sets its world size from the first graph it loads, and a later two-core graph fails with `NRT_INVALID in nrt_load()`, `World size of neff 2 is greater than world size of global communicator 1`. Run `decode` and `matrix` separately from `layer` onward.
- `CCOM WARN NET/OFI ... is EFA enabled?` is harmless on a single chip.
- `prepare_weights.py` took 3 minutes.
- All 16 blocks run end to end on `inf2.xlarge`. `model`, `greedy` and `speed` pass ([results](chip_first_run.json)): same top token as the CPU at 38 of 38 positions and the same 8 greedy tokens.
- Real generation: `What is the capital of France? Answer in one short sentence, then name one famous landmark there.` gave `The capital of France is Paris, and one famous landmark there is the Eiffel Tower.`
- Plain block speed: 4.13 s per token on the chip, 0.18 s on the host, 0.232 tokens/s. The pre-chip estimate of about 1.5 tokens/s was about 6 times too high.
- Host memory peaks at about 10.4 GB of 15 GB with everything loaded, so `inf2.xlarge` is enough.

## Throughput

Goal: tokens per second across all sequences. Latency per token doesn't matter.

- Layout-optimized block (`--optimize-layout`): 3.74 s per token on the chip, against 4.13 s plain.
- FP4 decode is the bottleneck: a real matrix decodes at 3.59 G values/s, and each core decodes 12.16 G values per token, so about 3.4 s of 3.74 s. Matrix math is very small in comparison.
- So batching comes first: all rows of a batch share one decode.
- Batch 32 compiled in 5.5 minutes on `utcs-life` with `--optimize-layout`, and the compiler's estimate is 15.65 GB per core. It loads in 114 s.
- Batch 32 on the chip: 4.10 s per step on the chip and 0.65 s on the host, so decode really is shared. On 32 short prompts with 32 new tokens each, including prompt steps and idle rows: 2.97 generated tokens/s. All repeated prompts gave the same output across rows. Only ever quote measured throughput, from `scripts/batch_generate.py`, not a figure projected from one step.
- Separating nibbles with shift and mask instead of `floor` raises decode from 3.59 to 5.97 G values/s on a real matrix (`scripts/bench_decode.py`). Avoiding the interleave alone barely helps: 3.70.
- Measured numbers live in [throughput.md](throughput.md). A short run undercounts batching if it's only one batch, because rows that finish early sit idle at the end.
- The compiler's memory total overstates what a run needs: batch 32 with the new decode is estimated at 17.0 GB per core, and all 16 blocks load and run.
- `packed >> 4` on uint8 compiles to an integer divide, not a shift.
- Testing tiers: `scripts/bench_decode.py` on one real matrix takes under a minute. `scripts/batch_generate.py --screen-blocks 1` times one real block at the real batch, and isn't a throughput result. The full `scripts/batch_generate.py` run gives the number that goes in [throughput.md](throughput.md).
- FP8 weights: the Tensor Engine reads FP8 directly next to BF16 inputs, so a small kernel skips decoding altogether. On one real matrix: 72.9 G values/s against 7.89. Whole-model chip time per step: 0.21 s at batch 16, against 2.25 s at batch 32 with decode. Weights take 12.2 GB per core against 6.9 GB.
- `.view(nl.float8_e5m2)` reads a uint8 tile as FP8 in a kernel. Use `neuronxcc.nki`, not the newer `nki` import, to match the SDK's own kernels.
- Instructions per block fell from about 545k to 58k, and program memory across all blocks from 1.2 GB to 0.1 GB.
- A sequence's text can differ slightly between builds, because rounding differs and greedy decoding follows near ties. The saved greedy row still matches.
- Real chip memory, from the table a failed load prints: a core fills at 15.94 GB. The scratchpad is shared across all loaded blocks (512 MB at batch 24), which is why the compiler's total runs high. Each loaded block also costs about 57 MB, mostly collective rings (39 MB) and code (9 MB), so about 0.9 GB across 16 blocks. Batch 24 at max length 2048 fails on the last blocks, and batch 32 at 512 fits.
- With the FP8 kernel, chip time barely grows with batch: 0.21 s per step at batch 16 and 0.24 s at batch 32.
- lm_head on the chip: 16 of 16 tokens match the host's, in 13.5 ms against 859 ms at batch 16, of which the chip graph is 8 ms. The chip returns the largest logit in each group of 32 tokens, and the host rechecks the top 4 groups from the checkpoint, so the host needs no 5 GB copy of lm_head.
- The FP8 kernel scales almost flat with batch on one matrix: 0.62 ms at batch 1 and 1.00 ms at batch 128. Memory is what limits batch.
- Memory limits at max length 512: batch 32 fits and batch 40 doesn't. With lm_head on the chip, batch 20 fits and batch 24 doesn't.
- `pkill` isn't always enough for a process holding the chip: a run can sit there holding both cores, and the next run fails with `Logical Neuron Core(s) not available`. Use `pkill -9`.
- With BF16 gated delta state, lm_head on the chip and max length 512, batch 32 fits and batch 36 doesn't: it fails allocating collective rings.
- The Tensor Engine runs the FP8 kernel in BF16. The compiler adds a cast from FP8 to BF16 on the Vector Engine for every weight tile (`src=fp8e5 dst=bfloat16` in the profile), so FP8 weights are cheap to widen rather than free. Moving those casts to other engines doesn't speed it up, so something else also limits it, possibly DMA.
- `neuron-profile view --output-format json` gives per-instruction data. `opcode` and `operands` on each instruction show data types, which is how the cast was found. The first all-reduce in a block takes about 2 ms and the rest about 0.05 ms, for about 32 ms per step. The cause isn't known.
- 16-layer blocks compile in about 4 minutes with FP8 and load in about 190 s. Whole-model graphs aren't practical on `inf2.xlarge`, because both cores' weights are in host RAM together while loading: about 24 GB for one graph against 15 GB of RAM.
- Don't write a queue script that only waits for a success message, because a job that dies leaves it waiting forever. Check the job is still running too. An out-of-memory kill during weight preparation idled the chip for about 30 minutes this way. `dmesg | grep -i oom` shows these.
- FP8 lm_head: rounding moves weights by about 4.5% on average. On 900 proxy vectors on the CPU, the true token's group was always in the top 4 groups, though the best group alone missed 37 times.
- Time host work on the instance, not a laptop. Combining the host's recheck into one `bmm` with a bit-shift widen was 6 times faster on an M4 Pro and 7 times slower on the EPYC host: 83 ms against 11.5 ms. PyTorch uses 2 threads on `inf2.xlarge`. Most of the "host" time in the throughput log is really the lm_head graph on the chip.
- `NEURON_SCRATCHPAD_PAGE_SIZE=128` (in MB) cuts the shared scratchpad from 512 MB to 256 MB per core. `scripts/batch_generate.py` now sets it.
- Memory limits with FP8 lm_head and 16-layer blocks: batch 56 fits at the default scratchpad and batch 60 misses by about 30 MB. At 128 MB pages, batch 60 fits and batch 64 doesn't.
- A fused kernel for the MLP (gate, up, SiLU and down, with activations staying on the chip) gives almost nothing: 1.64 ms against 1.66 ms on one real MLP at batch 56. Separate kernels are already close to the DMA limit. The time left is in gated delta and attention, at about 3.4 ms per layer each, of which only about 0.9 ms is reading weights.
- Reading weights looks DMA-limited at about 105 GB/s per core: casts wait on the DMA queues most of the time.
- Rewrites of the layers that didn't help, on one real layer at batch 56: gated delta state contraction as a matmul (3.75 ms in float32 and 3.52 ms in BF16, against 3.39 ms), and attention without repeating the KV cache (2.94 ms against 2.96 ms). Gated delta state is about 46 MB per layer per core at batch 60, read and written through the same DMA limit, so a hand-written kernel has little room. A mega kernel doesn't look worthwhile from here.
- Chip time per step fits about 0.118 s + 0.0031 s per row, so more rows keep helping until memory runs out.
- 32-layer blocks aren't practical on `inf2.xlarge`. Writing one weight file holds about 12 GB in host RAM and was killed for running out of memory, and loading would need both cores' 6 GB in RAM together, all for about a 5% gain. Three 32-layer compiles at once also tripped the memory guard on `utcs-life`.
- Host time per step swings between 0.03 s and 0.07 s with no code change across runs. Chip time also moves by several percent between runs, so compare builds on more than one run when the difference is small.
- Host time swings came from a cold disk cache. Loading the weights pushes the checkpoint's lm_head and embedding rows out of the cache, and each step then waits on the disk for scattered rows. `TensorSource.touch` reads them once after loading, in about 32 s, which holds host time at 0.026 s per step.
- Chip time doesn't always rise with batch, because the compiler's plan depends on the batch size. Batch 62 takes 0.287 s per step and batch 60 takes 0.304 s, in two runs each.
- Chip time per step at nearby batch sizes: 0.304 s at 60, 0.287 s at 62, 0.262 s at 63. The compiler's plan varies, so it's worth trying neighbouring sizes.
- Don't trust `--screen-blocks` for comparing builds: the same batch 63 build timed 0.065 s per block in one run and 0.114 s in another, while full benchmarks repeat to about 0.1%. An earlier conclusion from it, that layout optimization is worth almost 2x, was wrong.
- Batch 64 fits at max length 480, but its plan is slower: 0.313 s on the chip per step and 106.31 generated tok/s, against 0.262 s and 122.67 at batch 63 and max length 512.
- Full runs at batch 63: 122.67 generated tok/s with layout optimization, 122.65 with `-O2` as well, and 122.82 without layout optimization, so neither matters now that kernels read the weights. Layout optimization gave about 10% at batch 1, when weights were decoded in the graph.
- Full runs at nearby batches: 108.13 at 58, 110.38 at 59, 112.12 at 61, against 122.67 at 63. Batch 63 is a sweet spot in the compiler's plan. It's also why its token comparison differs a little from its neighbours': 31 of 40 sequences against 33.
- A separate prompt path isn't worth building. A prototype gated delta layer taking 4 tokens per row per call, sharing one weight read, took 8.69 ms against 2.24 ms for one token at batch 32, in two runs each. That saves about 3% per token, because the state still updates token by token and that's most of the layer's cost. MLPs would save more, but 48 of the 64 layers are gated delta, so prompt tokens would get maybe 20% cheaper before the extra graphs' memory cost.
- A fresh `inf2.xlarge` on 2026-11-05, set up from the README with the graphs from the `graphs-b63` release, gave 114.44 generated tok/s on its first full run, against 122.67 recorded. Chip time matched, 0.261 s per step against 0.262, and host time was 0.042 s per step against 0.022, so the gap is on the host. Load took 190 s and reading lm_head and the embeddings 38 s. Correctness passed: row 0 matched and 38 of 38 saved-passage positions were right. A later run on the same instance, with `scripts/benchmark.py`, gave 122.21 generated tok/s with row 0 matching, so the first run's gap didn't repeat.
- What limits streaming FP8 weights, from `nki.benchmark` on one 5120 x 8704 matrix at batch 64 on one core: the repo's kernel (BF16 activations, FP8 weights) ran at 95 G weights/s. BF16 weights in the same kernel also ran at 95 G weights/s, which is 192 GB/s, so HBM isn't the limit. The Tensor Engine alone, on weights already in SBUF, runs at 217 G weights/s for FP8 x FP8 or BF16 x BF16, but only 131 for BF16 x FP8, because the compiler widens every weight tile on the Vector Engine first. With weights as the stationary operand it reaches 424 G weights/s at batch 64. DMA reads from HBM reach 233 GB/s with 2 KB per partition per transfer. FP8 x FP8 with 2 KB loads ran the matrix in 238 us, against 478 us.
- NKI on NeuronCore-v2 has no mixed FP8 matmul: e4m3 x e5m2 fails to compile, so activations must be cast to the weights' e5m2.
- `neuronx-cc` 2.23 rejects FP8 types in XLA graphs for inf2 (`NCC_ESPP047: Data type F8E4M3FN is not supported`). FP8 is only usable inside NKI kernels, as uint8 tensors viewed as FP8 there.
- `neuron-profile view --output-format json` on a whole 16-layer block used all 15 GB of RAM, and the instance stopped answering SSH until it was rebooted. Use `summary-text` for whole blocks.
- One 16-layer block at batch 63 now compiles in about 110 s on `inf2.xlarge` itself, so compiling there works.

## Long context and prefill

- A KV cache write the compiler can't do in place costs a full copy of the cache per layer, as statically allocated tensors: 512 MB per 16-layer block at 2 rows and 32K positions, which ran the chip out of memory at load. `index_copy` and `scatter` both copied when attention read the scattered cache. The fix is NxD Inference's pattern: attend to the cache as it was plus the step's own key and value, and write the cache only at the end, where nothing reads the result. Allocated tensors in a 4-layer decode graph fell from 130 MB to 5 MB.
- Scatter writes should cover whole rows. With keys cached transposed (positions last), every element is its own write index. With positions in the middle, each index writes a 512-byte key.
- Rows that aren't being written must not all scatter to the same junk position: duplicate indices serialize. Prefill gives other rows the last `chunk` positions, distinct, so a build with a prefill graph serves `max_seq_len - chunk` positions.
- XLA attention over 32K positions keeps every score block live: a 4-layer prefill graph allocated 1.39 GB per attention layer. An NKI flash attention kernel keeps scores in SBUF and PSUM: 0.19 GB.
- The SDK's `flash_fwd` supports head dims up to 128 only, and this model's is 256, so `prefill_attention_body` is custom. At 32K positions it takes 8.7 ms per layer for a 512-token chunk reading pre-transposed keys, or 10.4 ms reading the slot's row by indirect DMA and transposing key blocks on the Tensor Engine, which avoids copying the row out.
- NKI loops: Python `range` in a kernel becomes a device loop, so a Python list can't be indexed by its variable. `nl.sequential_range` keeps code size flat as the cache grows: the attention kernel at 32K compiled and ran in 42 s instead of many minutes. There is no loop with a runtime trip count in this SDK, so cache blocks past a chunk's start can't be skipped.
- The NKI CPU simulator (`nki.simulate_kernel`) gives wrong results for PSUM accumulation inside `nl.affine_range` loops: an FP8 matmul that is exact on the chip came back half zeros. Check kernels on the chip with `nki.baremetal`.
- The 512-token FP8 matmul with device loops runs at 56 TFLOPS on real projection shapes, the Tensor Engine's peak, and compiles 3 times faster than the unrolled form.
- The chunked gated delta rule's triangular solve must not use a doubling series over the whole piece, `(I + L)(I + L^2)(I + L^4)...`: with strongly correlated keys its powers grow combinatorially and cancel, giving errors of 1e9 on CPU and NaN on real weights on the chip. Blocked forward substitution, 8x8 blocks by doubling and then merges, is exact to 1e-7 in the same cases. flash-linear-attention does the same.
- The chunked gated delta rule as XLA ops took about 19 ms per layer per 512 tokens. As an NKI kernel with the state kept in SBUF it takes 1.33 ms.
- NxD's `ModelBuilder` compiles a model's tags in parallel by default. Two large graphs at once ran the 15 GB instance out of memory, so `compile_graph` passes `max_workers=1`.
- Compiling the 512-token prefill graph for 16 layers took 20 to 30 minutes on `inf2.xlarge` while the chunked rule and attention were XLA ops, and about 5 minutes with them as kernels. The decode graph takes about 4.
- Never pass an NKI kernel two views of the same tensor, such as keys and their transpose. Inside the full layer the compiler handed one of them over in a layout the kernel didn't expect, and the gated delta state was corrupted from the second 128-token piece on, while every standalone test passed. The kernel now takes keys in one layout and transposes them itself.
- To find a bug like that, compare one real layer on the chip with the CPU on the same weights and input, then return intermediate values from a debug copy of the layer, stage by stage.
- NeuronCore-v2's `float8_e4m3` is the IEEE-style variant: the same bytes as OCP e4m3fn up to 240, its largest finite value, but `0x78` and above are infinity. 250, 300 and 448 all encode to `0x78`. Checkpoint scales calibrated for e4m3fn's 448 must be stretched by 448 / 240 on this chip, which keeps relative precision and loses nothing to overflow.
