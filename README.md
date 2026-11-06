# qwen38-27b-aws

Runs the text part of [RedHatAI/Qwen3.8-27B-MXFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4) on one AWS Inferentia2 chip (two NeuronCores), with its FP4 weights converted to FP8 and greedy decoding.

**Status: runs end to end on a real `inf2.xlarge`.** Currently 156.20 generated tokens/s at batch 63 with a maximum of 512 tokens per sequence, measured. See [the throughput log](docs/throughput.md) and [the field notes](docs/field_notes.md), which cover setup on the Amazon Linux Neuron AMI.

**Goal: maximum throughput, not latency.** Every design choice is judged by total tokens per second across all sequences. Time per token doesn't matter, so large batches and slow steps are fine.

## How it works

This describes the build measured in [the throughput log](docs/throughput.md), which also gives the effect of each change.

- **FP8 weights on a chip without FP4.** NeuronCore-v2 has no FP4 type (native MX compute is Trn3 only), and the weights would take about 22.7 GiB per core at BF16. So they are converted once, when prepared, to one FP8 byte per value. Two NKI kernels ([kernels.py](qwen38_inf2/kernels.py)) have the Tensor Engine read those bytes directly. The matmul kernel rounds its input to FP8 e5m2, because NeuronCore-v2 multiplies FP8 only by FP8, so no decode or weight cast runs on the chip. FP8 holds all but 0.0003% of scale groups exactly, and the rest are weights under 1e-4 that round to the nearest FP8 value.
- **Memory limits the batch.** Chip time per step fits about 0.118 s + 0.0031 s per row, so more rows keep helping until memory runs out. A core fills at 15.94 GiB, which the runtime's memory table labels GB:

  | | Per NeuronCore, at batch 62 |
  |---|---|
  | FP8 weights | 11.35 GiB |
  | FP8 `lm_head` | 0.59 GiB |
  | Gated delta state, BF16 | 2.26 GiB |
  | KV cache, 512 tokens | 0.97 GiB |
  | Code, collective rings and scratchpad | about 0.6 GiB |
  | Total | about 15.8 GiB |

  The code and rings row comes from the runtime's table at batch 64, which doesn't fit. The other rows come from file sizes and tensor shapes.

- **Text only.** The vision tower, MTP head and KV-cache FP8 scales are never read.
- **Two cores.** Each core owns a slice of the attention heads. Only the output projections need a collective, so there are two all-reduces per layer. Each core also holds half of `lm_head`'s tokens, and one more all-reduce joins the two halves of its output.
- **One graph, one token per row.** Each row of the fixed batch is an independent sequence at its own position. A prompt is fed token by token through the same graph, in the recurrent form of the gated delta rule, so there is no prefill graph or chunked solve. A row that finishes takes the next waiting prompt. Prompt plus output is limited to 512 tokens, because a shorter KV cache leaves room for more rows.
- **16 layers at a time, loaded 4 times.** The model is 16 repeats of the same 4 layers (3 gated delta, 1 full attention). We compile 16 layers once and load that graph 4 times, each with its own weights and caches. Each loaded graph costs chip memory for code and collective rings, so 4 graphs leave more room for rows than 16 did. Larger graphs aren't practical on `inf2.xlarge`. Both cores' weights sit in host RAM together while a graph loads, which is about 24 GB for a whole-model graph against 15 GB of RAM, and preparing 32-layer weights was killed for running out of memory.
- **`lm_head` on the chip, checked on the host.** A kernel returns the largest logit in each group of 32 tokens, using an FP8 copy of `lm_head`. The host then reads the top 4 groups from the checkpoint and picks the exact token, so it needs no 5 GB copy of `lm_head`, and the FP8 copy only has to keep the right group in the top 4.
- **Host side.** Embeddings and the greedy loop run on the host, reading embeddings from the mapped checkpoint. `scripts/batch_generate.py` reads `lm_head` and the embeddings once after loading, in about 32 s, because otherwise each step waits on the disk. It also sets 64 MB scratchpad pages, which reserve less chip memory than the runtime's default.
- **SDK pinned to Neuron 2.28.1.** From 2.29 on, NxD Inference and NKI 0.3 kernels dropped Inf2. This repo uses NxD, the compiler and the SDK's `neuronxcc.nki`, not NxD Inference.

Two earlier designs are still available. `--weight-format fp4`, the default in `scripts/compile.py`, keeps weights packed at 6.4 GiB per core and decodes them in the graph every step. It took 2.25 s per step on the chip at batch 32, against 0.21 s at batch 16 with FP8. Leaving out `--head` runs the final norm and `lm_head` on the host, which holds a 5 GB copy in host RAM and took 0.80 s per step at batch 32, against 0.04 s at batch 20 with `--head`.

### Differences from the checkpoint's intended serving

Weights are FP8, converted from the checkpoint's FP4. Matmul inputs are rounded to FP8 e5m2 and the KV cache runs at BF16, while the checkpoint was calibrated for MXFP4 activations and an FP8 cache. e5m2 keeps 2 mantissa bits and an exponent per value, against 1 mantissa bit and one exponent per 32 values for MXFP4, so it is the finer of the two. Gated delta state also runs at BF16, where the reference uses float32.

These choices change some outputs. Against an earlier float32-state build at batch 20, 33 of 40 sequences are identical, and part of that difference comes from batch size. On 324 tokens of well-known English text, BF16 and float32 state both get the real next token at 315 of 323 positions and pick the same token at 323 of 324 (`scripts/quality_check.py`). Famous text is an easy test, so this rules out large damage but not subtle damage. With FP8 matmul inputs the same check gives 316 of 323, and the same top token as the BF16-input build at 322 of 324. Nothing on real weights has been compared with MXFP4 activations and an FP8 cache.

## Run the measured build

These commands give the 156.20 tok/s build. The rest of this README covers the batch-1 FP4 graph used for first bring-up, which is what `scripts/generate.py` and `scripts/chip_check.py` take, and it explains how to set up the compiler, checkpoint and instance used here. [The field notes](docs/field_notes.md) cover installing the runtime and Python packages on the Amazon Linux Neuron AMI.

The graphs in [the `graphs-b63` release](https://github.com/Slug-ai/qwen38-27b-aws/releases/tag/graphs-b63) predate FP8 matmul inputs and give about 123 tok/s, so compile instead: the layers take about 2 minutes on the `inf2.xlarge` itself. To fetch the older graphs from the repo root:

```bash
curl -fL https://github.com/Slug-ai/qwen38-27b-aws/releases/download/graphs-b63/graphs_b63.tar.gz | tar xz
```

Compile the layers and `lm_head`:

```bash
python scripts/compile.py --config tests/fixtures/config.json --layers 16 --max-seq-len 512 --batch-size 63 \
  --weight-format fp8 --recurrent-state bf16 --output artifacts/block16_b63_s512_fp8_bf16state
python scripts/compile.py --config tests/fixtures/config.json --head --head-format fp8 --batch-size 63 --output artifacts/head_fp8_b63
```

Prepare the weights, which takes about 26 GB of disk on top of the 19 GB checkpoint:

```bash
python scripts/prepare_weights.py --model-dir model --compiled artifacts/block16_b63_s512_fp8_bf16state --output weights_fp8
python scripts/prepare_weights.py --model-dir model --compiled artifacts/head_fp8_b63 --output weights_fp8 --head-only
```

Run the benchmark on the instance. Loading takes about 3 minutes:

```bash
python scripts/batch_generate.py --model-dir model --compiled artifacts/block16_b63_s512_fp8_bf16state --head artifacts/head_fp8_b63 --weights weights_fp8
```

For your own prompts, add `--prompts-file prompts.txt --sequences N`, with one prompt per line. Row 0 always runs the saved greedy prompt, so `N` can be at most the number of lines plus one. `--show` sets how many outputs are printed.

`scripts/benchmark.py` takes the same arguments and runs the same work, but prints only one example answer, then the batch size, sequences, generated tokens, total time, throughput, tokens per second for one user, and whether row 0 matched. The per-user figure is each sequence's rate from its first new token to its last, averaged, so the wait while its prompt is read isn't in it. It skips the saved-passage check and the JSON results, so use `batch_generate.py` for rows of [the throughput log](docs/throughput.md).

```bash
python scripts/benchmark.py --model-dir model --compiled artifacts/block16_b63_s512_fp8_bf16state --head artifacts/head_fp8_b63 --weights weights_fp8
```

### Demo: load once, then keep taking prompts

`batch_generate.py` loads for about 3.5 minutes every run, because the chip keeps nothing after a process exits and the weights don't fit in the host's disk cache. `scripts/demo.py` loads once and then takes prompts until you quit:

```bash
python scripts/demo.py --model-dir model --compiled artifacts/block16_b63_s512_fp8_bf16state --head artifacts/head_fp8_b63 --weights weights_fp8
```

At the `>` prompt, type a message to see its answer as it's generated, or `@prompts.txt` to run a file of prompts together, one per line, which prints each answer and the generated tok/s. `quit` or Ctrl-D exits, and Ctrl-C stops a generation in progress. `--max-new-tokens` caps each answer, 256 by default, and prompt plus answer stops at 512 tokens.

One message uses one of the 63 rows, so it runs at the chip's step rate, about 3.5 tokens/s, and each prompt token takes a step before the answer starts. A file of prompts fills the rows, which is where the throughput comes from.

## Compile (no chip needed)

Compiling one 4-layer FP4 block takes about 10 minutes and peaks at 11.4 GB of RAM. The output is about 37 MB and can be copied to the instance. A single 64-layer FP4 graph would need about 184 GB of RAM and about 2.9 hours to compile, projected from measured compiles of 4 and 8 layers. Pick either route.

Docker, on any machine (on Apple Silicon it runs under Rosetta, which is slower):

```bash
docker build --platform=linux/amd64 -f docker/Dockerfile -t qwen38-neuron:2.28.1 .
docker run --rm --platform=linux/amd64 -v "$PWD":/workspace qwen38-neuron:2.28.1 \
  python scripts/compile.py --config tests/fixtures/config.json --layers 4 --max-seq-len 2048 --output artifacts/block4
```

Native on x86-64 Linux, with no root or Docker. This installs about 5 GB into `.neuron-env`, including its caches and temp files:

```bash
bash scripts/setup_native_env.sh
source .neuron-env/activate
bash scripts/run_guarded.sh python scripts/compile.py --config tests/fixtures/config.json --layers 4 --max-seq-len 2048 --output artifacts/block4
```

`run_guarded.sh` is for shared machines: it runs at the lowest priority and stops the job before memory runs out.

`--optimize-layout` turns on NxD's weight layout optimization, which rearranges weights once at load instead of every run. Use a different `--output` to keep both, such as `artifacts/block4_wlo`. It adds a load-time step that can only run on a chip, so try the plain block first. Compiler estimates per block, per core, before any chip has run:

| | Plain | `--optimize-layout` |
|---|---|---|
| Memory traffic | 1.83 GB | 0.97 GB |
| Instructions executed | 929k | 837k |
| Internal transfers | 11.9 GB | 10.5 GB |
| Weight copies per run | 25 | 0 |

`scripts/compile_report.py` prints these from any compile. They are useful for comparing designs, and they aren't speed measurements.

Then check the graph, which should end with `graph check passed`:

```bash
python scripts/inspect_graph.py artifacts/block4/compiler-work/decode/<timestamp> \
  --config tests/fixtures/config.json --layers 4 --header tests/fixtures/model.header.json
```

## Prepare weights (no chip or SDK needed)

Needs about 33 GB of disk: 19 GB for the checkpoint and 14 GB for the per-core files.

```bash
mkdir -p model
for f in model.safetensors config.json tokenizer.json; do
  curl -fL -o model/$f https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4/resolve/main/$f
done
python scripts/prepare_weights.py --model-dir model --compiled artifacts/block4 --output weights
```

## Try it on CPU

Runs the real weight files with one thread per core, a real sum across them, and the same block chain and greedy loop. It replaces only the compiled graph, using the PyTorch source it came from.

```bash
python scripts/generate.py --simulate --model-dir model --compiled artifacts/block4 --weights weights \
  --prompt "The capital of France is" --raw --max-new-tokens 8
```

This is slow, at about 50 seconds per token on an Apple Silicon Mac, and it says nothing about speed on a chip.

## On the instance

Start with `inf2.xlarge`. It has the same chip as `inf2.8xlarge` and is cheaper, but only 16 GiB of host RAM and 4 vCPUs, so compile elsewhere and copy `artifacts/block4` over. Move up to `inf2.8xlarge` (128 GiB) only if loading the weights runs out of host RAM. Allow at least 50 GB of disk.

```bash
# Ubuntu 22.04. Install the 2.28.1 driver and runtime.
sudo tee /etc/apt/sources.list.d/neuron.list <<< "deb https://apt.repos.neuron.amazonaws.com jammy main"
curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y "aws-neuronx-dkms=2.26.10.0*" "aws-neuronx-runtime-lib=2.30.51.0*" \
  "aws-neuronx-collectives=2.30.59.0*" "aws-neuronx-tools=2.28.23.0*"
neuron-ls  # should list one chip with two NeuronCores

# Python packages, with the same pins as the compiler image.
python3.11 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pip.repos.neuron.amazonaws.com -r docker/requirements-neuron.txt

# Run.
export PYTHONPATH=$PWD
python scripts/generate.py --model-dir model --compiled artifacts/block4 --weights weights \
  --prompt "What is the capital of France?" --max-new-tokens 32
```

Each step's time goes to stderr, and the script prints decode tokens per second at the end.

## On a chip for the first time

Don't start with `generate.py`. `scripts/chip_check.py` compares the chip with CPU results saved in `tests/golden`, one level at a time, and stops at the first failure with a hint about the likely cause.

Compile the small graphs on the same machine as the block, then copy `artifacts/` to the instance:

```bash
python scripts/compile_levels.py --config tests/fixtures/config.json --output artifacts/levels
```

On the instance, run the preflight first. It needs no device, so it also works in the Docker image. It runs NxD's argument routing for each graph and compares all 16 blocks of weight files with the graph:

```bash
python scripts/chip_check.py --preflight --model-dir model --weights weights --compiled artifacts/block4
```

Then run the check on the chip:

```bash
python scripts/chip_check.py --model-dir model --weights weights --compiled artifacts/block4 --output chip_report.json
```

| Level | What it adds | If it fails, suspect |
|---|---|---|
| `decode` | FP4 decode alone, which must be exact | floor, clamp or the uint8 cast |
| `matrix` | One real matrix on one core | Weight loading or BF16 matmul |
| `layer` | One gated delta layer, two cores, several tokens | The all-reduce, or state if only later tokens fail |
| `block` | The compiled 4-layer block, adding full attention | Attention or the KV cache |
| `model` | All 16 blocks, top tokens on a test passage | Block chaining or later blocks' weights |
| `greedy` | Real generation | Read the text first, since one near-tie changes every later token |
| `speed` | Real per-token time, split between chip and host | No pass or fail |

The chip won't match the CPU exactly, because it approximates sigmoid and exp with lookup tables and rounds differently. Token-level checks allow it 3 times the CPU's own BF16 error, and at least 2%. A top token may only change where the CPU's top two logits were within 0.25. Those limits were chosen before any chip has run, so treat a narrow failure as something to look at, not a verdict.

Run the plain block first, since it has fewer moving parts, then rerun with `--compiled artifacts/block4_wlo` to compare speed. Optimize only from these real numbers.

`scripts/make_golden.py` regenerates the saved results, and `--backend cpu` runs the check without a chip, which tests the script and not the hardware.

## Tests

On any machine, with no Neuron SDK:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
# compressed-tensors asks for a newer torch than the SDK's, but only its FP4 decoder is used, as a reference.
uv pip install --python .venv/bin/python --no-deps compressed-tensors==0.20.1
.venv/bin/python -m pytest tests
```

`tests/test_xla_graph.py` runs the model through XLA on CPU and is skipped without `torch_xla`. Run it in the Neuron environment, where the other tests can't run because their reference packages aren't installed:

```bash
python -m pytest tests/test_xla_graph.py
```

To check the real weights against the Hugging Face layers without downloading them (no disk, but about 13 GB of layers and 5 GB of `lm_head` over the network, which took about 2.5 hours on a slow connection):

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_real_weights.py
```

## What has been checked

Most of these checks predate FP8 weights, and they cover the model and the FP4 path. Each run of the measured build has its own correctness check, described in [the throughput log](docs/throughput.md).

| Check | Result |
|---|---|
| FP4 decode vs compressed-tensors, the code that wrote the checkpoint | Exact, for every byte value and a range of exponents |
| FP4 decode through XLA | Exact for every byte value |
| Greedy output on a tiny model vs Hugging Face `generate` | Identical |
| Logits on a tiny model vs Hugging Face | float32 error under 1e-5. BF16 differs only by rounding |
| Two simulated cores vs one core | Match to under 1e-5 |
| Chained blocks vs one whole graph, on a tiny model | Identical greedy output |
| All 64 real layers vs Hugging Face, in float32 ([results](docs/real_weights_check.json)) | Same top token at 38 of 38 positions, worst relative layer error 5.3e-6. Predicts `France is -> Paris`, `Japan is -> Tokyo` |
| Full greedy run on real weights: per-core files, two simulated cores, 16 chained blocks, BF16 | `The capital of France is` -> ` Paris.\nThe capital of Germany is` |
| BF16 vs float32 on real weights, on the CPU simulation ([results](docs/bf16_check.json)) | Same top token at 38 of 38 positions |
| Real layers 0 and 3 as two simulated cores vs one core | Absolute error about 1e-6 to 2e-6 |
| Real checkpoint header vs the full model | Every tensor maps, for both cores and all 16 blocks |
| Traced graph through XLA on CPU vs PyTorch, on a tiny model | float32 error under 1e-4 relative, at every token |
| Compile for `--target inf2` on two cores, real layer sizes | 4 and 8 layers compile, plain. The 4-layer block also compiles with `--optimize-layout` |
| BF16 vs float32 hidden states on real weights, one layer and one block | Differ by under 1% |
| Chip check script, fed deliberately broken results | Catches wrong decode, wrong math, NaN, confident token flips and state lost between tokens |
| Compiled FP4 graph structure | All caches persist as device state, all collectives present, every FP4 matrix decoded, no custom calls |
| Chip check script end to end on CPU, real weights, `model`, `greedy` and `speed` levels | All three pass. CPU results compared with themselves, so this only shows the script runs |

**On a real chip:** every level of `scripts/chip_check.py` passes on batch-1 graphs, and each row of [the throughput log](docs/throughput.md) runs real batches with a correctness check. The `chip_check.py` levels haven't been rerun since the decode changed, and they take batch-1 graphs. The Ubuntu install steps above are untested, since the instance used Amazon Linux.

The NKI simulator can't check the whole model, because it runs NKI kernels and not compiled graphs.
