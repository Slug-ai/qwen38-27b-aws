# qwen38-27b-aws

Runs the text part of [RedHatAI/Qwen3.8-27B-MXFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4) on one AWS Inferentia2 chip (two NeuronCores) with FP4 weights and greedy decoding.

**Status: not yet run on a chip.** Everything below "On the instance" is untested until we have AWS access. See [What has been checked](#what-has-been-checked).

## How it works

- **FP4 on a chip without FP4.** NeuronCore-v2 has no FP4 type (native MX compute is Trn3 only). Weights stay packed on the device and the graph expands them to BF16 for each matmul, using only float add, multiply, floor and clamp. This is what makes the model fit:

  | | Per NeuronCore (16 GB available) |
  |---|---|
  | Packed FP4 weights | 6.87 GB |
  | Same weights decoded to BF16 | about 24 GB |
  | KV cache at 2048 tokens | 0.07 GB |
  | Gated delta state | 0.08 GB |

- **Text only.** The vision tower, MTP head and KV-cache FP8 scales are never read.
- **Two cores.** Each core owns a slice of the attention heads. Only the output projections need a collective, so there are two all-reduces per layer.
- **One graph, one token.** A prompt is fed token by token through the same graph, in the recurrent form of the gated delta rule. There is no prefill graph or chunked solve.
- **One block, loaded 16 times.** The model is 16 repeats of the same 4 layers (3 gated delta, 1 full attention). We compile one block and load it once per repeat with that repeat's weights and caches. A single 64-layer graph would need about 184 GB of RAM and about 2.9 hours to compile, projected from measured compiles of 4 and 8 layers.
- **Host side.** Embeddings, the final norm, lm_head and the greedy loop run on the host, which saves 5 GB of device memory.
- **SDK pinned to Neuron 2.28.1.** From 2.29 on, NxD Inference and NKI 0.3 kernels dropped Inf2. This repo uses NxD and the compiler directly, not NxD Inference.

### Differences from the checkpoint's intended serving

Only the weights are FP4. Activations and the KV cache run at BF16, while the checkpoint was calibrated for FP4 activations and an FP8 cache. This should be at least as accurate, but it has not been measured. Throughput is unknown until it runs on a chip, and only batch size 1 is supported.

## Compile (no chip needed)

Compiling one block takes about 10 minutes and peaks at 11.4 GB of RAM. The output is about 37 MB and can be copied to the instance. Pick either route.

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

| Check | Result |
|---|---|
| FP4 decode vs compressed-tensors, the code that wrote the checkpoint | Exact, for every byte value and a range of exponents |
| FP4 decode through XLA | Exact for every byte value |
| Greedy output on a tiny model vs Hugging Face `generate` | Identical |
| Logits on a tiny model vs Hugging Face | float32 error under 1e-5. BF16 differs only by rounding |
| Two simulated cores vs one core | Match to under 1e-5 |
| Chained blocks vs one whole graph, on a tiny model | Identical greedy output |
| All 64 real layers vs Hugging Face, in float32 ([results](docs/real_weights_check.json)) | Same top token at 38 of 38 positions, worst relative layer error 5.3e-6. Predicts `France is -> Paris`, `Japan is -> Tokyo` |
| Real layers 0 and 3 as two simulated cores vs one core | Absolute error about 1e-6 to 2e-6 |
| Real checkpoint header vs the full model | Every tensor maps, for both cores and all 16 blocks |
| Traced graph through XLA on CPU vs PyTorch, on a tiny model | float32 error under 1e-4 relative, at every token |
| Compile for `--target inf2` on two cores, real layer sizes | 4 layers and 8 layers both compile |
| Compiled graph structure | All caches persist as device state, all collectives present, every FP4 matrix decoded, no custom calls |

**Not checked, because these need a chip:** whether the compiled graph computes the right values, whether 16 loaded copies of the block run together, weight loading and host RAM use on an instance, throughput, and the install steps above.

The NKI simulator can't help here, because it runs NKI kernels and not compiled graphs. This repo doesn't use NKI so far. A kernel is the likely next step if in-graph FP4 decode turns out to be too slow.
