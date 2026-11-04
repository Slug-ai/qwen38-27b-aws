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
