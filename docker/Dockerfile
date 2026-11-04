# Neuron SDK 2.28.1 compiler toolchain. Compiles for Inf2 on any machine, with no Neuron hardware.
# On Apple Silicon this runs under Rosetta emulation, because the SDK is x86-64 only.
FROM --platform=linux/amd64 python:3.11-slim-bookworm

# libnrt is needed just to import the SDK, even though nothing here runs on a device.
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends gcc g++ libgomp1 libarchive13 curl gnupg ca-certificates \
    && curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --dearmor -o /usr/share/keyrings/neuron.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com jammy main" \
        > /etc/apt/sources.list.d/neuron.list \
    && apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        "aws-neuronx-runtime-lib=2.30.51.0*" "aws-neuronx-collectives=2.30.59.0*" \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-neuron.txt /tmp/requirements-neuron.txt
RUN pip install -q --no-cache-dir uv \
    && uv pip install --system --no-cache --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pip.repos.neuron.amazonaws.com \
        -r /tmp/requirements-neuron.txt

WORKDIR /workspace
ENV PYTHONPATH=/workspace
ENV LD_LIBRARY_PATH=/opt/aws/neuron/lib
ENV NEURON_PLATFORM_TARGET_OVERRIDE=inf2
