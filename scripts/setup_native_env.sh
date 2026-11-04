#!/usr/bin/env bash
# Install the pinned Neuron SDK 2.28.1 compiler toolchain with no root and no Docker, on x86-64 Linux.
# Everything lands in .neuron-env inside this repo, including caches and temp files.
# Afterwards: source .neuron-env/activate
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/.neuron-env"
APT_REPO=https://apt.repos.neuron.amazonaws.com

mkdir -p "$ENV_DIR"/{home,tmp,cache,deb,runtime,python}
cat > "$ENV_DIR/activate" <<ACTIVATE
export HOME="$ENV_DIR/home"
export TMPDIR="$ENV_DIR/tmp"
export XDG_CACHE_HOME="$ENV_DIR/cache"
export PIP_CACHE_DIR="$ENV_DIR/cache/pip"
export UV_CACHE_DIR="$ENV_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ENV_DIR/python"
export HF_HOME="$ENV_DIR/cache/huggingface"
export LD_LIBRARY_PATH="$ENV_DIR/runtime/opt/aws/neuron/lib:\${LD_LIBRARY_PATH:-}"
export PATH="$ENV_DIR/venv/bin:$ENV_DIR/bootstrap/bin:\$PATH"
export PYTHONPATH="$ROOT"
export PYTHONWARNINGS=ignore
export NEURON_PLATFORM_TARGET_OVERRIDE=inf2
# The SDK insists libnrt is at /opt/aws/neuron. This environment only compiles and never touches a
# device, so skip that check here. Do not set this on a real Inf2 instance.
export NEURON_INTERNAL_SKIP_LIBNRT_CHECK=1
ACTIVATE
# shellcheck disable=SC1091
source "$ENV_DIR/activate"

python3 -m venv "$ENV_DIR/bootstrap"
"$ENV_DIR/bootstrap/bin/pip" install -q uv
uv venv -q --clear --python 3.11 "$ENV_DIR/venv"
# torch_xla links against libpython, which a relocatable Python keeps beside its own binary.
python_lib="$(dirname "$(dirname "$(readlink -f "$ENV_DIR/venv/bin/python")")")/lib"
echo "export LD_LIBRARY_PATH=\"$python_lib:\$LD_LIBRARY_PATH\"" >> "$ENV_DIR/activate"
# shellcheck disable=SC1091
source "$ENV_DIR/activate"
uv pip install -q --python "$ENV_DIR/venv/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pip.repos.neuron.amazonaws.com \
    -r "$ROOT/docker/requirements-neuron.txt"

# The SDK needs libnrt just to import. Unpack the packages rather than installing them.
curl -fsSL "$APT_REPO/dists/jammy/main/binary-amd64/Packages" -o "$ENV_DIR/deb/Packages"
for spec in "aws-neuronx-runtime-lib 2.30.51.0" "aws-neuronx-collectives 2.30.59.0"; do
    read -r package version <<<"$spec"
    entry="$(awk -v p="Package: $package" -v v="Version: $version" 'BEGIN{RS=""} index($0,p) && index($0,v)' "$ENV_DIR/deb/Packages")"
    filename="$(awk '/^Filename:/{print $2}' <<<"$entry")"
    sha256="$(awk '/^SHA256:/{print $2}' <<<"$entry")"
    deb="$ENV_DIR/deb/$(basename "$filename")"
    curl -fsSL "$APT_REPO/$filename" -o "$deb"
    echo "$sha256  $deb" | sha256sum -c --quiet
    dpkg -x "$deb" "$ENV_DIR/runtime"
done

python - <<'CHECK'
import subprocess, torch_neuronx, neuronxcc
print("torch-neuronx", torch_neuronx.__version__)
print(subprocess.run(["neuronx-cc", "--version"], capture_output=True, text=True).stdout.strip())
CHECK
