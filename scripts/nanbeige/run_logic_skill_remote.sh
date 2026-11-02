#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
export PATH="/usr/local/cuda/bin:$PATH"
APT=()
if [[ "$(id -u)" != 0 ]]; then APT=(sudo -n); fi
APT_STARTED=$SECONDS
until timeout 120s "${APT[@]}" apt-get -o DPkg::Lock::Timeout=110 update; do
  (( SECONDS - APT_STARTED < 600 )) || { echo 'Package update readiness timed out'; exit 2; }
  sleep 10
done
timeout 600s "${APT[@]}" env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=540 install -y build-essential gcc-11 g++-11 cmake git curl rsync python3-pip python3-venv
if [[ ! -x /usr/local/cuda-12.4/bin/nvcc ]]; then
  timeout 120s curl -fL --retry 3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb
  timeout 120s "${APT[@]}" dpkg -i cuda-keyring.deb
  timeout 300s "${APT[@]}" apt-get -o DPkg::Lock::Timeout=240 update
  timeout 1200s "${APT[@]}" env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=540 install -y --no-install-recommends cuda-toolkit-12-4
fi
export PATH="/usr/local/cuda-12.4/bin:$PATH" CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
nvcc --version
python3 -m venv .venv-prime
source .venv-prime/bin/activate
python - <<'PY'
import hashlib,json,pathlib
r=json.loads(pathlib.Path('bundle_manifest.json').read_text())
for name,digest in r['files'].items():
    assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest,name
print('Frozen bundle verified')
PY
timeout --signal=TERM --kill-after=30s 600s python -m pip install 'numpy==2.2.6' 'transformers==4.45.1' 'sentencepiece==0.2.0' 'protobuf==5.29.5'
mkdir -p tools/nanbeige/llama-source tools/nanbeige/runtime models
git -C tools/nanbeige/llama-source init
git -C tools/nanbeige/llama-source remote add origin https://github.com/ggml-org/llama.cpp.git
timeout 600s git -C tools/nanbeige/llama-source fetch --depth 1 origin 0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0
git -C tools/nanbeige/llama-source checkout --detach FETCH_HEAD
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')"
[[ "$ARCH" =~ ^[0-9]+$ ]] || { echo 'Expected one GPU architecture'; exit 2; }
timeout 120s cmake -S tools/nanbeige/llama-source -B tools/nanbeige/llama-build -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_BUILD_TYPE=Release
timeout --signal=TERM --kill-after=30s 1200s cmake --build tools/nanbeige/llama-build --target llama-server ggml-cuda ggml-cpu --parallel 6
cp -L tools/nanbeige/llama-build/bin/*.so* tools/nanbeige/runtime/
export LD_LIBRARY_PATH="${PWD}/tools/nanbeige/runtime:${LD_LIBRARY_PATH:-}"
cp tools/nanbeige/llama-build/bin/llama-server tools/nanbeige/runtime/
export HF_XET_HIGH_PERFORMANCE=1
timeout --signal=TERM --kill-after=30s 1800s python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
lock=json.loads(Path('artifacts/nanbeige_reasoning/comparator_sources/qwen35_27b/model_lock.json').read_text())
q=lock['quantized']; w=q['weights'][0]
print(hf_hub_download(repo_id=q['repo'],filename=w['filename'],revision=q['revision'],local_dir='models'))
PY
timeout --signal=TERM --kill-after=60s 18200s python -m experiments.nanbeige_reasoning.run_logic_skill_comparison --server tools/nanbeige/runtime/llama-server --model models/Qwen3.5-27B-Q4_K_M.gguf --wall-seconds 18000
