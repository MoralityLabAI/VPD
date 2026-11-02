#!/usr/bin/env bash
# Execute only inside the authorized temporary CUDA GPU workspace.
set -Eeuo pipefail
COHORT="${1:-pilot}"
case "$COHORT" in pilot|benchmark) ;; *) echo 'Expected pilot or benchmark'; exit 2 ;; esac
export HF_HOME="${PWD}/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || { echo 'CUDA development toolkit required before downloading weights'; exit 1; }
mkdir -p artifacts/nanbeige_reasoning/qwen_setup
nvidia-smi > artifacts/nanbeige_reasoning/qwen_setup/gpu.txt
nvcc --version > artifacts/nanbeige_reasoning/qwen_setup/nvcc.txt
timeout 600s python -m pip install -r experiments/nanbeige_looped/requirements.txt 'psutil>=7,<8'
if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  timeout 600s apt-get update
  timeout 600s apt-get install -y build-essential cmake git
fi
COMMIT=0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0
mkdir -p .llama-src
if [[ ! -d .llama-src/.git ]]; then
  git -C .llama-src init
  git -C .llama-src remote add origin https://github.com/ggml-org/llama.cpp.git
  timeout 300s git -C .llama-src fetch --depth 1 origin "$COMMIT"
  git -C .llama-src checkout --detach FETCH_HEAD
fi
[[ "$(git -C .llama-src rev-parse HEAD)" == "$COMMIT" ]] || { echo 'Unexpected runtime source commit'; exit 1; }
CUDA_ARCH="$(python -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(10 * major + minor)')"
[[ "$CUDA_ARCH" =~ ^[0-9]+$ ]] || { echo 'Invalid detected CUDA architecture'; exit 1; }
printf '%s\n' "$CUDA_ARCH" > artifacts/nanbeige_reasoning/qwen_setup/cuda_arch.txt
cmake -S .llama-src -B .llama-src/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
timeout --signal=TERM --kill-after=60s 3600s cmake --build .llama-src/build --target llama-server -j4
.llama-src/build/bin/llama-server --version > artifacts/nanbeige_reasoning/qwen_setup/runtime_version.txt 2>&1
mkdir -p models
python - <<'PY'
import json, subprocess, sys
from pathlib import Path
lock=json.loads(Path('artifacts/nanbeige_reasoning/comparator_sources/qwen35_27b/model_lock.json').read_text())
quant=lock['quantized']
assert len(quant['weights'])==1
weight=quant['weights'][0]
url=f"https://huggingface.co/{quant['repo']}/resolve/{quant['revision']}/{weight['filename']}"
subprocess.run([sys.executable,'scripts/nanbeige/download_file.py',url,str(Path('models')/weight['filename']),
               '--sha256',weight['sha256'],'--bytes',str(weight['bytes'])],check=True)
PY
INPUTS="artifacts/nanbeige_reasoning/qwen_comparator_${COHORT}_inputs_v1"
CONTEXT=32768
[[ "$COHORT" != pilot ]] || CONTEXT=10240
timeout --signal=TERM --kill-after=60s 22000s python -m experiments.nanbeige_reasoning.run_qwen_comparator \
  --inputs "$INPUTS" --sources artifacts/nanbeige_reasoning/comparator_sources/qwen35_27b \
  --model models/Qwen3.5-27B-Q4_K_M.gguf --server .llama-src/build/bin/llama-server \
  --output-dir "artifacts/nanbeige_reasoning/qwen35_27b_${COHORT}_8192_v1" \
  --context "$CONTEXT" --max-new-tokens 8192 --max-wall-seconds 21600
# A timed pause is incomplete and retains durable records for a later resume.
