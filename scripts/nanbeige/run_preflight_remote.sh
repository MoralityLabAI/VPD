#!/usr/bin/env bash
# Run inside a provisioned CUDA PyTorch pod, from the unpacked project root.
# The pod owner must terminate the pod after retrieving artifacts; timeout only
# bounds this process and does not terminate cloud billing.
set -euo pipefail
export HF_HOME="${PWD}/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p artifacts/nanbeige_looped
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
timeout --signal=TERM --kill-after=30s 600s python -m pip install -r experiments/nanbeige_looped/requirements.txt
timeout --signal=TERM --kill-after=60s 1800s python -m experiments.nanbeige_looped.preflight \
  --output-dir artifacts/nanbeige_looped/bf16_preflight --layers 11,21 --rows 32
