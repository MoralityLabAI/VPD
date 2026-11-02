#!/usr/bin/env bash
# Run on one temporary GPU. The outer launch controller owns pod termination.
set -euo pipefail
export HF_HOME="${PWD}/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p artifacts/nanbeige_looped
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
timeout --signal=TERM --kill-after=30s 600s python -m pip install -r experiments/nanbeige_looped/requirements.txt
timeout --signal=TERM --kill-after=60s 1500s python -m experiments.nanbeige_looped.preflight \
  --output-dir artifacts/nanbeige_looped/bf16_preflight --layers 11,21 --rows 16
# Reuse exactly the snapshot already downloaded by preflight.
MODEL_DIR="$(python - <<'PY'
from huggingface_hub import snapshot_download
from experiments.nanbeige_looped.preflight import MODEL_ID, REVISION
print(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
PY
)"
timeout --signal=TERM --kill-after=60s 900s python -m experiments.nanbeige_looped.run_multipass \
  --model-dir "$MODEL_DIR" --device cuda \
  --output-dir artifacts/nanbeige_looped/bf16_multipass_calibration \
  --layers 11,21 --components 8 --measure-components 2 \
  --vpd-steps 50 --warmup-steps 50 --sae-steps 1000 --rows 16 --seq-len 64
