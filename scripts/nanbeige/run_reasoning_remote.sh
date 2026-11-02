#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-capture}"
case "$MODE" in capture|cache-diagnostics) ;; *) echo 'Unsupported experiment mode'; exit 2 ;; esac
export HF_HOME="${PWD}/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
timeout --signal=TERM --kill-after=30s 600s python -m pip install -r experiments/nanbeige_looped/requirements.txt
MODEL_DIR="$(python - <<'PY'
from huggingface_hub import snapshot_download
from experiments.nanbeige_looped.preflight import MODEL_ID, REVISION
print(snapshot_download(MODEL_ID, revision=REVISION))
PY
)"
if [[ "$MODE" == cache-diagnostics ]]; then
  timeout --signal=TERM --kill-after=60s 600s python -m experiments.nanbeige_reasoning.cache_diagnostics \
    --model-dir "$MODEL_DIR" --output-dir artifacts/nanbeige_reasoning/cache_diagnostics
  exit 0
fi
timeout --signal=TERM --kill-after=60s 1800s python -m experiments.nanbeige_reasoning.run_capture \
  --model-dir "$MODEL_DIR" --questions artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl \
  --output-dir artifacts/nanbeige_reasoning/reasoning_capture --max-new-tokens 2048
timeout --signal=TERM --kill-after=60s 21600s python -m experiments.nanbeige_looped.run_multipass \
  --model-dir "$MODEL_DIR" --device cuda \
  --dataset artifacts/nanbeige_reasoning/reasoning_capture/calibration_tokens.jsonl \
  --output-dir artifacts/nanbeige_looped/bf16_multipass_calibration \
  --layers 11,21 --components 64 --measure-components 1 \
  --vpd-steps 50 --warmup-steps 50 --sae-steps 1000 --rows 8 \
  --training-window generated_prefixes --max-example-tokens 4096
