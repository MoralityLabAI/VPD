#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${PWD}/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
timeout --signal=TERM --kill-after=30s 600s python -m pip install -r experiments/nanbeige_looped/requirements.txt
MODEL_DIR="$(timeout --signal=TERM --kill-after=30s 1800s python - <<'PY'
import hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download
contract = json.loads(Path('artifacts/nanbeige_reasoning/cross_precision_vpd_inputs_v2/contract.json').read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
if sha(Path(contract['dataset'])) != contract['dataset_sha256']:
    raise ValueError('Frozen trajectory hash mismatch')
model = Path(snapshot_download(contract['model_id'], revision=contract['revision'],
                              allow_patterns=list(contract['checkpoint_files'])))
for name, expected in contract['checkpoint_files'].items():
    if sha(model / name) != expected:
        raise ValueError(f'Checkpoint asset hash mismatch: {name}')
print(model)
PY
)"
timeout --signal=TERM --kill-after=30s 600s python -m experiments.nanbeige_looped.saved_tensor_control \
  --device cuda --output artifacts/nanbeige_reasoning/saved_tensor_control.json
timeout --signal=TERM --kill-after=60s 21600s python -m experiments.nanbeige_looped.run_multipass \
  --model-dir "$MODEL_DIR" --device cuda \
  --dataset artifacts/nanbeige_reasoning/q4_reasoning_pilot_v1/calibration_tokens.jsonl \
  --output-dir artifacts/nanbeige_looped/bf16_multipass_calibration \
  --layers 11,21 --components 64 --measure-components 1 \
  --vpd-steps 50 --warmup-steps 50 --sae-steps 1000 --rows 8 \
  --training-window generated_prefixes --max-example-tokens 4096 --saved-tensors-on-cpu
