#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${PWD}/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
timeout --signal=TERM --kill-after=30s 600s python -m pip install -r experiments/nanbeige_looped/requirements.txt
MODEL_DIR="$(timeout --signal=TERM --kill-after=30s 1800s python - <<'PY'
import hashlib,json
from pathlib import Path
from huggingface_hub import snapshot_download
contract=json.loads(Path('artifacts/nanbeige_reasoning/vpd_precision_inputs_v1/contract.json').read_text())
model=Path(snapshot_download(contract['model_id'],revision=contract['revision'],allow_patterns=list(contract['checkpoint_files'])))
for name,expected in contract['checkpoint_files'].items():
    h=hashlib.sha256()
    with (model/name).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('Checkpoint hash mismatch: '+name)
print(model)
PY
)"
timeout --signal=TERM --kill-after=60s 2850s python -m experiments.nanbeige_reasoning.run_vpd_precision_control \
  --model-dir "$MODEL_DIR" --inputs artifacts/nanbeige_reasoning/vpd_precision_inputs_v1 \
  --output artifacts/nanbeige_reasoning/vpd_precision_control_v1 --max-wall-seconds 2700
