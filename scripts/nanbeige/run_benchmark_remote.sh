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
contract=json.loads(Path('artifacts/nanbeige_reasoning/full_benchmark_inputs_v1/contract.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()
if sha(Path(contract['questions']))!=contract['questions_sha256']:
    raise ValueError('Full benchmark question hash mismatch')
model=Path(snapshot_download(contract['model_id'],revision=contract['revision'],
                            allow_patterns=list(contract['checkpoint_files'])))
for name,expected in contract['checkpoint_files'].items():
    if sha(model/name)!=expected:
        raise ValueError(f'Checkpoint asset hash mismatch: {name}')
print(model)
PY
)"
timeout --signal=TERM --kill-after=60s 24000s python -m experiments.nanbeige_reasoning.run_benchmark \
  --model-dir "$MODEL_DIR" --questions artifacts/nanbeige_reasoning/benchmark_v1/questions.jsonl \
  --output-dir artifacts/nanbeige_reasoning/bf16_full_benchmark_v1 \
  --loops 2,1 --max-new-tokens 8192 --batch-size 8 --max-prefill-tokens 32768 \
  --max-total-tokens 65536 --max-wall-seconds 21600
