#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/nanbeige-reasoning
[[ ! -e artifacts/nanbeige_reasoning/prime_q4_cache_gate_v1 ]] || exit 2
python3 - <<'PY'
import os, pathlib, signal
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try: args=(p/'cmdline').read_bytes().split(b'\0')
    except OSError: continue
    if args and args[0].split(b'/')[-1]==b'curl' and any(b'/loqy/Nanbeige4.2-3B-GGUF/resolve/' in a for a in args):
        os.kill(int(p.name),signal.SIGTERM)
        print('Stopped stalled model curl',p.name,flush=True)
PY
source .venv-prime/bin/activate
export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
export LD_LIBRARY_PATH="$PWD/tools/nanbeige/runtime:${LD_LIBRARY_PATH:-}"
timeout --signal=TERM --kill-after=30s 1800s python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id='loqy/Nanbeige4.2-3B-GGUF',filename='Nanbeige4.2-3B-Q4_K_M.gguf',
    revision='5ae468e507f641599c25f34f85772212f7e00dc7',local_dir='models'))
PY
python -m scripts.nanbeige.prepare_prime_q4_runtime
timeout --signal=TERM --kill-after=30s 3600s python -m experiments.nanbeige_reasoning.check_cached_steering --chunk-size 1 --output artifacts/nanbeige_reasoning/prime_q4_cache_gate_v1
python -m scripts.nanbeige.verify_prime_q4_gate
timeout --signal=TERM --kill-after=60s 21720s python -m experiments.nanbeige_reasoning.run_q4_feature_answers --gate artifacts/nanbeige_reasoning/prime_q4_cache_gate_v1 --context-check artifacts/nanbeige_reasoning/q4_answer_context_v1 --output artifacts/nanbeige_reasoning/prime_q4_feature_answers_v1 --wall-seconds 21600 --max-new-tokens 8192
