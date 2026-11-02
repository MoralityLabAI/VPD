"""Create an isolated bounded launcher and hash-verified public-input payload."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/nanbeige_reasoning'
SCRIPTS=ROOT/'scripts/nanbeige'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def save(name,text):
    (SCRIPTS/name).write_text(text,encoding='utf-8',newline='\n')


def main():
    # Reuse the proven lifecycle while giving this experiment distinct files.
    launch=(SCRIPTS/'launch_prime_q4_durable.sh').read_text()
    launch=launch.replace('NAME="nanbeige-reasoning-${STAMP}"','NAME="hermes-logic-${STAMP}"')
    launch=launch.replace('prime-${STAMP}','logic-prime-${STAMP}')
    launch=launch.replace('prime_q4_answers_bundle','logic_skill_bundle')
    launch=launch.replace('monitor_prime_q4.py','monitor_logic_skill.py')
    launch=launch.replace('start_prime_q4_worker.py','start_logic_skill_worker.py')
    launch=launch.replace('All 70 outputs','All 259 outputs')
    launch=launch.replace('run_prime_q4_answers_remote.sh','run_logic_skill_remote.sh')
    save('launch_logic_skill.sh',launch)
    save('launch_logic_skill_bounded.sh',
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'export NANBEIGE_COMPUTE_PROFILE=lambda_a100 NANBEIGE_REASONING_REGION=US '
        'NANBEIGE_MAX_HOURLY=1.99 NANBEIGE_LAUNCH_BOUND_SECONDS=19800 NANBEIGE_IMAGE=cuda_12_4_pytorch_2_4\n'
        'unset NANBEIGE_LAUNCHER_FROZEN\n'
        'exec bash /mnt/c/projects/VPD/scripts/nanbeige/launch_logic_skill.sh\n')
    worker=(SCRIPTS/'prime_q4_worker.sh').read_text().replace('run_prime_q4_answers_remote.sh','run_logic_skill_remote.sh')
    save('logic_skill_worker.sh',worker)
    save('start_logic_skill_worker.py',(SCRIPTS/'start_prime_q4_worker.py').read_text().replace('prime_q4_worker.sh','logic_skill_worker.sh'))
    start=(SCRIPTS/'start_prime_q4_controller.py').read_text()
    start=start.replace('durable_wsl_','logic_durable_wsl_').replace('launch_prime_q4_durable_authorized.sh','launch_logic_skill_bounded.sh')
    start=start.replace('durable_controller_current.json','logic_controller_current.json')
    start=start.replace('max_hours=6','max_hours=5.5')
    save('start_logic_skill_controller.py',start)
    monitor=(SCRIPTS/'monitor_prime_q4.py').read_text()
    begin=monitor.index('def verify_finished(root):'); end=monitor.index('\n\ndef backup(',begin)
    monitor=monitor[:begin]+'''def verify_finished(root):
    run=root/'nanbeige_reasoning/logic_skill_comparison_outputs_v1'
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='logic_skill_generation_complete' and manifest['records']==259
    for name,h in manifest['files'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h,name
    rows=[json.loads(p.read_text()) for p in sorted(run.glob('[0-9][0-9][0-9].json'))]
    assert len(rows)==len({r['call_id'] for r in rows})==259
    assert json.loads((run/'cleanup.json').read_text())['server_stopped']
    return dict(status='all_259_outputs_retrieved_and_hash_verified',outputs=259)
'''+monitor[end:]
    monitor=monitor.replace("nanbeige_reasoning/prime_q4_runtime/contract.json", "nanbeige_reasoning/logic_skill_comparison_outputs_v1/contract.json")
    save('monitor_logic_skill.py',monitor)
    remote=(SCRIPTS/'run_prime_q4_answers_remote.sh').read_text()
    remote=remote[:remote.index('export HF_XET_HIGH_PERFORMANCE=1')]
    remote=remote.replace('-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_SERVER=ON')
    remote=remote.replace('--target llama ggml-cuda ggml-cpu','--target llama-server ggml-cuda ggml-cpu')
    remote+= '''cp tools/nanbeige/llama-build/bin/llama-server tools/nanbeige/runtime/
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
'''
    save('run_logic_skill_remote.sh',remote)
    protocol=json.loads((BASE/'logic_skill_comparison_v1/protocol.json').read_text())
    files=[ROOT/'experiments/nanbeige_reasoning/run_logic_skill_comparison.py']
    files.extend(SCRIPTS/name for name in ('run_logic_skill_remote.sh','start_logic_skill_worker.py','logic_skill_worker.sh'))
    files.extend(BASE/'logic_skill_comparison_v1'/name for name in ('calls.jsonl','protocol.json'))
    sources=BASE/'comparator_sources/qwen35_27b'
    files.append(sources/'model_lock.json')
    files.extend(sources/'official'/name for name in protocol['tokenizer_assets'])
    files.extend(sources/'official'/name for name in ('config.json','generation_config.json'))
    mapping={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    assert len(mapping)==len(files)
    manifest=BASE/'logic_bundle_manifest.json'
    manifest.write_text(json.dumps(dict(files=mapping),indent=2)+'\n')
    target=BASE/'logic_skill_bundle.tar.gz'
    with tarfile.open(target,'w:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        tar.add(manifest,arcname='bundle_manifest.json')
    with tarfile.open(target) as tar:
        for name,h in mapping.items():assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==h
    receipt=dict(sha256=sha(target),bytes=target.stat().st_size,files=mapping,
                 model_weights_included=False,grader_sidecar_included=False,
                 current_evaluation_reference_answers_included=False,
                 synthetic_training_hints_included=True)
    target.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='logic_cloud_bundle_verified',files=len(files),bytes=receipt['bytes'])))


if __name__=='__main__':main()
