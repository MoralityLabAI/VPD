#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" =~ ^prime-[0-9]{8}T[0-9]{6}Z-[0-9]+$ ]] || exit 2
RUN=/mnt/c/projects/VPD/artifacts/nanbeige_reasoning/$1
read -r SSH_HOST SSH_PORT < "$RUN/ssh_target.txt"
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes \
  -o "UserKnownHostsFile=$RUN/known_hosts" -i /home/duganist/.ssh/id_rsa \
  -p "$SSH_PORT" "$SSH_HOST" python3 - <<'PY'
import json, pathlib, subprocess
root=pathlib.Path('/home/ubuntu/nanbeige-reasoning/artifacts/nanbeige_reasoning')
out={}
for name in ['prime_q4_cache_gate_v1/results.json', 'prime_q4_cache_gate_v1/verification.json',
             'q4_answer_context_v1/verification.json','prime_q4_feature_answers_v1/progress.json',
             'prime_q4_feature_answers_v1/cleanup.json','prime_q4_feature_answers_v1/interrupted.json']:
    p=root/name
    if p.exists():
        value=json.loads(p.read_text())
        if name.endswith('results.json'):
            value=[{k:r[k] for k in ['condition','allclose','greedy_tokens_equal','max_abs_logit_difference']} for r in value]
        out[name]=value
run=root/'prime_q4_feature_answers_v1'
rows=[json.loads(p.read_text()) for p in run.glob('[0-9][0-9]_*.json')]
out['answer_tokens_saved']=sum(len(r['generated']) for r in rows)
out['answers_complete']=sum(bool(r['complete']) for r in rows)
progress=out.get('prime_q4_feature_answers_v1/progress.json',{})
pid=progress.get('pid')
if isinstance(pid,int) and pid>0:
    proc=pathlib.Path('/proc')/str(pid)
    try:
        command=(proc/'cmdline').read_bytes().split(b'\0')
        stats=subprocess.run(['ps','-p',str(pid),'-o','pid=,etimes=,pcpu=,stat=,comm='],capture_output=True,text=True)
        out['inference_process']=dict(pid=pid,exists=stats.returncode==0,
            expected_module=b'experiments.nanbeige_reasoning.run_q4_feature_answers' in command,
            process_stats=stats.stdout.strip())
    except FileNotFoundError:
        out['inference_process']=dict(pid=pid,exists=False)
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True).strip()
print(json.dumps(out))
PY
