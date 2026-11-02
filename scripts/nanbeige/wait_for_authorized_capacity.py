"""Bounded capacity retries for an already-authorized experiment.

Retries only confirmed no-offer exits before any pod creation attempt. A launch
that reaches creation is never retried, even if provisioning or inference fails.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['benchmark','qwen-pilot'])
    a=p.parse_args()
    if os.name!='posix':raise SystemExit('Run with WSL Python')
    root=Path(__file__).resolve().parents[2]
    base=root/'artifacts/nanbeige_reasoning'
    authorization=json.loads((base/'paid_launch_upload_review_block.json').read_text())
    if authorization['status']!='upload_authorized' or not authorization['compute_authorized']:
        raise SystemExit('Explicit compute and reviewed upload approval required')
    script={'benchmark':'launch_benchmark_runpod_ada.sh','qwen-pilot':'launch_qwen_pilot_runpod_a100.sh'}[a.mode]
    out=base/'capacity_waits'/f'{a.mode}-{time.time_ns()}'
    out.mkdir(parents=True)
    started=time.monotonic()
    attempts=[]

    def save(status,**extra):
        report=dict(status=status,mode=a.mode,updated_unix=time.time(),
                    capacity_wait_limit_seconds=3600,retry_interval_seconds=300,
                    attempts=attempts,**extra)
        (out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(status=status,mode=a.mode,output=str(out),**extra)),flush=True)

    save('capacity_wait_started')
    while time.monotonic()-started<3600:
        log=out/f'attempt_{len(attempts)+1:02d}.log'
        with log.open('w') as stream:
            child=subprocess.Popen(['bash',str(root/'scripts/nanbeige'/script)],cwd=root,
                                   stdout=stream,stderr=subprocess.STDOUT)
            save('launcher_running',launcher_pid=child.pid,log=str(log))
            code=child.wait()
        output=log.read_text(errors='replace')
        matches=re.findall(r'^Multipass launch directory: (.+)$',output,re.M)
        run=Path(matches[0]).resolve() if len(matches)==1 else None
        if run is not None and (not run.is_relative_to(base.resolve()) or not run.name.startswith('prime-')):
            raise RuntimeError('Unexpected launcher run path')
        attempted=run is not None and ((run/'create.log').exists() or (run/'pod_id.txt').exists())
        attempts.append(dict(run=str(run) if run else None,exit_code=code,creation_attempted=attempted,log=str(log)))
        if attempted or code==0:
            save('launch_finished' if code==0 else 'launch_needs_attention',exit_code=code)
            return
        no_offer=re.search(r'^No available US runpod (?:RTX6000Ada 48GB|A100 80GB) below the experimental hourly ceiling$',output,re.M)
        if not no_offer or run is None:
            save('unexpected_prelaunch_failure',exit_code=code)
            return
        remaining=3600-(time.monotonic()-started)
        if remaining<=0:break
        save('waiting_for_capacity',retry_in_seconds=min(300,remaining))
        time.sleep(min(300,remaining))
    save('capacity_wait_expired_without_launch')


if __name__=='__main__':main()
