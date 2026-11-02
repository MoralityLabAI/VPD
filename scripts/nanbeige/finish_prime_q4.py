"""Wait for a backed-up full cohort, then verify and grade locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.nanbeige.monitor_prime_q4 import verify_finished, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--native-cleanup-audit',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();root=Path(__file__).resolve().parents[2]
    log=run/'postprocess.log';status=run/'postprocess_status.json'
    started=time.monotonic()
    while not (run/'completion_verification.json').exists():
        if (run/'termination.json').exists():
            write(status,dict(status='stopped_without_complete_cohort',updated_unix=time.time()))
            return
        if time.monotonic()-started>22500:
            write(status,dict(status='postprocess_wait_limit_reached',updated_unix=time.time()))
            return
        write(status,dict(status='waiting_for_complete_local_backup',updated_unix=time.time()))
        time.sleep(30)
    received=run/'retrieved/artifacts'
    verify_finished(received)
    scientific=received/'nanbeige_reasoning'
    audit=run/'independent_verification'
    audit.mkdir(exist_ok=False)
    gate=scientific/'prime_q4_cache_gate_v1'
    shutil.copytree(gate,audit/'gate')
    def winpath(path):
        return subprocess.check_output(['wslpath','-w',str(path)],text=True).strip() if os.name=='posix' else str(path)
    python=root/'.venv/Scripts/python.exe'
    commands=[
        [str(python),'-m','scripts.nanbeige.verify_cached_steering',winpath(audit/'gate')],
        [str(python),'-m','experiments.nanbeige_reasoning.grade_q4_feature_answers',
         '--run',winpath(scientific/'prime_q4_feature_answers_v1'),
         '--context-check',winpath(scientific/'q4_answer_context_v1'),
         '--output',winpath(run/'grading'),
         '--native-cleanup-audit',winpath(args.native_cleanup_audit)],
    ]
    with log.open('ab') as stream:
        for command in commands:
            subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
    manifest=run/'grading/manifest.json'
    value=json.loads(manifest.read_text())
    assert value['status']=='q4_feature_answers_independently_decoded_and_graded' and value['records']==70
    for name,digest in value['files'].items():
        assert hashlib.sha256((run/'grading'/name).read_bytes()).hexdigest()==digest
    write(status,dict(status='independently_verified_and_graded',records=70,updated_unix=time.time(),
        grading_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),paper_update_pending=True))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        import sys
        if len(sys.argv)>1:
            target=Path(sys.argv[1]).resolve()
            if target.is_dir():
                write(target/'postprocess_status.json',dict(status='postprocess_failed',
                    error=f'{type(exc).__name__}: {exc}',updated_unix=time.time()))
        traceback.print_exc()
        raise
