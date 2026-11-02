"""Run checkpointed answer blocks; resume only clean wall-limit stops, never thermal faults."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.nanbeige_reasoning.run_q4_feature_answers import atomic,read_record
from experiments.nanbeige_reasoning.temperature_guard_burst import gpu_temperature


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_tokens(output):
    return sum(len(read_record(p)['generated']) for p in output.glob('[0-9][0-9]_*.json'))


def assess_stop(output,returncode,previous_tokens):
    cleanup=read(output/'cleanup.json')
    if not cleanup['model_closed'] or not cleanup['context_closed']:
        raise RuntimeError('Model cleanup was not confirmed')
    manifest=read(output/'manifest.json')
    if manifest['status']=='q4_feature_answers_complete' and returncode==0:return 'complete'
    if manifest['status']!='q4_feature_answers_partial':raise RuntimeError('Unexpected answer manifest')
    if returncode:
        error=read(output/'interrupted.json')['error']
        if error!='RuntimeError: Guard wall budget reached':
            raise RuntimeError('Automatic continuation stopped: '+error)
    elif not manifest['wall_budget_reached']:
        raise RuntimeError('Partial run did not stop at its wall limit')
    if saved_tokens(output)<=previous_tokens:
        raise RuntimeError('Wall-limited attempt added no saved tokens; automatic continuation stopped')
    return 'resume'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-attempts',type=int,default=0,help='Zero permits as many progressing blocks as needed.')
    args=parser.parse_args()
    root=Path('artifacts/nanbeige_reasoning')
    output=args.output; receipts=output/'supervision';receipts.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-X','utf8','-m','experiments.nanbeige_reasoning.run_q4_feature_answers',
        '--gate',str(root/'q4_cached_steering_burst_v1'),'--context-check',str(root/'q4_answer_context_v1'),
        '--output',str(output),'--wall-seconds','7200','--max-new-tokens','8192']
    policy=dict(command=command,max_attempts=args.max_attempts,supervisor_sha256=sha(Path(__file__)),
        automatic_resume='Only confirmed cleanup plus wall-limit termination with increased saved token count.',
        thermal_fault_policy='Stop and require review; never automatically retry a sensor, cooling or hard-stop failure.')
    atomic(receipts/'policy.json',policy)
    attempt=0;child=None
    try:
        while not args.max_attempts or attempt<args.max_attempts:
            if (receipts/'PAUSE').exists():
                atomic(receipts/'status.json',dict(status='paused_before_next_attempt',attempts=attempt));return
            started=time.monotonic()
            while gpu_temperature()>78:
                if time.monotonic()-started>180:raise RuntimeError('GPU did not cool to 78 C before launch')
                time.sleep(2)
            previous=saved_tokens(output)
            attempt+=1
            folder=receipts/f'attempt-{time.time_ns()}'
            folder.mkdir()
            with (folder/'stdout.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                atomic(receipts/'status.json',dict(status='running',attempt=attempt,child_pid=child.pid,
                    started_unix=time.time(),attempt_directory=str(folder),saved_tokens_before=previous))
                returncode=child.wait();child=None
            for name in ['manifest.json','contract.json','cleanup.json','interrupted.json','progress.json']:
                if (output/name).exists():shutil.copyfile(output/name,folder/name)
            atomic(folder/'receipt.json',dict(returncode=returncode,
                files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
            action=assess_stop(output,returncode,previous)
            atomic(receipts/'status.json',dict(status='complete' if action=='complete' else 'wall_limit_checkpointed',
                attempt=attempt,returncode=returncode,saved_tokens=saved_tokens(output),attempt_directory=str(folder)))
            if action=='complete':
                grading=output.with_name(output.name+'_grading')
                subprocess.run([sys.executable,'-X','utf8','-m','experiments.nanbeige_reasoning.grade_q4_feature_answers',
                    '--run',str(output),'--output',str(grading)],check=True)
                atomic(receipts/'status.json',dict(status='complete_and_independently_graded',attempts=attempt,
                    generation_manifest_sha256=sha(output/'manifest.json'),grading_manifest_sha256=sha(grading/'manifest.json')))
                return
        atomic(receipts/'status.json',dict(status='requested_attempt_limit_reached',attempts=attempt,saved_tokens=saved_tokens(output)))
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=30)
        atomic(receipts/'status.json',dict(status='stopped_requires_review',attempts=attempt,error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':main()
