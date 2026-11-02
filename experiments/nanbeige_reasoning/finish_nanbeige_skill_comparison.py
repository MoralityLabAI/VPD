"""Follow one live benchmark, optionally resume a wall stop once, then grade and pair."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

from experiments.nanbeige_receipt_replay.io import ROOT,read_json,save,digest


def wall_resume_allowed(run):
    """Never restart a thermal, sensor, decoding or unknown runtime failure."""
    manifest=read_json(run/'manifest.json')
    if manifest['status']!='interrupted' or manifest['records']<1:return False
    parent=read_json(run/'runtime/server_stopped.json');worker=read_json(run/'runtime/worker_stopped.json')
    return (parent['stopped'] and worker['stopped'] and parent.get('guard_error') in (
        'RuntimeError: RuntimeError: Guard wall budget reached',
        'RuntimeError: Guard wall budget reached',
        'TimeoutError: Worker wall budget expired'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','run','resume-output','grading','comparison','output',
                 'qwen-preparation','qwen-run','qwen-grading'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source=a.run.resolve();candidates=[]
    for process in psutil.process_iter(['pid','ppid','cmdline']):
        args=process.info['cmdline'] or []
        if ('experiments.nanbeige_reasoning.run_nanbeige_skill_comparison' in args and '--output' in args
                and Path(args[args.index('--output')+1]).resolve()==source):candidates.append(process)
    parents={proc.ppid() for proc in candidates};leaves=[proc for proc in candidates if proc.pid not in parents]
    if len(leaves)!=1:raise RuntimeError('Expected one currently live benchmark process')
    process=leaves[0];birth=process.create_time()
    for path in (a.resume_output,a.grading,a.comparison,a.output):
        if path.exists():raise FileExistsError(f'Use a fresh output: {path}')
    a.output.mkdir(parents=True)
    state=dict(status='waiting_on_live_generation',source_pid=process.pid,source_created=birth,
        source=str(source),stages=[],paid_provider=False,max_wall_resumes=1,code_sha256=digest(Path(__file__)))
    def status(value):state['status']=value;save(a.output/'status.json',state)
    def run(module,*args):
        command=[sys.executable,'-m','experiments.nanbeige_reasoning.'+module,*map(str,args)]
        entry=dict(module=module,command=command,started=time.time());state['stages'].append(entry)
        status('running_'+module)
        log=a.output/f'{len(state["stages"]):02d}_{module}.log'
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONIOENCODING='utf-8'),creationflags=subprocess.CREATE_NO_WINDOW)
        entry.update(returncode=result.returncode,finished=time.time(),log=str(log),log_sha256=digest(log))
        if result.returncode:raise RuntimeError(f'{module} exited {result.returncode}; inspect {log}')
    status('waiting_on_live_generation')
    try:
        deadline=time.monotonic()+18600
        while True:
            try:
                if process.create_time()!=birth:raise RuntimeError('Source PID was reused')
                process.wait(timeout=10);break
            except psutil.NoSuchProcess:break
            except psutil.TimeoutExpired:
                if time.monotonic()>deadline:raise TimeoutError('Source wait budget expired; no restart performed')
        manifest=read_json(source/'manifest.json');state['source_manifest_sha256']=digest(source/'manifest.json')
        if manifest['status']!='logic_skill_generation_complete':
            if not wall_resume_allowed(source):status('stopped_interrupted_generation_requires_inspection');return
            # Runner verifies every completed reply and refuses unrecorded completed outputs.
            run('run_nanbeige_skill_comparison','--inputs',a.inputs,'--output',a.resume_output,
                '--reuse-run',source,'--wall-seconds',18000)
            source=a.resume_output.resolve()
        if read_json(source/'manifest.json')['status']!='logic_skill_generation_complete':
            status('stopped_generation_incomplete');return
        state['completed_run']=str(source)
        run('grade_logic_skill_comparison','--preparation',a.inputs,'--run',source,'--output',a.grading)
        run('compare_skill_models','--qwen-preparation',a.qwen_preparation,'--qwen-run',a.qwen_run,
            '--qwen-grading',a.qwen_grading,'--nanbeige-preparation',a.inputs,'--nanbeige-run',source,
            '--nanbeige-grading',a.grading,'--output',a.comparison)
        state['comparison_verification_sha256']=digest(a.comparison/'verification.json')
        status('matched_campsite_comparison_complete')
    except Exception as exc:
        state['error']=f'{type(exc).__name__}: {exc}';status('stopped_error');raise


if __name__=='__main__':main()
