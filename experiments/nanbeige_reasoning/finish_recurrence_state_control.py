"""Finish analysis after the exact observed CPU worker exits; never restart inference."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import psutil
from .intervention_change_fidelity import read,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True);p.add_argument('--birth',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path('artifacts/nanbeige_reasoning');run=root/'recurrence_state_control_v1'
    out=a.output;out.mkdir(exist_ok=False)
    write(out/'status.json',dict(status='waiting_for_observed_worker',pid=a.pid,birth=a.birth))
    start=time.monotonic()
    while psutil.pid_exists(a.pid):
        process=psutil.Process(a.pid)
        if process.create_time()!=a.birth:break
        if time.monotonic()-start>8100:raise TimeoutError('Observed worker did not exit within bounded wait')
        time.sleep(2)
    manifest=read(run/'manifest.json')
    if manifest['status']!='completed':
        write(out/'status.json',dict(status='stopped_incomplete_inference',pid=a.pid,birth=a.birth));return
    for module,args in [
        ('verify_recurrence_state_control',['--run',str(run),'--output',str(root/'recurrence_state_validation_v1')]),
        ('analyze_recurrence_state_control',['--run',str(run),'--output',str(root/'recurrence_state_analysis_v1')])]:
        write(out/'status.json',dict(status='running',stage=module,pid=a.pid,birth=a.birth))
        with (out/f'{module}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-m',f'experiments.nanbeige_reasoning.{module}',*args],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write(out/'status.json',dict(status='failed',stage=module,exit_code=result.returncode));return
    write(out/'status.json',dict(status='completed',new_inference_calls=0))


if __name__=='__main__':main()
