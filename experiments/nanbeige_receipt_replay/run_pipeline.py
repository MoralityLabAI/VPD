"""Bounded local execution of the already-authorized stages, with persisted gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from .io import ROOT,digest,save,read_json


def read(path):return read_json(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--capability',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-seconds',type=int,default=7500);p.add_argument('--pilot-wall-seconds',type=int,default=10800)
    p.add_argument('--transport',choices=['server','native'],default='native')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting_for_capability',stages=[],paid_provider=False,source_sha256=digest(Path(__file__)))
    def update(status):state['status']=status;save(a.output/'status.json',state)
    def run(module,*args):
        command=[sys.executable,'-m','experiments.nanbeige_receipt_replay.'+module,*map(str,args)]
        state['stages'].append(dict(module=module,command=command,started_unix=time.time()))
        update('running_'+module)
        result=subprocess.run(command,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
        state['stages'][-1].update(returncode=result.returncode,finished_unix=time.time())
        if result.returncode:raise RuntimeError(f'{module} exited {result.returncode}')
    update('waiting_for_capability');deadline=time.monotonic()+a.wait_seconds
    try:
        while True:
            path=a.capability/'summary.json';stopped=a.capability/'runtime/server_stopped.json'
            gate=read(path) if path.exists() else None
            if stopped.exists():
                if gate is None or not gate.get('gate_passed'):
                    state['capability']=gate;update('stopped_capability_not_passed');return
                break
            if time.monotonic()>=deadline:raise TimeoutError('Capability wait budget reached')
            time.sleep(10)
        state['capability']=gate;state['capability_sha256']=digest(path)
        run('run_pilot','--output',a.pilot,'--capability',path,'--worlds-per-split',1,
            '--max-new-tokens',4096,'--context',12288,'--wall-seconds',a.pilot_wall_seconds,'--transport',a.transport)
        pilot=read(a.pilot/'summary.json');state['pilot']=pilot
        run('validate_episodes','--episodes',a.pilot/'episodes','--output',a.output/'episode_replay.json')
        if pilot['status']!='complete':update('stopped_pilot_'+pilot['status']);return
        run('prepare_captures','--episodes',a.pilot/'episodes','--output',a.output/'capture_inputs.json')
        run('capture_decisions','--inputs',a.output/'capture_inputs.json','--output',a.output/'captures','--wall-seconds',3600)
        run('probes','--captures',a.output/'captures','--output',a.output/'probes')
        run('render_probes','--probes',a.output/'probes','--output',a.output/'figures')
        run('sae_analysis','--captures',a.output/'captures','--output',a.output/'sae_quality')
        candidates=read(a.output/'probes/intervention_candidates.json')
        state['interventions']=dict(status='pending_pretrained_followup' if candidates['candidates'] else 'not_triggered',
            reason=candidates['status'],trained_or_modified_model=False)
        update('observational_analysis_complete')
    except Exception as exc:
        state['error']=f'{type(exc).__name__}: {exc}';update('stopped_error');raise


if __name__=='__main__':main()
