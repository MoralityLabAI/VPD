"""Finish one authorized experiment after its verified live generator exits.

This is a finite pipeline, not a recurring job. Each stage must succeed before
the next starts; inference remains independent from grading and SAE fitting.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil


def resume_command(command,run):
    module='experiments.nanbeige_reasoning.run_q4_reasoning'
    arguments=list(command[command.index(module):])
    output=Path(arguments[arguments.index('--output-dir')+1]).resolve()
    if output!=Path(run).resolve():
        raise ValueError('Observed generator output differs from the pipeline run')
    if '--max-wall-seconds' in arguments:
        arguments[arguments.index('--max-wall-seconds')+1]='14400'
    else:
        arguments += ['--max-wall-seconds','14400']
    return arguments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller-pid',type=int)
    p.add_argument('--controller-created',type=float)
    p.add_argument('--resume-completed-generation',action='store_true')
    p.add_argument('--start-at',choices=['grading','activation_replay','sae_fitting','artifact_verification'],default='grading')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--extended-run',type=Path)
    args=p.parse_args()
    if not args.resume_completed_generation and (args.controller_pid is None or args.controller_created is None):
        p.error('Provide the live generator identity or --resume-completed-generation')
    if args.start_at!='grading' and not args.resume_completed_generation:
        p.error('--start-at requires a completed generation run')
    state=args.run/'pipeline_state.json'
    def save(**value):
        state.write_text(json.dumps(dict(updated_unix=time.time(),**value),indent=2),encoding='utf-8')
    def run_stage(name,command):
        print('Starting',name,flush=True)
        with (args.run/f'pipeline_{name}.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8','-m',*command],stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            save(status='running_stage',stage=name,child_pid=child.pid,child_created=psutil.Process(child.pid).create_time())
            try:
                code=child.wait()
            except BaseException:
                child.terminate();child.wait(timeout=15);raise
            if code:raise RuntimeError(f'{name} exited {code}; inspect its stage log')
    try:
        if args.resume_completed_generation:
            manifest=json.loads((args.run/'manifest.json').read_text())
            save(status='resuming_completed_generation',start_at=args.start_at)
        else:
            process=psutil.Process(args.controller_pid)
            if process.create_time()!=args.controller_created or 'experiments.nanbeige_reasoning.run_q4_reasoning' not in process.cmdline():
                raise ValueError('Generator identity mismatch')
            resume=resume_command(process.cmdline(),args.run)
            save(status='waiting_for_live_generator',generator_pid=process.pid,generator_created=process.create_time())
            print(f'Waiting for verified live generator PID {process.pid}',flush=True)
            while True:
                try:exit_code=process.wait(timeout=60);break
                except psutil.TimeoutExpired:continue
            if exit_code!=0:
                raise RuntimeError(f'Original generator exited {exit_code}; inspect its failure before resuming')
            manifest=json.loads((args.run/'manifest.json').read_text())
            if manifest['status']=='paused_time_budget' and manifest['records']<manifest['expected_records']:
                # One continuation preserves the contract and saved greedy prefix.
                run_stage('resume_generation',resume)
                manifest=json.loads((args.run/'manifest.json').read_text())
        if manifest['status']!='generation_complete' or manifest['records']!=manifest['expected_records']:
            raise ValueError('Generator stopped without completing both conditions; resume generation first')
        def analysis_stages(run,prefix=''):
            capture=run/'q4_activations'
            return [(prefix+'grading',['experiments.nanbeige_reasoning.scoring',str(run)]),
                (prefix+'activation_replay',['experiments.nanbeige_reasoning.replay_q4_reasoning',str(run),'--output-dir',str(capture)]),
                (prefix+'sae_fitting',['experiments.nanbeige_reasoning.fit_q4_reasoning_saes',str(capture),'--output-dir',str(run/'q4_reasoning_saes')]),
                (prefix+'artifact_verification',['experiments.nanbeige_reasoning.verify_q4_reasoning',str(run)])]
        # Complete the initial budget's mechanistic measurements before the
        # substantially longer follow-up; both budgets keep separate artifacts.
        stages=analysis_stages(args.run)
        stages=stages[next(i for i,(name,_) in enumerate(stages) if name==args.start_at):]
        if args.extended_run:
            selected=args.extended_run
            stages += [('extended_generation',['experiments.nanbeige_reasoning.run_q4_reasoning',
                '--output-dir',str(selected),'--context','10240','--max-new-tokens','8192',
                '--max-wall-seconds','43200'])]
            stages += analysis_stages(selected,'extended_')
        for name,command in stages:
            run_stage(name,command)
            if name=='extended_generation':
                extended=json.loads((selected/'manifest.json').read_text())
                if extended['status']!='generation_complete' or extended['records']!=extended['expected_records']:
                    raise RuntimeError('Extended generation is incomplete; resume before downstream analysis')
        save(status='q4_pilot_pipeline_complete',scope='Paired Q4 pilot, same-model activation replay and shared SAEs; broader benchmark and VPD remain pending')
        print('Completed Q4 pilot pipeline',flush=True)
    except BaseException as exc:
        save(status='pipeline_needs_attention',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
