"""Wait on a verified capability process, preserve completed replies, and continue once."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from .io import ROOT,save,read_json


def read(path):return read_json(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','capability','pilot','pipeline','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recovered',type=Path)
    p.add_argument('--capability-wall-seconds',type=int,default=7200)
    p.add_argument('--resume-completed-directly',action='store_true')
    p.add_argument('--stop-after-current',action='store_true')
    a=p.parse_args();source=a.source.resolve()
    if a.capability_wall_seconds<=0:p.error('--capability-wall-seconds must be positive')
    if not a.resume_completed_directly and a.recovered is None:p.error('--recovered is required for decode-only recovery')
    if a.stop_after_current and not a.resume_completed_directly:p.error('--stop-after-current requires --resume-completed-directly')
    candidates=[]
    for process in psutil.process_iter(['pid','ppid','cmdline']):
        try:
            args=process.info['cmdline'] or []
            if ('experiments.nanbeige_receipt_replay.run_capability' in args and '--output' in args
                    and Path(args[args.index('--output')+1]).resolve()==source):candidates.append(process)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    parents={r.ppid() for r in candidates}
    leaves=[r for r in candidates if r.pid not in parents]
    if len(leaves)!=1:raise RuntimeError('Expected one currently live source capability worker')
    process=leaves[0];birth=process.create_time()
    for path in (a.recovered,a.capability,a.pipeline,a.output):
        if path is not None and path.exists():raise FileExistsError(f'Use a fresh continuation path: {path}')
    a.output.mkdir(parents=True)
    state=dict(status='waiting_on_live_capability',source_pid=process.pid,source_created=birth,
               source=str(source),stages=[],model_calls_by_recovery=0,
               capability_wall_seconds=a.capability_wall_seconds)
    def status(value):state['status']=value;save(a.output/'status.json',state)
    def run(module,*args):
        command=[sys.executable,'-m','experiments.nanbeige_receipt_replay.'+module,*map(str,args)]
        state['stages'].append(dict(module=module,command=command,started=time.time()));status('running_'+module)
        result=subprocess.run(command,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
        state['stages'][-1].update(returncode=result.returncode,finished=time.time())
        if result.returncode:raise RuntimeError(f'{module} returned {result.returncode}')
    status('waiting_on_live_capability')
    try:
        if a.stop_after_current:
            if not process.is_running() or process.create_time()!=birth or (source/'runtime/server_stopped.json').exists():
                raise RuntimeError('Source is no longer verified live before boundary request')
            boundary=source/'runtime/stop_requested.json'
            if boundary.exists():raise RuntimeError('A source stop was already requested')
            # Native worker reads this only between requests, never inside a generation.
            save(boundary,dict(stop=True,reason='Finish current response, then reuse all completed answers with adaptive thermal pacing'))
            state['stop_after_current_requested']=True;status('waiting_for_response_boundary')
        deadline=time.monotonic()+7500
        while True:
            try:
                if process.create_time()!=birth:raise RuntimeError('Source PID was reused')
                process.wait(timeout=10);break
            except psutil.NoSuchProcess:break
            except psutil.TimeoutExpired:
                if time.monotonic()>deadline:raise TimeoutError('Source process wait budget expired')
        stopped=source/'runtime/server_stopped.json'
        worker=source/'runtime/worker_stopped.json'
        if not stopped.exists() or not worker.exists() or not read(stopped)['stopped'] or not read(worker)['stopped']:
            raise RuntimeError('Source exited without verified worker cleanup; no restart performed')
        summary=read(source/'summary.json');state['source_summary']=summary
        if summary['gate_passed']:
            status('source_gate_passed_existing_pipeline_continues');return
        native_results=list((source/'runtime/results').glob('*.json'))
        complete=[f for f in native_results if read(f).get('completed')]
        if not complete:status('stopped_no_completed_native_response');return
        if a.resume_completed_directly:
            if any(not read(f).get('completed') or read(f).get('error') for f in native_results):
                status('stopped_incomplete_native_response');return
            rows=[read(f) for f in source.glob('question_*.json') if 'response' in read(f)]
            represented={(ROOT/r['response']['call_record']).resolve() for r in rows}
            if any(f.resolve() not in represented for f in complete):
                status('stopped_unrecorded_completed_response_requires_recovery');return
            recovered=dict(model_responses=len(rows),correct=sum(r['correct'] for r in rows))
            reuse_source=source;state['reused_completed_summary']=recovered
        else:
            run('recover_native_capability','--source',source,'--output',a.recovered)
            recovered=read(a.recovered/'summary.json');state['recovered_summary']=recovered
            reuse_source=a.recovered
        if recovered['model_responses']-recovered['correct']>=2:
            status('stopped_gate_impossible_preserved_answers');return
        run('run_capability','--output',a.capability,'--transport','native','--max-new-tokens',4096,
            '--wall-seconds',a.capability_wall_seconds,'--reuse-checks',reuse_source)
        if not read(a.capability/'summary.json')['gate_passed']:
            status('stopped_capability_not_passed');return
        run('run_pipeline','--capability',a.capability,'--pilot',a.pilot,'--output',a.pipeline,'--transport','native')
        state['pipeline_status']=read(a.pipeline/'status.json')['status']
        status('finished_'+state['pipeline_status'])
    except Exception as exc:
        state['error']=f'{type(exc).__name__}: {exc}';status('stopped_error');raise


if __name__=='__main__':main()
