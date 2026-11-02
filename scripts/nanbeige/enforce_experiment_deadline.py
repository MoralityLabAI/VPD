"""Enforce the user's finite cutoff on explicitly identified local experiments."""
import argparse
import datetime
import json
from pathlib import Path
import time

import psutil


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def identified(row):
    try:
        process=psutil.Process(row['pid'])
        if abs(process.create_time()-row['created'])>.01:return None
        if row['module'] not in process.cmdline():return None
        return process
    except psutil.NoSuchProcess:return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    args=p.parse_args()
    contract=json.loads(args.contract.read_text(encoding='utf-8'))
    output=args.contract.parent
    targets=contract['targets']
    allowed={'experiments.nanbeige_reasoning.finish_q4_pipeline',
             'experiments.nanbeige_reasoning.control_sae_loss_balance'}
    if any(r['module'] not in allowed for r in targets):raise ValueError('Unrecognized target module')
    deadline=datetime.datetime.fromisoformat(contract['compute_cutoff_utc']).timestamp()
    write(output/'state.json',dict(status='watching_identified_experiments',cutoff_utc=contract['compute_cutoff_utc']))
    while time.time()<deadline:
        if not any(identified(r) is not None for r in targets):
            write(output/'state.json',dict(status='all_identified_experiments_finished_before_cutoff'))
            print('Identified experiments finished before cutoff',flush=True)
            return
        time.sleep(min(5,max(0,deadline-time.time())))
    owned={}
    for row in targets:
        root=identified(row)
        if root is None:continue
        for process in [root,*root.children(recursive=True)]:
            try:owned[process.pid]=dict(pid=process.pid,created=process.create_time(),name=process.name())
            except psutil.NoSuchProcess:pass
    stopped=[]
    # Parents first prevent the finite pipelines from starting another stage.
    for row in owned.values():
        try:
            process=psutil.Process(row['pid'])
            if abs(process.create_time()-row['created'])>.01:continue
            process.terminate()
            stopped.append(dict(**row,termination_requested=True))
        except psutil.NoSuchProcess:pass
    remaining=[]
    for row in stopped:
        try:
            process=psutil.Process(row['pid'])
            if abs(process.create_time()-row['created'])>.01:continue
            process.wait(timeout=10)
        except psutil.NoSuchProcess:pass
        except psutil.TimeoutExpired:remaining.append(row['pid'])
    write(output/'state.json',dict(status='user_deadline_enforced' if not remaining else 'termination_needs_attention',
        stopped=stopped,remaining_pids=remaining,cutoff_utc=contract['compute_cutoff_utc'],
        note='Incomplete artifacts are deadline-censored, not successful completed experiments.'))
    print(json.dumps(dict(status='deadline_processed',stopped=len(stopped),remaining=remaining)),flush=True)


if __name__=='__main__':main()
