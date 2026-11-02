"""Run the included twelve mechanics checks on the existing local model."""
import argparse
import json
from pathlib import Path
from .backend import NanbeigeBackend,save
from .capability_check import build_checks
from .ledger_eval import split_thought
from .io import digest


def grade_reply(check,reply):
    """Grade an unchanged completed response; also used for decode-only recovery."""
    if reply['finish_reason']=='length':raise ValueError('Generation truncated')
    _,text=split_thought(reply);text=text.strip()
    if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'):text=text[3:-3].strip()
    pred=json.loads(text);expected=check['expected_DO_NOT_SEND']
    valid=isinstance(pred,dict) and set(pred)==set(expected) and all(type(pred[k])==type(expected[k]) for k in expected)
    return dict(parsed=pred,valid=valid,correct=valid and pred==expected)


def reusable_checks(folder,checks,cap,device,transport='server'):
    if folder is None:return {}
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['checks']==checks and protocol['max_new_tokens']==cap and protocol['temperature']==.6
    assert protocol.get('transport','server')==transport,'A changed inference path requires a fresh capability gate'
    runtime=json.loads((folder/'runtime/runtime_contract.json').read_text(encoding='utf-8'))
    from .backend import Q4_SHA256
    assert runtime['model_sha256']==Q4_SHA256 and runtime['loops']==2
    assert runtime.get('execution_device','cuda')==device
    result={}
    for i,check in enumerate(checks):
        path=folder/f'question_{i:02d}.json'
        if not path.exists():continue
        row=json.loads(path.read_text(encoding='utf-8'))
        assert row['id']==check['id'] and row['messages']==check['messages']
        # Retain EVERY completed response, including wrong/censored answers.
        # Only a missing backend response may be retried after interruption.
        if 'response' in row:
            response=row['response']
            result[i]=dict(response=response,source=str(path.resolve()),sha256=digest(path))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-new-tokens',type=int,default=2048);p.add_argument('--wall-seconds',type=int,default=3600)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--reuse-checks',type=Path)
    p.add_argument('--transport',choices=['server','native'],default='native')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[];checks=build_checks()
    reused=reusable_checks(a.reuse_checks,checks,a.max_new_tokens,a.device,a.transport)
    save(a.output/'protocol.json',dict(checks=checks,max_new_tokens=a.max_new_tokens,temperature=.6,execution_device=a.device,cpu_threads=a.cpu_threads,transport=a.transport,
        clarification='physical_after counts delivered crates only; execute no later correction during the hypothetical command',
        reused_checks={i:{k:v for k,v in item.items() if k!='response'} for i,item in reused.items()},
        stop_rule='Stop when two checks fail, making the 11/12 gate impossible; preserve not-run counts'))
    from .native_backend import NativeNanbeigeBackend
    backend_type=NativeNanbeigeBackend if a.transport=='native' else NanbeigeBackend
    with backend_type(a.output/'runtime',context=max(8192,a.max_new_tokens+2048),wall_seconds=a.wall_seconds,
            gpu_layers='0' if a.device=='cpu' else 'all',cpu_threads=a.cpu_threads) as generate:
        for i,check in enumerate(checks):
            row=dict(check)
            try:
                if i in reused:
                    reply=reused[i]['response'];row['reused_from']={k:v for k,v in reused[i].items() if k!='response'}
                else:
                    reply=generate(messages=check['messages'],seed=709+i,metadata=dict(case_id=check['id'],phase='capability'),temperature=.6,max_new_tokens=a.max_new_tokens)
                row['response']=reply
                row.update(grade_reply(check,reply))
            except Exception as exc:row.update(valid=False,correct=False,error=f'{type(exc).__name__}: {exc}')
            interrupted='response' not in row
            row['outcome_status']='runtime_interrupted' if interrupted else 'correct' if row['correct'] else 'censored' if row['response'].get('finish_reason')=='length' else 'incorrect_or_invalid'
            rows.append(row);save(a.output/f'question_{i:02d}.json',row)
            impossible=sum(not r['correct'] for r in rows)>=2
            save(a.output/'summary.json',dict(assigned=12,completed=len(rows),not_run=12-len(rows),valid=sum(r['valid'] for r in rows),correct=sum(r['correct'] for r in rows),
                gate_passed=len(rows)==12 and sum(r['correct'] for r in rows)>=11,status='stopped_runtime_interruption' if interrupted else 'complete' if len(rows)==12 else 'stopped_gate_impossible' if impossible else 'running',
                model_responses=sum('response' in r for r in rows),reused_responses=sum('reused_from' in r for r in rows),runtime_interruptions=sum(r['outcome_status']=='runtime_interrupted' for r in rows)))
            print(check['id'],'INTERRUPTED' if interrupted else 'PASS' if row['correct'] else 'FAIL',flush=True)
            if interrupted or impossible:break


if __name__=='__main__':main()
