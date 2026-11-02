"""Run prepared matched Campsite calls through the existing cached Nanbeige backend."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from experiments.nanbeige_receipt_replay.io import ROOT,digest,save
from experiments.nanbeige_receipt_replay.native_backend import NativeNanbeigeBackend
from experiments.nanbeige_receipt_replay.backend import Q4_SHA256
from .skill_resume import SETTINGS,load_reusable_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,default=18000)
    parser.add_argument('--reuse-run',type=Path,help='Reuse every completed reply from a verified stopped run into a fresh output')
    args=parser.parse_args()
    cfg=json.loads((args.inputs/'protocol.json').read_text(encoding='utf-8'))
    assert cfg['model']=='Nanbeige4.2-3B' and cfg['thinking_enabled'] is False
    assert digest(args.inputs/'calls.jsonl')==cfg['calls_sha256']
    assert digest(args.inputs/'cases.json')==cfg['cases_sha256']
    for name,checksum in cfg['tokenizer_assets'].items():assert digest(ROOT/cfg['tokenizer_root']/name)==checksum
    calls=[json.loads(line) for line in (args.inputs/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(calls)==len({r['call_id'] for r in calls})==cfg['unique_model_calls']
    reused=load_reusable_run(args.reuse_run,args.inputs,calls,cfg,Q4_SHA256)
    args.output.mkdir(parents=True,exist_ok=False)
    contract=dict(model=cfg['model'],protocol_sha256=digest(args.inputs/'protocol.json'),
                  calls_sha256=digest(args.inputs/'calls.jsonl'),runner_sha256=digest(Path(__file__)),
                  expected_records=len(calls),generation_settings={k:cfg[k] for k in
                  SETTINGS},
                  reused_run=str(args.reuse_run.resolve()) if args.reuse_run else None,
                  reused_manifest_sha256=digest(args.reuse_run/'manifest.json') if args.reuse_run else None,
                  comparison='Matched user contents and output cap; native templates and compute differ')
    save(args.output/'contract.json',contract)
    completed=0
    try:
        manager=(NativeNanbeigeBackend(args.output/'runtime',context=cfg['context'],wall_seconds=args.wall_seconds,
                                   enable_thinking=False) if len(reused)<len(calls) else nullcontext(None))
        with manager as backend:
            # Check every prepared sequence before making any model call.
            for call in calls:
                if backend is None:break
                prompt=backend.render(call['messages'])
                assert prompt==call['rendered_prompt']
                assert backend.tokenizer.encode(prompt,add_special_tokens=False)==call['token_ids']
            for index,call in enumerate(calls):
                if index in reused:
                    item=reused[index];result=dict(item['row'])
                    result.update(contract_sha256=digest(args.output/'contract.json'),
                                  reused_from={k:v for k,v in item.items() if k!='row'})
                    save(args.output/f'{index:03d}.json',result)
                    completed+=1
                    save(args.output/'progress.json',dict(completed=completed,expected=len(calls),last_call_id=call['call_id'],reused=True))
                    continue
                reply=backend(messages=call['messages'],seed=cfg['seed'],temperature=cfg['temperature'],
                              max_new_tokens=cfg['max_new_tokens'],metadata=dict(call_id=call['call_id'],phase='matched_campsite'))
                assert reply['input_ids']==call['token_ids']
                result=dict(call_id=call['call_id'],id=call['id'],arm=call['arm'],
                    contract_sha256=digest(args.output/'contract.json'),
                    prompt_token_ids_sha256=hashlib.sha256(json.dumps(call['token_ids']).encode()).hexdigest(),
                    prompt_tokens=len(call['token_ids']),generated_tokens=len(reply['output_ids']),
                    generated_token_ids=reply['output_ids'],generated_text=reply['tokenizer_decoded_output'],
                    native_text=reply['raw_decoded_output'],max_token_censored=reply['finish_reason']=='length',
                    elapsed_seconds=reply['elapsed_seconds'],response=reply)
                save(args.output/f'{index:03d}.json',result)
                completed+=1
                save(args.output/'progress.json',dict(completed=completed,expected=len(calls),last_call_id=call['call_id']))
    finally:
        stopped=args.output/'runtime/server_stopped.json'
        cleanup=json.loads(stopped.read_text(encoding='utf-8')) if stopped.exists() else {}
        save(args.output/'cleanup.json',dict(server_stopped=cleanup.get('stopped',len(reused)==len(calls)),records=completed))
        save(args.output/'manifest.json',dict(status='logic_skill_generation_complete' if completed==len(calls) else 'interrupted',
            model=cfg['model'],records=completed,expected_records=len(calls),
            reused_records=min(completed,len(reused)),new_model_responses=max(0,completed-len(reused)),
            files={str(p.relative_to(args.output)).replace('\\','/'):digest(p) for p in args.output.rglob('*')
                   if p.is_file() and p.name!='manifest.json'}))


if __name__=='__main__':main()
