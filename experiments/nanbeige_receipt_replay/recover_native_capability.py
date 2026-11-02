"""Recover ALL completed native responses after a text-adapter failure, without inference."""
import argparse
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .capability_check import build_checks
from .io import ROOT,digest,save
from .native_backend import NativeNanbeigeBackend
from .backend import Q4_SHA256
from .run_capability import grade_reply
from .native_text import verify_native_text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.source/'protocol.json').read_text(encoding='utf-8'))
    checks=build_checks()
    assert protocol['checks']==checks and protocol['transport']=='native'
    runtime=args.source/'runtime'
    assert json.loads((runtime/'server_stopped.json').read_text(encoding='utf-8'))['stopped']
    assert json.loads((runtime/'worker_stopped.json').read_text(encoding='utf-8'))['stopped']
    execution=json.loads((runtime/'runtime_contract.json').read_text(encoding='utf-8'))
    assert execution['model_sha256']==Q4_SHA256 and execution['loops']==2 and execution['enable_thinking'] is True
    decoder=Tokenizer.from_file(str(ROOT/'models/Nanbeige4.2-3B-tokenizer/tokenizer.json'))
    selected=[]
    for index,check in enumerate(checks):
        name=f'{index+1:05d}.json';result_path=runtime/'results'/name
        if not result_path.exists():continue
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if not result['completed']:continue  # Partial sequences remain interruptions.
        request_path=runtime/'requests'/name
        request=json.loads(request_path.read_text(encoding='utf-8'))
        assert result['request_sha256']==digest(request_path) and not result.get('error')
        assert request['messages']==check['messages'] and request['seed']==709+index
        assert request['temperature']==protocol['temperature'] and request['max_new_tokens']==protocol['max_new_tokens']
        verify_native_text(decoder.decode(result['output_ids'],skip_special_tokens=False),result['raw_decoded_output'])
        selected.append((index,check,request_path,result_path,request))
    if not selected:raise ValueError('No completed native responses available for recovery')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.source/'protocol.json',args.output/'protocol.json')
    backend=NativeNanbeigeBackend(args.output/'runtime',context=execution['context'])
    # Do not enter the context manager: only consume already completed local replies.
    assert backend.process is None
    backend.tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Nanbeige4.2-3B-tokenizer',use_fast=False,local_files_only=True)
    backend.output_decoder=decoder
    rows=[];provenance=[]
    for index,check,request_path,result_path,request in selected:
        dest=args.output/'runtime/results'/result_path.name;dest.parent.mkdir(exist_ok=True)
        shutil.copy2(result_path,dest)
        assert backend.render(check['messages'])==request['rendered_prompt']
        assert backend.tokenizer.encode(request['rendered_prompt'],add_special_tokens=False)==request['input_ids']
        backend.calls=index
        reply=backend(messages=check['messages'],seed=request['seed'],metadata=request['metadata'],
                      temperature=request['temperature'],max_new_tokens=request['max_new_tokens'])
        row=dict(check,response=reply)
        try:row.update(grade_reply(check,reply))
        except Exception as exc:row.update(valid=False,correct=False,error=f'{type(exc).__name__}: {exc}')
        row['outcome_status']='correct' if row['correct'] else 'censored' if reply['finish_reason']=='length' else 'incorrect_or_invalid'
        row['recovery']=dict(source_result=str(result_path.resolve()),sha256=digest(result_path),new_model_calls=0)
        save(args.output/f'question_{index:02d}.json',row);rows.append(row)
        provenance.append(dict(question=index,request_sha256=digest(request_path),result_sha256=digest(result_path)))
    shutil.copy2(runtime/'runtime_contract.json',args.output/'runtime/runtime_contract.json')
    save(args.output/'summary.json',dict(assigned=12,completed=len(rows),not_run=12-len(rows),
        valid=sum(r['valid'] for r in rows),correct=sum(r['correct'] for r in rows),
        gate_passed=len(rows)==12 and sum(r['correct'] for r in rows)>=11,
        status='completed_native_responses_recovered',model_responses=len(rows),runtime_interruptions=0,
        new_model_calls=0,source=str(args.source.resolve())))
    save(args.output/'recovery_receipt.json',dict(source=str(args.source.resolve()),new_model_calls=0,
        recovered_completed_responses=len(rows),all_completed_responses_retained=True,source_records=provenance,
        source_runtime_contract_sha256=digest(runtime/'runtime_contract.json'),decoder_sha256=digest(ROOT/'models/Nanbeige4.2-3B-tokenizer/tokenizer.json'),
        code_sha256=digest(Path(__file__)),scope='Decode-only recovery of completed native tokens; original answers and grades are not selected or regenerated'))
    print(json.dumps(dict(recovered=len(rows),correct=sum(r['correct'] for r in rows),model_calls=0)))


if __name__=='__main__':main()
