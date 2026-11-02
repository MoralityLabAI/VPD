"""Validate backed-up answer checkpoints without scoring an incomplete cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from experiments.nanbeige_reasoning.answer_decoding import AnswerDecoder
from experiments.nanbeige_reasoning.grade_q4_feature_answers import validate_record
from experiments.nanbeige_reasoning.run_q4_feature_answers import read_record
from experiments.nanbeige_reasoning.native_text_cleanup import verify_native_text,verify_cleanup_receipt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('cloud_run',type=Path)
    parser.add_argument('--native-cleanup-audit',type=Path,required=True);args=parser.parse_args()
    verify_cleanup_receipt(args.native_cleanup_audit)
    root=args.cloud_run/'retrieved/artifacts/nanbeige_reasoning'
    run=root/'prime_q4_feature_answers_v1';contract=read(run/'contract.json');runtime=read(run/'runtime.json')
    qpath=Path('artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl')
    assert sha(qpath)==contract['question_sha256']
    questions={r['id']:r for r in map(json.loads,qpath.read_text(encoding='utf-8').splitlines())}
    arms={r['name']:r for r in contract['conditions']};contract_sha=sha(run/'contract.json')
    context=root/'q4_answer_context_v1/verification.json'
    assert sha(context)==contract['context_check_verification_sha256']
    tokenizer=Path('models/Nanbeige4.2-3B-tokenizer')
    for name,digest in contract['tokenizer_files'].items():assert sha(tokenizer/name)==digest
    decoder=AnswerDecoder(tokenizer);records=[]
    for p in sorted(run.glob('[0-9][0-9]_*.json')):
        row=read_record(p);arm=arms[row['condition']];q=questions[row['id']]
        assert row['id'] in contract['evaluation_ids'] and row['contract_sha256']==contract_sha
        if row['complete']:
            validate_record(row,q,arm,contract,runtime,decoder,native_cleanup_verified=True)
        else:
            tokens=row['generated'];count=len(tokens)
            assert 0<count<contract['max_new_tokens']
            assert all(type(t) is int and 0<=t<runtime['vocab_size'] and t not in runtime['eog_token_ids'] for t in tokens)
            assert not row['stopped_on_eog'] and not row['max_token_censored']
            processed=len(q['token_ids'])+count-1
            assert row['prompt_tokens']==len(q['token_ids']) and row['model_processed_tokens']==processed
            assert row['mutation_count']==processed*len(arm['passes'])
            verify_native_text(decoder.decode(tokens),row['generated_text'])
        records.append(dict(file=p.name,payload_sha256=row['payload_sha256'],complete=row['complete'],tokens=len(row['generated'])))
    assert len({(r['file']) for r in records})==len(records)
    result=dict(status='live_backups_independently_verified_without_grading',verified_unix=time.time(),
        generation_contract_sha256=contract_sha,records=records,complete=sum(r['complete'] for r in records),
        tokens=sum(r['tokens'] for r in records),verifier_sha256=sha(Path(__file__)))
    result['native_cleanup_audit_sha256']=sha(args.native_cleanup_audit)
    out=args.cloud_run/'live_audits';out.mkdir(exist_ok=True)
    path=out/f'{time.time_ns()}.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],complete=result['complete'],records=len(records),tokens=result['tokens'],receipt=str(path))))

if __name__=='__main__':main()
