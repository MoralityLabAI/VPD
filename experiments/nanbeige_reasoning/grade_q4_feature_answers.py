"""Verify and grade complete local feature-steering cohorts, outside inference."""
import argparse
import json
from pathlib import Path
from .check_llama_steering import sha,read,write
from .compare_models import outcome_tables
from .scoring import score
from .run_q4_feature_answers import read_record
from .answer_decoding import AnswerDecoder,DECODING_POLICY
from .native_text_cleanup import verify_native_text,verify_cleanup_receipt


def validate_record(row, question, arm, contract, runtime, tokenizer, *, native_cleanup_verified=False):
    if not row['complete'] or row['id']!=question['id'] or row['condition']!=arm['name']:
        raise ValueError('Incomplete or mismatched condition')
    tokens=row['generated'];prefix=question['token_ids'];count=len(tokens)
    if not 1<=count<=contract['max_new_tokens'] or any(type(t) is not int or not 0<=t<runtime['vocab_size'] for t in tokens):
        raise ValueError('Invalid generated tokens')
    eos=set(runtime['eog_token_ids'])
    stopped=tokens[-1] in eos
    if any(t in eos for t in tokens[:-1]) or row['stopped_on_eog']!=stopped:
        raise ValueError('Invalid stop token accounting')
    capped=not stopped and count==contract['max_new_tokens']
    if not (stopped or capped) or row['max_token_censored']!=capped:
        raise ValueError('Incomplete or mislabeled stopping condition')
    processed=len(prefix)+count-1
    if row['prompt_tokens']!=len(prefix) or row['model_processed_tokens']!=processed or row['mutation_count']!=processed*len(arm['passes']):
        raise ValueError('Incorrect token or invocation accounting')
    decoded=tokenizer.decode(tokens,skip_special_tokens=False)
    if native_cleanup_verified:
        verify_native_text(decoded,row['generated_text'])
    elif decoded!=row['generated_text']:
        raise ValueError('C API text differs from independent tokenizer decoding')
    tail=tokenizer.decode(prefix[-8:],skip_special_tokens=False)
    return ('<think>' if '<think>' in tail and '</think>' not in tail else '')+decoded


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--context-check',type=Path,default=Path('artifacts/nanbeige_reasoning/q4_answer_context_v1'))
    parser.add_argument('--native-cleanup-audit',type=Path)
    args=parser.parse_args()
    if args.native_cleanup_audit:verify_cleanup_receipt(args.native_cleanup_audit)
    manifest=read(args.run/'manifest.json');contract=read(args.run/'contract.json')
    if manifest['status']!='q4_feature_answers_complete': raise ValueError('Full cohort must finish before comparative grading')
    for name,digest in manifest['files'].items():
        path=(args.run/name).resolve()
        if path.parent!=args.run.resolve() or sha(path)!=digest: raise ValueError('Generation source hash mismatch')
    root=Path('artifacts/nanbeige_reasoning/pilot_v1');qpath=root/'questions.jsonl';gpath=root/'graders.jsonl'
    assert sha(qpath)==contract['question_sha256'] and sha(args.run/'contract.json')==manifest['contract_sha256']
    questions=[json.loads(s) for s in qpath.read_text(encoding='utf-8').splitlines()]
    graders=[json.loads(s) for s in gpath.read_text(encoding='utf-8').splitlines()]
    q={r['id']:r for r in questions};g={r['id']:r for r in graders}
    assert len(q)==len(questions) and len(g)==len(graders) and set(q)==set(g)
    ids=contract['evaluation_ids'];arms=contract['conditions']
    assert len(ids)==len(set(ids)) and set(ids)=={r['id'] for r in questions if r['calibration_split']=='eval'}
    assert len({r['name'] for r in arms})==len(arms)
    assert manifest['completed']==manifest['expected']==len(ids)*len(arms)
    tokenizer_path=Path('models/Nanbeige4.2-3B-tokenizer')
    for name,digest in contract['tokenizer_files'].items(): assert sha(tokenizer_path/name)==digest
    if contract.get('schema')=='nanbeige.muon_feature_answers.v2':
        verification_path=args.context_check/'verification.json'
        assert sha(verification_path)==contract['context_check_verification_sha256']
        verification=read(verification_path)
        assert verification['decoding_policy']==DECODING_POLICY
        assert verification['decoder_sha256']==sha(Path(__file__).with_name('answer_decoding.py'))
        tokenizer=AnswerDecoder(tokenizer_path)
    else:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,use_fast=False,trust_remote_code=True)
    runtime=read(args.run/'runtime.json');pending=[]
    for i,qid in enumerate(ids):
        for arm in arms:
            name=f'{i:02d}_{arm["name"]}.json'
            assert name in manifest['files']
            row=read_record(args.run/name) if contract.get('schema')=='nanbeige.muon_feature_answers.v2' else read(args.run/name)
            assert row['contract_sha256']==sha(args.run/'contract.json')
            text=validate_record(row,q[qid],arm,contract,runtime,tokenizer,
                native_cleanup_verified=bool(args.native_cleanup_audit))
            pending.append((row,text))
    args.output.mkdir(parents=True,exist_ok=False)
    graded=[]
    for row,text in pending:
        raw=tokenizer.decode(row['generated'],skip_special_tokens=False)
        prefix=text[:-len(raw)] if raw else text
        native_grade=score(g[row['id']],prefix+row['generated_text'])
        graded.append(dict(id=row['id'],arm=row['condition'],grade=score(g[row['id']],text),
            native_render_grade=native_grade,native_render_differs=raw!=row['generated_text'],
            max_token_censored=row['max_token_censored'],generated_tokens=len(row['generated']),elapsed_seconds=row['elapsed_seconds']))
    write(args.output/'graded.json',graded)
    tables=outcome_tables(questions,graded,[a['name'] for a in arms])
    write(args.output/'tables.json',tables)
    write(args.output/'manifest.json',dict(status='q4_feature_answers_independently_decoded_and_graded',records=len(graded),
        generation_manifest_sha256=sha(args.run/'manifest.json'),graders_sha256=sha(gpath),
        native_cleanup_audit_sha256=sha(args.native_cleanup_audit) if args.native_cleanup_audit else None,
        primary_scoring_text='Original raw HF AnswerDecoder output, with prompt-open thinking delimiter restored',
        native_render_grade_disagreements=sum(r['grade']!=r['native_render_grade'] for r in graded),
        published_math_verifier_sha256=sha(Path('artifacts/nanbeige_reasoning/sources/research_environments/environments/math/i3_math/i3_math/verify.py')),
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['grade_q4_feature_answers.py','answer_decoding.py','native_text_cleanup.py','scoring.py','math_verify_stdin.py','compare_models.py','safe_expression.py']},
        files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()},
        limits=['Single feature/dose, post-hoc cohort; not a model-scale or routing result.','Complete ARC tasks are aggregated across every test input.']))
    print(json.dumps(dict(status='graded',records=len(graded))))


if __name__=='__main__': main()
