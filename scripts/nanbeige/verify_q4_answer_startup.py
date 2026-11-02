"""Verify a saved startup checkpoint without scoring or changing a running experiment."""
import argparse
import json
from pathlib import Path
from experiments.nanbeige_reasoning.answer_decoding import AnswerDecoder,DECODING_POLICY
from experiments.nanbeige_reasoning.check_llama_steering import read,sha,write
from experiments.nanbeige_reasoning.run_q4_feature_answers import read_record


def main():
    root=Path('artifacts/nanbeige_reasoning')
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=root/'q4_feature_answers_v1')
    parser.add_argument('--context-check',type=Path,default=root/'q4_answer_context_v1')
    parser.add_argument('--output',type=Path,default=root/'q4_feature_answers_startup_verification.json')
    args=parser.parse_args();run=args.run
    contract=read(run/'contract.json');row=read_record(run/'00_baseline.json')
    assert row['contract_sha256']==sha(run/'contract.json')
    qpath=root/'pilot_v1/questions.jsonl';assert sha(qpath)==contract['question_sha256']
    question=next(json.loads(s) for s in qpath.read_text(encoding='utf-8').splitlines() if json.loads(s)['id']==row['id'])
    assert row['condition']=='baseline' and question['calibration_split']=='eval'
    tokens=row['generated'];runtime=read(run/'runtime.json');eog=set(runtime['eog_token_ids'])
    assert 1<=len(tokens)<=contract['max_new_tokens']
    assert all(type(t) is int and 0<=t<runtime['vocab_size'] for t in tokens)
    assert not any(t in eog for t in tokens[:-1])
    stopped=tokens[-1] in eog;capped=len(tokens)==contract['max_new_tokens'] and not stopped
    assert row['complete']==(stopped or capped) and row['stopped_on_eog']==stopped and row['max_token_censored']==capped
    assert row['prompt_tokens']==len(question['token_ids'])
    assert row['model_processed_tokens']==len(question['token_ids'])+len(tokens)-1 and row['mutation_count']==0
    verification_path=args.context_check/'verification.json'
    assert sha(verification_path)==contract['context_check_verification_sha256']
    assert read(verification_path)['decoding_policy']==DECODING_POLICY
    tokenizer_path=Path('models/Nanbeige4.2-3B-tokenizer')
    for name,digest in contract['tokenizer_files'].items():assert sha(tokenizer_path/name)==digest
    assert AnswerDecoder(tokenizer_path).decode(tokens)==row['generated_text']
    receipt=dict(status='answer_generation_startup_checkpoint_verified',question_id=row['id'],condition=row['condition'],
        generated_tokens=len(tokens),complete=row['complete'],source_payload_sha256=row['payload_sha256'],
        generation_contract_sha256=sha(run/'contract.json'),verifier_sha256=sha(Path(__file__)),
        scope='Checkpoint hash, stop accounting, processed tokens, mutation count and independent text decoding; no task grading.')
    write(args.output,receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
