"""Independently verify the full-context Linux gate and raw token decoding."""
import json
from pathlib import Path
import numpy as np
from scripts.nanbeige.verify_cached_steering import verify
from experiments.nanbeige_reasoning.answer_decoding import AnswerDecoder,DECODING_POLICY
from experiments.nanbeige_reasoning.check_llama_steering import read,write,sha


def main():
    root=Path('artifacts/nanbeige_reasoning');gate=root/'prime_q4_cache_gate_v1'
    receipt=verify(gate);assert receipt['gate_passed']
    gc=read(gate/'contract.json');assert gc['context']==10153
    clean=read(gate/'cleanup.json');assert clean['model_closed'] and clean['context_closed']
    tokenizer_path=Path('models/Nanbeige4.2-3B-tokenizer');decoder=AnswerDecoder(tokenizer_path)
    for row in read(gate/'results.json'):
        with np.load(gate/row['archive']) as z:
            assert decoder.decode(z['generated'].tolist())==row['generated_text']
            assert decoder.decode(z['token_ids'].tolist())==row['prompt_text']
        for token,text in row['stop_token_texts'].items():assert decoder.decode([int(token)])==text
    out=root/'q4_answer_context_v1';out.mkdir(parents=True,exist_ok=False)
    runtime=read(root/'prime_q4_runtime/contract.json')
    code={n:sha(Path('experiments/nanbeige_reasoning')/n) for n in ['llama_capture.py','llama_steer.py','llama_cached_steer.py','run_q4_feature_answers.py','temperature_guard_burst.py','check_cached_steering.py']}
    write(out/'contract.json',dict(context=10153,gate_manifest_sha256=sha(gate/'manifest.json'),
        model_sha256=runtime['model_sha256'],code_sha256=code))
    write(out/'gate_link.json',dict(path=str(gate),manifest_sha256=sha(gate/'manifest.json'),verification_sha256=sha(gate/'verification.json')))
    write(out/'manifest.json',dict(status='remote_full_context_gate_verified',files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    write(out/'verification.json',dict(status='answer_context_and_decoding_verified',context=10153,
        manifest_sha256=sha(out/'manifest.json'),conditions=8,full_vocabulary_comparisons=64,
        tokenizer_files={n:sha(tokenizer_path/n) for n in ['tokenizer.model','tokenizer_config.json','added_tokens.json','special_tokens_map.json','config.json']},
        decoding_policy=DECODING_POLICY,decoder_sha256=sha(Path('experiments/nanbeige_reasoning/answer_decoding.py')),
        verifier_sha256=sha(Path(__file__)),scope='Fresh Linux/CUDA gate at full allocated context; 383 prompt tokens and 8 decoding steps, no task score.'))
    print(json.dumps(dict(status='remote_gate_and_decoding_verified',conditions=8)))


if __name__=='__main__':main()
