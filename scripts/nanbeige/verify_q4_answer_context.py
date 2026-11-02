"""Independent array and HF token-decoding verification; separate from C-API inference."""
import json
from pathlib import Path

import numpy as np
from experiments.nanbeige_reasoning.answer_decoding import AnswerDecoder,DECODING_POLICY

from experiments.nanbeige_reasoning.check_llama_steering import read,sha,write


def main():
    root=Path('artifacts/nanbeige_reasoning');run=root/'q4_answer_context_v1';gate=root/'q4_cached_steering_burst_v1'
    manifest=read(run/'manifest.json');contract=read(run/'contract.json')
    assert manifest['status']=='answer_context_preflight_passed'
    for name,digest in manifest['files'].items():assert sha(run/name)==digest
    assert contract['gate_manifest_sha256']==sha(gate/'manifest.json')
    for name,digest in contract['code_sha256'].items():assert sha(Path('experiments/nanbeige_reasoning')/name)==digest
    tokenizer_path=Path('models/Nanbeige4.2-3B-tokenizer')
    tokenizer_files={n:sha(tokenizer_path/n) for n in ['tokenizer.model','tokenizer_config.json','added_tokens.json','special_tokens_map.json','config.json']}
    tokenizer=AnswerDecoder(tokenizer_path)
    rows=read(run/'results.json');assert [r['condition'] for r in rows]==['baseline','identity','both_minus']
    for row in rows:
        assert sha(run/row['archive'])==row['sha256']
        with np.load(run/row['archive']) as z,np.load(gate/row['archive']) as old:
            np.testing.assert_array_equal(z['cached'],old['cached'])
            np.testing.assert_array_equal(z['generated'],z['cached'].argmax(1))
            assert z['cached'].shape==(8,166144) and len(z['token_ids'])==383
            if row['condition']!='identity':np.testing.assert_array_equal(z['full_last'][0],z['cached'][-1])
            assert row['mutation_count']==390*(0 if row['condition']=='baseline' else 2)
            assert tokenizer.decode(z['generated'].tolist(),skip_special_tokens=False)==row['generated_text']
            assert tokenizer.decode(z['token_ids'].tolist(),skip_special_tokens=False)==row['prompt_text']
            for token,text in row['stop_token_texts'].items():assert tokenizer.decode([int(token)],skip_special_tokens=False)==text
    cleanup=read(run/'cleanup.json');assert cleanup['model_closed'] and cleanup['context_closed']
    receipt=dict(status='answer_context_and_decoding_verified',context=contract['context'],conditions=3,
        cached_logit_comparisons=24,fresh_prefix_last_step_comparisons=2,manifest_sha256=sha(run/'manifest.json'),
        tokenizer_files=tokenizer_files,decoding_policy=DECODING_POLICY,
        decoder_sha256=sha(Path('experiments/nanbeige_reasoning/answer_decoding.py')),
        verifier_sha256=sha(Path(__file__)),scope=contract['scope'])
    write(run/'verification.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
