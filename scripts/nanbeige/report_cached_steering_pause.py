"""Report completed cache controls while preserving the user-requested pause."""
import json
from pathlib import Path
import numpy as np
from scripts.nanbeige.verify_cached_steering import verify, read, sha


def main():
    root=Path('artifacts/nanbeige_reasoning')
    batched=root/'q4_cached_steering_v1'
    failed=verify(batched)
    assert not failed['gate_passed']
    serial=root/'q4_cached_steering_serial_v1'
    pause=read(serial/'user_pause.json')
    assert pause['status']=='paused_by_user_process_stopped' and not pause['resume_authorized']
    assert not (serial/'manifest.json').exists(), 'Do not apply a partial-run report to a completed run'
    contract=read(serial/'contract.json');other=read(batched/'contract.json')
    assert contract['chunk_size']==1 and other['chunk_size']==64
    assert {k:v for k,v in contract.items() if k!='chunk_size'}=={k:v for k,v in other.items() if k!='chunk_size'}
    records=read(serial/'results.json')
    assert [r['condition'] for r in records]==pause['completed_conditions']==['baseline','identity']
    baseline=None;prefix=None
    for row in records:
        assert sha(serial/row['archive'])==row['sha256']
        with np.load(serial/row['archive']) as z:
            a,b=z['cached'],z['full_prefix']
            assert a.shape==b.shape==(8,166144) and np.isfinite(a).all() and np.isfinite(b).all()
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(a.argmax(1),z['generated'])
            assert row['max_abs_logit_difference']==0 and row['allclose'] and row['greedy_tokens_equal']
            if row['condition']=='baseline': baseline=a.copy();prefix=z['token_ids'].copy()
            else:
                np.testing.assert_array_equal(a,baseline)
                np.testing.assert_array_equal(prefix,z['token_ids'])
                assert row['identity_write_equals_baseline'] and row['mutation_count']==2*(len(prefix)+7)
    questions=[json.loads(s) for s in (root/'pilot_v1/questions.jsonl').read_text(encoding='utf-8').splitlines()]
    q=next(r for r in questions if r['id']==contract['question_id'])
    assert q['calibration_split']=='train'
    np.testing.assert_array_equal(prefix,q['token_ids'])
    receipt=dict(status='paused_cache_controls_saved_arrays_verified',serial_conditions_checked=2,
        serial_gate_complete=False,serial_max_abs_error=0.,training_prompt_tokens=len(prefix),
        batched_gate_passed=False,batched_conditions_checked=8,
        batched_max_abs_error=max(r['max_abs_logit_difference'] for r in failed['conditions']),
        batched_greedy_decisions_equal=sum(8 for r in failed['conditions'] if r['greedy_tokens_equal']),
        batched_manifest_sha256=sha(batched/'manifest.json'),
        serial_files={n:sha(serial/n) for n in ['contract.json','results.json','baseline.npz','identity.npz','user_pause.json']},
        reporter_sha256=sha(Path(__file__)),model_inference_performed=False)
    (root/'cached_steering_pause_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    paragraph='''A subsequent calibration-only cache test uses the full 383-token training prompt and eight greedy decoding steps under a baseline, identity writes and six signed pass-specific conditions. Interventions apply at every processed position. For each step, cached logits are compared with a fresh evaluation of the same complete token prefix. With 64-token prefill batches and single-token cached decoding, all eight conditions fail the fixed numerical gate (absolute and relative tolerances both 1e-4), despite agreement on all 64 greedy decisions. Maximum absolute logit discrepancy is 3.6051 for the baseline and reaches 3.7099 across conditions. Identity writes still reproduce the baseline exactly. This failure prevents using that execution path for the planned answer-intervention experiment; greedy agreement alone is insufficient for its gate.

Using single-token processing throughout prefill, cached decoding and fresh-prefix replay gives bitwise-identical logits in the two completed conditions: baseline and identity writes. Their outputs also match each other exactly. The run was stopped after these two controls; the six signed serial conditions remain incomplete. Independent reload checks reproduce the completed results without rerunning inference. This comparison identifies an execution-schedule discrepancy but does not establish its underlying kernel or numerical cause. It does not validate the full serial intervention gate, long-context generation or answer accuracy. The failed complete run is retained in `q4_cached_steering_v1`; partial serial evidence is in `q4_cached_steering_serial_v1`, with a source-linked receipt in `cached_steering_pause_verification.json`.

'''
    p=root/'PAPER.md';text=p.read_text(encoding='utf-8')
    anchor='## 5. Prospective tests and limits'
    start=text.find('A subsequent calibration-only cache test uses the full 383-token')
    if start!=-1: text=text[:start]+text[text.index(anchor,start):]
    text=text.replace(anchor,paragraph+anchor,1)
    p.write_text(text,encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__': main()
