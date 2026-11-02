"""Independently reload cache parity measurements without model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))

def verify(run):
    manifest=read(run/'manifest.json');contract=read(run/'contract.json')
    for name,digest in manifest['files'].items(): assert sha(run/name)==digest
    results=read(run/'results.json')
    names=['baseline','identity','first_plus','first_minus','second_plus','second_minus','both_plus','both_minus']
    assert [r['condition'] for r in results]==names and manifest['conditions']==8
    assert contract['atol']==contract['rtol']==1e-4 and contract['generated_steps']==8
    baseline=None;checks=[]
    for row in results:
        with np.load(run/row['archive']) as z:
            cached=z['cached'];full=z['full_prefix']
            assert cached.shape==full.shape==(8,166144)
            assert np.isfinite(cached).all() and np.isfinite(full).all()
            np.testing.assert_array_equal(cached.argmax(1),z['generated'])
            assert np.allclose(cached,full,rtol=1e-4,atol=1e-4)==row['allclose']
            assert np.array_equal(cached.argmax(1),full.argmax(1))==row['greedy_tokens_equal']
            error=float(np.abs(cached.astype(np.float64)-full.astype(np.float64)).max())
            assert error==row['max_abs_logit_difference']
            if row['condition']=='baseline': baseline=cached.copy()
            if row['condition']=='identity':
                assert np.array_equal(cached,baseline)==row['identity_write_equals_baseline']
            expected=(len(z['token_ids'])+7)*(0 if row['condition']=='baseline' else 2 if row['condition']=='identity' or row['condition'].startswith('both') else 1)
            assert expected==row['mutation_count']
            checks.append(dict(condition=row['condition'],max_abs_logit_difference=error,
                allclose=row['allclose'],greedy_tokens_equal=row['greedy_tokens_equal']))
    passed=all(r['allclose'] and r['greedy_tokens_equal'] and r['identity_write_equals_baseline'] is not False for r in results)
    assert manifest['status']==('cached_steering_gate_passed' if passed else 'cached_steering_gate_failed')
    receipt=dict(status='cached_steering_saved_arrays_verified',gate_passed=passed,conditions=checks,
        manifest_sha256=sha(run/'manifest.json'),verifier_sha256=sha(Path(__file__)))
    (run/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    print(json.dumps(verify(parser.parse_args().run)))
