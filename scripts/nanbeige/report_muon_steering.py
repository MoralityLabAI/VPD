"""Reload the calibration steering archives and incorporate bounded results."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('artifacts/nanbeige_reasoning')
OUT = ROOT/'q4_steering_preflight_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))

def main():
    manifest = read(OUT/'manifest.json')
    assert manifest['status'] == 'pretrained_q4_residual_steering_preflight_passed'
    for name, digest in manifest['files'].items(): assert sha(OUT/name) == digest
    assert read(OUT/'cleanup.json') == dict(model_closed=True, context_closed=True)
    contract = read(OUT/'contract.json')
    direction = np.load(OUT/'selected_direction.npz')['direction']
    records = read(OUT/'results.json')
    checks = 0
    with np.load(OUT/'baseline.npz') as base:
        sites = [k for k in base.files if '__pass_' in k]
        for row in records:
            assert sha(OUT/row['archive']) == row['sha256']
            with np.load(OUT/row['archive']) as z:
                np.testing.assert_array_equal(z['token_ids'], contract['token_ids'])
                np.testing.assert_array_equal(z['token_positions'], contract['retained_positions'])
                delta = float(np.abs(z['logits'].astype(np.float64)-base['logits']).max())
                assert delta == row['max_abs_logit_change']
                assert bool(z['logits'].argmax()!=base['logits'].argmax()) == row['argmax_changed']
                assert row['mutations'] == len(row['passes'])
                for i, invocation in enumerate(row['passes']):
                    expected = z[f'before_{i}'] + np.float32(row['alpha'])*direction
                    np.testing.assert_array_equal(expected,z[f'after_{i}'])
                    np.testing.assert_array_equal(z[f'after_{i}'],z[f'model.layers.21__pass_{invocation}'][-1])
                    checks += 2
                for site in sites:
                    np.testing.assert_array_equal(base[site][:-1],z[site][:-1]); checks += 1
                    if row['alpha']==0 or (row['passes']==[1] and site.endswith('__pass_0')) or (site.startswith('model.layers.') and site.endswith('__pass_0') and int(site.split('.')[2].split('__')[0])<21):
                        np.testing.assert_array_equal(base[site],z[site]); checks += 1
                if row['alpha']==0: np.testing.assert_array_equal(base['logits'],z['logits'])
    receipt = dict(status='saved_steering_arrays_verified',manifest_sha256=sha(OUT/'manifest.json'),
        conditions=len(records),array_checks=checks,verifier_sha256=sha(Path(__file__)),
        limitation='Independent reload checks saved arithmetic and causal ordering; inference itself was not rerun.')
    (OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    section = '''### 4.15 Local Muon-feature steering implementation control

We additionally test pass-specific residual steering in the pretrained Q4 runtime, using the frozen physical-layer-22 Muon dictionary. The direction is the unit decoder vector with the largest mean absolute pass-to-pass feature difference across the eight training inputs only. This selects a numerically changing feature without attributing a semantic meaning. The test uses only the first 96 prompt tokens of one calibration input, perturbing position 95 at layer 22 before loop normalization. It does not use a held-out task or a generated reasoning trace.

The pinned ggml activation-write API applies signed additive doses of +0.1 and -0.1 on the first pass, second pass or both. A zero-dose control writes identical activation values back to both passes. Full-vocabulary logits match exactly between capture-disabled and capture-enabled baselines, zero writes preserve captured activations and logits exactly, and clearing steering restores the baseline exactly. All earlier retained positions remain unchanged; second-pass-only steering preserves all captured first-pass activations. Backend readback confirms each requested write, and the post-intervention capture agrees with it.

Across the six nonzero conditions, maximum absolute next-token logit changes range from 0.0143 to 0.8051; none changes the greedy next token. These values demonstrate that the runtime applies the intervention and propagates it to the output. They are neither an answer-accuracy result nor evidence that this selected feature controls reasoning. The truncated calibration prompt, one feature and one dose magnitude do not support comparisons of causal importance between passes. Cached generation parity, semantic controls and task-level steering remain untested by this preflight.

The seven condition archives and their source/code hashes are retained in `q4_steering_preflight_v1`. A separate reload verifies saved intervention arithmetic, causal ordering, token identity and full-vocabulary logit differences; runtime checks and cleanup are recorded in its manifest and cleanup receipt. This residual-direction intervention complements the parameter-component interventions in Section 4.14; its coefficients use a different basis and are not directly comparable.

'''
    paper = ROOT/'PAPER.md'
    content = paper.read_text(encoding='utf-8')
    if '### 4.15 Local Muon-feature' in content:
        start = content.index('### 4.15 Local Muon-feature')
        content = content[:start] + content[content.index('## 5. Prospective',start):]
    content = content.replace('## 5. Prospective tests and limits',section+'## 5. Prospective tests and limits')
    paper.write_text(content,encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__': main()
