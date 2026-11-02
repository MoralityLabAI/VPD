"""Independently recompute paired SAE artifacts with NumPy float64 arithmetic.

Does not rerun training or establish feature semantics. Partial mode is explicit
and writes its receipt outside the running experiment directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true')
    a = p.parse_args()
    if a.receipt.resolve().is_relative_to(a.control.resolve()):
        raise ValueError('Receipt must be outside experiment directory')
    c = read(a.control/'contract.json')
    m = read(a.fit/'manifest.json')
    assert digest(a.fit/'manifest.json') == c['source_manifest_sha256']
    assert digest(a.fit/'examples.json') == m['artifacts']['examples.json']
    examples = read(a.fit/'examples.json')
    assert c['training_ids'] == [r['id'] for r in examples if r['split']=='train']
    assert c['evaluation_ids'] == [r['id'] for r in examples if r['split']=='eval']
    assert not set(c['training_ids']) & set(c['evaluation_ids'])
    data, positions = {}, {}
    for r in examples:
        name = m['baseline_archives'][r['id']]
        assert digest(a.fit/name) == c['source_files'][name]
        with np.load(a.fit/name,allow_pickle=False) as z:
            positions[r['id']] = z['token_positions']
            np.testing.assert_array_equal(z['token_ids'],np.asarray(r['token_ids'])[z['token_positions']])
            data[r['id']] = [z[f"{c['site']}__pass_{i}"].astype(np.float64) for i in (0,1)]
    train = [np.concatenate([data[k][i] for k in c['training_ids']]) for i in (0,1)]
    variance = np.array([np.square(x-x.mean(0)).mean() for x in train])
    np.testing.assert_allclose(variance,c['training_variance'],rtol=1e-12)
    inv = 1/variance
    np.testing.assert_allclose(inv/inv.mean(),c['arms']['inverse_variance_mse'],rtol=1e-12)
    assert c['arms']['raw_mse'] == [1.,1.]
    conditions, paired, archive_checks, quality_checks = [], {}, 0, 0
    for seed in c['seeds']:
        batch_file = a.control/f'batches_{seed}.npy'
        if not batch_file.exists():
            assert a.allow_partial
            continue
        batches = np.load(batch_file,allow_pickle=False)
        np.testing.assert_array_equal(batches,np.random.default_rng(seed).integers(64,size=(1000,64)))
        batch_sha = hashlib.sha256(batches.tobytes()).hexdigest()
        for arm in c['arms']:
            out = a.control/f'{seed}_{arm}'
            if not (out/'manifest.json').exists():
                assert a.allow_partial
                continue
            manifest = read(out/'manifest.json')
            assert manifest['status']=='sae_loss_condition_complete'
            for name, expected in manifest['files'].items():
                assert Path(name).name==name and digest(out/name)==expected
            assert manifest['seed']==seed and manifest['arm']==arm
            assert manifest['weights']==c['arms'][arm]
            assert manifest['completed_steps']==1000 and manifest['batches_sha256']==batch_sha
            initial=manifest['initial_parameters_sha256']
            assert seed not in paired or paired[seed]==initial
            paired[seed]=initial
            history=read(out/'history.json')
            assert [h['step'] for h in history]==list(range(1,1001))
            for h in history:
                np.testing.assert_allclose(h['objective'],np.mean(np.array(h['pass_mse'])*manifest['weights']),rtol=2e-6,atol=1e-7)
            with np.load(out/'dictionary.npz',allow_pickle=False) as z:
                w=z['encoder_weight'].astype(np.float64);b=z['encoder_bias'].astype(np.float64)
                decoder=z['decoder'].astype(np.float64);bias=z['pre_bias'].astype(np.float64)
                assert int(z['top_k'])==32 and str(z['activation_key'])==c['site']
            assert w.shape==(6144,3072) and decoder.shape==(6144,3072)
            np.testing.assert_allclose(np.linalg.norm(decoder,axis=1),1,rtol=1e-5,atol=1e-6)
            records=read(out/'records.json')
            assert [r['id'] for r in records]==[r['id'] for r in examples]
            encoded={}; max_feature_error=0.
            for r, original in zip(records,examples):
                assert r['split']==original['split']
                assert digest(out/r['archive'])==r['sha256']
                with np.load(out/r['archive'],allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['token_positions'],positions[r['id']])
                    np.testing.assert_array_equal(saved['delta_z'],saved['pass_1']-saved['pass_0'])
                    encoded[r['id']]=[]
                    for i,x in enumerate(data[r['id']]):
                        positive=np.maximum((x-bias)@w.T+b,0)
                        selected=np.argpartition(positive,-32,axis=1)[:,-32:]
                        actual=np.zeros_like(positive)
                        np.put_along_axis(actual,selected,np.take_along_axis(positive,selected,axis=1),axis=1)
                        expected=saved[f'pass_{i}'].astype(np.float64)
                        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-3)
                        max_feature_error=max(max_feature_error,float(np.max(np.abs(actual-expected))))
                        encoded[r['id']].append(actual)
                archive_checks+=1
            quality={}
            for split, ids in [('train',c['training_ids']),('eval',c['evaluation_ids'])]:
                quality[split]={}
                for i in (0,1):
                    x=np.concatenate([data[k][i] for k in ids])
                    z=np.concatenate([encoded[k][i] for k in ids])
                    sse=np.square(z@decoder+bias-x).sum()
                    ev=float(1-sse/np.square(x-x.mean(0)).sum())
                    measured=dict(mse=float(sse/x.size),explained_variance=ev,mean_l0=float((z>0).sum(1).mean()),rows=len(x))
                    expected=manifest['quality'][split][f'pass_{i}']
                    for key in measured:
                        np.testing.assert_allclose(measured[key],expected[key],rtol=2e-5,atol=2e-6)
                    quality[split][f'pass_{i}']=measured
                    quality_checks+=1
            conditions.append(dict(seed=seed,arm=arm,manifest_sha256=digest(out/'manifest.json'),quality=quality,max_feature_absolute_error=max_feature_error))
    complete=(a.control/'manifest.json').exists() and len(conditions)==6
    if not a.allow_partial:assert complete
    if complete:
        root=read(a.control/'manifest.json')
        assert root['status']=='sae_loss_balance_control_complete' and root['conditions']==6
        assert root['contract_sha256']==digest(a.control/'contract.json')
        for name,expected in root['files'].items():
            path=(a.control/name).resolve()
            assert path.is_relative_to(a.control.resolve()) and digest(path)==expected
    receipt=dict(status='complete_control_independently_verified' if complete else 'partial_completed_conditions_verified',
        completed_conditions=len(conditions),feature_archives_checked=archive_checks,quality_checks=quality_checks,
        contract_sha256=digest(a.control/'contract.json'),verifier_sha256=digest(Path(__file__)),conditions=conditions,
        scope='Float64 NumPy encoder/top-k/decoder and pass-delta recomputation, split and hash checks, paired batch/initialization receipts. Training not rerun; no feature-semantic or behavioral validation.')
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('conditions','scope')}))


if __name__=='__main__':main()
