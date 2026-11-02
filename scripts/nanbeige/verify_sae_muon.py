"""Independent NumPy verification of the completed paired Muon SAE pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_files(root, files):
    for name, expected in files.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve()) and digest(path) == expected, name


def check_delta(z0, z1, delta):
    np.testing.assert_array_equal(delta, z1 - z0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    if args.receipt.resolve().is_relative_to(args.control.resolve()):
        raise ValueError('Verification receipt must be outside the experiment')
    root = read(args.control / 'manifest.json')
    c = read(args.control / 'contract.json')
    assert root['status'] == 'sae_muon_control_complete'
    assert root['conditions'] == 2 * len(c['sites'])
    assert root['contract_sha256'] == digest(args.control / 'contract.json')
    check_files(args.control, root['files'])
    source = Path(c['source_run'])
    cap = source / 'q4_activations'
    assert digest(source / 'mechanistic_verification.json') == c['source_verification_sha256']
    assert digest(cap / 'manifest.json') == c['capture_manifest_sha256']
    m = read(cap / 'manifest.json')
    assert digest(cap / 'records.jsonl') == m['files']['records.jsonl']
    examples = [json.loads(s) for s in (cap / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert c['training_ids'] == [r['id'] for r in examples if r['split'] == 'train']
    assert c['evaluation_ids'] == [r['id'] for r in examples if r['split'] == 'eval']
    assert not set(c['training_ids']) & set(c['evaluation_ids'])
    check_files(cap, c['source_archives'])
    assert c['steps'] > 0 and c['features'] == 6144 and c['top_k'] == 32
    assert digest(args.control / 'batches.npy') == c['batches_sha256']
    batches = np.load(args.control / 'batches.npy', allow_pickle=False)
    np.testing.assert_array_equal(batches, np.random.default_rng(c['seed']).integers(256, size=(c['steps'], 64)))
    results, comparisons = [], []
    archives = 0
    for site_index, site in enumerate(c['sites']):
        values = {}
        for r in examples:
            with np.load(cap / r['archive'], allow_pickle=False) as z:
                values[r['id']] = [z[f'{site}__pass_{i}'].astype(np.float64) for i in (0, 1)]
        initial = []
        paired = {}
        for arm in ['adamw', 'muon_hybrid']:
            out = args.control / f'{site_index:02d}_{arm}'
            manifest = read(out / 'manifest.json')
            check_files(out, manifest['files'])
            assert manifest['status'] == 'sae_optimizer_condition_complete'
            assert manifest['site'] == site and manifest['arm'] == arm
            assert manifest['completed_steps'] == c['steps'] and manifest['batches_sha256'] == c['batches_sha256']
            initial.append(manifest['initial_parameters_sha256'])
            groups = manifest['parameter_groups']
            if arm == 'muon_hybrid':
                assert set(groups['muon']) == {'encoder.weight', 'decoder.weight'}
                assert set(groups['adamw']) == {'pre_bias', 'encoder.bias'}
            history = read(out / 'history.json')
            assert [h['step'] for h in history] == list(range(1, c['steps'] + 1))
            for h in history:
                np.testing.assert_allclose(h['objective'], np.mean(h['pass_mse']), rtol=2e-6, atol=1e-7)
            with np.load(out / 'dictionary.npz', allow_pickle=False) as z:
                w = z['encoder_weight'].astype(np.float64)
                b = z['encoder_bias'].astype(np.float64)
                decoder = z['decoder'].astype(np.float64)
                bias = z['pre_bias'].astype(np.float64)
                assert int(z['top_k']) == 32 and str(z['activation_key']) == site
            assert w.shape == decoder.shape == (6144, 3072)
            np.testing.assert_allclose(np.linalg.norm(decoder, axis=1), 1, rtol=1e-5, atol=1e-6)
            records = read(out / 'records.json')
            assert [r['id'] for r in records] == [r['id'] for r in examples]
            encoded = {}
            for r, original in zip(records, examples):
                assert r['split'] == original['split'] and digest(out / r['archive']) == r['sha256']
                encoded[r['id']] = []
                with np.load(out / r['archive'], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['token_positions'], original['positions'])
                    check_delta(saved['pass_0'], saved['pass_1'], saved['delta_z'])
                    for i, x in enumerate(values[r['id']]):
                        positive = np.maximum((x - bias) @ w.T + b, 0)
                        indices = np.argpartition(positive, -32, axis=1)[:, -32:]
                        actual = np.zeros_like(positive)
                        np.put_along_axis(actual, indices, np.take_along_axis(positive, indices, axis=1), axis=1)
                        np.testing.assert_allclose(actual, saved[f'pass_{i}'], rtol=1e-4, atol=1e-3)
                        encoded[r['id']].append(actual)
                archives += 1
            quality = {}
            for split, ids in [('train', c['training_ids']), ('eval', c['evaluation_ids'])]:
                quality[split] = {}
                zs = []
                for i in (0, 1):
                    x = np.concatenate([values[key][i] for key in ids])
                    z = np.concatenate([encoded[key][i] for key in ids]); zs.append(z)
                    sse = np.square(z @ decoder + bias - x).sum()
                    measured = dict(mse=float(sse / x.size), explained_variance=float(1 - sse / np.square(x - x.mean(0)).sum()),
                        mean_l0=float((z > 0).sum(axis=1).mean()), rows=len(x), active_features=int(np.any(z > 0, axis=0).sum()),
                        inactive_features_on_this_split=int((~np.any(z > 0, axis=0)).sum()))
                    expected = manifest['quality'][split][f'pass_{i}']
                    for key in measured:
                        np.testing.assert_allclose(measured[key], expected[key], rtol=2e-5, atol=2e-6)
                    quality[split][f'pass_{i}'] = measured
                z = np.concatenate(zs)
                both = dict(active_features=int(np.any(z > 0, axis=0).sum()), inactive_features_on_this_split=int((~np.any(z > 0, axis=0)).sum()), mean_l0=float((z > 0).sum(axis=1).mean()))
                assert both == manifest['quality'][split]['both_pass_usage']
                quality[split]['both_pass_usage'] = both
            result = dict(site=site, arm=arm, quality=quality, condition_manifest_sha256=digest(out / 'manifest.json'))
            results.append(result); paired[arm] = quality
        assert len(set(initial)) == 1
        comparisons.append(dict(site=site, heldout_ev_change_muon_minus_adamw=[paired['muon_hybrid']['eval'][f'pass_{i}']['explained_variance'] - paired['adamw']['eval'][f'pass_{i}']['explained_variance'] for i in (0, 1)],
            training_inactive_feature_change_muon_minus_adamw=paired['muon_hybrid']['train']['both_pass_usage']['inactive_features_on_this_split'] - paired['adamw']['train']['both_pass_usage']['inactive_features_on_this_split']))
    receipt = dict(status='muon_control_independently_verified', completed_conditions=len(results), feature_archives_checked=archives,
        pooled_quality_checks=4 * len(results), contract_sha256=digest(args.control / 'contract.json'), verifier_sha256=digest(__file__),
        conditions=results, paired_changes=comparisons,
        scope='Independent NumPy float64 re-encoding, reconstruction, feature usage and pass differences; provenance and paired initialization/batches checked. Training not rerun, single seed, no semantic or behavioral validation.')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['conditions', 'scope']}))


if __name__ == '__main__':
    main()
