"""Recompute saved intervention arithmetic and shared-SAE encodings after retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.nanbeige_looped.sae import SharedSAE, encode_rows


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_file(root, name, expected):
    path = (root / name).resolve()
    if path.parent != root.resolve() or sha(path) != expected:
        raise ValueError(f'Artifact identity mismatch: {name}')
    return path


def verify_effect_arithmetic(arrays, sites, alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Invalid intervention coefficient')
    checks = 0
    for site in sites:
        for kind in ('h', 'z'):
            np.testing.assert_array_equal(arrays[f'baseline__{site}__pass_delta__{kind}'],
                arrays[f'baseline__{site}__pass_1__{kind}'] - arrays[f'baseline__{site}__pass_0__{kind}'])
            checks += 1
            for invocation in (0, 1):
                key = f'{site}__pass_{invocation}__{kind}'
                base = arrays[f'baseline__{key}']
                for scope in ('first', 'second', 'both'):
                    plus, minus = arrays[f'{scope}_plus__{key}'], arrays[f'{scope}_minus__{key}']
                    np.testing.assert_array_equal(arrays[f'{scope}__{key}__positive_effect'], plus - base)
                    np.testing.assert_array_equal(arrays[f'{scope}__{key}__central_difference'],
                                                  (plus - minus) / (2 * alpha))
                    checks += 2
                np.testing.assert_array_equal(arrays[f'interaction__{key}'],
                    arrays[f'both_plus__{key}'] - arrays[f'first_plus__{key}']
                    - arrays[f'second_plus__{key}'] + base)
                checks += 1
            for sign in ('plus', 'minus'):
                np.testing.assert_array_equal(arrays[f'second_{sign}__{site}__pass_0__{kind}'],
                                               arrays[f'baseline__{site}__pass_0__{kind}'])
                checks += 1
    return checks


def load_sae(path):
    with np.load(path, allow_pickle=False) as saved:
        features, width = saved['encoder_weight'].shape
        sae = SharedSAE(width, features, int(saved['top_k']))
        with torch.no_grad():
            sae.encoder.weight.copy_(torch.from_numpy(saved['encoder_weight']))
            sae.encoder.bias.copy_(torch.from_numpy(saved['encoder_bias']))
            sae.decoder.weight.copy_(torch.from_numpy(saved['decoder'].T))
            sae.pre_bias.copy_(torch.from_numpy(saved['pre_bias']))
    return sae.eval().requires_grad_(False)


def verify(root):
    root = root.resolve()
    manifest = read(root / 'manifest.json')
    if manifest['status'] != 'calibration_complete':
        raise ValueError('Calibration is incomplete')
    for name, expected in manifest['artifacts'].items():
        path = checked_file(root, name, expected)
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as arrays:
                for key in arrays.files:
                    if np.issubdtype(arrays[key].dtype, np.number) and not np.isfinite(arrays[key]).all():
                        raise ValueError(f'Nonfinite array: {name}:{key}')
    examples = read(root / 'examples.json')
    by_id = {r['id']: r for r in examples}
    if len(by_id) != len(examples):
        raise ValueError('Duplicate example IDs')
    train = {r['id'] for r in examples if r['split'] == 'train'}
    evaluation = {r['id'] for r in examples if r['split'] == 'eval'}
    if (not train or not evaluation or len(train) != manifest['train_examples']
            or len(evaluation) != manifest['eval_examples']):
        raise ValueError('Invalid split cardinalities')
    for field in ('pair_id', 'token_ids'):
        if {json.dumps(by_id[i][field]) for i in train} & {json.dumps(by_id[i][field]) for i in evaluation}:
            raise ValueError('Training/evaluation overlap')
    baseline = {}
    rows = manifest['calibration_settings']['tail_rows']
    for row in examples:
        with np.load(root / manifest['baseline_archives'][row['id']], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['token_ids'], row['token_ids'][-rows:])
            np.testing.assert_array_equal(saved['token_positions'],
                np.arange(max(0, len(row['token_ids']) - rows), len(row['token_ids'])))
            baseline[row['id']] = {k: saved[k].copy() for k in saved.files}
    quality = read(root / 'sae_quality.json')
    if set(quality) != set(manifest['sites']):
        raise ValueError('Missing SAE site')
    dictionaries, quality_checks = {}, []
    for site, entry in quality.items():
        if set(entry['train_example_ids']) != train:
            raise ValueError('SAE training IDs differ from split')
        dictionaries[site] = load_sae(checked_file(root, entry['dictionary'], entry['sha256']))
        for invocation in (0, 1):
            values = np.concatenate([baseline[r['id']][f'{site}__pass_{invocation}']
                                     for r in examples if r['id'] in evaluation])
            _, observed = encode_rows(dictionaries[site], values)
            expected = entry['held_out_quality'][f'pass_{invocation}']
            for key in ('mse', 'explained_variance', 'mean_l0', 'rows'):
                if observed[key] is None or expected[key] is None:
                    if observed[key] is not expected[key]:
                        raise ValueError('Undefined quality differs')
                else:
                    np.testing.assert_allclose(observed[key], expected[key], rtol=1e-5, atol=1e-5)
            quality_checks.append({'site': site, 'pass': invocation, **observed})
    effects = read(root / 'effects.json')
    training = read(root / 'vpd_training.json')
    expected_effects = {(i, site, component) for i in evaluation
                        for site in training['physical_matrices']
                        for component in manifest['calibration_settings']['measured_component_indices']}
    if (len(effects) != len(expected_effects)
            or {(r['example_id'], r['physical_module'], r['component']) for r in effects} != expected_effects):
        raise ValueError('Missing or duplicated held-out intervention')
    arithmetic_checks = encoded_checks = 0
    for record in effects:
        example = by_id[record['example_id']]
        with np.load(checked_file(root, record['archive'], record['sha256']), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['token_ids'], example['token_ids'][-rows:])
            np.testing.assert_array_equal(arrays['token_positions'], baseline[example['id']]['token_positions'])
            arithmetic_checks += verify_effect_arithmetic(arrays, manifest['sites'], record['alpha'])
            for site in manifest['sites']:
                for invocation in (0, 1):
                    key = f'{site}__pass_{invocation}'
                    np.testing.assert_array_equal(arrays[f'baseline__{key}__h'], baseline[example['id']][key])
                    for condition in ('baseline', 'first_plus', 'first_minus', 'second_plus',
                                      'second_minus', 'both_plus', 'both_minus'):
                        z, _ = encode_rows(dictionaries[site], arrays[f'{condition}__{key}__h'])
                        np.testing.assert_allclose(arrays[f'{condition}__{key}__z'], z, rtol=1e-5, atol=1e-4)
                        encoded_checks += 1
    result = {'status': 'multipass_saved_effects_independently_verified',
              'manifest_sha256': sha(root / 'manifest.json'), 'effect_records': len(effects),
              'difference_checks': arithmetic_checks, 'sae_reencodings': encoded_checks,
              'heldout_sae_quality': quality_checks,
              'scope': 'Saved-array arithmetic, encoding, alignment and provenance; not rerun model interventions or semantic validation.'}
    (root / 'independent_verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = verify(args.run)
    print(json.dumps({k: v for k, v in result.items() if k != 'heldout_sae_quality'}))
