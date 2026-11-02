"""Frozen SAE transfer to verified BF16 VPD interventions; no fitting/inference.

Torch produces the archives; a separate NumPy float64 encoder independently
checks every encoding and pooled reconstruction. Precision/position shift is
measured, not assumed harmless. Neither feature semantics nor behavior is tested.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import torch

from experiments.nanbeige_looped.sae import encode_rows
from scripts.nanbeige.verify_multipass_results import load_sae

CONDITIONS = ['baseline', 'first_plus', 'first_minus', 'second_plus', 'second_minus', 'both_plus', 'both_minus']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def numpy_encoder(dictionary, x):
    positive = np.maximum((x.astype(np.float64)-dictionary['pre_bias']) @ dictionary['encoder_weight'].T + dictionary['encoder_bias'], 0)
    k = int(dictionary['top_k'])
    selected = np.argpartition(positive, -k, axis=1)[:, -k:]
    z = np.zeros_like(positive)
    np.put_along_axis(z, selected, np.take_along_axis(positive, selected, axis=1), axis=1)
    return z


def numpy_quality(dictionary, x, z):
    x = x.astype(np.float64)
    sse = np.square(z @ dictionary['decoder'] + dictionary['pre_bias'] - x).sum()
    return dict(mse=float(sse/x.size), explained_variance=float(1-sse/np.square(x-x.mean(0)).sum()),
                mean_l0=float((z>0).sum(axis=1).mean()), rows=len(x))


def check_feature_arithmetic(arrays):
    checks = 0
    for condition in CONDITIONS:
        np.testing.assert_array_equal(arrays[f'{condition}__pass_difference'], arrays[f'{condition}__pass_1']-arrays[f'{condition}__pass_0'])
        checks += 1
        if condition != 'baseline':
            for i in (0, 1):
                np.testing.assert_array_equal(arrays[f'{condition}__delta_{i}'], arrays[f'{condition}__pass_{i}']-arrays[f'baseline__pass_{i}'])
                checks += 1
    for condition in ['second_plus', 'second_minus']:
        np.testing.assert_array_equal(arrays[f'{condition}__pass_0'], arrays['baseline__pass_0'])
        checks += 1
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    fit = read(args.fit/'manifest.json')
    fit_check = read(args.fit/'independent_verification.json')
    lineage = read(args.fit/'cross_precision_verification.json')
    assert fit_check['status'] == 'multipass_saved_effects_independently_verified'
    assert lineage['status'] == 'cross_precision_reasoning_vpd_verified'
    assert fit_check['manifest_sha256'] == lineage['manifest_sha256'] == sha(args.fit/'manifest.json')
    control = read(args.control/'contract.json')
    verified = read(args.verification)
    assert verified['status'] == 'muon_control_independently_verified' and verified['completed_conditions'] == 6
    assert verified['contract_sha256'] == sha(args.control/'contract.json')
    for name in ['examples.json', 'effects.json']:
        assert sha(args.fit/name) == fit['artifacts'][name]
    examples = read(args.fit/'examples.json')
    effects = read(args.fit/'effects.json')
    assert [r['id'] for r in examples if r['split']=='train'] == control['training_ids']
    assert [r['id'] for r in examples if r['split']=='eval'] == control['evaluation_ids']
    original = {r['id']: r for r in examples}
    source_run = Path(control['source_run'])
    rows = [json.loads(s) for s in (source_run/'generations.jsonl').read_text(encoding='utf-8').splitlines()]
    native = {r['id']: r for r in rows if r['loops']==2}
    capture_manifest = read(source_run/'q4_activations/manifest.json')
    assert sha(source_run/'q4_activations/manifest.json') == control['capture_manifest_sha256']
    assert sha(source_run/'generations.jsonl') == capture_manifest['generation_source_sha256']
    baselines = {}
    for r in examples:
        assert native[r['id']]['calibration_split'] == r['split']
        assert native[r['id']]['token_ids'][:len(r['token_ids'])] == r['token_ids']
        name = fit['baseline_archives'][r['id']]
        assert sha(args.fit/name) == fit['artifacts'][name]
        with np.load(args.fit/name, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['token_ids'], np.asarray(r['token_ids'])[saved['token_positions']])
            baselines[r['id']] = {k: saved[k] for k in saved.files}
    expected = {(r['id'], site+'.mlp.down_proj', 0) for r in examples if r['split']=='eval' for site in ['model.layers.11', 'model.layers.21']}
    assert {(r['example_id'], r['physical_module'], r['component']) for r in effects} == expected and len(effects)==20
    for r in effects:
        assert r['alpha']==.05 and sha(args.fit/r['archive']) == r['sha256'] == fit['artifacts'][r['archive']]
    args.output.mkdir(parents=True, exist_ok=False)
    contract = dict(schema='nanbeige.frozen_muon_vpd_transfer.v1', fit_manifest_sha256=sha(args.fit/'manifest.json'),
        fit_verification_sha256=sha(args.fit/'independent_verification.json'),
        muon_verification_sha256=sha(args.verification), muon_contract_sha256=sha(args.control/'contract.json'),
        source_generation_sha256=sha(source_run/'generations.jsonl'), source_code_sha256=sha(__file__),
        sites=fit['sites'], arms=['adamw', 'muon_hybrid'], conditions=CONDITIONS,
        token_alignment='All18BF16source trajectories are exact prefixes of their longer native Q4 trajectories; original train/eval assignments retained.',
        scope='Frozen Q4-trained dictionaries on BF16 baseline and intervention activations at initial-budget endpoint positions. No new fit, model inference, steering or answer outcome.',
        precision_and_position_shift=True, posthoc=True, training_unchanged=True)
    write(args.output/'contract.json', contract)
    summaries, records = [], []
    encoding_checks = quality_checks = arithmetic_checks = 0
    for site in fit['sites']:
        index = control['sites'].index(site)
        for arm in contract['arms']:
            condition = args.control/f'{index:02d}_{arm}'
            source_manifest = read(condition/'manifest.json')
            previous = next(r for r in verified['conditions'] if r['site']==site and r['arm']==arm)
            assert sha(condition/'manifest.json') == previous['condition_manifest_sha256']
            assert sha(condition/'dictionary.npz') == source_manifest['files']['dictionary.npz']
            sae = load_sae(condition/'dictionary.npz')
            with np.load(condition/'dictionary.npz', allow_pickle=False) as z:
                dictionary = {k: z[k].astype(np.float64) for k in ['encoder_weight', 'encoder_bias', 'decoder', 'pre_bias', 'top_k']}
            quality = {}
            for split in ['train', 'eval']:
                quality[split] = {}
                for i in (0, 1):
                    x = np.concatenate([baselines[r['id']][f'{site}__pass_{i}'] for r in examples if r['split']==split])
                    actual, measured = encode_rows(sae, x)
                    independent = numpy_encoder(dictionary, x)
                    np.testing.assert_allclose(actual, independent, rtol=1e-4, atol=1e-3)
                    q = numpy_quality(dictionary, x, independent)
                    for key, value in q.items():
                        np.testing.assert_allclose(value, measured[key], rtol=2e-5, atol=2e-6)
                    quality[split][f'pass_{i}'] = q
                    quality_checks += 1
            summary = dict(site=site, arm=arm, baseline_transfer_quality=quality,
                dictionary_sha256=sha(condition/'dictionary.npz'), dictionary_manifest_sha256=sha(condition/'manifest.json'))
            summaries.append(summary)
            print(json.dumps(dict(site=site, arm=arm, heldout_ev=[quality['eval'][f'pass_{i}']['explained_variance'] for i in (0,1)])), flush=True)
            for effect_index, effect in enumerate(effects):
                arrays = {}
                with np.load(args.fit/effect['archive'], allow_pickle=False) as hidden:
                    arrays['token_positions'] = hidden['token_positions']
                    arrays['token_ids'] = hidden['token_ids']
                    baseline = baselines[effect['example_id']]
                    np.testing.assert_array_equal(arrays['token_positions'], baseline['token_positions'])
                    np.testing.assert_array_equal(arrays['token_ids'], baseline['token_ids'])
                    for invocation in (0, 1):
                        key = f'{site}__pass_{invocation}'
                        np.testing.assert_array_equal(hidden[f'baseline__{key}__h'], baseline[key])
                        for name in CONDITIONS:
                            x = hidden[f'{name}__{key}__h']
                            z, _ = encode_rows(sae, x)
                            arrays[f'{name}__pass_{invocation}'] = z
                    for name in CONDITIONS:
                        arrays[f'{name}__pass_difference'] = arrays[f'{name}__pass_1'] - arrays[f'{name}__pass_0']
                        if name != 'baseline':
                            for invocation in (0, 1):
                                arrays[f'{name}__delta_{invocation}'] = arrays[f'{name}__pass_{invocation}'] - arrays[f'baseline__pass_{invocation}']
                    destination = args.output/f'{site.replace(".", "_")}_{arm}_{effect_index:02d}.npz'
                    np.savez_compressed(destination, **arrays)
                    with np.load(destination, allow_pickle=False) as reload:
                        arithmetic_checks += check_feature_arithmetic(reload)
                        for name in CONDITIONS:
                            for invocation in (0, 1):
                                x = hidden[f'{name}__{site}__pass_{invocation}__h']
                                independent = numpy_encoder(dictionary, x)
                                np.testing.assert_allclose(reload[f'{name}__pass_{invocation}'], independent, rtol=1e-4, atol=1e-3)
                                encoding_checks += 1
                records.append(dict(site=site, arm=arm, example_id=effect['example_id'], physical_module=effect['physical_module'],
                    component=effect['component'], alpha=effect['alpha'], source_archive=effect['archive'],
                    source_archive_sha256=effect['sha256'], archive=destination.name, sha256=sha(destination)))
            del sae, dictionary
    write(args.output/'quality.json', summaries)
    write(args.output/'records.json', records)
    assert len(records)==120 and encoding_checks==1680 and quality_checks==24 and arithmetic_checks==2520
    write(args.output/'manifest.json', dict(status='frozen_sae_vpd_transfer_verified', feature_archives=len(records),
        independently_reencoded_conditions=encoding_checks, pooled_quality_checks=quality_checks, feature_arithmetic_checks=arithmetic_checks,
        contract_sha256=sha(args.output/'contract.json'),
        files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='frozen_sae_vpd_transfer_verified', archives=len(records), encoding_checks=encoding_checks, quality_checks=quality_checks)), flush=True)


if __name__ == '__main__':
    main()
