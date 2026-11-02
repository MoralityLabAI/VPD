"""Training-span, sparse-reconstruction and matched-runtime diagnostics.

Exploratory analysis following observed SAE failures. All projection controls
are fitted only to the original training split; every requested rank is saved.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest, read, rows
from scripts.nanbeige.verify_multipass_results import load_sae


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def energy(x):
    return float(np.square(np.asarray(x, dtype=np.float64)).sum())


def quality(x, reconstruction):
    squared_error = energy(x-reconstruction)
    variance = energy(x-x.mean(0))
    return dict(squared_error=squared_error, variance=variance,
        mse=squared_error/x.size, explained_variance=1-squared_error/variance if variance else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--q4-captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = read(args.fit/'manifest.json')
    manifest_hash = digest(args.fit/'manifest.json')
    for name, status in [('independent_verification.json', 'multipass_saved_effects_independently_verified'),
                         ('cross_precision_verification.json', 'cross_precision_reasoning_vpd_verified')]:
        verification = read(args.fit/name)
        if verification['status'] != status or verification['manifest_sha256'] != manifest_hash:
            raise ValueError('Verified generated-prefix calibration required')
    capture = read(args.q4_captures/'manifest.json')
    if (capture['status'] != 'q4_reasoning_capture_complete'
            or digest(args.q4_captures/'records.jsonl') != capture['files']['records.jsonl']):
        raise ValueError('Verified complete Q4 capture required')
    examples = read(args.fit/'examples.json')
    metadata = {r['id']: r for r in rows(args.q4_captures/'records.jsonl')}
    if len(metadata) != len(examples) or set(metadata) != {r['id'] for r in examples}:
        raise ValueError('Capture cohorts differ')
    training = [r for r in examples if r['split'] == 'train']
    evaluation = [r for r in examples if r['split'] == 'eval']
    if len(training) != 8 or len(evaluation) != 10:
        raise ValueError('Unexpected pilot split')
    sites = ['model.layers.11', 'model.layers.21', 'model.norm']
    bf16, q4 = {}, {}
    source_files = {}
    for name in ('examples.json', 'sae_quality.json'):
        if digest(args.fit/name) != manifest['artifacts'][name]:
            raise ValueError('Changed fit metadata')
    for row in examples:
        name = manifest['baseline_archives'][row['id']]
        if digest(args.fit/name) != manifest['artifacts'][name]:
            raise ValueError('Changed BF16 activation archive')
        source_files['bf16/'+name] = digest(args.fit/name)
        record = metadata[row['id']]
        path = (args.q4_captures/record['archive']).resolve()
        if not path.is_relative_to(args.q4_captures.resolve()) or digest(path) != record['sha256']:
            raise ValueError('Changed Q4 activation archive')
        source_files['q4/'+record['archive']] = record['sha256']
        if record['split'] != row['split']:
            raise ValueError('Split mismatch')
        with np.load(args.fit/name, allow_pickle=False) as b, np.load(path, allow_pickle=False) as q:
            positions = b['token_positions']
            np.testing.assert_array_equal(q['token_ids'], row['token_ids'])
            np.testing.assert_array_equal(b['token_ids'], np.asarray(row['token_ids'])[positions])
            matching = np.searchsorted(q['token_positions'], positions)
            np.testing.assert_array_equal(q['token_positions'][matching], positions)
            bf16[row['id']] = {f'{site}__pass_{p}': b[f'{site}__pass_{p}'].astype(np.float64)
                              for site in sites for p in (0, 1)}
            q4[row['id']] = {f'{site}__pass_{p}': q[f'{site}__pass_{p}'][matching].astype(np.float64)
                            for site in sites for p in (0, 1)}
    args.output.mkdir(parents=True, exist_ok=False)
    contract = dict(schema='nanbeige.sae_generalization_diagnostic.v1', exploratory_posthoc=True,
        fit_manifest_sha256=manifest_hash, q4_capture_manifest_sha256=digest(args.q4_captures/'manifest.json'),
        requested_ranks=[0, 8, 32, 64, 'full_training_rank'],
        training_ids=[r['id'] for r in training], evaluation_ids=[r['id'] for r in evaluation],
        fitted_data='Original BF16 training activations only, both passes with equal row counts',
        precision_comparison='Same exact tokens and positions; quantization and runtime effects are not isolated',
        source_code_sha256=digest(Path(__file__)), source_files=source_files,
        no_hyperparameter_selection=True, no_model_inference=True, no_new_SAE_training=True)
    write(args.output/'contract.json', contract)
    results, paired = {}, []
    orthogonality_checks = identity_checks = 0
    max_normalized_projection_identity_error = 0.
    saved_quality = read(args.fit/'sae_quality.json')
    for index, site in enumerate(sites):
        splits = {name: [np.concatenate([bf16[r['id']][f'{site}__pass_{p}'] for r in selected])
                         for p in (0, 1)] for name, selected in [('train', training), ('eval', evaluation)]}
        joined = np.concatenate(splits['train'])
        mean = joined.mean(0)
        _, singular, vh = torch.linalg.svd(torch.from_numpy(joined-mean), full_matrices=False)
        singular, vh = singular.numpy(), vh.numpy()
        tolerance = singular[0]*max(joined.shape)*np.finfo(np.float64).eps
        rank = int((singular > tolerance).sum())
        basis = vh[:rank].T.copy()
        np.testing.assert_allclose(basis.T@basis, np.eye(rank), rtol=1e-10, atol=1e-10)
        orthogonality_checks += 1
        ranks = sorted({0, min(8, rank), min(32, rank), min(64, rank), rank})
        dictionary_name = saved_quality[site]['dictionary']
        if digest(args.fit/dictionary_name) != saved_quality[site]['sha256']:
            raise ValueError('Changed sparse dictionary')
        sae = load_sae(args.fit/dictionary_name)
        sparse_outputs = {}
        result = dict(training_rows_per_pass=len(splits['train'][0]), training_rank=rank,
            singular_value_tolerance=tolerance, training_pass_variances=[energy(x-x.mean(0)) for x in splits['train']],
            quality=[], matched_runtime=[])
        arrays = dict(training_mean=mean, training_basis=basis, singular_values=singular)
        train_features = None
        for split, values in splits.items():
            features_by_pass = []
            for p, x in enumerate(values):
                prefix = f'{split}_pass_{p}'
                with torch.inference_mode():
                    reconstruction, features = sae(torch.from_numpy(x.astype(np.float32)))
                reconstruction, features = reconstruction.numpy().astype(np.float64), features.numpy()
                arrays[prefix+'__sae_reconstruction'] = reconstruction.astype(np.float32)
                sparse_outputs[prefix] = reconstruction
                features_by_pass.append(features)
                if split == 'eval':
                    expected = saved_quality[site]['held_out_quality'][f'pass_{p}']
                    observed = quality(x, reconstruction)
                    np.testing.assert_allclose(observed['mse'], expected['mse'], rtol=1e-5, atol=1e-5)
                    np.testing.assert_allclose(observed['explained_variance'], expected['explained_variance'], rtol=1e-5, atol=1e-5)
                for k in ranks:
                    selected = basis[:, :k]
                    coordinates = (x-mean)@selected
                    predicted = mean + coordinates@selected.T
                    measure = quality(x, predicted)
                    # Pythagoras computes projection error without using predicted rows.
                    total, projected = energy(x-mean), energy(coordinates)
                    normalized_error = abs(measure['squared_error']-(total-projected))/max(total+projected, 1e-30)
                    # Full training-rank projection has near-zero residual; its
                    # energy subtraction needs a tolerance scaled to its operands.
                    if normalized_error > 128*np.finfo(np.float64).eps:
                        raise ValueError('Projection energy identity disagrees beyond float64 roundoff')
                    max_normalized_projection_identity_error = max(max_normalized_projection_identity_error, normalized_error)
                    identity_checks += 1
                    result['quality'].append(dict(split=split, invocation=p, method='shared_training_pca', rank=k, **measure))
                result['quality'].append(dict(split=split, invocation=p, method='pass_training_mean',
                    **quality(x, np.broadcast_to(splits['train'][p].mean(0), x.shape))))
                residual = x-reconstruction
                parallel_energy = energy(residual@basis)
                orthogonal_energy = energy(residual-(residual@basis)@basis.T)
                np.testing.assert_allclose(parallel_energy+orthogonal_energy, energy(residual), rtol=1e-10, atol=1e-7)
                identity_checks += 1
                result['quality'].append(dict(split=split, invocation=p, method='saved_shared_sae',
                    **quality(x, reconstruction), error_energy_inside_training_span=parallel_energy,
                    error_energy_outside_training_span=orthogonal_energy,
                    reconstruction_to_target_norm_ratio=float(np.sqrt(energy(reconstruction)/energy(x))),
                    active_features=int((features > 0).any(0).sum()),
                    mean_l0=float((features > 0).sum(1).mean())))
            if split == 'train':
                train_features = np.concatenate(features_by_pass)
                result['training_active_features'] = int((train_features > 0).any(0).sum())
            else:
                test_features = np.concatenate(features_by_pass)
                result['eval_active_features'] = int((test_features > 0).any(0).sum())
                result['eval_active_but_never_positive_on_training'] = int(((test_features > 0).any(0) & ~(train_features > 0).any(0)).sum())
        for split, selected_rows in [('train', training), ('eval', evaluation)]:
            for p in (0, 1):
                x = np.concatenate([bf16[r['id']][f'{site}__pass_{p}'] for r in selected_rows])
                y = np.concatenate([q4[r['id']][f'{site}__pass_{p}'] for r in selected_rows])
                result['matched_runtime'].append(dict(split=split, invocation=p,
                    relative_difference_norm=float(np.sqrt(energy(y-x)/energy(x))),
                    pooled_cosine=float(np.sum(x*y)/np.sqrt(energy(x)*energy(y)))))
        for row in examples:
            for p in (0, 1):
                x, y = bf16[row['id']][f'{site}__pass_{p}'], q4[row['id']][f'{site}__pass_{p}']
                paired.append(dict(id=row['id'], split=row['split'], site=site, invocation=p,
                    relative_difference_norm=float(np.sqrt(energy(y-x)/energy(x))),
                    pooled_cosine=float(np.sum(x*y)/np.sqrt(energy(x)*energy(y)))))
        np.savez_compressed(args.output/f'diagnostic_{index:02d}.npz', **arrays)
        results[site] = result
        write(args.output/'summary.json', results)
        print(json.dumps(dict(site=site, training_rank=rank, eval_quality=[r for r in result['quality']
            if r['split']=='eval' and (r['method']=='saved_shared_sae' or r.get('rank')==rank)])), flush=True)
    write(args.output/'matched_runtime.json', paired)
    write(args.output/'manifest.json', dict(status='sae_generalization_diagnostic_complete',
        contract_sha256=digest(args.output/'contract.json'), orthogonality_checks=orthogonality_checks,
        projection_and_residual_identity_checks=identity_checks, matched_runtime_records=len(paired),
        max_normalized_projection_identity_error=max_normalized_projection_identity_error,
        projection_identity_tolerance=128*np.finfo(np.float64).eps,
        limitations=['Post-hoc pilot diagnostic; no new validation cohort or semantic claims.',
                     'PCA controls are dense and do not match SAE sparsity or provide feature labels.',
                     'Matching token inputs does not isolate quantization from implementation/runtime effects.',
                     'Projection rank is reported in full; no held-out rank selection.'],
        files={p.name:digest(p) for p in args.output.iterdir() if p.is_file()}))


if __name__ == '__main__':
    main()
