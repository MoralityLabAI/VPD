"""Project verified Q4 activations into a verified generated-prefix BF16 basis.

This computes transfer coordinates from saved inputs; it does not rerun the
model, fit Q4 weights, or establish behavioral effects.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest, read, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = read(args.captures / 'manifest.json')
    fit = read(args.fit / 'manifest.json')
    fit_hash = digest(args.fit / 'manifest.json')
    for name, status in [('independent_verification.json', 'multipass_saved_effects_independently_verified'),
                         ('cross_precision_verification.json', 'cross_precision_reasoning_vpd_verified')]:
        receipt = read(args.fit / name)
        if receipt['status'] != status or receipt['manifest_sha256'] != fit_hash:
            raise ValueError('A verified generated-prefix fit is required')
    if (capture['status'] != 'q4_reasoning_capture_complete'
            or fit['status'] != 'calibration_complete'
            or fit['vpd_training_window']['kind'] != 'generated_prefixes'):
        raise ValueError('Incomplete or mismatched source experiment')
    metadata = args.captures / 'records.jsonl'
    factors = args.fit / 'vpd_factors.npz'
    if (digest(metadata) != capture['files']['records.jsonl']
            or digest(factors) != fit['artifacts']['vpd_factors.npz']
            or digest(args.fit / 'examples.json') != fit['artifacts']['examples.json']):
        raise ValueError('Source artifact changed')
    records = rows(metadata)
    examples = {r['id']: r for r in read(args.fit / 'examples.json')}
    if len(records) != len(examples) or {r['id'] for r in records} != set(examples):
        raise ValueError('Capture and fitted trajectory cohorts differ')
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    max_coordinate_error = 0.
    with np.load(factors, allow_pickle=False) as basis:
        for index, record in enumerate(records):
            source = (args.captures / record['archive']).resolve()
            if not source.is_relative_to(args.captures.resolve()) or digest(source) != record['sha256']:
                raise ValueError('Activation source or hash mismatch')
            example = examples[record['id']]
            if record['split'] != example['split']:
                raise ValueError('Split mismatch')
            with np.load(source, allow_pickle=False) as activations:
                positions = activations['token_positions']
                if np.any(positions < 0) or np.any(positions >= len(example['token_ids'])):
                    raise ValueError('Invalid absolute token positions')
                np.testing.assert_array_equal(activations['token_ids'], example['token_ids'])
                for layer in (11, 21):
                    site = f'model.layers.{layer}.mlp.down_proj'
                    v, u = basis[site + '__V'], basis[site + '__U']
                    if v.shape[0] != 10752 or u.shape != (v.shape[1], 3072):
                        raise ValueError('Unexpected factor dimensions')
                    inputs = [activations[f'{site}.input__pass_{p}'] for p in (0, 1)]
                    coordinates = [x @ v for x in inputs]
                    for x, c in zip(inputs, coordinates):
                        if not np.isfinite(c).all():
                            raise ValueError('Nonfinite coordinate')
                        reference = x.astype(np.float64) @ v.astype(np.float64)
                        error = float(np.linalg.norm(c - reference) / max(np.linalg.norm(reference), 1e-30))
                        if error > 1e-5:
                            raise ValueError('Float64 coordinate recomputation disagrees')
                        max_coordinate_error = max(max_coordinate_error, error)
                    c0, c1 = coordinates
                    y0, y1 = c0 @ u, c1 @ u
                    path = args.output / f'layer_{layer + 1}_{index:03d}.npz'
                    np.savez_compressed(path, coordinates_0=c0, coordinates_1=c1,
                        delta_coordinates=c1-c0, basis_output_0=y0, basis_output_1=y1,
                        delta_basis_output=y1-y0, token_positions=positions,
                        token_ids=activations['token_ids'][positions])
                    with np.load(path, allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['delta_coordinates'], c1-c0)
                        np.testing.assert_array_equal(saved['delta_basis_output'], y1-y0)
                    summaries.append(dict(id=record['id'], site=site, split=record['split'],
                        components=int(v.shape[1]), rows=len(positions), archive=path.name,
                        sha256=digest(path), coordinate_difference_norm=float(np.linalg.norm(c1-c0)),
                        basis_output_difference_norm=float(np.linalg.norm(y1-y0))))
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2)+'\n', encoding='utf-8')
    manifest = dict(status='reasoning_vpd_coordinate_transfer_verified', records=len(summaries),
        capture_manifest_sha256=digest(args.captures / 'manifest.json'),
        capture_metadata_sha256=digest(metadata), fit_manifest_sha256=fit_hash,
        source_factors_sha256=digest(factors), source_code_sha256=digest(Path(__file__)),
        source_basis_weights='BF16', input_activation_weights='Q4_K_M',
        max_relative_float64_coordinate_error=max_coordinate_error,
        interpretation='Saved Q4 inputs projected into a frozen BF16 generated-prefix basis. '
                       'Not a BF16 activation capture, Q4 parameter fit, model rerun or answer intervention; '
                       'residual weights are omitted from diagnostic basis outputs.',
        files={p.name: digest(p) for p in args.output.iterdir() if p.is_file()})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}))


if __name__ == '__main__':
    main()
