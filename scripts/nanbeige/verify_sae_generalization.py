"""Check saved training-span diagnostics using a separate QR construction."""
import argparse
import json
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np

from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    args = p.parse_args()
    report = read(args.diagnostic/'manifest.json')
    if report['status'] != 'sae_generalization_diagnostic_complete':
        raise ValueError('Completed diagnostic required')
    for name, expected in report['files'].items():
        if Path(name).name != name or digest(args.diagnostic/name) != expected:
            raise ValueError('Changed diagnostic file')
    contract = read(args.diagnostic/'contract.json')
    if digest(args.fit/'manifest.json') != contract['fit_manifest_sha256']:
        raise ValueError('Fit identity changed')
    manifest = read(args.fit/'manifest.json')
    examples = read(args.fit/'examples.json')
    training = [r for r in examples if r['id'] in contract['training_ids']]
    evaluation = [r for r in examples if r['id'] in contract['evaluation_ids']]
    if (set(r['id'] for r in training) & set(r['id'] for r in evaluation)
            or len(training) != 8 or len(evaluation) != 10):
        raise ValueError('Invalid diagnostic split')
    data = {}
    for row in examples:
        name = manifest['baseline_archives'][row['id']]
        if digest(args.fit/name) != contract['source_files']['bf16/'+name]:
            raise ValueError('Changed source activation')
        with np.load(args.fit/name, allow_pickle=False) as a:
            data[row['id']] = {k:a[k].astype(np.float64) for k in a.files if '__pass_' in k}
    summary = read(args.diagnostic/'summary.json')
    checks = []
    with np.errstate(over='raise', invalid='raise'):
        for index, (site, result) in enumerate(summary.items()):
            train = [np.concatenate([data[r['id']][f'{site}__pass_{i}'] for r in training]) for i in (0,1)]
            joined = np.concatenate(train)
            mean = joined.mean(0)
            rank = result['training_rank']
            if rank != len(joined)-1:
                raise ValueError('QR control expects the observed full centered row rank')
            # Centering makes the last row the negative sum of the others.
            # QR of the remaining independent rows recovers the same span as
            # SVD without reusing the saved right singular vectors.
            orthogonal, triangular = np.linalg.qr((joined[:-1]-mean).T, mode='reduced')
            if np.min(np.abs(np.diag(triangular))) < 1e-9:
                raise ValueError('QR control rows are unexpectedly dependent')
            with np.load(args.diagnostic/f'diagnostic_{index:02d}.npz',allow_pickle=False) as saved:
                np.testing.assert_allclose(saved['training_mean'],mean,rtol=1e-12,atol=1e-12)
                for split, selected in [('train',training),('eval',evaluation)]:
                    for i in (0,1):
                        x = np.concatenate([data[r['id']][f'{site}__pass_{i}'] for r in selected])
                        projection = mean + ((x-mean)@orthogonal)@orthogonal.T
                        sse = float(np.square(x-projection).sum())
                        expected = next(r for r in result['quality'] if r['split']==split
                            and r['invocation']==i and r['method']=='shared_training_pca' and r['rank']==rank)
                        np.testing.assert_allclose(sse,expected['squared_error'],rtol=1e-9,atol=1e-7)
                        reconstruction = saved[f'{split}_pass_{i}__sae_reconstruction'].astype(np.float64)
                        sae_sse = float(np.square(x-reconstruction).sum())
                        expected_sae = next(r for r in result['quality'] if r['split']==split
                            and r['invocation']==i and r['method']=='saved_shared_sae')
                        np.testing.assert_allclose(sae_sse,expected_sae['squared_error'],rtol=1e-12,atol=1e-9)
                        checks.append(dict(site=site,split=split,invocation=i,qr_projection_sse=sse,
                            saved_sae_sse=sae_sse))
    receipt = dict(status='training_span_diagnostic_qr_verified',
        diagnostic_manifest_sha256=digest(args.diagnostic/'manifest.json'),
        checks=checks, source_code_sha256=digest(Path(__file__)),
        scope='Independent QR full-training-span reconstruction and saved SAE reconstruction errors. '
              'Not independent verification of all truncated PCA ranks or matched-runtime metrics.')
    (args.diagnostic/'qr_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],qr_controls=len(checks),saved_sae_checks=len(checks))))


if __name__=='__main__':
    main()
