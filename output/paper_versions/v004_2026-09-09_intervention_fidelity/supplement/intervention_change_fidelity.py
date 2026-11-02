"""Measure SAE reconstruction of archived VPD intervention changes on CPU.

No model execution or dictionary fitting. Metrics use saved nonlinear encodings
and float64 decoding. Zero-signal strata have undefined fidelity and cosine.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/nanbeige_reasoning'
CONDITIONS = ['first_plus', 'first_minus', 'second_plus', 'second_minus', 'both_plus', 'both_minus']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def decode(z, decoder):
    # Exact sparse multiplication, retaining every nonzero saved coordinate.
    return np.stack([row[row != 0] @ decoder[row != 0] for row in z])


def totals(actual, reconstructed):
    a, r = actual.astype(np.float64), reconstructed.astype(np.float64)
    return dict(signal_energy=float(np.sum(a*a)), reconstructed_energy=float(np.sum(r*r)),
                mismatch_energy=float(np.sum((a-r)**2)), dot=float(np.sum(a*r)), coordinates=a.size)


def scores(t):
    s, r, m, dot = [t[k] for k in ['signal_energy', 'reconstructed_energy', 'mismatch_energy', 'dot']]
    np.testing.assert_allclose(m, s+r-2*dot, rtol=1e-10, atol=1e-9)
    return dict(fidelity=1-m/s if s else None,
                cosine=dot/np.sqrt(s*r) if s and r else None,
                reconstructed_to_actual_norm=np.sqrt(r/s) if s else None,
                status='nonzero_signal' if s else 'zero_signal')


def self_checks():
    a=np.array([[1., 2.]])
    assert scores(totals(a,a))['fidelity']==1
    assert scores(totals(a,np.zeros_like(a)))['fidelity']==0
    assert scores(totals(a,-a))['fidelity']==-3
    assert scores(totals(a,2*a))['fidelity']==0
    assert scores(totals(a,2*a))['cosine']==1
    assert scores(totals(np.zeros_like(a),a))['fidelity'] is None
    # Large identical reconstruction errors cancel in a paired change.
    h0=np.array([[101.,102.]]);h1=h0+a
    r0=h0-100;r1=h1-100
    np.testing.assert_array_equal(h1-h0,(r1-r0)+((h1-r1)-(h0-r0)))
    assert scores(totals(h1-h0,r1-r0))['fidelity']==1
    return 8


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE/'intervention_change_fidelity_v1')
    args=p.parse_args()
    fit=BASE/'prime-20260906T204058Z-561/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration'
    transfer=BASE/'muon_vpd_transfer_v1'; control=BASE/'muon_sae_control_v1'
    fm=read(fit/'manifest.json');tm=read(transfer/'manifest.json');tc=read(transfer/'contract.json')
    cc=read(control/'contract.json');quality=read(transfer/'quality.json')
    assert sha(fit/'manifest.json')==tc['fit_manifest_sha256']
    assert sha(control/'contract.json')==tc['muon_contract_sha256']
    assert sha(transfer/'contract.json')==tm['contract_sha256']
    for name in ['records.json','quality.json']:
        assert sha(transfer/name)==tm['files'][name]
    assert sha(fit/'effects.json')==fm['artifacts']['effects.json']
    effects={r['archive']:r for r in read(fit/'effects.json')}
    records=read(transfer/'records.json')
    args.output.mkdir(parents=True,exist_ok=False)
    contract=dict(question='How faithfully does each frozen SAE reconstruct the changes caused by a shared parameter intervention?',
        created_utc=datetime.now(timezone.utc).isoformat(),source_code_sha256=sha(__file__),
        transfer_manifest_sha256=sha(transfer/'manifest.json'),fit_manifest_sha256=sha(fit/'manifest.json'),
        formula='F_delta = 1 - sum ||delta_h - delta_reconstruction||^2 / sum ||delta_h||^2',
        reference='Predicting zero intervention change',pooling='Sum energies and inner products over all 80 held-out token positions in each stratum; do not average token ratios',
        strata=['perturbed physical matrix','capture site','capture invocation','signed intervention condition','dictionary optimizer'],
        precision='Archived BF16-exact hidden states and archived float32 nonlinear feature encodings; float64 decoding and metrics',
        primary_filter='None; report all nonzero signals. Exact zero signal gives null fidelity and cosine.',
        resolution_sensitivity='Also report coordinate-restricted metrics where absolute actual change exceeds one larger adjacent baseline BF16 step. This is a descriptive sensitivity, not a validated noise floor.',
        inference='Post-hoc descriptive analysis on the existing ten-input/eight-task cohort and one paired SAE seed; no independent samples from repeated conditions',
        new_model_execution=False,new_dictionary_training=False)
    write(args.output/'contract.json',contract)
    n_tests=self_checks();rows=[];groups=defaultdict(list);files={};checked=0
    for site in cc['sites']:
        for arm in ['adamw','muon_hybrid']:
            folder=control/f'{cc["sites"].index(site):02d}_{arm}'
            dm=read(folder/'manifest.json');dp=folder/'dictionary.npz'
            q=next(x for x in quality if x['site']==site and x['arm']==arm)
            assert sha(dp)==dm['files']['dictionary.npz']==q['dictionary_sha256']
            files[str(dp.relative_to(ROOT))]=sha(dp)
            with np.load(dp,allow_pickle=False) as d:
                decoder=d['decoder'].astype(np.float64);bias=d['pre_bias'].astype(np.float64)
            for rec in [r for r in records if r['site']==site and r['arm']==arm]:
                hp=fit/rec['source_archive'];zp=transfer/rec['archive'];effect=effects[hp.name]
                assert sha(hp)==rec['source_archive_sha256']==fm['artifacts'][hp.name]
                assert sha(zp)==rec['sha256']==tm['files'][zp.name]
                assert effect['example_id']==rec['example_id'] and effect['alpha']==.05
                files[str(hp.relative_to(ROOT))]=sha(hp);files[str(zp.relative_to(ROOT))]=sha(zp)
                with np.load(hp,allow_pickle=False) as hs,np.load(zp,allow_pickle=False) as zs:
                    for key in ['token_ids','token_positions']:
                        np.testing.assert_array_equal(hs[key],zs[key])
                    for inv in [0,1]:
                        h0=hs[f'baseline__{site}__pass_{inv}__h'].astype(np.float64)
                        z0=zs[f'baseline__pass_{inv}'].astype(np.float64)
                        r0=decode(z0,decoder)+bias
                        _,exponents=np.frexp(np.abs(h0))
                        step=np.exp2(np.where(h0==0,-133,np.maximum(exponents-8,-133)))
                        for condition in CONDITIONS:
                            raw=hs[f'{condition}__{site}__pass_{inv}__h']
                            assert raw.dtype==np.float32 and not np.any(raw.view(np.uint32)&65535)
                            h1=raw.astype(np.float64);z1=zs[f'{condition}__pass_{inv}'].astype(np.float64)
                            actual=h1-h0
                            reconstructed=decode(z1-z0,decoder)
                            r1=decode(z1,decoder)+bias
                            np.testing.assert_allclose(reconstructed,r1-r0,rtol=1e-9,atol=1e-11)
                            error_change=(h1-r1)-(h0-r0)
                            np.testing.assert_allclose(actual,reconstructed+error_change,rtol=1e-9,atol=1e-11)
                            t=totals(actual,reconstructed)
                            np.testing.assert_allclose(t['mismatch_energy'],np.sum(error_change**2),rtol=1e-9,atol=1e-10)
                            mask=np.abs(actual)>step
                            restricted=totals(actual[mask],reconstructed[mask])
                            changed=actual!=0
                            switched=np.any((z1!=0)!=(z0!=0),axis=1)
                            row=dict(example_id=rec['example_id'],task_id=effect['pair_id'],physical_module=rec['physical_module'],
                                site=site,invocation=inv+1,arm=arm,condition=condition,component=0,alpha=.05 if condition.endswith('plus') else -.05,
                                tokens=len(h0),**t,**scores(t),
                                baseline_sse=float(np.sum((h0-r0)**2)),intervention_sse=float(np.sum((h1-r1)**2)),
                                changed_coordinates=int(changed.sum()),within_one_bf16_step=int(np.sum(changed & ~mask)),
                                support_changed_tokens=int(switched.sum()),resolution_restricted={**restricted,**scores(restricted)})
                            rows.append(row);groups[(rec['physical_module'],site,inv+1,arm,condition)].append(row)
                            checked+=3
            print(json.dumps(dict(site=site,arm=arm,rows=len(rows))),flush=True)
    pooled=[]
    fields=['signal_energy','reconstructed_energy','mismatch_energy','dot','coordinates']
    for key,subset in groups.items():
        t={k:sum(r[k] for r in subset) for k in fields}
        rt={k:sum(r['resolution_restricted'][k] for r in subset) for k in fields}
        changed=sum(r['changed_coordinates'] for r in subset)
        vals=[r['fidelity'] for r in subset if r['fidelity'] is not None]
        row=dict(zip(['physical_module','site','invocation','arm','condition'],key))
        row.update(examples=len(subset),tasks=len({r['task_id'] for r in subset}),tokens=sum(r['tokens'] for r in subset),**t,**scores(t),
            changed_coordinates=changed,within_one_bf16_step=sum(r['within_one_bf16_step'] for r in subset),
            support_changed_tokens=sum(r['support_changed_tokens'] for r in subset),
            per_input_fidelity_min=min(vals) if vals else None,per_input_fidelity_median=float(np.median(vals)) if vals else None,
            per_input_fidelity_max=max(vals) if vals else None,
            signal_energy_above_one_step_fraction=rt['signal_energy']/t['signal_energy'] if t['signal_energy'] else None,
            resolution_restricted={**rt,**scores(rt)})
        pooled.append(row)
    pairs=[]
    for a in [r for r in pooled if r['arm']=='adamw']:
        m=next(r for r in pooled if r['arm']=='muon_hybrid' and all(r[k]==a[k] for k in ['physical_module','site','invocation','condition']))
        assert a['signal_energy']==m['signal_energy']
        pairs.append({k:a[k] for k in ['physical_module','site','invocation','condition']} | dict(adamw_fidelity=a['fidelity'],muon_fidelity=m['fidelity'],
            fidelity_difference=m['fidelity']-a['fidelity'] if a['fidelity'] is not None else None,
            adamw_cosine=a['cosine'],muon_cosine=m['cosine']))
    assert len(rows)==1440 and len(pooled)==144 and len(pairs)==72
    write(args.output/'per_input.json',rows);write(args.output/'pooled.json',pooled);write(args.output/'paired.json',pairs)
    write(args.output/'sources.json',files)
    manifest=dict(status='intervention_change_fidelity_complete',created_utc=datetime.now(timezone.utc).isoformat(),
        input_rows=len(rows),pooled_strata=len(pooled),paired_strata=len(pairs),nonzero_paired_strata=sum(r['fidelity_difference'] is not None for r in pairs),
        muon_higher_fidelity_strata=sum(r['fidelity_difference'] is not None and r['fidelity_difference']>0 for r in pairs),
        accounting_and_dual_decode_checks=checked,analytical_self_checks=n_tests,source_files_checked=len(files),
        checks_scope='Float64 decoder-difference versus separate reconstruction, reconstruction-error identity, independent energy formula; not independent model execution',
        files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()})
    write(args.output/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    main()
