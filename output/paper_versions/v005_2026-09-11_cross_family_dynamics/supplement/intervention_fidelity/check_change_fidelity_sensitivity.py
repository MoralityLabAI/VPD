"""Layer-22 second-pass precision and top-k-support diagnostics on saved states."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from collections import defaultdict
from pathlib import Path
import numpy as np
from intervention_change_fidelity import BASE, ROOT, CONDITIONS, read, sha, write, decode, totals, scores


def main():
    output=BASE/'intervention_change_fidelity_v1/sensitivity'
    output.mkdir(exist_ok=False)
    fit=BASE/'prime-20260906T204058Z-561/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration'
    transfer=BASE/'muon_vpd_transfer_v1';control=BASE/'muon_sae_control_v1'
    parent=read(output.parent/'manifest.json')
    for name in ['per_input.json','pooled.json','sources.json']:
        assert sha(output.parent/name)==parent['files'][name]
    sources=read(output.parent/'sources.json')
    records=[r for r in read(transfer/'records.json') if r['site']=='model.layers.21']
    original=read(output.parent/'per_input.json');groups=defaultdict(list);rows=[];checks=0;max_encoding_diff=0.
    for arm in ['adamw','muon_hybrid']:
        path=control/f'00_{arm}/dictionary.npz'
        assert sha(path)==sources[str(path.relative_to(ROOT))]
        with np.load(path,allow_pickle=False) as d:
            w=d['encoder_weight'].astype(np.float64);b=d['encoder_bias'].astype(np.float64)
            bias=d['pre_bias'].astype(np.float64);decoder=d['decoder'].astype(np.float64);k=int(d['top_k'])
        for rec in [r for r in records if r['arm']==arm]:
            hp=fit/rec['source_archive'];zp=transfer/rec['archive']
            assert sha(hp)==sources[str(hp.relative_to(ROOT))] and sha(zp)==sources[str(zp.relative_to(ROOT))]
            with np.load(hp,allow_pickle=False) as hs,np.load(zp,allow_pickle=False) as zs:
                names=['baseline']+CONDITIONS
                x=np.concatenate([hs[f'{name}__model.layers.21__pass_1__h'].astype(np.float64) for name in names])
                positive=np.maximum((x-bias)@w.T+b,0)
                indices=np.argpartition(positive,-k,axis=1)[:,-k:]
                z=np.zeros_like(positive)
                np.put_along_axis(z,indices,np.take_along_axis(positive,indices,axis=1),axis=1)
                h0=x[:8];z0=z[:8];saved0=zs['baseline__pass_1'].astype(np.float64)
                for j,name in enumerate(names):
                    saved=zs[f'{name}__pass_1'].astype(np.float64)
                    np.testing.assert_allclose(z[j*8:(j+1)*8],saved,rtol=1e-4,atol=1e-3)
                    max_encoding_diff=max(max_encoding_diff,float(np.max(np.abs(z[j*8:(j+1)*8]-saved))))
                    checks+=1
                    if name=='baseline':continue
                    actual=x[j*8:(j+1)*8]-h0
                    prediction=decode(z[j*8:(j+1)*8]-z0,decoder)
                    saved_prediction=decode(saved-saved0,decoder)
                    source=next(r for r in original if r['example_id']==rec['example_id'] and r['physical_module']==rec['physical_module'] and r['site']==rec['site'] and r['invocation']==2 and r['arm']==arm and r['condition']==name)
                    np.testing.assert_allclose(totals(actual,saved_prediction)['mismatch_energy'],source['mismatch_energy'],rtol=1e-12,atol=1e-10)
                    flipped=np.any((saved!=0)!=(saved0!=0),axis=1)
                    t=totals(actual,prediction)
                    row=dict(example_id=rec['example_id'],physical_module=rec['physical_module'],arm=arm,condition=name,**t,**scores(t),
                        saved_support_changed_tokens=int(flipped.sum()),
                        mismatch_on_support_changed_tokens=float(np.sum((actual[flipped]-saved_prediction[flipped])**2)),
                        total_saved_mismatch=source['mismatch_energy'],
                        stable_support=totals(actual[~flipped],saved_prediction[~flipped]))
                    rows.append(row);groups[(rec['physical_module'],arm,name)].append(row)
        print(arm+' float64 re-encoding complete',flush=True)
    pooled=[]
    fields=['signal_energy','reconstructed_energy','mismatch_energy','dot','coordinates']
    for key,rs in groups.items():
        t={k:sum(r[k] for r in rs) for k in fields}
        st={k:sum(r['stable_support'][k] for r in rs) for k in fields}
        mismatch=sum(r['total_saved_mismatch'] for r in rs)
        pooled.append(dict(zip(['physical_module','arm','condition'],key))|dict(**t,**scores(t),
            support_changed_tokens=sum(r['saved_support_changed_tokens'] for r in rs),
            saved_mismatch_fraction_on_support_changed_tokens=sum(r['mismatch_on_support_changed_tokens'] for r in rs)/mismatch,
            stable_support={**st,**scores(st)}))
    write(output/'per_input.json',rows);write(output/'pooled.json',pooled)
    write(output/'manifest.json',dict(status='layer22_second_pass_sensitivity_complete',source_code_sha256=sha(__file__),
        parent_manifest_sha256=sha(output.parent/'manifest.json'),encoding_checks=checks,max_absolute_encoding_difference=max_encoding_diff,
        source_files='Validated against parent sources.json; all shared inputs and dictionaries unchanged',
        scope='Post-hoc numerical precision sensitivity and descriptive partition by top-k support change, not a causal support intervention',
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('complete',flush=True)


if __name__=='__main__':main()
