"""Independent direct delta-z decoding check of all primary L22 metrics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
from collections import defaultdict
import numpy as np
from .intervention_change_fidelity import read,write,sha


def main():
    root=Path('artifacts/nanbeige_reasoning');run=root/'recurrence_state_control_v1';analysis=root/'recurrence_state_analysis_v1'
    manifest=read(analysis/'manifest.json')
    for name,h in manifest['files'].items():assert sha(analysis/name)==h,name
    recs=read(run/'records.json');index={r['archive']:i for i,r in enumerate(recs)}
    baselines={(r['id'],r['history']):index[r['archive']] for r in recs if r['condition']=='baseline'}
    saved={(r['id'],r['history'],r['edited_layer'],r['dose'],r['optimizer'],r['encoding']):r
           for r in read(analysis/'per_input.json') if r['site']=='model.layers.21'}
    x=[]
    for r in recs:
        with np.load(run/r['archive'],allow_pickle=False) as a:x.append(a[f'model.layers.21__pass_{r["loops"]-1}'][0].astype(np.float64))
    x=np.stack(x);checks=0;max_error=0.;groups=defaultdict(lambda:[0.,0.])
    for opt in ('adamw','muon_hybrid'):
        dp=root/'muon_sae_control_v1'/('00_'+opt)/'dictionary.npz'
        assert sha(dp)==read(analysis/'dictionary_sources.json')[str(dp)]
        with np.load(dp,allow_pickle=False) as d:decoder=d['decoder'].astype(np.float64)
        with np.load(analysis/f'encodings_model.layers.21_{opt}.npz',allow_pickle=False) as z:
            assert list(z['source_archives'])==[r['archive'] for r in recs]
            for field,mode in [('full_top_k','top_k'),('frozen_baseline_support','frozen_support')]:
                values=z[field]
                assert values.shape==(450,6144) and np.isfinite(values).all()
                assert np.all(np.count_nonzero(values,axis=1)<=32)
                for i,r in enumerate(recs):
                    if r['condition']=='baseline':continue
                    j=baselines[r['id'],r['history']];dz=values[i]-z['full_top_k'][j];active=np.flatnonzero(dz)
                    predicted=dz[active]@decoder[active] if len(active) else np.zeros(3072)
                    actual=x[i]-x[j]
                    signal=float(actual@actual);mismatch=float((actual-predicted)@(actual-predicted))
                    reference=saved[r['id'],r['history'],r['layer'],r['dose'],opt,mode]
                    np.testing.assert_allclose([signal,mismatch],[reference['signal_energy'],reference['mismatch_energy']],rtol=1e-8,atol=1e-10)
                    if signal:
                        f=1-mismatch/signal;max_error=max(max_error,abs(f-reference['fidelity']))
                        np.testing.assert_allclose(f,reference['fidelity'],rtol=1e-7,atol=1e-8)
                    else:assert reference['fidelity'] is None
                    key=(r['history'],r['layer'],r['dose'],opt,mode)
                    groups[key][0]+=signal;groups[key][1]+=mismatch;checks+=1
    pooled=[r for r in read(analysis/'pooled.json') if r['site']=='model.layers.21']
    for r in pooled:
        s,m=groups[r['history'],r['edited_layer'],r['dose'],r['optimizer'],r['encoding']]
        np.testing.assert_allclose([s,m],[r['signal_energy'],r['mismatch_energy']],rtol=1e-8,atol=1e-9)
    out=root/'recurrence_state_analysis_validation_v1';out.mkdir(exist_ok=False)
    write(out/'verification.json',dict(status='primary_fidelity_independently_verified',primary_per_input_checks=checks,
        primary_pooled_checks=len(pooled),max_absolute_per_input_fidelity_difference=max_error,
        method='Directly decode saved feature differences, then calculate residual squared norms; compare with paired reconstruction differences and pooled metrics.',
        analysis_manifest_sha256=sha(analysis/'manifest.json'),code_sha256=sha(Path(__file__)),new_model_inference=False))
    print(dict(checks=checks,pooled=len(pooled),max_error=max_error))


if __name__=='__main__':main()
