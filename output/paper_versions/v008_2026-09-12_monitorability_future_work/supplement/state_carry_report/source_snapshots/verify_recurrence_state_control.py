"""Recheck captured causal-order and state-reset controls from immutable arrays."""
import argparse
from pathlib import Path
import numpy as np
from .intervention_change_fidelity import read,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=read(a.run/'manifest.json');protocol=read(a.run/'protocol.json')
    assert m['status']=='completed' and not protocol['canary']
    for name,h in m['files'].items():assert sha(a.run/name)==h
    records=read(a.run/'records.json');by={(r['id'],r['history'],r.get('layer'),r.get('dose')):r for r in records}
    assert len(by)==len(records)==450
    ids=[q['id'] for q in protocol['questions']]
    expected={(i,h,None,None) for i in ids for h in ('one_pass','carry','reset')}
    expected|={(i,h,l,d) for i in ids for h in ('one_pass','carry','reset') for l in (12,22) for d in [0.]+protocol['doses']}
    assert set(by)==expected
    parity=[];zero_checks=upstream_checks=write_checks=0
    def load(r):
        assert sha(a.run/r['archive'])==r['sha256']
        with np.load(a.run/r['archive'],allow_pickle=False) as f:return {k:f[k] for k in f.files}
    for identifier in ids:
        for layer,dose in [(None,None)]+[(l,d) for l in (12,22) for d in [0.]+protocol['doses']]:
            one=load(by[identifier,'one_pass',layer,dose]);reset=load(by[identifier,'reset',layer,dose])
            errors=[]
            for il in range(22):
                x,y=one[f'model.layers.{il}__pass_0'],reset[f'model.layers.{il}__pass_1']
                np.testing.assert_allclose(x,y,rtol=2e-5,atol=2e-4);errors.append(float(np.max(np.abs(x-y))))
            np.testing.assert_allclose(one['model.norm__pass_0'],reset['model.norm__pass_1'],rtol=2e-5,atol=2e-4)
            np.testing.assert_allclose(one['logits'],reset['logits'],rtol=2e-5,atol=2e-4)
            parity.append(dict(id=identifier,layer=layer,dose=dose,max_state_error=max(errors),max_logit_error=float(np.max(np.abs(one['logits']-reset['logits'])))))
        for history in ('one_pass','carry','reset'):
            baseline=load(by[identifier,history,None,None]);loops=1 if history=='one_pass' else 2
            for layer in (12,22):
                for dose in [0.]+protocol['doses']:
                    current=load(by[identifier,history,layer,dose])
                    for key,x in current.items():assert np.isfinite(x).all(),key
                    actual=current['write_after'].astype(np.float64)-current['write_before']
                    np.testing.assert_allclose(np.linalg.norm(actual),abs(dose),rtol=1e-3,atol=1e-5);write_checks+=1
                    for pidx in range(loops):
                        for il in range(22):
                            if dose==0 or pidx<loops-1 or il<layer-1:
                                key=f'model.layers.{il}__pass_{pidx}'
                                np.testing.assert_array_equal(current[key],baseline[key]);upstream_checks+=1
                    if dose==0:
                        np.testing.assert_array_equal(current['logits'],baseline['logits']);zero_checks+=1
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'parity.json',parity)
    write(a.output/'verification.json',dict(status='saved_controls_verified',source_manifest_sha256=sha(a.run/'manifest.json'),
        records=450,reset_one_pass_pairs=len(parity),identity_logit_checks=zero_checks,upstream_state_checks=upstream_checks,
        matched_write_norm_checks=write_checks,max_reset_state_error=max(r['max_state_error'] for r in parity),
        max_reset_logit_error=max(r['max_logit_error'] for r in parity),new_model_inference=False,
        implementation_sha256=sha(Path(__file__)),files={'parity.json':sha(a.output/'parity.json')}))
    print('Saved state controls verified',flush=True)


if __name__=='__main__':main()
