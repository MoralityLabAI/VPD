"""Frozen-SAE analysis of the native equal-executed-depth carry/reset assay."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .intervention_change_fidelity import read,write,sha,totals,scores,decode


def encode(x,w,b,pre,k,support=None,return_positive=False):
    positive=np.maximum((x-pre)@w.T+b,0.)
    if support is None:
        indices=np.argpartition(positive,-k,axis=1)[:,-k:]
        support=np.zeros_like(positive,dtype=bool)
        np.put_along_axis(support,indices,True,axis=1)
    result=(np.where(support,positive,0.),support)
    return (*result,positive) if return_positive else result


def pool(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rs in groups.items():
        t={field:sum(r[field] for r in rs) for field in ('signal_energy','reconstructed_energy','mismatch_energy','dot','coordinates')}
        result.append(dict(zip(keys,key))|dict(inputs=len(rs),**t,**scores(t)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run;out=a.output
    m=read(run/'manifest.json');protocol=read(run/'protocol.json')
    assert m['status']=='completed' and not protocol['canary']
    for name,digest in m['files'].items():assert sha(run/name)==digest,name
    assert m['protocol_sha256']==sha(run/'protocol.json')
    records=read(run/'records.json');assert len(records)==450
    control=Path('artifacts/nanbeige_reasoning/muon_sae_control_v1');cfg=read(control/'contract.json')
    assert sha(control/'contract.json')==protocol['dictionary_contract_sha256']
    assert {q['id'] for q in protocol['questions']}==set(cfg['evaluation_ids'])
    baseline={(r['id'],r['history']):r for r in records if r['condition']=='baseline'}
    loaded={};logit_states={};logit_effects=[]
    for r in records:
        with np.load(run/r['archive'],allow_pickle=False) as f:
            loaded[r['archive']]={site:f[f'{site}__pass_{r["loops"]-1}'].astype(np.float64) for site in cfg['sites']}
            logits=f['logits'].astype(np.float64)
            if r['condition']=='baseline':logit_states[r['id'],r['history']]=logits
            else:
                base=logit_states[r['id'],r['history']]
                def logsoft(x):
                    x=x-x.max()
                    return x-np.log(np.exp(x).sum())
                p0,p1=logsoft(base),logsoft(logits)
                logit_effects.append(dict(id=r['id'],history=r['history'],layer=r['layer'],dose=r['dose'],
                    rms_logit_change=float(np.sqrt(np.mean((logits-base)**2))),
                    baseline_to_intervention_kl=float(np.sum(np.exp(p0)*(p0-p1))),
                    argmax_changed=bool(base.argmax()!=logits.argmax())))
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(source_manifest_sha256=sha(run/'manifest.json'),
        primary='Full nonlinear frozen SAE encodings of complete baseline and intervention states; pooled zero-change-reference fidelity.',
        support_control='Freeze baseline top-k candidate support; retain ReLU and decoder. This changes the measuring instrument only, not the model.',
        state_direction_swap='At final layer 22 after a layer-12 +1 write, cross carry/reset baseline states with carry/reset response directions, normalized to L2=1. Offline representational assay; no new model states claimed.',
        statistics='Descriptive pooled signal/mismatch energy over 10 inputs. Related ARC grids and repeated doses/masks are not independent replications. No population confidence interval.',
        limits=['Equal executed depth does not equal effective dependent depth.','No separately trained untied model.','New CPU Q4 prompt-end distribution, not earlier BF16 continuation interventions.','Residual VPD-direction writes, not fixed-coefficient full-sequence weight edits.'],
        code_sha256=sha(Path(__file__))))
    rows=[];quality=[];crossed=[];sources={}
    for site in cfg['sites']:
        for optimizer in ('adamw','muon_hybrid'):
            folder=control/f'{cfg["sites"].index(site):02d}_{optimizer}'
            dp=folder/'dictionary.npz';dm=read(folder/'manifest.json')
            assert sha(dp)==dm['files']['dictionary.npz']
            sources[str(dp)]=sha(dp)
            with np.load(dp,allow_pickle=False) as d:
                w=d['encoder_weight'].astype(np.float64);b=d['encoder_bias'].astype(np.float64)
                dec=d['decoder'].astype(np.float64);pre=d['pre_bias'].astype(np.float64);k=int(d['top_k'])
            x=np.concatenate([loaded[r['archive']][site] for r in records])
            z,support,positive=encode(x,w,b,pre,k,return_positive=True)
            frozen_rows=z.copy()
            ix={r['archive']:i for i,r in enumerate(records)}
            rhat=decode(z,dec)+pre
            for history in ('one_pass','carry','reset'):
                indices=[i for i,r in enumerate(records) if r['condition']=='baseline' and r['history']==history]
                h=x[indices];recon=rhat[indices];error=float(np.sum((h-recon)**2));center=float(np.sum((h-h.mean(axis=0))**2))
                quality.append(dict(site=site,optimizer=optimizer,history=history,inputs=len(indices),state_error_energy=error,
                    state_energy=float(np.sum(h*h)),centered_state_energy=center,explained_variance=1-error/center if center else None,
                    relative_rmse=float(np.sqrt(error/np.sum(h*h)))))
            for i,r in enumerate(records):
                if r['condition']=='baseline':continue
                j=ix[baseline[r['id'],r['history']]['archive']]
                delta=x[i:i+1]-x[j:j+1]
                # Reuse identical batched encoder arithmetic: only support varies.
                frozen=np.where(support[j:j+1],positive[i:i+1],0.)
                frozen_rows[i:i+1]=frozen
                for mode,prediction in [('top_k',rhat[i:i+1]-rhat[j:j+1]),('frozen_support',decode(frozen-z[j:j+1],dec))]:
                    t=totals(delta,prediction)
                    row=dict(id=r['id'],history=r['history'],edited_layer=r['layer'],dose=r['dose'],site=site,optimizer=optimizer,
                        encoding=mode,**t,**scores(t),actual_write_norm=r['actual_write_norm'],
                        support_changed=bool(np.any((z[i]!=0)!=(z[j]!=0))))
                    if r['dose']==0:assert t['signal_energy']==t['mismatch_energy']==0
                    rows.append(row)
            if site=='model.layers.21':
                by={(r['id'],r['history'],r.get('layer'),r.get('dose')):i for i,r in enumerate(records) if r['condition']=='write'}
                for identifier in cfg['evaluation_ids']:
                    for state in ('carry','reset'):
                        j=ix[baseline[identifier,state]['archive']];h=x[j:j+1]
                        for donor in ('carry','reset'):
                            base=ix[baseline[identifier,donor]['archive']];i=by[identifier,donor,12,1.]
                            delta=x[i:i+1]-x[base:base+1];norm=np.linalg.norm(delta)
                            if norm==0:raise ValueError('Cannot normalize a zero propagated effect')
                            delta=delta/norm
                            current,_=encode(h+delta,w,b,pre,k)
                            predicted=decode(current-z[j:j+1],dec)
                            t=totals(delta,predicted)
                            crossed.append(dict(id=identifier,optimizer=optimizer,state_history=state,direction_history=donor,**t,**scores(t)))
            np.savez_compressed(out/f'encodings_{site}_{optimizer}.npz',
                full_top_k=z,frozen_baseline_support=frozen_rows,
                full_top_k_candidate_support=support,
                source_archives=np.asarray([r['archive'] for r in records]))
            print(site+' '+optimizer+' analyzed',flush=True)
    pooled=pool(rows,['history','edited_layer','dose','site','optimizer','encoding'])
    swaps=pool(crossed,['optimizer','state_history','direction_history'])
    contrasts=[]
    lookup={(r['history'],r['edited_layer'],r['dose'],r['site'],r['optimizer'],r['encoding']):r for r in pooled}
    for r in pooled:
        if r['history']!='carry' or not r['signal_energy']:continue
        c=lookup['reset',r['edited_layer'],r['dose'],r['site'],r['optimizer'],r['encoding']]
        contrasts.append(dict(edited_layer=r['edited_layer'],dose=r['dose'],site=r['site'],optimizer=r['optimizer'],encoding=r['encoding'],
            carry_fidelity=r['fidelity'],reset_fidelity=c['fidelity'],carry_minus_reset=r['fidelity']-c['fidelity'] if c['fidelity'] is not None else None))
    write(out/'per_input.json',rows);write(out/'pooled.json',pooled);write(out/'state_quality.json',quality)
    write(out/'crossed_per_input.json',crossed);write(out/'crossed_pooled.json',swaps);write(out/'contrasts.json',contrasts)
    write(out/'dictionary_sources.json',sources)
    write(out/'logit_effects.json',logit_effects)
    primary=[r for r in pooled if r['site']=='model.layers.21' and r['encoding']=='top_k' and r['dose']!=0 and r['history']!='one_pass']
    lines=['# Recurrent-state carry control','',
        'Actual CPU Nanbeige inference on ten complete held-out prompts. Carry and reset both execute 44 layers. Reset removes the first pass state before the second pass; it does not preserve effective dependent depth or simulate independently trained untied weights. One-pass parity validates the reset.','',
        '| Edited layer | Dose | SAE | Carry F | Reset F |','|---|---:|---|---:|---:|']
    for r in primary:
        if r['history']!='carry':continue
        other=lookup['reset',r['edited_layer'],r['dose'],r['site'],r['optimizer'],'top_k']
        lines.append(f'| {r["edited_layer"]} | {r["dose"]:+g} | {r["optimizer"]} | {r["fidelity"]:.4f} | {other["fidelity"]:.4f} |')
    lines+=['','Capture is the final prompt token at layer 22, final executed pass. Dose is absolute L2 residual-write magnitude along normalized VPD component-0 output direction. All six signed magnitudes and both edited layers are retained. Scores pool energy across ten inputs; conditions are not independent replications.','',
        'state_quality.json reports frozen-dictionary reconstruction on this distribution. pooled.json also includes final normalization and layer 12, with full top-k and fixed-baseline-support encodings. crossed_pooled.json separates baseline state and propagated direction in an offline L2=1 assay; those counterfactual vectors were not passed through the model.','',
        'This experiment tests state carry within one trained checkpoint. It cannot establish that looped training causes the SAE failure or isolate recurrence from effective depth.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'manifest.json',dict(status='completed',model_archive_records=len(records),per_input_rows=len(rows),
        new_model_inference=False,source_manifest_sha256=sha(run/'manifest.json'),
        files={f.name:sha(f) for f in out.iterdir() if f.is_file()}))


if __name__=='__main__':main()
