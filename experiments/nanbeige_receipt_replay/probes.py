"""Linear readouts with world/template holdout and training-label permutations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import json
from pathlib import Path
import numpy as np
from .io import save,digest,ROOT
from .capture_contract import load_captures


def metrics(y,score):
    y=np.asarray(y,dtype=int);score=np.asarray(score)
    if set(y.tolist())!={0,1}:return dict(auroc=None,balanced_accuracy=None)
    pos=score[y==1];neg=score[y==0]
    auc=float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean())
    return dict(auroc=auc,balanced_accuracy=float(.5*((pos>=0).mean()+(neg<0).mean())))


def fit_linear(x,y,ridge=1.):
    """Class-balanced ridge classification; scaling sees training states only."""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=int)
    if set(y.tolist())!={0,1}:raise ValueError('Two training classes required')
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    z=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    counts=np.bincount(y,minlength=2);weight=len(y)/(2*counts[y]);root=np.sqrt(weight)
    zw=z*root[:,None]
    beta=zw.T@np.linalg.solve(zw@zw.T+ridge*np.eye(len(x)),(2*y-1)*root)
    return dict(mean=mean,scale=scale,coef=beta[:-1],intercept=np.asarray(beta[-1]),ridge=np.asarray(ridge))


def predict(model,x):
    return ((np.asarray(x)-model['mean'])/model['scale'])@model['coef']+model['intercept']


def permute_within_world(y,worlds,rng):
    result=np.array(y,copy=True)
    for world in sorted(set(worlds)):
        ids=np.flatnonzero(np.asarray(worlds)==world);result[ids]=rng.permutation(result[ids])
    return result


def clustered_interval(y,score,worlds,rng,n=1000):
    unique=sorted(set(worlds))
    if len(unique)<4:return dict(status='not_reported_fewer_than_four_test_worlds',worlds=len(unique))
    values=[]
    for _ in range(n):
        selected=rng.choice(unique,len(unique),replace=True)
        ids=np.concatenate([np.flatnonzero(np.asarray(worlds)==w) for w in selected])
        m=metrics(y[ids],score[ids])
        if m['auroc'] is not None:values.append([m['auroc'],m['balanced_accuracy']])
    if not values:return dict(status='undefined_resampled_classes')
    interval=np.quantile(values,[.025,.975],axis=0)
    return dict(status='exploratory_world_cluster_percentile',resamples=n,valid_resamples=len(values),
        auroc=interval[:,0].tolist(),balanced_accuracy=interval[:,1].tolist())


def cohort(records,target,anchor):
    chosen=[]
    for r in records:
        if not any(x['anchor']==anchor for x in r['anchors']):continue
        if target=='receipt_already_credited':
            if r['variant'] not in ['main','pending']:continue
            label=r['labels'][target]
        elif target=='duplicate_choice_within_opportunity':
            if r['variant']!='main':continue
            label=r['labels']['will_attempt_duplicate_credit']
        else:
            if r['variant'] not in ['main','pending'] or r['chosen_action']!='POST':continue
            label=r['labels']['will_attempt_duplicate_credit']
        if label is not None:chosen.append((r,int(label)))
    return chosen


def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--permutations',type=int,default=100);p.add_argument('--seed',type=int,default=791)
    a=p.parse_args()
    if a.permutations < 1:p.error('--permutations must be positive')
    m,records,data=load_captures(a.captures)
    a.output.mkdir(parents=True,exist_ok=False)
    train_worlds={r['world_id'] for r in records if r['split']=='dev'};test_worlds={r['world_id'] for r in records if r['split']=='holdout'}
    train_templates={r['template_id'] for r in records if r['split']=='dev'};test_templates={r['template_id'] for r in records if r['split']=='holdout'}
    assert not train_worlds&test_worlds and not train_templates&test_templates
    save(a.output/'contract.json',dict(model='Class-balanced linear ridge classifier; L2=1 including intercept',
        preprocessing='Training-only per-coordinate mean/std; zero-variance coordinates scale to one',
        split='All dev worlds/templates train; all holdout worlds/templates test. No random token-row split.',
        train_worlds=sorted(train_worlds),test_worlds=sorted(test_worlds),train_templates=sorted(train_templates),test_templates=sorted(test_templates),
        permutations=a.permutations,permutation_scheme='Shuffle training labels within each world; refit identical probe; evaluate on unchanged heldout labels. Descriptive null baseline, not a causal test.',
        seed=a.seed,capture_manifest_sha256=digest(a.captures/'manifest.json'),code_sha256=digest(Path(__file__)),
        capture_validator_sha256=digest(Path(__file__).with_name('capture_contract.py')),
        semantic_localization_claim=False,primary_anchor='prompt_end',secondary_anchor='pre_action, after native reasoning; potentially text-mediated',
        targets=dict(receipt_already_credited='Input fact; main versus legitimate pending controls crossed with condition',
            duplicate_choice_within_opportunity='Behavior within the same vulnerable, already-credited opportunity; excludes legal/blocked cases',
            duplicate_choice_post_only='Action-controlled secondary comparison of legitimate and duplicate POST decisions')))
    results=[];(a.output/'models').mkdir()
    for anchor in ['prompt_end','pre_action']:
        for target in ['receipt_already_credited','duplicate_choice_within_opportunity','duplicate_choice_post_only']:
            selected=cohort(records,target,anchor)
            train=[x for x in selected if x[0]['split']=='dev'];test=[x for x in selected if x[0]['split']=='holdout']
            yt=np.asarray([y for _,y in train],dtype=int);ye=np.asarray([y for _,y in test],dtype=int)
            wt=[r['world_id'] for r,_ in train];we=[r['world_id'] for r,_ in test]
            enough=set(yt.tolist())=={0,1} and set(ye.tolist())=={0,1}
            for layer in range(1,23):
                for invocation in [1,2]:
                    row=dict(target=target,anchor=anchor,physical_layer=layer,pass_number=invocation,
                        train_examples=len(train),test_examples=len(test),train_worlds=len(set(wt)),test_worlds=len(set(we)),
                        train_class_counts=np.bincount(yt,minlength=2).tolist(),test_class_counts=np.bincount(ye,minlength=2).tolist())
                    if not enough:
                        row.update(status='insufficient_class_variation',auroc=None,balanced_accuracy=None);results.append(row);continue
                    key=f'model.layers.{layer-1}__pass_{invocation-1}'
                    def states(items):
                        return np.stack([data[r['episode_id']][key][next(x['array_row'] for x in r['anchors'] if x['anchor']==anchor)] for r,_ in items])
                    xt,xe=states(train),states(test);model=fit_linear(xt,yt);score=predict(model,xe)
                    row.update(status='evaluated',**metrics(ye,score))
                    filename=f'{target}__{anchor}__L{layer:02d}_P{invocation}.npz'
                    np.savez_compressed(a.output/'models'/filename,**model)
                    row['probe_archive']='models/'+filename
                    row['probe_sha256']=digest(a.output/row['probe_archive'])
                    row['test_predictions']=[dict(episode_id=r['episode_id'],world_id=r['world_id'],label=y,score=float(s)) for (r,y),s in zip(test,score)]
                    rng=np.random.default_rng(a.seed)
                    null=[]
                    for _ in range(a.permutations):
                        shuffled=permute_within_world(yt,wt,rng)
                        null.append(metrics(ye,predict(fit_linear(xt,shuffled),xe)))
                    row['permutation_baseline']=dict(auroc_mean=float(np.mean([x['auroc'] for x in null])),
                        auroc_p95=float(np.quantile([x['auroc'] for x in null],.95)),balanced_accuracy_mean=float(np.mean([x['balanced_accuracy'] for x in null])),repeats=len(null))
                    row['confidence_interval']=clustered_interval(ye,score,we,np.random.default_rng(a.seed))
                    row['exploratory_candidate']=bool(target=='duplicate_choice_within_opportunity' and anchor=='prompt_end' and layer in (12,22)
                        and row['auroc']>=.75 and row['balanced_accuracy']>=.65 and row['auroc']>row['permutation_baseline']['auroc_p95'])
                    results.append(row)
            print(anchor,target,'train',len(train),'test',len(test),'classes adequate',enough,flush=True)
    save(a.output/'heatmap_data.json',results)
    candidates=[r for r in results if r.get('exploratory_candidate')]
    save(a.output/'intervention_candidates.json',dict(status='exploratory_candidates' if candidates else 'no_qualified_behavior_candidate',
        note='Heldout scores inspected across layers; any resulting intervention analysis is exploratory, not confirmation on an untouched test set.',candidates=candidates))
    save(a.output/'manifest.json',dict(status='linear_probe_analysis_complete',rows=len(results),evaluated=sum(r['status']=='evaluated' for r in results),
        files={str(p.relative_to(a.output)).replace('\\','/'):digest(p) for p in a.output.rglob('*') if p.is_file()}))


if __name__=='__main__':main()
