"""Task-grouped descriptions of saved, token-aligned recurrent states.

No model inference, dictionary fitting, or benchmark-score inference.
"""
import argparse
from collections import defaultdict,Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/nanbeige_reasoning'
REGIONS=['prompt','early','middle','late']
LANES=['arc_agi_1','arc_agi_2','intellect3_logic','intellect3_math']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def paired_metrics(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert a.shape==b.shape and a.ndim==2 and np.isfinite(a).all() and np.isfinite(b).all()
    n0=np.linalg.norm(a,axis=1);n1=np.linalg.norm(b,axis=1)
    assert np.all(n0>0) and np.all(n1>0)
    return dict(mean_cosine=float(np.mean(np.sum(a*b,axis=1)/(n0*n1))),
        mean_relative_change=float(np.mean(np.linalg.norm(b-a,axis=1)/n0)),
        mean_pass1_norm=float(n0.mean()),mean_pass2_norm=float(n1.mean()))


def grouped_states(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['lane'],r['calibration_split'],r['region'],r['physical_layer'],r['group_id'])].append(r)
    tasks=[]
    metrics=['mean_cosine','mean_relative_change','mean_pass1_norm','mean_pass2_norm']
    for (lane,split,region,layer,group),items in groups.items():
        tasks.append(dict(lane=lane,calibration_split=split,region=region,physical_layer=layer,group_id=group,
            inputs=len(items),**{k:float(np.mean([x[k] for x in items])) for k in metrics}))
    cells=defaultdict(list)
    for r in tasks:cells[(r['lane'],r['calibration_split'],r['region'],r['physical_layer'])].append(r)
    pooled=[]
    for (lane,split,region,layer),items in sorted(cells.items()):
        row=dict(lane=lane,calibration_split=split,region=region,physical_layer=layer,task_groups=len(items),inputs=sum(r['inputs'] for r in items))
        for key in metrics:
            values=[r[key] for r in items]
            row.update({key:float(np.mean(values)),key+'_min':float(min(values)),key+'_max':float(max(values))})
        pooled.append(row)
    return tasks,pooled


def render(output,rows):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained',sharey='row')
    for col,lane in enumerate(LANES):
        for region in REGIONS:
            values=sorted([r for r in rows if r['lane']==lane and r['calibration_split']=='eval' and r['region']==region],key=lambda r:r['physical_layer'])
            for row,key in enumerate(['mean_cosine','mean_relative_change']):
                ax=axes[row,col];line=ax.plot([r['physical_layer'] for r in values],[r[key] for r in values],label=region,lw=1.5)[0]
                ax.fill_between([r['physical_layer'] for r in values],[r[key+'_min'] for r in values],[r[key+'_max'] for r in values],alpha=.10,color=line.get_color())
                ax.set(xticks=[1,6,12,17,22],xlabel='Physical layer');ax.grid(alpha=.15)
            axes[0,col].set_title(lane.replace('_',' ')+'\n'+str(values[0]['task_groups'])+' held-out task groups',fontsize=11)
    axes[0,0].set_ylabel('Cosine: pass 1 vs pass 2');axes[1,0].set_ylabel('Change norm / pass-1 norm')
    axes[0,0].set_ylim(0,1)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Saved paired states across task families\nEqual task-group means; shading is the observed group range, not a confidence interval',fontsize=13)
    fig.savefig(output/'heldout_pass_dynamics.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    directory=DATA/'dataset_index_v1';manifest=read(directory/'manifest.json');assets=read(directory/'assets.json')
    assert sha(directory/'examples.jsonl')==manifest['index_sha256'] and sha(directory/'assets.json')==manifest['assets_sha256']
    examples=[json.loads(s) for s in (directory/'examples.jsonl').read_text(encoding='utf-8').splitlines()]
    verified={}
    def checked(name):
        path=(ROOT/name).resolve();assert path.is_relative_to(ROOT)
        if name not in verified:
            assert sha(path)==assets[name]['sha256'],name
            verified[name]=assets[name]['sha256']
        return path
    trajectories={};states=[];features=[];coverage=[]
    for example in examples:
        trace_ref=next(x for x in example['trajectories'] if x['selector']['loops']==2)
        name=trace_ref['file']
        if name not in trajectories:trajectories[name]={(r['id'],r['loops']):r for r in map(json.loads,checked(name).read_text(encoding='utf-8').splitlines())}
        trace=trajectories[name][(example['id'],2)]
        meta=example['activations'];assert meta['variant']=='extended_8192_native_two_pass'
        common={k:example[k] for k in ['id','lane','group_id','calibration_split','source_split']}
        common.update(max_token_censored=trace['max_token_censored'],thinking_closed=trace['thinking_closed'])
        coverage.append(dict(**common,generated_tokens=trace['generated_tokens']))
        with np.load(checked(meta['file']),allow_pickle=False) as archive:
            positions=archive['token_positions'];np.testing.assert_array_equal(positions,meta['token_positions'])
            np.testing.assert_array_equal(archive['token_ids'],trace['token_ids'])
            region_rows={region:np.flatnonzero((positions>=meta['regions'][region][0])&(positions<meta['regions'][region][1])) for region in REGIONS}
            assert all(len(v)==8 for v in region_rows.values())
            for layer in range(1,23):
                first=archive[f'model.layers.{layer-1}__pass_0'];second=archive[f'model.layers.{layer-1}__pass_1']
                assert first.shape==second.shape==(32,3072)
                for region,ids in region_rows.items():
                    states.append(dict(**common,region=region,physical_layer=layer,token_pairs=len(ids),**paired_metrics(first[ids],second[ids])))
            for feature in example['saes']:
                assert feature['variant']==meta['variant'];checked(feature['dictionary'])
                with np.load(checked(feature['file']),allow_pickle=False) as z:
                    np.testing.assert_array_equal(positions,z['token_positions'])
                    np.testing.assert_array_equal(z['pass_1']-z['pass_0'],z['delta_z'])
                    for region,ids in region_rows.items():
                        first=z['pass_0'][ids];second=z['pass_1'][ids];u=first!=0;v=second!=0
                        union=(u|v).sum(1);assert np.all(union>0)
                        features.append(dict(**common,region=region,site=feature['site'],optimizer=feature['optimizer'],dictionary_sha256=assets[feature['dictionary']]['sha256'],
                            token_pairs=len(ids),mean_support_jaccard=float(np.mean((u&v).sum(1)/union)),
                            mean_l0_pass1=float(u.sum(1).mean()),mean_l0_pass2=float(v.sum(1).mean()),
                            mean_feature_change_norm=float(np.linalg.norm(second.astype(np.float64)-first,axis=1).mean())))
    tasks,pooled=grouped_states(states)
    assert len(states)==18*22*4 and len(features)==18*9*4
    for name,rows in [('per_input_states',states),('per_task_states',tasks),('grouped_states',pooled),('per_input_features',features),('coverage',coverage)]:
        save(a.output/(name+'.json'),rows);write_csv(a.output/(name+'.csv'),rows)
    render(a.output,pooled)
    save(a.output/'manifest.json',dict(status='saved_pass_dynamics_described',examples=len(examples),physical_layers=22,passes=2,state_windows=len(states),feature_windows=len(features),
        checked_source_assets=verified,source_index_sha256=manifest['index_sha256'],code_sha256=sha(Path(__file__)),
        grouping='Average ARC test inputs within source task before averaging task groups. No token-row inferential sample size.',
        inferential_intervals=False,model_inference=False,semantic_localization_claim=False,reasoning_gain_claim=False,
        files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()}))
    print(json.dumps(dict(state_windows=len(states),feature_windows=len(features),verified_assets=len(verified),censored=sum(r['max_token_censored'] for r in coverage))))


if __name__=='__main__':main()
