"""Compare terminal-layer and loop-normalized geometry on the same saved rows."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .summarize_pass_dynamics import ROOT,DATA,REGIONS,read,save,sha,paired_metrics,write_csv


def main():
    source=DATA/'pass_dynamics_by_task_v1';out=DATA/'pass_boundary_geometry_v1';out.mkdir(exist_ok=False)
    source_manifest=read(source/'manifest.json')
    examples=[json.loads(s) for s in (DATA/'dataset_index_v1/examples.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sha(DATA/'dataset_index_v1/examples.jsonl')==source_manifest['source_index_sha256']
    rows=[]
    for e in examples:
        meta=e['activations'];path=ROOT/meta['file']
        assert sha(path)==source_manifest['checked_source_assets'][meta['file']]
        with np.load(path,allow_pickle=False) as z:
            pos=z['token_positions'];np.testing.assert_array_equal(pos,meta['token_positions'])
            for region in REGIONS:
                start,end=meta['regions'][region];ids=np.flatnonzero((pos>=start)&(pos<end));assert len(ids)==8
                raw=paired_metrics(z['model.layers.21__pass_0'][ids],z['model.layers.21__pass_1'][ids])
                norm=paired_metrics(z['model.norm__pass_0'][ids],z['model.norm__pass_1'][ids])
                rows.append(dict(id=e['id'],lane=e['lane'],group_id=e['group_id'],calibration_split=e['calibration_split'],region=region,
                    raw_cosine=raw['mean_cosine'],normalized_cosine=norm['mean_cosine'],
                    raw_relative_change=raw['mean_relative_change'],normalized_relative_change=norm['mean_relative_change'],
                    raw_pass1_norm=raw['mean_pass1_norm'],raw_pass2_norm=raw['mean_pass2_norm'],
                    normalized_pass1_norm=norm['mean_pass1_norm'],normalized_pass2_norm=norm['mean_pass2_norm']))
    metrics=[k for k in rows[0] if k.startswith(('raw_','normalized_'))]
    by_task=defaultdict(list)
    for r in rows:by_task[(r['lane'],r['calibration_split'],r['region'],r['group_id'])].append(r)
    cells=defaultdict(list)
    for (lane,split,region,group),items in by_task.items():cells[(lane,split,region)].append({k:float(np.mean([r[k] for r in items])) for k in metrics})
    pooled=[dict(lane=lane,calibration_split=split,region=region,task_groups=len(items),**{k:float(np.mean([r[k] for r in items])) for k in metrics}) for (lane,split,region),items in sorted(cells.items())]
    for name,data in [('per_input',rows),('grouped',pooled)]:save(out/(name+'.json'),data);write_csv(out/(name+'.csv'),data)
    save(out/'manifest.json',dict(status='saved_normalization_boundary_comparison_complete',source_manifest_sha256=sha(source/'manifest.json'),code_sha256=sha(Path(__file__)),
        rows=len(rows),new_model_inference=False,intervention_performed=False,
        interpretation='A change of geometry across the learned normalization readout is not a behavioral or semantic intervention.',
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps([r for r in pooled if r['calibration_split']=='eval' and r['region']=='late']))


if __name__=='__main__':main()
