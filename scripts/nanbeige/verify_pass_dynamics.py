"""Independent scalar arithmetic checks for the prespecified terminal/middle sites."""
import math
from pathlib import Path
import numpy as np
from .summarize_pass_dynamics import ROOT,DATA,read,save,sha


def main():
    out=DATA/'pass_dynamics_by_task_v1';manifest=read(out/'manifest.json')
    for name,digest in manifest['files'].items():assert sha(out/name)==digest,name
    import json
    examples=[json.loads(s) for s in (DATA/'dataset_index_v1/examples.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={(r['id'],r['physical_layer']):r for r in read(out/'per_input_states.json') if r['region']=='late'}
    checks=0;maximum=0.
    for e in examples:
        meta=e['activations'];path=ROOT/meta['file'];assert sha(path)==manifest['checked_source_assets'][meta['file']]
        with np.load(path,allow_pickle=False) as z:
            start,end=meta['regions']['late'];ids=[i for i,p in enumerate(z['token_positions']) if start<=p<end]
            for layer in (12,22):
                cosines=[];changes=[]
                for i in ids:
                    a=z[f'model.layers.{layer-1}__pass_0'][i].tolist();b=z[f'model.layers.{layer-1}__pass_1'][i].tolist()
                    n0=math.sqrt(math.fsum(x*x for x in a));n1=math.sqrt(math.fsum(x*x for x in b))
                    cosines.append(math.fsum(x*y for x,y in zip(a,b))/(n0*n1))
                    changes.append(math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b)))/n0)
                row=expected[(e['id'],layer)]
                for key,values in [('mean_cosine',cosines),('mean_relative_change',changes)]:
                    difference=abs(math.fsum(values)/len(values)-row[key]);maximum=max(maximum,difference)
                    assert difference<1e-10,(e['id'],layer,key,difference)
                    checks+=1
    boundary=read(DATA/'pass_boundary_geometry_v1/manifest.json')
    assert boundary['source_manifest_sha256']==sha(out/'manifest.json')
    for name,digest in boundary['files'].items():assert sha(DATA/'pass_boundary_geometry_v1'/name)==digest,name
    save(out/'verification.json',dict(status='saved_analysis_verified',analysis_manifest_sha256=sha(out/'manifest.json'),
        boundary_manifest_sha256=sha(DATA/'pass_boundary_geometry_v1/manifest.json'),output_hashes_checked=len(manifest['files'])+len(boundary['files']),
        scalar_metric_checks=checks,max_absolute_error=maximum,scope='Every example at physical layers 12/22, late sampled window; independent math.fsum scalar calculation.',
        verifier_sha256=sha(Path(__file__)),new_inference=False))
    print({'scalar_checks':checks,'max_absolute_error':maximum})


if __name__=='__main__':main()
