"""Render held-out layer/pass readouts, retaining undefined cells as missing."""
import argparse
import json
import os
from pathlib import Path
from .io import ROOT,digest,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.probes/'manifest.json').read_text(encoding='utf-8'))
    source=a.probes/'heatmap_data.json'
    assert digest(source)==manifest['files']['heatmap_data.json']
    rows=json.loads(source.read_text(encoding='utf-8'))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    targets=[('receipt_already_credited','Receipt already credited'),
        ('duplicate_choice_within_opportunity','Duplicate choice: vulnerable cases'),
        ('duplicate_choice_post_only','Duplicate choice: POST actions only')]
    anchors=[('prompt_end','Before reasoning'),('pre_action','Before action code')]
    for metric,title in [('auroc','Held-out AUROC'),('balanced_accuracy','Held-out balanced accuracy')]:
        fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
        cmap=plt.colormaps['viridis'].copy();cmap.set_bad('#d5d5d5')
        for j,(anchor,anchor_title) in enumerate(anchors):
            for i,(target,target_title) in enumerate(targets):
                selected=[r for r in rows if r['anchor']==anchor and r['target']==target]
                values=np.full((22,2),np.nan)
                for r in selected:
                    if r.get(metric) is not None:values[r['physical_layer']-1,r['pass_number']-1]=r[metric]
                ax=axes[j,i];im=ax.imshow(values,vmin=0,vmax=1,cmap=cmap,aspect='auto')
                ax.set(xticks=[0,1],xticklabels=['Pass 1','Pass 2'],yticks=[0,5,11,16,21],yticklabels=[1,6,12,17,22],ylabel='Physical layer')
                count=selected[0] if selected else {}
                ax.set_title(f'{target_title}\n{anchor_title}; n={count.get("train_examples",0)}/{count.get("test_examples",0)} train/test',fontsize=10)
                if not np.isfinite(values).any():ax.text(.5,.5,'Unavailable: class variation',transform=ax.transAxes,ha='center',rotation=90,color='#333333')
        fig.colorbar(im,ax=axes,label=title,shrink=.8)
        fig.suptitle(title+' · whole-world and template holdout\nGray = undefined; 0.5 is chance; decoding does not establish semantic localization',fontsize=12)
        fig.savefig(a.output/f'{metric}.png',dpi=180);fig.savefig(a.output/f'{metric}.pdf');plt.close(fig)
    notes=['# Held-out linear probes','',
        'Each row below describes a cohort, not an independent test at every layer. Multiple layers and both passes are inspected; any selected location is exploratory. The pre-action anchor follows generated reasoning and can therefore contain an explicit textual plan.','',
        '| Target | Anchor | Train/test episodes | Train/test worlds | Train classes 0/1 | Test classes 0/1 | Status |',
        '|---|---|---:|---:|---|---|---|']
    for target,_ in targets:
        for anchor,_ in anchors:
            selected=[r for r in rows if r['target']==target and r['anchor']==anchor]
            if not selected:continue
            r=selected[0]
            notes.append(f'| {target} | {anchor} | {r["train_examples"]}/{r["test_examples"]} | {r["train_worlds"]}/{r["test_worlds"]} | {r["train_class_counts"]} | {r["test_class_counts"]} | {r["status"]} |')
    notes+=['','Per-cell scores, held-out predictions, permutation-label baselines and available world-cluster intervals are in the source heatmap_data.json. A small number of held-out worlds limits generalization even when many paired conditions are present.']
    (a.output/'READOUTS.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    save(a.output/'manifest.json',dict(source_sha256=digest(source),code_sha256=digest(Path(__file__)),
        files={x.name:digest(x) for x in a.output.iterdir() if x.is_file()}))


if __name__=='__main__':main()
