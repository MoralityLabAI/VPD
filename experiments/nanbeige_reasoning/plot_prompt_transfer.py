"""Plot measured prompt-only pass differences and frozen SAE transfer quality."""
import os
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path('artifacts/nanbeige_reasoning/mplconfig').resolve())
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('artifacts/nanbeige_reasoning')
    source=root/'local_prompt_capture_v2/frozen_sae_transfer'
    hidden=json.loads((source/'all_layer_hidden_differences.json').read_text())
    features=json.loads((source/'summary.json').read_text())
    records=list(dict.fromkeys(r['id'] for r in hidden))
    sites=[f'model.layers.{i}' for i in range(22)]+['model.norm']
    values={(r['id'],r['site']):r['cosine'] for r in hidden}
    labels={r['id']:r['lane'].replace('arc_agi_','ARC ').replace('intellect3_','I3 ')+': '+r['split'] for r in hidden}
    fig,(ax,quality)=plt.subplots(2,1,figsize=(12,8),gridspec_kw={'height_ratios':[1.15,1]},layout='constrained')
    matrix=np.asarray([[values[(record,site)] for site in sites] for record in records])
    im=ax.imshow(matrix,aspect='auto',vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_yticks(range(len(records)),[labels[r] for r in records])
    ax.set_xticks(range(23),list(map(str,range(1,23)))+['Norm'])
    ax.set_xlabel('Physical layer (1-based); normalization at the loop boundary')
    ax.set_title('Aligned prompt activations: cosine between first and second pass',loc='left',weight='bold')
    fig.colorbar(im,ax=ax,label='Cosine',fraction=.025,pad=.02)
    selected=sorted({r['site'] for r in features})
    x=np.arange(len(records))
    colors=['#2266aa','#ce6c23','#3c8a60']
    for i,site in enumerate(selected):
        rows={r['id']:r for r in features if r['site']==site}
        name={'model.layers.11':'Layer 12','model.layers.21':'Layer 22','model.norm':'Norm'}[site]
        for p,style in [(0,'--'),(1,'-')]:
            quality.plot(x,[rows[r][f'pass_{p}_quality']['explained_variance'] for r in records],
                style,marker='o',markersize=4,color=colors[i],label=f'{name}, pass {p+1}')
    quality.axhline(0,color='#777777',linewidth=.7)
    quality.set_xticks(x,[labels[r].replace(': ','\n') for r in records],fontsize=9)
    quality.set_ylabel('Explained variance (1 − SSE / SST)')
    quality.set_title('Frozen arithmetic SAEs transfer poorly to the broader prompt positions',loc='left',weight='bold')
    quality.legend(ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.24))
    quality.grid(axis='y',alpha=.2)
    fig.suptitle('Nanbeige4.2-3B · native BF16 · two fixed passes · prompt-only capture',fontsize=14)
    out=root/'figures'
    out.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'prompt_pass_transfer.{ext}',dpi=180)
    print('Saved prompt transfer figure for',len(records),'examples')


if __name__=='__main__':
    main()
