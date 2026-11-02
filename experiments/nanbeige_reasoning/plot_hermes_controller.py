"""Plot verified external repair controls and retain paired outcome counts."""
import argparse
import json
import os
from pathlib import Path
from .audit_retrieval_exposure import read,ROOT
from .fetch_sources import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    args=p.parse_args()
    root=args.run
    verification=json.loads((root/'verification.json').read_text())
    if verification['status']!='controller_outputs_independently_verified':
        raise ValueError('Controller verification required')
    manifest=json.loads((root/'manifest.json').read_text())
    if sha(root/'rows.jsonl')!=manifest['files']['rows.jsonl'] or sha(root/'summary.json')!=manifest['files']['summary.json']:
        raise ValueError('Verified results changed')
    rows=read(root/'rows.jsonl')
    summary=json.loads((root/'summary.json').read_text())
    paired=[]
    for subset in ('synthetic_dev','public_campsite'):
        selected={(r['id'],r['arm']):r for r in rows if r['slice']==subset}
        ids=sorted({r['id'] for r in rows if r['slice']==subset})
        for first,second in [('exact_empty','exact_retrieval'),('local_retrieval_passes_2','local_retrieval_passes_4'),('local_retrieval_passes_4','local_retrieval_passes_8')]:
            counts=dict(both_correct=0,first_only=0,second_only=0,both_incorrect=0,identical_output=0)
            for key in ids:
                a,b=selected[key,first],selected[key,second]
                field=('both_correct' if a['official_valid'] and b['official_valid'] else
                    'first_only' if a['official_valid'] else 'second_only' if b['official_valid'] else 'both_incorrect')
                counts[field]+=1
                counts['identical_output']+=int(a['action']==b['action'])
            paired.append(dict(slice=subset,first=first,second=second,cases=len(ids),**counts))
    (root/'paired_outcomes.json').write_text(json.dumps(paired,indent=2))
    os.environ.setdefault('MPLCONFIGDIR',str((ROOT/'mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True)
    caps=(0,1,2,4,8)
    for ax,subset,title in zip(axes,('public_campsite','synthetic_dev'),('Public Campsite (109 cases)','Synthetic development (32 cases)')):
        values=summary[subset]
        total=values['retrieval_only']['cases']
        counts=[values[f'local_retrieval_passes_{n}']['official_valid'] for n in caps]
        ax.plot(caps,[100*n/total for n in counts],'-o',color='#187e99',label='Local repair from retrieval')
        for x,count in zip(caps,counts):
            ax.annotate(str(count),(x,100*count/total),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9,
                bbox=dict(facecolor='white',edgecolor='none',pad=1))
        ax.axhline(100*values['exact_empty']['official_valid']/total,color='#6c757d',linestyle='--',label='Search + repair, empty start')
        ax.axhline(100*values['exact_retrieval']['official_valid']/total,color='#a04274',linestyle=':',linewidth=2,label='Search + repair, retrieved start')
        ax.set(title=title,xlabel='Configured local-repair pass cap',xticks=caps,ylim=(-4,108),yticks=(0,25,50,75,100))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Officially accepted answers (%)')
    fig.suptitle('External controller ablation: zero language-model calls\nPoint labels show accepted-case counts',fontsize=13,y=.97)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncols=3,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,top=.76,bottom=.22,wspace=.12)
    for suffix in ('png','pdf','svg'):
        fig.savefig(root/f'controller_passes.{suffix}',dpi=180)
    plt.close(fig)
    report=dict(status='verified_controller_figure_created',source_manifest_sha256=sha(root/'manifest.json'),
        verification_sha256=sha(root/'verification.json'),implementation_sha256=sha(Path(__file__)),
        files={name:sha(root/name) for name in ('paired_outcomes.json','controller_passes.png','controller_passes.pdf','controller_passes.svg')},
        scope='Local symbolic repair caps, not transformer loops. Search-enabled arms may fall back to eight local sweeps; all observed outputs independently verified.')
    (root/'figure_provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(paired,indent=2))


if __name__=='__main__':
    main()
