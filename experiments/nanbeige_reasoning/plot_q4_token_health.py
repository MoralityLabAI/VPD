"""Publication artifacts for completed paired output-token diagnostics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .fetch_sources import sha


def main():
    root=Path('artifacts/nanbeige_reasoning')
    run=root/'q4_reasoning_pilot_v1'
    source=run/'generation_health.json'
    data=json.loads(source.read_text())
    if data['status']!='paired_generation_health_complete' or data['paired_inputs']!=18:
        raise ValueError('This figure requires the completed18-input pilot')
    if data['source_complete_line_prefix_sha256']!=sha(run/'generations.jsonl'):
        raise ValueError('Generation source changed')
    colors={'arc_agi_1':'#3476A8','arc_agi_2':'#D07A32','intellect3_logic':'#388768','intellect3_math':'#8B62A8'}
    names={'arc_agi_1':'ARC-AGI-1','arc_agi_2':'ARC-AGI-2','intellect3_logic':'INTELLECT logic','intellect3_math':'INTELLECT math'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(8.2,4.6))
    keys=['unigram_entropy_bits','longest_identical_token_run']
    labels=['Unigram entropy (bits)','Longest identical-token run']
    for ax,key,label in zip(axes,keys,labels):
        for row in data['pairs']:
            color=colors[row['lane']]
            ax.scatter(row['loops_1'][key],row['loops_2'][key],s=47,edgecolor=color,
                facecolor=color if row['calibration_split']=='eval' else 'white',linewidth=1.35,zorder=3)
        if key=='unigram_entropy_bits':
            ax.set_xlim(0,8);ax.set_ylim(0,8);ax.plot([0,8],[0,8],ls='--',color='#888888',lw=.8)
        else:
            ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.8,3000);ax.set_ylim(.8,3000)
            ax.plot([1,3000],[1,3000],ls='--',color='#888888',lw=.8)
        ax.set_xlabel('One pass');ax.set_ylabel('Two passes');ax.set_title(label,pad=10)
        ax.grid(alpha=.15);ax.set_box_aspect(1)
    handles=[Line2D([],[],marker='o',color='none',markerfacecolor=color,markeredgecolor=color,label=names[lane]) for lane,color in colors.items()]
    handles.extend([Line2D([],[],marker='o',color='none',markerfacecolor='white',markeredgecolor='#555555',label='Calibration input'),
        Line2D([],[],marker='o',color='none',markerfacecolor='#555555',markeredgecolor='#555555',label='Evaluation input')])
    fig.suptitle('Paired Q4 pilot: all 36 continuations reach the 2,048-token cap',fontsize=12,y=.98)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,columnspacing=1.6)
    fig.subplots_adjust(left=.09,right=.98,top=.83,bottom=.23,wspace=.34)
    out=root/'figures';out.mkdir(exist_ok=True)
    files=[]
    for suffix in ('png','pdf','svg'):
        path=out/f'q4_token_health.{suffix}'
        fig.savefig(path,dpi=220,facecolor='white');files.append(path)
    plt.close(fig)
    receipt=dict(status='completed_pilot_token_health_figure',source_sha256=sha(source),
        generation_sha256=sha(run/'generations.jsonl'),inputs=18,independent_eval_units=8,
        interpretation='Descriptive token statistics. No completed final answers at this cap; no reasoning-compression or accuracy-benefit claim.',
        files={p.name:sha(p) for p in files})
    (out/'q4_token_health.provenance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
