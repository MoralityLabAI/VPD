"""Publication figures derived from the independently recomputed solver audit."""
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path('artifacts/nanbeige_reasoning/.mpl-cache').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path('artifacts/nanbeige_reasoning')
    audit=json.loads((root/'hermes_audit/summary.json').read_text())
    n=audit['rows']
    arms=audit['historical_arms']
    totals=audit['totals']
    labels=['27B vanilla\n(historical)','27B + TRM\n(historical)',
            'Hermes solver\nfirst candidate','Hermes solver\n+ official filter']
    exact=[arms['vanilla']['exact'],arms['logic_skill_trm']['exact'],totals['first_exact'],totals['official_filtered_exact']]
    official=[arms['vanilla']['official_valid'],arms['logic_skill_trm']['official_valid'],totals['first_official_valid'],totals['official_filtered_valid']]
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    x=np.arange(len(labels))
    for offset,values,label,color in [(-.19,exact,'Canonical target match','#547895'),(.19,official,'Official verifier accepted','#bf713e')]:
        bars=ax.bar(x+offset,np.array(values)/n*100,width=.36,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v}/{n}' for v in values],padding=4,fontsize=10)
    ax.set(xticks=x,xticklabels=labels,ylabel='Percent of the 109 Campsite cases',ylim=(0,115),
           title='Same source cases, different reasoning systems and scoring contracts')
    ax.set_yticks([0,25,50,75,100])
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',frameon=False)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
    destination=root/'figures'
    destination.mkdir(exist_ok=True)
    for extension in ('png','pdf','svg'):
        fig.savefig(destination/f'hermes_campsite_audit.{extension}',dpi=180)
    plt.close(fig)
    print(destination/'hermes_campsite_audit.png')

if __name__=='__main__':
    main()
