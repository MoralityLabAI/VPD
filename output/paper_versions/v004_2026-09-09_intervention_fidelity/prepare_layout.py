"""Prepare v004 equations and a version-specific renderer, without model work."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[3]
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib')
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

v=Path(__file__).resolve().parent
pairs=json.loads((ROOT/'artifacts/nanbeige_reasoning/intervention_change_fidelity_v1/paired.json').read_text())
s=(v/'PAPER.md').read_text(encoding='utf-8')
for inv in [1,2]:
    wins=sum(r['site']=='model.norm' and r['invocation']==inv and r['fidelity_difference'] is not None and r['fidelity_difference']>0 for r in pairs)
    s=s.replace('{{NORM%d_WINS}}'%inv,str(wins))
assert '{{' not in s
(v/'PAPER.md').write_text(s,encoding='utf-8')
renderer=(ROOT/'output/paper_versions/v003_2026-09-09_mechanistic_essay/render_paper.py').read_text(encoding='utf-8')
renderer=renderer.replace('Version 3 | Mechanistic interpretation | 9 September 2026','Version 4 | Intervention fidelity | 9 September 2026')
renderer=renderer.replace("if text.startswith('Appendix B.'):","if text.startswith('Appendix '):")
renderer=renderer.replace('Shared weights and state-conditioned roles across transformer passes; research manuscript','Tracking shared parameter effects through transformer passes; intervention-change fidelity')
renderer=renderer.replace('def markup(text):', "def markup(text):\n    text=text.replace('F_delta', r'\\(F_{\\Delta}\\)')")
(v/'render_paper.py').write_text(renderer,encoding='utf-8')
(v/'equations').mkdir(exist_ok=True)
plt.rcParams.update({'mathtext.fontset':'stix','font.family':'STIXGeneral'})
items={
    'local_response':(1.0,[(.73,r'$\delta h_p=\alpha C_k u_p$'),(.25,r'$\left.\frac{\partial y}{\partial\alpha}\right|_{\alpha=0}=J_{p\rightarrow y}\,C_k u_p$')]),
    'accounting':(.9,[(.76,r'$\hat h=D(E(h)),\qquad e(h)=h-\hat h$'),(.25,r'$\Delta h=\left[\hat h_1-\hat h_0\right]+\left[e(h_1)-e(h_0)\right]=\Delta\hat h+\Delta e$')]),
    'fidelity':(.75,[(.5,r'$F_{\Delta}=1-\frac{\sum_i\Vert\Delta h_i-\Delta\hat h_i\Vert_2^2}{\sum_i\Vert\Delta h_i\Vert_2^2}$')])}
for name,(height,lines) in items.items():
    fig=plt.figure(figsize=(6.5,height),dpi=300,facecolor='white')
    for y,formula in lines:
        fig.text(.04,y,formula,fontsize=15,va='center')
    fig.savefig(v/'equations'/f'{name}.png',dpi=300)
    plt.close(fig)
print('Prepared equations and renderer; manuscript words',len(s.split()))
