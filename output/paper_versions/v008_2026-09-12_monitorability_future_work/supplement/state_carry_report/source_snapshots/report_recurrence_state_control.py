"""Create a compact scientific report from the completed, checked control."""
import os
os.environ.setdefault('MPLCONFIGDIR','C:/projects/VPD/tmp/matplotlib')
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from .intervention_change_fidelity import read,sha,write


def main():
    root=Path('artifacts/nanbeige_reasoning');analysis=root/'recurrence_state_analysis_v1'
    validation=read(root/'recurrence_state_validation_v1/verification.json')
    manifest=read(analysis/'manifest.json')
    for name,h in manifest['files'].items():assert sha(analysis/name)==h
    assert validation['source_manifest_sha256']==manifest['source_manifest_sha256']
    rows=read(analysis/'pooled.json');quality=read(analysis/'state_quality.json')
    crossed=read(analysis/'crossed_pooled.json');logs=read(analysis/'logit_effects.json')
    out=root/'recurrence_state_report_v2';out.mkdir(exist_ok=False)
    get=lambda hist,layer,dose,opt,mode='top_k':next(r for r in rows if (r['history'],r['edited_layer'],r['dose'],r['site'],r['optimizer'],r['encoding'])==(hist,layer,dose,'model.layers.21',opt,mode))
    fig,axes=plt.subplots(2,2,figsize=(10,7.5),sharex=True,layout='constrained')
    for i,layer in enumerate((12,22)):
        for j,opt in enumerate(('adamw','muon_hybrid')):
            ax=axes[i,j]
            all_values=[]
            for history,color in [('carry','#b25036'),('reset','#24759c')]:
                for sign,style in [(1,'-'),(-1,'--')]:
                    doses=[.1,1.,10.];values=[get(history,layer,sign*d,opt)['fidelity'] for d in doses]
                    all_values.extend(values)
                    ax.plot(doses,values,style,marker='o',color=color,label=history+(' +' if sign==1 else ' -'))
            ax.axhline(0,color='black',lw=.7);ax.set_xscale('log')
            if i==1:
                ax.set_yscale('symlog',linthresh=1)
                bottom=min(all_values)*1.25
                ax.set_ylim(bottom,.1)
                ax.set_yticks([v for v in (-100,-10,-1,0) if bottom<=v<=.1])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
            ax.set_title(f'Write at L{layer}; {opt.replace("_"," ")}')
            ax.set_ylabel('Change fidelity'+(' (symmetric-log axis)' if i==1 else ''));ax.grid(alpha=.18)
            if i==1:ax.set_xlabel('Absolute L2 write magnitude')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Final-layer SAE change fidelity: carried versus reset state\n10 full held-out prompts; capture at final prompt token; descriptive pooled energy')
    fig.savefig(out/'carry_reset_fidelity.png',dpi=180);plt.close(fig)
    lines=['# Does recurrent state cause the reconstruction problem?','',
        '**The failure does not require second-pass state carry, but carry changes what the SAE can track.** The controlled results separate a limitation of the sparse measurement from a change in the model\'s response to an intervention.','',
        'In the one-pass reference and the equivalent 44-layer reset control, Muon reconstructs layer-22 baseline states with 0.862 explained variance. Nevertheless, all six local signed-dose conditions have negative intervention-change fidelity. For the -1 write, holding the same hidden-state change fixed while freezing the baseline feature support improves fidelity from -52.37 to approximately -0.0005. This removes the large error spike without making the reconstruction meaningfully accurate. The spike is attributable to the SAE\'s support-selection step in this assay; it does not require a second executed pass.','',
        'State carry still matters for propagated effects. With the same layer-12 write magnitude, all six Muon conditions measured at layer 22 have positive fidelity under reset (0.047 to 0.572) and negative fidelity under carry (-0.300 to -0.046). This is not a universal worsening: AdamW has higher fidelity with carry in ten of the twelve local/propagated conditions, although every one remains negative. The effect depends on the dictionary and the response being measured.','',
        'The unit-norm direction-swap assay further distinguishes the starting state from the propagated change. For Muon, carry-derived directions have fidelity around 0.05-0.06 on either baseline, while reset-derived directions reach about 0.35-0.38 on either baseline. In this specific offline control, changing the response direction has a much larger effect than swapping the baseline state. Thus poorer baseline state reconstruction alone is insufficient to explain the difference: state carry also changes the direction of the model\'s downstream response relative to the dictionary.','',
        'This experiment manipulates state carry within the existing Nanbeige checkpoint. It tests a specific pathway through which recurrence could affect measurement fidelity. It does not compare separately trained looped and unlooped models.','',
        'The control executes both 22-layer passes, then replaces each token\'s boundary state with its original embedding before the second pass. Carry and reset therefore execute 44 layers with the same weights and token sequence. Reset removes cross-pass information and reduces effective dependent depth; executed depth and effective depth must not be conflated. A separate 22-layer one-pass run validates the reset.','',
        f'All {validation["reset_one_pass_pairs"]} baseline/intervention pairs passed the reset-versus-one-pass check. Maximum residual-state error was {validation["max_reset_state_error"]:.6g}; maximum logit error was {validation["max_reset_logit_error"]:.6g}. There were {validation["identity_logit_checks"]} identity-write checks, {validation["upstream_state_checks"]} upstream-state checks and {validation["matched_write_norm_checks"]} write-norm checks.','',
        '## Matched state and write controls','',
        'The cohort is the original ten held-out inputs (eight task groups), spanning both ARC families and INTELLECT-3 logic/math. Full prompts are used, with no new generated reasoning. Writes affect the final prompt token at layer 12 or 22, in the final executed pass. They follow the normalized output direction of already-fitted VPD component 0, with absolute norms 0.1, 1 and 10 in both signs. These are residual interventions along a VPD direction, not repetitions of the paper\'s fixed-coefficient weight edits.','',
        '![Matched-dose reconstruction fidelity](carry_reset_fidelity.png)','',
        '## Frozen dictionary quality on the new distribution','',
        '| Site | Dictionary | One-pass EV | Carry EV | Reset EV |','|---|---|---:|---:|---:|']
    for site in ('model.layers.11','model.layers.21','model.norm'):
        for opt in ('adamw','muon_hybrid'):
            q={r['history']:r for r in quality if r['site']==site and r['optimizer']==opt}
            lines.append(f'| {site} | {opt} | {q["one_pass"]["explained_variance"]:.4f} | {q["carry"]["explained_variance"]:.4f} | {q["reset"]["explained_variance"]:.4f} |')
    lines+=['','EV uses the across-input mean at that site/history as reference. Ten prompt-end vectors give a small, previously inspected sample; these are descriptive scores. Frozen SAE weights are identical across histories.','',
        '## Does failure remain after removing state carry?','',
        '| Dictionary at L22 | Negative F: carry / 12 | Negative F: reset / 12 | Carry higher F / 12 | Frozen support higher F: carry / 12 |','|---|---:|---:|---:|---:|']
    for opt in ('adamw','muon_hybrid'):
        pairs=[(get('carry',l,d,opt),get('reset',l,d,opt),get('carry',l,d,opt,'frozen_support')) for l in (12,22) for d in (-.1,.1,-1.,1.,-10.,10.)]
        lines.append(f'| {opt} | {sum(c["fidelity"]<0 for c,r,f in pairs)} | {sum(r["fidelity"]<0 for c,r,f in pairs)} | {sum(c["fidelity"]>r["fidelity"] for c,r,f in pairs)} | {sum(f["fidelity"]>c["fidelity"] for c,r,f in pairs)} |')
    lines+=['','Each column counts repeated conditions, not independent tasks. Negative F means reconstruction mismatch exceeds actual change energy. The complete signed-dose table is in the analysis directory.','',
        'Freezing the baseline top-k support changes only the dictionary measurement: it retains the same encoder, ReLU and decoder while preventing new support selection. It is not an intervention on the model\'s internal computation.','',
        '## Baseline state versus propagated direction','',
        'For a layer-12 +1 write, an additional offline assay crosses carry/reset baseline states with the corresponding L22 response directions, normalizing each response to L2=1. This keeps magnitude fixed at the measurement site. Counterfactual vectors are re-encoded through the frozen dictionary; they were not passed through the model.','',
        '| Dictionary | Baseline state | Direction donor | F |','|---|---|---|---:|']
    for r in crossed:lines.append(f'| {r["optimizer"]} | {r["state_history"]} | {r["direction_history"]} | {r["fidelity"]:.4f} |')
    lines+=['','## Downstream endpoint and limits','',
        'Next-token logits are preserved for every model replay. Among 120 nonzero-dose conditions per history, the argmax changes in two one-pass/reset conditions and none of the carry conditions. Maximum baseline-to-intervention KL is 0.00269 for one-pass/reset and 0.03726 for carry. Distributional effects therefore occur without an argmax change, but these are not solved-task scores or evidence of a semantic mediator.','',
        'The experiment can show whether the reconstruction problem requires cross-pass state carry in this checkpoint and whether carry changes its severity at a fixed direct-write magnitude. It cannot isolate looped training, architectural weight sharing or effective depth as a general cause. A matched independently trained untied model remains necessary for that stronger comparison. It also uses CPU Q4 prompt-end states; the paper\'s earlier BF16 continuation interventions are a different distribution and precision setting.','',
        'Prior manuscript v006 remains unchanged. Source records, exact prompts and token IDs, runtime/model hashes, all-layer states, component directions, checks and frozen-SAE outputs are retained in the adjacent recurrence_state directories.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for name in ('run_recurrence_state_control.py','recurrence_state_control.py','llama_capture.py','llama_steer.py',
                 'analyze_recurrence_state_control.py','verify_recurrence_state_control.py','report_recurrence_state_control.py',
                 'test_recurrence_state_analysis.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshots/name)
    write(out/'manifest.json',dict(status='completed',analysis_manifest_sha256=sha(analysis/'manifest.json'),
        validation_sha256=sha(root/'recurrence_state_validation_v1/verification.json'),code_sha256=sha(Path(__file__)),
        files={str(f.relative_to(out)).replace('\\','/'):sha(f) for f in out.rglob('*') if f.is_file()}))
    print(str(out/'REPORT.md'))


if __name__=='__main__':main()
