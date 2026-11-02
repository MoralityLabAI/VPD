"""Integrate verified state-carry controls in a new, non-overwriting manuscript."""
from pathlib import Path
import json
import shutil
from .build_closeout_v006 import ROOT,read,sha,save

VERSION='v007_2026-09-12_state_carry_controls'


def main():
    archive=ROOT/'output/paper_versions';index=read(archive/'index.json')
    old=archive/'v006_2026-09-12_experiment_closeout';out=archive/VERSION
    if out.exists():raise FileExistsError('Preserve numbered revisions')
    previous={}
    for v in index['versions']:
        for name,entry in v['files'].items():
            f=archive/v['version']/name;assert sha(f)==entry['sha256'],f
            previous[str(f.relative_to(ROOT))]=sha(f)
    data=ROOT/'artifacts/nanbeige_reasoning'
    run=data/'recurrence_state_control_v1';analysis=data/'recurrence_state_analysis_v1'
    report=data/'recurrence_state_report_v2';close=read(data/'recurrence_state_closeout_v1/verification.json')
    for folder in (run,analysis,report):
        for name,h in read(folder/'manifest.json')['files'].items():assert sha(folder/name)==h,name
    assert sha(run/'manifest.json')==close['source_manifest_sha256']
    assert sha(report/'manifest.json')==close['report_manifest_sha256']
    validation=read(data/'recurrence_state_validation_v1/verification.json')
    numeric=read(data/'recurrence_state_analysis_validation_v1/verification.json')
    assert validation['max_reset_state_error']==validation['max_reset_logit_error']==0
    assert numeric['primary_per_input_checks']==1680
    rows=read(analysis/'pooled.json');quality=read(analysis/'state_quality.json');swaps=read(analysis/'crossed_pooled.json')
    def get(hist,layer,dose,opt='muon_hybrid',mode='top_k'):
        return next(r for r in rows if (r['history'],r['edited_layer'],r['dose'],r['site'],r['optimizer'],r['encoding'])==(hist,layer,dose,'model.layers.21',opt,mode))
    doses=(-.1,.1,-1.,1.,-10.,10.)
    assert all(get('carry',12,d)['fidelity']<0<get('reset',12,d)['fidelity'] for d in doses)
    assert all(get(h,22,d,opt)['fidelity']<0 for h in ('one_pass','reset') for d in doses for opt in ('adamw','muon_hybrid'))
    state_table=['| Final L22 state | AdamW EV | Muon-hybrid EV |','|---|---:|---:|']
    for history,title in [('one_pass','One pass, 22 layers'),('carry','State carry, 44 layers'),('reset','State reset, 44 layers')]:
        values={r['optimizer']:r['explained_variance'] for r in quality if r['site']=='model.layers.21' and r['history']==history}
        state_table.append(f'| {title} | {values["adamw"]:.3f} | {values["muon_hybrid"]:.3f} |')
    crossed_table=['| Baseline state | Carry-derived direction | Reset-derived direction |','|---|---:|---:|']
    for history,title in [('carry','Carry'),('reset','Reset')]:
        values={r['direction_history']:r['fidelity'] for r in swaps if r['optimizer']=='muon_hybrid' and r['state_history']==history}
        crossed_table.append(f'| {title} | {values["carry"]:.3f} | {values["reset"]:.3f} |')
    section='''## 7. Separating state carry from sparse reconstruction

The earlier measurements leave two explanations entangled. A sparse encoder may turn a small model change into a large feature transition through its own selection rule. Recurrent computation may also change the directions along which the model responds, placing those changes outside what a particular dictionary reconstructs well. These mechanisms can coexist. To distinguish them, we intervene separately on the state carried between passes and on the support used by the measuring instrument.

The new experiment uses the complete prompts of all ten existing held-out inputs, representing eight task groups across the four families. A native one-pass reference executes 22 layers. The carry condition executes the original two passes. The reset condition also executes 44 layers, but replaces every token's loop-boundary state with its original embedding before the second pass. This replacement applies to the whole prefix, so second-pass attention cannot recover carried information from earlier cached tokens. Carry and reset hold the executed layer count and weights fixed. They do not hold effective dependent depth fixed, and they are not separately trained architectures. Expanding shared weights into identical layer copies would preserve the same computation and would not provide that training comparison.

At the final prompt token, we apply signed residual writes with absolute L2 magnitudes 0.1, 1 and 10, targeting layer 12 or 22 on the final executed pass. Each write follows the normalized output direction of already-fitted VPD component 0 for that layer. These are controlled writes along a parameter-component direction, not repetitions of the earlier fixed-coefficient weight edits across a sequence. Matching the write magnitude allows us to ask how the same local displacement is represented and propagated under different state histories.

The reset reproduces its one-pass reference exactly across all 150 matched baseline and intervention pairs: saved states at every layer, final normalization states, and logits agree. Identity writes preserve outputs, and interventions leave upstream states unchanged. These checks validate the control's implementation; they do not establish a claim about the merits of looping. Frozen dictionary quality on the new prompt-end distribution is:

STATE_TABLE

The reconstruction problem remains when the second pass receives no carried state. For local layer-22 writes, both dictionaries have negative change fidelity in all six signed conditions under one-pass execution and reset. Muon's state EV of 0.862 therefore coexists with failure to reconstruct these changes better than the zero-change reference. This separates a property of the measuring instrument from any requirement to execute a second pass. It leaves open whether the checkpoint's training on recurrence helped produce its weights or representations.

We can intervene directly on one part of that instrument. The full encoder selects top-k support separately for each state. The support control instead holds the baseline candidate support fixed while retaining the same positive encoder responses and decoder. Both measurements use identical batched encoder arithmetic and the same observed hidden-state changes. For the one-pass layer-22 write of -1, this changes Muon fidelity from -52.367 to approximately -0.000543. AdamW's large -10-write spike similarly falls from -10.520 to about -0.024. The error amplification is attributable to support selection in this assay. Removing it does not make the remaining reconstruction accurate: the frozen-support score still lies near the zero-change reference.

State carry nevertheless changes the model responses that the dictionary must describe. For every tested layer-12 write measured at layer 22, Muon fidelity is positive under reset (0.047 to 0.572) and negative under carry (-0.300 to -0.046). Freezing support makes those carry scores positive but small (0.022 to 0.074), while reset scores remain higher (0.313 to 0.655). The comparison identifies both a selection-induced error and a persistent difference in recoverability under the fixed-support instrument. It does not support a universal penalty from recurrence: AdamW has higher fidelity with carry in ten of the twelve local/propagated conditions, although all remain negative.

![Figure 2. Final-layer change fidelity at matched direct-write magnitudes. Each point pools ten prompt-end observations from eight task groups. Solid lines are positive writes; dashed lines are negative writes. Upper panels show effects propagated from layer 12; lower panels show local layer-22 writes and use symmetric-log vertical axes. The same frozen dictionary is used across histories within each panel. These are repeated conditions, not independent task replications.](figures/carry_reset_fidelity.png)

A further control asks whether the difference mainly follows the baseline state or the response direction. For the layer-12 +1 write, we cross the two layer-22 baseline states with the two corresponding response directions, rescaling each direction to unit norm before re-encoding. These are offline vectors, not additional states produced by the model. Muon's pooled change fidelity is:

CROSSED_TABLE

In this assay, carrying the response direction from one condition to the other changes fidelity much more than exchanging baseline states. Carry-derived directions remain relatively poorly recovered on either baseline, while reset-derived directions remain substantially more recoverable. The lower state EV under carry is therefore insufficient by itself to explain the observed difference. The computation also produces a different downstream response direction relative to the dictionary. This result concerns the specified direction, magnitude and cohort; it is not a general decomposition of the causes of reconstruction error.

These controls sharpen the paper's account of a shared operation. Recurrence does not have to create the sparse reconstruction failure for invocation history to matter. State carry changes the transformation that follows a local write; the dictionary then describes that response with a fidelity that depends on both its learned coordinates and its selection rule. A semantic explanation would need to trace a relation through this chain while distinguishing the model's transformation from the measuring instrument's distortion. The new experiment establishes parts of that distinction, without assigning a semantic identity to the component.

'''.replace('STATE_TABLE','\n'.join(state_table)).replace('CROSSED_TABLE','\n'.join(crossed_table))
    text=(old/'PAPER.md').read_text(encoding='utf-8')
    text=text.replace('Version 6 | 12 September 2026','Version 7 | 12 September 2026')
    abstract_start=text.index('How faithfully can we track');abstract_end=text.index('\n\n## 1.',abstract_start)
    text=text[:abstract_start]+'''How faithfully can we track a shared parameter's effects through successive passes? We investigate this question in Nanbeige4.2-3B using paired invocation states, parameter-component interventions and frozen sparse autoencoder dictionaries. Across small ARC-AGI-1, ARC-AGI-2, INTELLECT-3 logic and math cohorts, final-layer pass states diverge in direction even before generation. A paired optimizer control improves second-pass state reconstruction, yet most measured parameter effects remain poorly reconstructed. A new state-carry control separates two sources of difficulty. Removing cross-pass state carry while retaining 44 executed layers reproduces a one-pass reference exactly. Reconstruction failure persists, and freezing the SAE's baseline support removes large error spikes without making local changes faithfully recoverable. State carry also changes downstream response directions: for six signed layer-12 writes, Muon reconstructs the layer-22 change better than a zero-change reference under reset but worse under carry. An offline direction-swap assay preserves much of that difference across exchanged baseline states. These results locate a sparse-measurement failure that does not require a second executed pass and an invocation-dependent change in which model effects the dictionary can describe. They test state carry within one checkpoint, not the effect of training a looped architecture. We retain unexplained changes alongside feature responses so that a future semantic account remains answerable to the computation it aims to explain.''' + text[abstract_end:]
    target='## 7. What a semantic interpretation would now have to explain'
    assert target in text;text=text.replace(target,section+'## 8. What a semantic interpretation would now have to explain',1)
    marker='We use Nanbeige4.2-3B because its weights and execution are available for direct inspection.'
    text=text.replace(marker,'A controlled state-carry experiment now separates two parts of that difficulty. Reconstruction failures persist when the second pass is reset to the original embeddings, and an intervention on sparse support removes large error spikes. At the same time, carrying the first pass\'s state changes downstream response directions and their recoverability. The distinction lets us investigate the measurement failure without assuming either that recurrence caused it or that recurrence is irrelevant to it.\n\n'+marker,1)
    start=text.index('This revision closes the associated experiments at their observed endpoints.')
    end=text.index('\n\n',start)
    text=text[:start]+'''This revision adds the completed CPU Q4 state-carry experiment and frozen-support analysis in Section 7. It preserves the earlier all-layer geometry, BF16 parameter-intervention analysis and optimizer control. The new study holds full prompts fixed and uses matched-magnitude residual writes along fitted VPD directions. It does not add dictionary training, a semantic restoration experiment or a separately trained architecture comparison. Version 6 preserves the partial Campsite comparison and terminal ledger gate; those endpoints remain unchanged.''' +text[end:]
    text=text.replace('Primary F_delta is calculated from summed signal and mismatch energies over all 80 positions in each stratum.',
        'For the original BF16 intervention analysis, primary F_delta is calculated from summed signal and mismatch energies over all 80 positions in each stratum. The new state-carry study instead pools ten prompt-end positions per stratum, as detailed in Appendix E.')
    text=text.replace('That control remains prospective; the archived endpoint norms do not supply it.',
        'Section 7 adds matched-magnitude residual writes along a VPD direction. It does not measure or equalize the original fixed-coefficient projection writes, so the archived endpoint norms still do not separate those factors.')
    # Keep the gate's complete record in Appendix D, with a short main-text boundary.
    start=text.index('That experiment first needs an agent that can express')
    end=text.index('\n\nThe emerging account of recurrence',start)
    text=text[:start]+'''The ledger experiment did not pass its comprehension gate: one question produced a correct final answer, two exhausted the thinking budget, and nine were not run. No ledger episode has completed. It therefore supplies no result about concealment, semantic probes or mediation. Appendix D preserves the protocol and terminal evidence. The state-carry controls in Section 7 establish measurement and propagation effects on other inputs; they cannot fill that semantic evidential gap.''' +text[end:]
    text+='''
## Appendix E. State-carry control and measurement checks

The source run contains 450 saved conditions: three histories on ten inputs, each with a baseline, two identity writes, and twelve nonzero writes. Both ARC families contribute three inputs from two task groups; logic and math contribute two inputs each. Inputs are the previously inspected SAE holdout, not a newly sampled benchmark. The native CPU runtime uses the pinned Q4_K_M checkpoint, q8_0 KV cache and two threads. No continuation is generated. The last-token cache entry is removed before each intervention and the preceding cache is reused within its history condition. Prefix caches are rebuilt when history changes.

All 150 reset/one-pass pairs match bit-for-bit at the saved layer states and logits; final-normalization parity also passes. The independent check covers 60 identity-logit comparisons, 13,240 upstream-state invariance comparisons and 420 direct-write norms, including zero-dose controls. These counts describe engineering controls on repeated conditions, not independent evidence of generalization. Every physical layer invocation is captured at the measured token, together with exact prompt token IDs, history, intervention site and dose. Both the intended and observed residual writes are retained.

The six frozen dictionaries retain their original weights and top-32 sparsity. Analysis uses float64 encoder arithmetic and decoding of complete states, retaining both full top-k and frozen-baseline-support encodings with exact source-archive alignment. Primary F_delta pools signal and mismatch energies over ten positions; zero signal remains undefined. The full analysis contains 5,040 per-input metric rows. A separate implementation directly decodes saved feature differences and checks all 1,680 primary layer-22 per-input metrics and 168 pooled strata; its maximum absolute fidelity difference is below 3e-12. Five focused unit tests cover support switching, zero effects, shared encoder arithmetic and energy pooling.

The state/direction crossing is a secondary offline assay for the layer-12 +1 write, specified before inspecting carry/reset SAE results. It was not an externally preregistered study. It normalizes the propagated layer-22 response to L2=1 before crossing it with carry/reset baselines. Baseline state, response direction and magnitude are therefore explicit factors at the measuring instrument. It does not establish that the crossed vectors would be reached naturally, or that changing those vectors would restore a behavior. Full and frozen-support comparisons share the same positive encoder responses to keep support selection separate from matrix-kernel arithmetic differences.

Saved logits provide a limited downstream endpoint. Among 120 nonzero-dose conditions per history, the next-token argmax changes in two one-pass/reset conditions and none of the carry conditions. Maximum baseline-to-intervention KL is 0.00269 for one-pass/reset and 0.03726 for carry. Logit distributions can change without an argmax change. These are not task-success measurements or evidence of a semantic mediator.

These CPU Q4 prompt-end states are distinct from the earlier BF16 continuation states, even for common question IDs. Residual writes match absolute norm along a fitted VPD output direction; they do not reproduce a fixed-coefficient weight edit across positions. Carry/reset matches executed layer count, not effective dependent depth, and does not isolate the effect of training with tied weights.

The supplement includes analysis, both encoding streams, independent validations, figures and source snapshots. Native states are retained under artifacts/nanbeige_reasoning/recurrence_state_control_v1; their protocol, records and hash manifest are copied into state_carry_run. Earlier manuscript versions remain unchanged.
'''
    out.mkdir()
    for directory in ('equations','figures','supplement'):shutil.copytree(old/directory,out/directory)
    shutil.copy2(report/'carry_reset_fidelity.png',out/'figures/carry_reset_fidelity.png')
    for name,folder in [('state_carry_analysis',analysis),('state_carry_validation',data/'recurrence_state_validation_v1'),
                        ('state_carry_metric_validation',data/'recurrence_state_analysis_validation_v1'),('state_carry_report',report)]:
        shutil.copytree(folder,out/'supplement'/name)
    evidence=out/'supplement/state_carry_run';evidence.mkdir()
    for name in ('protocol.json','manifest.json','records.json','cleanup.json'):shutil.copy2(run/name,evidence/name)
    shutil.copy2(data/'recurrence_state_closeout_v1/verification.json',evidence/'closeout.json')
    for name in ('recurrence_state_analysis_plan.json','recurrence_state_analysis_logging_amendment.json'):
        shutil.copy2(data/name,evidence/name)
    (out/'PAPER.md').write_text(text,encoding='utf-8')
    renderer=(old/'render_paper.py').read_text(encoding='utf-8').replace('Version 6 | Shared parameter effects and experiment closeout | 12 September 2026',
        'Version 7 | State carry and intervention fidelity | 12 September 2026')
    (out/'render_paper.py').write_text(renderer,encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    save(out/'content_verification.json',dict(status='source_verified_render_pending',previous_index_sha256=sha(archive/'index.json'),
        preserved_previous_files=previous,state_table=state_table,crossed_table=crossed_table,
        native_manifest_sha256=sha(run/'manifest.json'),analysis_manifest_sha256=sha(analysis/'manifest.json'),
        new_native_conditions=450,full_benchmark_completed=False,ledger_episodes=0))
    print(json.dumps(dict(version=VERSION,previous_files_verified=len(previous))))


if __name__=='__main__':main()
