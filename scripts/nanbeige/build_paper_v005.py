"""Create an incremented mechanistic manuscript from verified saved evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'v005_2026-09-11_cross_family_dynamics'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    archive = ROOT / 'output/paper_versions'
    index = read(archive / 'index.json')
    old = archive / 'v004_2026-09-09_intervention_fidelity'
    out = archive / VERSION
    if out.exists():
        raise FileExistsError('Never overwrite an existing numbered revision')
    previous = {}
    for version in index['versions']:
        for name, entry in version['files'].items():
            path = archive / version['version'] / name
            actual = sha(path)
            assert actual == entry['sha256'], path
            previous[str(path.relative_to(ROOT))] = actual
    data = ROOT / 'artifacts/nanbeige_reasoning'
    dynamics = data / 'pass_dynamics_by_task_v1'
    boundary = data / 'pass_boundary_geometry_v1'
    verified = read(dynamics / 'verification.json')
    assert sha(dynamics / 'manifest.json') == verified['analysis_manifest_sha256']
    assert sha(boundary / 'manifest.json') == verified['boundary_manifest_sha256']
    for folder in (dynamics, boundary):
        for name, digest in read(folder / 'manifest.json')['files'].items():
            assert sha(folder / name) == digest, name
    source_assets = read(dynamics / 'manifest.json')['checked_source_assets']
    for name, digest in source_assets.items():
        assert sha(ROOT / name) == digest, name
    evidence = read(dynamics / 'goal_evidence_audit.json')
    comparison = data / 'logic-prime-20260909T042949Z-897/grading_v2/summary.json'
    cases = data / 'logic_skill_comparison_v1/cases.json'
    assert sha(comparison) == evidence['comparator_summary_sha256']
    assert sha(cases) == evidence['cases_sha256']
    grouped = read(dynamics / 'grouped_states.json')
    normalized = read(boundary / 'grouped.json')
    families = [('arc_agi_1', 'ARC-AGI-1'), ('arc_agi_2', 'ARC-AGI-2'),
                ('intellect3_logic', 'INTELLECT-3 logic'), ('intellect3_math', 'INTELLECT-3 math')]
    table = ['| Family | Inputs / groups | L12 cosine | L22 cosine | After loop norm |',
             '|---|---:|---:|---:|---:|']
    numeric_rows = []
    for lane, title in families:
        rows = {r['physical_layer']: r for r in grouped
                if r['lane'] == lane and r['calibration_split'] == 'eval' and r['region'] == 'late'}
        norm = next(r for r in normalized if r['lane'] == lane and r['calibration_split'] == 'eval' and r['region'] == 'late')
        assert rows[22]['task_groups'] == 2
        assert abs(rows[22]['mean_cosine'] - norm['raw_cosine']) < 1e-12
        table.append(f"| {title} | {rows[22]['inputs']} / 2 | {rows[12]['mean_cosine']:.3f} | {rows[22]['mean_cosine']:.3f} | {norm['normalized_cosine']:.3f} |")
        numeric_rows.append(dict(family=lane,layer12=rows[12],layer22=rows[22],normalization=norm))
    new_section = '''## 3. The terminal pass difference across task families

Recurrence makes parameter identity a poor substitute for computational identity. The same weights are reused, but their place in the computation changes: an intermediate invocation supplies a later pass, while the final invocation supplies the output path. This suggests looking for a systematic difference between their states before assigning any feature a semantic role. We tested that observational prerequisite across ARC-AGI-1, ARC-AGI-2, INTELLECT-3 logic and INTELLECT-3 math.

The dataset contains 18 measured inputs: five in each ARC family and four in each INTELLECT-3 family. Eight inputs provide the dictionary-calibration set and ten are held out from that calibration. Each trajectory has paired captures from all 22 physical layers, with eight sampled token positions in each of four regions: the prompt suffix and early, middle and late generated text. A pair compares the two invocations at the same token in a fixed, recorded sequence replayed through the native two-pass model. It does not compare independently generated one-pass and two-pass solutions.

Figure 1 shows a common profile across the held-out task groups. Pass states become relatively aligned through the middle of the network, then diverge sharply in direction at the final physical layer. The late-window cosine at layer 12 ranges from 0.799 to 0.869 across families, compared with 0.230 to 0.307 at layer 22. These two sites were already selected for VPD in the existing study; they were not selected from this new profile.

![Figure 1. Paired invocation states across the 22 shared layers. Lines are equal-weight task-group means of tokenwise cosine; shading is the observed range of two held-out task groups per family, not a confidence interval. Prompt suffix and late-generation windows each contain eight positions per input. Both invocations use the same recorded sequence.](figures/pass_alignment.png)

The prompt suffix already shows the terminal decline, before any sampled continuation. The observation therefore cannot identify a late reasoning discovery, and it cannot explain successful reasoning by itself. It supports a narrower hypothesis: the terminal shared operation encounters different representational conditions depending on whether it feeds another pass or the final output path. The saved geometry motivates that hypothesis; an intervention that separates these roles is still needed to test it.

TABLE

The table reports late-generation task-group means. Multiple ARC test grids belonging to one task are averaged within that task before averaging groups. Each family has only two held-out groups; the ten held-out inputs represent eight task groups. Late generated text can include an answer as well as reasoning. The regional sampling identifies positions in a sequence, not semantic stages of a solution. The holdout concerns dictionary fitting on this inspected cohort; it is not a claim of uncontaminated benchmark testing. The INTELLECT-3 sources include public reinforcement-learning training data.

Measurement location matters. Comparing the saved states after loop normalization leaves the terminal directions poorly aligned, with cosines from 0.184 to 0.229. Yet the relative change, computed as the norm of the pass difference divided by the first-pass norm, rises from 0.95-0.98 at the raw terminal layer to 4.84-5.15 after normalization. These are observations at two sites, not an intervention on normalization. They show that a large relative effect depends on the reference scale as well as the direction of change. Neither cosine nor norm identifies what information the operation represents.

The cross-family profile gives a reason to take invocation-conditioned dictionary quality seriously. A common physical layer can present a common dictionary with substantially different states. The accompanying frozen-SAE data retain feature-support overlap and feature-change norms separately from hidden-state geometry, but sparse support differences do not establish new concepts. The next question is whether a dictionary can describe both state distributions and, more stringently, the changes produced by controlled parameter edits.

'''.replace('TABLE', '\n'.join(table))
    text = (old / 'PAPER.md').read_text(encoding='utf-8')
    text = text.replace('Reconstruction quality and intervention fidelity', 'Invocation geometry, reconstruction quality and intervention fidelity', 1)
    text = text.replace('Version 4 | 9 September 2026', 'Version 5 | 11 September 2026', 1)
    start = text.index('How faithfully can we track', text.index('## Abstract'))
    end = text.index('\n\n## 1.', start)
    text = text[:start] + '''How faithfully can we track a shared parameter's effects through successive passes? We investigate this question in Nanbeige4.2-3B using paired invocation states, fixed parameter interventions and frozen sparse autoencoder dictionaries. Across small ARC-AGI-1, ARC-AGI-2, INTELLECT-3 logic and math cohorts, pass-state alignment declines sharply at the final physical layer. The same profile appears before generation, so it does not establish a reasoning-specific mechanism. A paired optimizer control improves layer-22 second-pass state reconstruction from -0.180 to 0.453 explained variance, with an advantage that persists on saved BF16 intervention baselines. Muon-hybrid also improves intervention-change fidelity in all 12 signed conditions measured at that invocation. Yet 11 conditions still have greater reconstruction mismatch than predicting no change. Together these observations distinguish the geometry of successive states, the quality of their sparse descriptions and fidelity to an experimental effect. We develop an account that retains unexplained changes alongside feature responses, specifying the further controls needed to connect a reusable parameter operation to a semantic relation and its downstream consequence.''' + text[end:]
    needle = 'Our initial measurements exposed a specific difficulty.'
    text = text.replace(needle, 'Across four task families, the saved trajectories show why this distinction matters. Mid-network pass states are relatively aligned, while the final shared layer has a different directional profile. This difference is present in the input prompt as well as the generated continuation. It invites an explanation of how an operation is used at successive invocations, while leaving the semantic content of that operation unresolved.\n\n' + needle, 1)
    text = re.sub(r'^## ([3-6])\.', lambda m: f'## {int(m[1])+1}.', text, flags=re.M)
    text = text.replace('## 4. Improving the description', new_section + '## 4. Improving the description', 1)
    text = text.replace('The new analysis gives a qualified answer.', 'The intervention-change analysis gives a qualified answer.', 1)
    text = text.replace('The revised evidence changes the next mechanistic question.', 'The geometry and intervention evidence sharpen the next mechanistic question.', 1)
    semantic = 'The emerging account of recurrence is one of reusable operations acting on evolving representations.'
    text = text.replace(semantic, '''The receipt-replay environment provides a prospective way to make such relations explicit. Its input fact is whether a shipment receipt has already been credited; its behavioral target is whether the model will request duplicate credit. Legitimate uncredited receipts require the same submission action, separating action identity from the meaning of that action in context. Matched pressure and announced-auditor conditions could test whether a choice remains recoverable internally when its textual evidence changes. The implemented harness retains these labels separately and prepares pass-resolved linear probes and frozen-SAE diagnostics. No ledger episode has completed, so it supplies a test design, not a measured monitorability or mediation result.

''' + semantic, 1)
    old_scope = 'The new contribution in this revision is CPU analysis of existing archives, including float64 re-encoding of the layer-22 second-pass states. No model inference, dictionary training, immediate-write experiment, matched-write intervention, or semantic restoration experiment was added. Version 2 preserves the full experimental record; version 3 preserves the earlier interpretive manuscript.'
    new_scope = '''This revision adds task-grouped analysis of existing all-layer captures and a loop-normalization comparison. It retains the intervention-change analysis introduced in version 4, including its float64 re-encoding check. No new model inference, dictionary training, immediate-write experiment, matched-write intervention or semantic restoration experiment was added. Version 2 preserves the detailed technical record; versions 3 and 4 preserve the earlier mechanistic manuscripts.

The geometry analysis contains 1,584 input/layer/region windows and 648 input/dictionary/region feature windows. It verifies 190 source assets and exact recorded token correspondence. Its independent scalar check covers 72 metrics across every input at layers 12 and 22 in the late window, agreeing within 2.23e-16. These windows are repeated observations of 18 inputs, not additional independent tasks. The cross-family summaries are descriptive on a previously inspected cohort, even when using the dictionary-calibration holdout.

Thirteen of the 18 two-pass continuations reached the output cap: two ARC-AGI-1, five ARC-AGI-2, three logic and three math inputs. The dataset preserves useful internal trajectories, but their geometry is not evidence of solved tasks or a benefit from additional passes. A loop-count comparison would require an explicit behavioral endpoint and a controlled generation protocol beyond paired invocation observations.'''
    assert old_scope in text
    text = text.replace(old_scope, new_scope)
    text = text.replace('The hidden states come from BF16 teacher-forced analysis of saved Q4-generated sequences. They are not BF16 self-generated trajectories. The dictionaries were trained on Q4 states at a different selection of positions. This transfer setting is part of the measured result, and the present analysis does not isolate its contributing shifts. Causal-order checks apply to a fixed token sequence; subsequent first-pass states during free generation can change after an earlier intervention changes the generated tokens.',
                        'The geometry uses native Q4 captures of extended sequences generated with an 8,192-token cap. Interventions use BF16 teacher-forced replay of the earlier 2,048-token-cap sequences, at their final eight positions. Matching task IDs therefore do not align token rows across these analyses. The dictionaries were fitted on extended Q4 states; precision and position distribution shift together on transfer. Causal-order controls hold the token sequence fixed. They do not constrain how later first-pass states change during free generation after a perturbation changes an earlier token.')
    text = text.replace('All paths below are relative to the VPD workspace. The analysis directory includes', 'All paths below are relative to the VPD workspace. The intervention-change analysis directory includes', 1)
    text = text.replace('| New intervention-change analysis |', '| Intervention-change analysis |', 1)
    text = text.replace('| CPU analysis implementation |', '| All-layer task-grouped dynamics | `artifacts/nanbeige_reasoning/pass_dynamics_by_task_v1/` |\n| Loop-normalization comparison | `artifacts/nanbeige_reasoning/pass_boundary_geometry_v1/` |\n| Versioned activation dataset index | `artifacts/nanbeige_reasoning/dataset_index_v1/` |\n| CPU analysis implementation |', 1)
    text += '''
## Appendix C. What the skills comparison can establish

The mechanistic dataset and the external skills benchmark answer different experimental questions. Reusing a physical transformer operation concerns internal computation. A skill prompt, learned route and bounded symbolic repair change the procedure around model inference. Their comparison needs shared tasks and explicit accounting for generation and tool computation; a success rate alone cannot identify the role of recurrence.

The verified fresh comparison contains 109 Campsite logic puzzles using Qwen3.5-27B Q4. Its archived official-grader totals are:

| Procedure | Correct / puzzles | Fraction |
|---|---:|---:|
| Vanilla generation | 0 / 109 | 0.0% |
| Plain logic skill | 13 / 109 | 11.9% |
| Routed candidate before repair | 11 / 109 | 10.1% |
| Routed final with repair | 35 / 109 | 32.1% |
| Plain candidate with routed repair | 34 / 109 | 31.2% |

The protocol requests answer-only generation with thinking disabled, unlike the Nanbeige thinking-mode trajectories. All 109 vanilla calls reached their 1,024-token output cap. The resulting zero is a result of this bounded protocol, not an estimate of unrestricted reasoning ability. The repaired results combine candidate generation with external computation, including a 200,000-node exact-search limit; several rows reuse candidates rather than representing independent runs. The route also used historical outcome information and 128 synthetic positive-only support examples, limiting a clean generalization claim. Version 2 preserves the full protocol and grading qualifications.

Only one question identifier overlaps the 18-input Nanbeige cohort, and identifier overlap does not establish matched prompts, generation budgets or tool access. The fresh comparison covers Campsite logic only. The available artifacts therefore do not establish that TRM-infused skills clear all four requested task families, nor do they support a matched Nanbeige-versus-27B ranking across ARC, logic and math. That benchmark remains incomplete. Its absence limits performance claims without removing the directly measured pass-state and intervention-fidelity observations.

The supplement preserves the grading source, frozen protocol and overlap audit. No benchmark score here is inferred from an activation metric.
'''
    out.mkdir()
    (out / 'PAPER.md').write_text(text, encoding='utf-8')
    renderer = (old / 'render_paper.py').read_text(encoding='utf-8')
    renderer = renderer.replace('Version 4 | Intervention fidelity | 9 September 2026', 'Version 5 | Invocation geometry and intervention fidelity | 11 September 2026')
    renderer = renderer.replace('transformer passes; intervention-change fidelity', 'transformer passes; cross-family geometry and intervention-change fidelity')
    renderer = renderer.replace("default=Path('artifacts/nanbeige_reasoning/PAPER.md')", "default=Path(__file__).resolve().with_name('PAPER.md')")
    renderer = renderer.replace("default=Path('output/pdf/nanbeige_hermes_research_draft.pdf')", "default=Path(__file__).resolve().with_name('paper.pdf')")
    renderer = renderer.replace('KeepTogether, PageBreak', 'KeepTogether, PageBreak, CondPageBreak')
    renderer = renderer.replace('story.append(PageBreak())', 'story.append(CondPageBreak(150))')
    renderer = renderer.replace("    story=[];headings=[];table_cells=[]", "    styles.add(ParagraphStyle(name='PaperAppendixBody',parent=styles['PaperBody'],spaceAfter=4))\n    story=[];headings=[];table_cells=[]\n    in_appendix=False")
    renderer = renderer.replace("if text.startswith('Appendix '):", "if text.startswith('Appendix '):\n                in_appendix=True")
    renderer = renderer.replace("story.append(Paragraph(markup(' '.join(paragraph)),styles['PaperBody']))", "story.append(Paragraph(markup(' '.join(paragraph)),styles['PaperAppendixBody' if in_appendix else 'PaperBody']))")
    (out / 'render_paper.py').write_text(renderer, encoding='utf-8')
    shutil.copytree(old / 'equations', out / 'equations')
    shutil.copytree(old / 'supplement', out / 'supplement/intervention_fidelity')
    shutil.copytree(dynamics, out / 'supplement/pass_dynamics')
    shutil.copytree(boundary, out / 'supplement/boundary_geometry')
    shutil.copy2(old / 'literature_verification.json', out / 'supplement/inherited_literature_verification.json')
    (out / 'supplement/comparator').mkdir()
    shutil.copy2(comparison, out / 'supplement/comparator/summary.json')
    shutil.copy2(data / 'logic_skill_comparison_v1/protocol.json', out / 'supplement/comparator/protocol.json')
    shutil.copy2(__file__, out / 'build_paper_v005.py')
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), sharex=True, sharey=True, layout='constrained')
    for ax, (lane, title) in zip(axes.flat, families):
        for region, label, color, style in [('prompt', 'Prompt suffix', '#24638b', '-'),
                                             ('late', 'Late generation', '#ac5919', '--')]:
            rows = sorted([r for r in grouped if r['lane'] == lane and r['calibration_split'] == 'eval'
                           and r['region'] == region], key=lambda r: r['physical_layer'])
            assert len(rows) == 22
            xs = [r['physical_layer'] for r in rows]
            ax.plot(xs, [r['mean_cosine'] for r in rows], color=color, ls=style, lw=1.4, label=label)
            ax.fill_between(xs, [r['mean_cosine_min'] for r in rows], [r['mean_cosine_max'] for r in rows],
                            color=color, alpha=.12, linewidth=0)
        ax.set_title(title, fontsize=10)
        ax.set(xlim=(1, 22), ylim=(0, 1), xticks=[1, 6, 12, 18, 22], yticks=[0, .25, .5, .75, 1])
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(loc='lower left', fontsize=8, frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Pass-state cosine')
    for ax in axes[1, :]:
        ax.set_xlabel('Physical layer')
    (out / 'figures').mkdir()
    fig.savefig(out / 'figures/pass_alignment.png', dpi=300)
    plt.close(fig)
    receipt = dict(status='manuscript_prepared_render_pending', timestamp=datetime.now(timezone.utc).isoformat(),
                   previous_index_sha256=sha(archive / 'index.json'), preserved_previous_files=previous,
                   source_assets_rechecked=len(source_assets), numeric_rows=numeric_rows,
                   source_analysis_verification=verified, comparator_summary_sha256=sha(comparison),
                   manuscript_sha256=sha(out / 'PAPER.md'), new_model_calls=0,
                   benchmark_complete=False, paper_scope='Mechanistic observations and measurement fidelity; no reasoning-gain claim')
    (out / 'content_verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(version=VERSION, words=len(text.split()), source_assets_rechecked=len(source_assets))))


if __name__ == '__main__':
    main()
