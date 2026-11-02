"""Seal the descriptive analysis and audit its coverage against the full objective."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from .summarize_pass_dynamics import ROOT,DATA,LANES,read,save,sha,render


def main():
    out=DATA/'pass_dynamics_by_task_v1';boundary=DATA/'pass_boundary_geometry_v1'
    pooled=read(out/'grouped_states.json');coverage=read(out/'coverage.json');norm=read(boundary/'grouped.json')
    render(out,pooled)
    comparison=DATA/'logic-prime-20260909T042949Z-897';verification=read(comparison/'FINAL_VERIFICATION.json')
    graded_path=comparison/'grading_v2/summary.json'
    expected=verification['files'][str(graded_path.relative_to(ROOT))]
    assert sha(graded_path)==expected
    cases_path=DATA/'logic_skill_comparison_v1/cases.json';protocol_path=DATA/'logic_skill_comparison_v1/protocol.json'
    protocol=read(protocol_path);assert sha(cases_path)==protocol['cases_sha256']
    cases=read(cases_path);graded=read(graded_path)
    ids={r['id'] for r in coverage};overlap=sorted(ids&{r['id'] for r in cases})
    versions=read(ROOT/'output/paper_versions/index.json')
    version=next(v for v in versions['versions'] if v['version']==versions['current_manuscript'])
    paper=ROOT/'output/paper_versions'/version['version']
    for name in ['PAPER.md','paper.pdf']:assert sha(paper/name)==version['files'][name]['sha256']
    audit=dict(previous_goal_turn='progress: implemented and verified the ledger harness; empirical GPU execution remained incomplete',
        this_goal_turn='progress: computed new task-grouped hidden-state/feature statistics and a normalization-boundary control from saved tensors',
        new_inference=False,goal_complete=False,
        requirements=[
            dict(requirement='Nanbeige activation dataset spanning ARC-AGI-1, ARC-AGI-2, INTELLECT-3 logic and math',status='verified_small_pilot',
                examples_by_lane=dict(Counter(r['lane'] for r in coverage)),physical_layers=22,passes=2,checked_assets=190),
            dict(requirement='Pass-resolved VPD and SAE mechanistic analysis',status='existing_measurements_plus_new_descriptive_cross_family_analysis',
                limitation='Semantic behavioral mediation is not established. New profiles are observations, not interventions.'),
            dict(requirement='Paper emphasizing mechanistic interpretation',status='existing_v004_manuscript_hash_verified',path=str(paper.relative_to(ROOT)),
                new_analysis_incorporated_into_pdf=False),
            dict(requirement='Benchmark against TRM-infused skills and the 27B comparator across requested task families',status='incomplete',
                verified_comparator='Fresh 109-puzzle Qwen3.5-27B Campsite comparison only',pilot_identifier_overlap=overlap,
                matched_prompt_and_compute_comparison=False,missing_families=['arc_agi_1','arc_agi_2','intellect3_math'],
                limitation='Identifier overlap does not establish equal prompts, tools, thinking budgets or runtime conditions.'),
            dict(requirement='Receipt-replay behavioral pilot and controlled latent-variable probes',status='incomplete_thermal_block',behavioral_episodes=0)],
        comparator_summary_sha256=sha(graded_path),cases_sha256=sha(cases_path),checked_at=datetime.now(timezone.utc).isoformat())
    save(out/'goal_evidence_audit.json',audit)
    lines=['# Shared operations across four task families','',
        'The same physical layer can receive a different state on its second invocation. The useful mechanistic question is how that difference is organized across the network, and whether it helps explain an intervention’s consequences. We examined the saved Nanbeige trajectories across ARC-AGI-1, ARC-AGI-2, INTELLECT-3 logic and INTELLECT-3 math, preserving the token correspondence between passes. Both states come from the same native two-pass replay. Separately generated one-pass trajectories are outside this analysis; these statistics do not measure a loop-count ablation.','',
        'Across these small cohorts, the middle layers have relatively aligned pass states, followed by a sharp decline in alignment at the final physical layer. The table shows the late-generation window at the two layers already selected for VPD in the existing study; those sites were not chosen from this new plot. Each entry averages task groups equally. Multiple ARC test grids from one task are averaged within that task first.','',
        '| Family | Held-out inputs / task groups | Layer 12 cosine | Layer 22 cosine | Cosine after loop normalization |',
        '|---|---:|---:|---:|---:|']
    for lane in LANES:
        selected=[r for r in pooled if r['lane']==lane and r['calibration_split']=='eval' and r['region']=='late']
        a=next(r for r in selected if r['physical_layer']==12);b=next(r for r in selected if r['physical_layer']==22)
        c=next(r for r in norm if r['lane']==lane and r['calibration_split']=='eval' and r['region']=='late')
        lines.append(f'| {lane} | {a["inputs"]} / {a["task_groups"]} | {a["mean_cosine"]:.3f} | {b["mean_cosine"]:.3f} | {c["normalized_cosine"]:.3f} |')
    lines+=['','![Task-grouped paired-state profiles](heldout_pass_dynamics.png)','',
        'This profile is also visible at the prompt suffix, before the sampled continuation. It therefore cannot be attributed specifically to successful reasoning or to a late discovery in the generated text. It instead motivates a hypothesis about how the shared computation is used at different points in the loop. Controlled interventions and a behavioral endpoint are needed to turn that hypothesis into a mechanistic explanation.','',
        'The loop-normalization control preserves low terminal directional alignment, but changes the scale of the relative difference substantially: the late-window change divided by the first-pass norm is about 0.95–0.98 before normalization and 4.84–5.15 after it. This cautions against reading an effect magnitude without specifying the measurement site and reference scale. Neither number identifies a semantic operation.','',
        'The accompanying frozen-SAE tables describe feature-support overlap and feature-change norms separately from actual hidden-state changes. They use matching positions and the same dictionary within each pass comparison. Feature IDs are not aligned across dictionaries, and low feature overlap is not evidence of a new reasoning concept. The manuscript’s state-reconstruction and intervention-change-fidelity checks remain necessary before interpreting those coordinates.','',
        'There are 18 inputs, 1,584 layer/region state windows and 648 dictionary/region feature windows. Those windows do not create independent tasks. Each family has only two held-out task groups, so the plotted bands show observed group ranges, not confidence intervals. Thirteen of the 18 two-pass continuations reached the output cap; all five ARC-AGI-2 inputs did. No answer score or reasoning advantage is inferred from these state differences.','',
        'The comparison requirement remains open. The verified fresh Campsite run reports plain-skill success on 13/109 puzzles and routed final success on 35/109; all 109 vanilla calls were capped. Only '+str(len(overlap))+' pilot question identifier overlaps that cohort. These are not matched cross-model results, and the available comparison does not cover ARC or math. The original claim that the skills cleared those evaluations is not established by these artifacts.','',
        'This analysis is a candidate supplement to the existing mechanistic manuscript. It adds observations across all four requested task families without changing the model or treating the unfinished ledger pilot as evidence. The full goal remains incomplete; see `goal_evidence_audit.json`.','',
        '## Reproduction and verification','',
        'Run `python -m scripts.nanbeige.summarize_pass_dynamics --output <fresh-directory>` from the project root. Source archives, dictionaries and exact two-pass token sequences are checked against the existing dataset index. `check_pass_boundary_geometry.py` performs the linked normalization control. CSV and JSON exports retain input IDs, task groups, calibration splits, regions, output-censoring flags and measurement sites. The source snapshots are stored in `source/`.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'source').mkdir(exist_ok=True)
    for name in ['summarize_pass_dynamics.py','check_pass_boundary_geometry.py','test_pass_dynamics.py','finish_pass_dynamics_report.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    m=read(out/'manifest.json');m['code_sha256']=sha(Path(__file__).with_name('summarize_pass_dynamics.py'))
    m['visual_review']='Eight panels inspected; labels and ranges legible; shared vertical scales by metric; task-group ranges explicitly not CIs.'
    m['files']={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save(out/'manifest.json',m)
    b=read(boundary/'manifest.json');b['source_manifest_sha256']=sha(out/'manifest.json');save(boundary/'manifest.json',b)
    print(json.dumps(dict(analysis_complete=True,goal_complete=False,overlapping_comparator_ids=overlap)))


if __name__=='__main__':main()
