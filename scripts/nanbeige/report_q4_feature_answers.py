"""Prepare paper-ready tables/figure only from a verified, complete cohort."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from experiments.nanbeige_reasoning.check_llama_steering import read, sha, write
from experiments.nanbeige_reasoning.compare_models import outcome_tables

LABELS = {
    'baseline':'Baseline', 'second_plus':'Pass 2 +0.1', 'second_minus':'Pass 2 -0.1',
    'first_plus':'Pass 1 +0.1', 'first_minus':'Pass 1 -0.1',
    'both_plus':'Both +0.1', 'both_minus':'Both -0.1',
}


def summarize(questions, graded, arms):
    expected={(q['id'],a) for q in questions if q['calibration_split']=='eval' for a in arms}
    keys=[(r['id'],r['arm']) for r in graded]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('Reporting requires every unique evaluation input/arm')
    tables=outcome_tables(questions,graded,arms)
    if any(r['state']=='missing' for r in tables['units']):raise ValueError('Missing complete task')
    grouped={arm:[r for r in tables['units'] if r['arm']==arm] for arm in arms}
    baseline={(r['lane'],r['evaluation_slice'],r['unit']):r for r in grouped['baseline']}
    summary=[]
    for arm in arms:
        rows=[r for r in graded if r['arm']==arm]
        outcomes=Counter(r['state'] for r in grouped[arm])
        fixes=regressions=paired=0
        for r in grouped[arm]:
            b=baseline[r['lane'],r['evaluation_slice'],r['unit']]['state']
            if b in ('correct','incorrect') and r['state'] in ('correct','incorrect'):
                paired+=1
                fixes+=b=='incorrect' and r['state']=='correct'
                regressions+=b=='correct' and r['state']=='incorrect'
        summary.append(dict(arm=arm,units=len(grouped[arm]),correct=outcomes['correct'],
            incorrect=outcomes['incorrect'],grader_errors=outcomes['grader_error'],
            paired_units=paired,fixes=fixes,regressions=regressions,
            input_records=len(rows),capped_inputs=sum(r['max_token_censored'] for r in rows),
            generated_tokens=sum(r['generated_tokens'] for r in rows),
            median_generated_tokens=statistics.median(r['generated_tokens'] for r in rows),
            native_grade_record_differences=sum(r['grade']!=r['native_render_grade'] for r in rows),
            native_score_differences=sum((r['grade']['status'],r['grade']['score']) !=
                (r['native_render_grade']['status'],r['native_render_grade']['score']) for r in rows)))
    return tables,summary


def figure(tables, arms, output, *, diagnostic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    units=sorted({(r['lane'],r['evaluation_slice'],r['unit']) for r in tables['units']})
    lookup={(r['lane'],r['evaluation_slice'],r['unit'],r['arm']):r for r in tables['units']}
    matrix=np.array([[{'incorrect':0,'correct':1,'grader_error':2}[lookup[*u,a]['state']] for a in arms] for u in units])
    fig,ax=plt.subplots(figsize=(9,5.2))
    ax.imshow(matrix,cmap=ListedColormap(['#f1d9cd','#a9d5c1','#d5d5df']),vmin=0,vmax=2,aspect='auto')
    labels=[]
    for lane,split,unit in units:
        short=unit.split('/')[-2] if '/' in unit else unit.removeprefix('id:')
        lane_label={'arc_agi_1':'ARC-AGI-1','arc_agi_2':'ARC-AGI-2','intellect3_logic':'Logic','intellect3_math':'Math'}[lane]
        labels.append(f'{lane_label} / {short}'+(' (historical)' if split=='historical_eval' else ''))
    ax.set_yticks(range(len(units)),labels);ax.set_xticks(range(len(arms)),[LABELS[a] for a in arms],rotation=30,ha='right')
    for i,u in enumerate(units):
        for j,a in enumerate(arms):
            row=lookup[*u,a]
            ax.text(j,i,{'correct':'1','incorrect':'0','grader_error':'E'}[row['state']]+
                ('*' if row['censored_inputs'] else ''),ha='center',va='center',fontsize=11)
    ax.set_title(('SYNTHETIC LAYOUT CHECK - ' if diagnostic else '')+'Single-feature steering: paired task outcomes',pad=14)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.02,.025,'1 = correct; 0 = incorrect; E = grader error. * = at least one input reached the output cap.\nARC requires all test inputs correct. One post-hoc feature/dose; no independent seed replication.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,1))
    for ext in ('png','svg','pdf'):fig.savefig(output/f'feature_answer_outcomes.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('cloud_run',type=Path)
    parser.add_argument('--grading-name',default='grading')
    parser.add_argument('--output-name',default='paper_results');args=parser.parse_args()
    run=args.cloud_run;grading=run/args.grading_name;manifest=read(grading/'manifest.json')
    if manifest['status']!='q4_feature_answers_independently_decoded_and_graded' or manifest['records']!=70:
        raise ValueError('Complete verified grading is required')
    for name,digest in manifest['files'].items():
        path=(grading/name).resolve()
        assert path.parent==grading.resolve() and sha(path)==digest
    source=run/'retrieved/artifacts/nanbeige_reasoning/prime_q4_feature_answers_v1'
    assert sha(source/'manifest.json')==manifest['generation_manifest_sha256']
    generation=read(source/'manifest.json');contract=read(source/'contract.json')
    assert generation['status']=='q4_feature_answers_complete' and generation['completed']==generation['expected']==70
    assert sha(source/'contract.json')==generation['contract_sha256']
    qpath=Path('artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl')
    assert sha(qpath)==contract['question_sha256']
    questions=list(map(json.loads,qpath.read_text(encoding='utf-8').splitlines()))
    graded=read(grading/'graded.json');arms=[a['name'] for a in contract['conditions']]
    assert set(arms)==set(LABELS)
    tables,summary=summarize(questions,graded,arms)
    assert tables==read(grading/'tables.json'), 'Independent table recomputation differs'
    gate=read(run/'independent_verification/gate/verification.json')
    assert gate['gate_passed'] and gate['status']=='cached_steering_saved_arrays_verified'
    output=run/args.output_name;output.mkdir(exist_ok=False)
    write(output/'summary.json',dict(arms=summary,slices=tables['arms'],pairs=tables['pairs']))
    figure(tables,arms,output)
    text=['### Completed single-feature answer-steering extension','',
        'All 70 continuations are complete: ten held-out input records under seven conditions, aggregating to eight evaluation units per condition. '
        'ARC uses complete tasks; the historical logic input remains a separate slice. The primary scores use raw HF token decoding, '
        'with native rendering checked independently. This is a post-hoc, single-feature/dose experiment.','',
        '| Condition | Correct / units | Grader errors | Fixes / regressions vs baseline | Capped inputs / 10 |',
        '|---|---:|---:|---:|---:|']
    for row in summary:
        text.append(f"| {LABELS[row['arm']]} | {row['correct']}/{row['units']} | {row['grader_errors']} | {row['fixes']} / {row['regressions']} | {row['capped_inputs']}/10 |")
    text+=['','Correct/units retains grader errors in the displayed cohort denominator; paired changes exclude units with a grader error in either arm. '
        'Censoring counts input records, not ARC tasks. The pooled eight-unit table is descriptive; lane/split results are retained below.','',
        '| Lane / split | Condition | Correct / units | Grader errors |','|---|---|---:|---:|']
    for row in tables['arms']:
        text.append(f"| {row['lane']} / {row['evaluation_slice']} | {LABELS[row['arm']]} | {row['correct']}/{row['units']} | {row['grader_error']} |")
    total_tokens=sum(r['generated_tokens'] for r in summary)
    native_differences=sum(r['native_score_differences'] for r in summary)
    text+=['',f'The continuations contain {total_tokens:,} generated tokens. Native-rendering and raw-decoding grades differ in status or binary score on {native_differences}/70 inputs. '
        'Full grading-record differences are retained separately. Generated-token totals do not measure equal compute across intervention arms.','',
        'These results neither identify feature semantics nor establish a recurrence benefit, an adaptive routing mechanism, or a scale threshold. '
        'A zero-intervention baseline here is two-pass Nanbeige; this experiment is distinct from the earlier one-pass ablation and the unmatched historical Hermes/27B audit.','']
    (output/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    write(output/'manifest.json',dict(status='complete_feature_answer_paper_material_prepared',
        grading_manifest_sha256=sha(grading/'manifest.json'),generation_contract_sha256=sha(source/'contract.json'),
        reporter_sha256=sha(Path(__file__)),visual_review_required=True,paper_integration_pending=True,
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='paper_material_prepared',output=str(output))))


if __name__=='__main__':main()
