"""Grade completed model arms on one fixed cohort with task-level pairing.

Each arm supplies its own exact model-facing tokenized questions. Raw question
identity is checked against the shared cohort before scores are compared.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

from .fetch_sources import sha
from .scoring import score


def outcome_tables(questions,graded,labels):
    """Keep missing outputs and grader failures separate from binary failures."""
    if len(labels)!=len(set(labels)) or len(labels)<1:raise ValueError('Duplicate or empty arm labels')
    lookup={}
    for row in graded:
        key=(row['arm'],row['id'])
        if key in lookup:raise ValueError('Duplicate graded arm/input')
        if row['arm'] not in labels:raise ValueError('Unregistered arm')
        lookup[key]=row
    groups=defaultdict(dict)
    seen=set()
    for q in questions:
        if q['id'] in seen:raise ValueError('Duplicate question ID')
        seen.add(q['id'])
        if q['calibration_split']!='eval':continue
        unit=q['group_id'] if q['lane'].startswith('arc_') else q['id']
        groups[(q['lane'],q['split'])].setdefault(unit,[]).append(q['id'])
    if any(row['id'] not in seen for row in graded):raise ValueError('Graded input outside cohort')
    summaries,pairs,details=[],[],[]
    for (lane,split),units in sorted(groups.items()):
        outcomes={}
        for unit,ids in sorted(units.items()):
            for arm in labels:
                selected=[lookup.get((arm,key)) for key in ids]
                if any(row is None for row in selected):state='missing'
                elif any(row['grade']['status']!='graded' for row in selected):state='grader_error'
                else:
                    if any(row['grade']['score'] not in (0,1) for row in selected):
                        raise ValueError('Binary success comparison requires binary grades')
                    state='correct' if all(row['grade']['score']==1 for row in selected) else 'incorrect'
                outcomes[arm,unit]=state
                details.append(dict(lane=lane,evaluation_slice=split,arm=arm,unit=unit,test_inputs=len(ids),
                    state=state,censored_inputs=sum(bool(row and row.get('max_token_censored')) for row in selected)))
        for arm in labels:
            counts={state:sum(outcomes[arm,u]==state for u in units) for state in ('correct','incorrect','missing','grader_error')}
            denominator=counts['correct']+counts['incorrect']
            summaries.append(dict(lane=lane,evaluation_slice=split,arm=arm,
                unit='complete_arc_task' if lane.startswith('arc_') else 'question',units=len(units),
                graded_units=denominator,**counts,accuracy=counts['correct']/denominator if denominator else None))
        for left,right in itertools.combinations(labels,2):
            counts=dict(both_correct=0,left_only_correct=0,right_only_correct=0,both_incorrect=0,missing=0,grader_error=0)
            for unit in units:
                a,b=outcomes[left,unit],outcomes[right,unit]
                if 'missing' in (a,b):key='missing'
                elif 'grader_error' in (a,b):key='grader_error'
                elif a==b:key='both_correct' if a=='correct' else 'both_incorrect'
                else:key='left_only_correct' if a=='correct' else 'right_only_correct'
                counts[key]+=1
            denominator=len(units)-counts['missing']-counts['grader_error']
            pairs.append(dict(lane=lane,evaluation_slice=split,left=left,right=right,units=len(units),
                paired_graded_units=denominator,**counts,
                left_minus_right=(counts['left_only_correct']-counts['right_only_correct'])/denominator if denominator else None))
    return dict(arms=summaries,pairs=pairs,units=details)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--arm',nargs=4,action='append',required=True,metavar=('LABEL','RUN','CONDITION','MODEL_QUESTIONS'))
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    labels=[a[0] for a in args.arm]
    if len(labels)!=len(set(labels)):raise ValueError('Duplicate arm labels')
    qpath=args.cohort/'questions.jsonl';gpath=args.cohort/'graders.jsonl'
    questions=list(map(json.loads,qpath.read_text(encoding='utf-8').splitlines()))
    graders=list(map(json.loads,gpath.read_text(encoding='utf-8').splitlines()))
    q={r['id']:r for r in questions};g={r['id']:r for r in graders}
    if len(q)!=len(questions) or len(g)!=len(graders) or set(q)!=set(g):
        raise ValueError('Invalid cohort question/grader identity')
    if args.output_dir.exists():raise ValueError('Choose a new comparison output directory')
    # Validate every arm before running any verifier or writing outputs.
    pending=[];provenance=[]
    for label,run_name,condition,model_input_name in args.arm:
        run=Path(run_name);inputs=Path(model_input_name)
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest['status'] not in ('generation_complete','capture_complete'):
            raise ValueError(f'Arm {label} is incomplete')
        source=run/'generations.jsonl'
        if manifest['files']['generations.jsonl']!=sha(source):raise ValueError('Generation hash mismatch')
        recorded_question_hash=manifest.get('question_sha256',manifest.get('contract',{}).get('question_sha256'))
        if recorded_question_hash!=sha(inputs):raise ValueError('Arm model-input hash mismatch')
        arm_rows=list(map(json.loads,inputs.read_text(encoding='utf-8').splitlines()))
        arm_q={r['id']:r for r in arm_rows}
        if len(arm_q)!=len(arm_rows) or set(arm_q)!=set(q):raise ValueError('Arm cohort differs')
        for key,original in q.items():
            if any(arm_q[key].get(field)!=value for field,value in original.items() if field!='token_ids'):
                raise ValueError(f'Question identity changed: {label}/{key}')
        rows=[r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines()) if r['condition']==condition]
        if len(rows)!=len(q) or {r['id'] for r in rows}!=set(q):raise ValueError('Missing/duplicate selected arm records')
        for row in rows:
            original=q[row['id']]
            if any(row[field]!=original[field] for field in ('lane','family','split','calibration_split')):
                raise ValueError('Generation metadata differs from cohort')
            prefix=arm_q[row['id']]['token_ids']
            if row['prompt_tokens']!=len(prefix) or row['token_ids'][:len(prefix)]!=prefix:
                raise ValueError('Generated token prefix differs from model input')
            if len(row['token_ids'])-len(prefix)!=row['generated_tokens']:
                raise ValueError('Generated token count differs from payload')
            pending.append((label,row))
        provenance.append(dict(arm=label,run=str(run),condition=condition,manifest_sha256=sha(run/'manifest.json'),
            generations_sha256=sha(source),model_questions_sha256=sha(inputs)))
    args.output_dir.mkdir(parents=True)
    graded=[]
    with (args.output_dir/'graded.jsonl').open('w',encoding='utf-8') as stream:
        for label,row in pending:
            item=dict(arm=label,id=row['id'],grade=score(g[row['id']],row['scoring_text']),
                max_token_censored=row['max_token_censored'],generated_tokens=row['generated_tokens'],
                prompt_tokens=row['prompt_tokens'],elapsed_seconds=row.get('elapsed_seconds'))
            graded.append(item);stream.write(json.dumps(item)+'\n');stream.flush()
    tables=outcome_tables(questions,graded,labels)
    (args.output_dir/'tables.json').write_text(json.dumps(tables,indent=2))
    manifest=dict(status='model_comparison_graded',cohort_questions_sha256=sha(qpath),grader_sidecar_sha256=sha(gpath),
        arms=provenance,scoring_implementation_sha256=sha(Path(__file__).with_name('scoring.py')),
        comparison_implementation_sha256=sha(Path(__file__)),
        limitations=['Public cohort; no private-test or pretraining-clean claim.',
            'Different trained models/tokenizers and inference systems do not isolate scale or recurrence.',
            'Time and token totals require separate hardware/tool accounting before compute-normalized comparison.'],
        files={f.name:sha(f) for f in args.output_dir.iterdir() if f.is_file()})
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(status=manifest['status'],arms=tables['arms'],pairs=tables['pairs']),indent=2))


if __name__=='__main__':main()
