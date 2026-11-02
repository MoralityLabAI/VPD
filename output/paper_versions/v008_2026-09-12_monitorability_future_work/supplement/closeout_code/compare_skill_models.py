"""Compare completed matched Campsite studies without treating missing calls as failures."""
import argparse
import json
from pathlib import Path

from .compare_models import outcome_tables
from experiments.nanbeige_receipt_replay.io import digest,save


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def load_study(preparation,run,grading,allow_partial=False):
    cfg=read(preparation/'protocol.json');verification=read(grading/'verification.json')
    expected='partial_grading_and_correction_complete' if allow_partial else 'grading_and_correction_complete'
    if verification['status']!=expected:raise ValueError('Unexpected grading status')
    if digest(preparation/'protocol.json')!=verification['protocol_sha256']:raise ValueError('Grading protocol differs')
    if digest(run/'manifest.json')!=verification['generation_manifest_sha256']:raise ValueError('Generation manifest differs')
    for folder,files in ((grading,verification['files']),(run,read(run/'manifest.json')['files'])):
        for name,checksum in files.items():
            if digest(folder/name)!=checksum:raise ValueError(f'Changed artifact: {folder/name}')
    manifest=read(run/'manifest.json')
    if allow_partial:
        if manifest['status']!='interrupted' or not 0<manifest['records']<cfg['unique_model_calls']:
            raise ValueError('Expected an interrupted partial study')
    elif manifest['status']!='logic_skill_generation_complete' or manifest['records']!=cfg['unique_model_calls']:
        raise ValueError('All planned model calls must finish before comparison')
    for name,field in (('calls.jsonl','calls_sha256'),('cases.json','cases_sha256')):
        if digest(preparation/name)!=cfg[field]:raise ValueError('Prepared inputs differ')
    return cfg,read(preparation/'cases.json'),read(grading/'rows.json'),verification


def matched_tables(cases,left,right,arms):
    ids=[r['id'] for r in cases]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate case IDs')
    expected={(identifier,arm) for identifier in ids for arm in arms}
    graded=[]
    for model,rows in (('qwen27b',left),('nanbeige',right)):
        keys=[(r['id'],r['arm']) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Missing, duplicate or extra graded cases')
        for r in rows:
            if r['official_score'] not in (0,1):raise ValueError('Expected binary official grades')
            graded.append(dict(id=r['id'],arm=f'{model}/{r["arm"]}',grade=dict(status='graded',score=r['official_score']),
                               max_token_censored=r['max_token_censored']))
    questions=[dict(id=i,lane='intellect3_logic',split='selected_historical_campsite',calibration_split='eval',group_id=i) for i in ids]
    labels=[f'{model}/{arm}' for model in ('qwen27b','nanbeige') for arm in arms]
    tables=outcome_tables(questions,graded,labels)
    selected=[]
    for arm in arms:
        row=next(r for r in tables['pairs'] if r['left']==f'qwen27b/{arm}' and r['right']==f'nanbeige/{arm}')
        selected.append(dict(procedure=arm,puzzles=row['units'],both_correct=row['both_correct'],
            qwen_only_correct=row['left_only_correct'],nanbeige_only_correct=row['right_only_correct'],
            both_incorrect=row['both_incorrect'],nanbeige_minus_qwen=-row['left_minus_right'],
            qwen_capped=sum(r['arm']==arm and r['max_token_censored'] for r in left),
            nanbeige_capped=sum(r['arm']==arm and r['max_token_censored'] for r in right)))
    return tables,selected


def partial_matched_tables(cases,left,right,arms):
    expected={(c['id'],arm) for c in cases for arm in arms}
    for rows in (left,right):
        keys=[(r['id'],r['arm']) for r in rows]
        if len(keys)!=len(set(keys)) or not set(keys)<=expected:
            raise ValueError('Duplicate or unknown graded case')
    tables={};selected=[]
    for arm in arms:
        ids={r['id'] for r in left if r['arm']==arm}&{r['id'] for r in right if r['arm']==arm}
        if not ids:continue
        subset=[c for c in cases if c['id'] in ids]
        l=[r for r in left if r['id'] in ids and r['arm']==arm]
        r=[r for r in right if r['id'] in ids and r['arm']==arm]
        table,rows=matched_tables(subset,l,r,[arm])
        tables[arm]=table
        for row in rows:row['matched_ids']=sorted(ids)
        selected.extend(rows)
    return tables,selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen-preparation',type=Path,required=True);p.add_argument('--qwen-run',type=Path,required=True)
    p.add_argument('--qwen-grading',type=Path,required=True)
    p.add_argument('--nanbeige-preparation',type=Path,required=True);p.add_argument('--nanbeige-run',type=Path,required=True)
    p.add_argument('--nanbeige-grading',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial-nanbeige',action='store_true')
    a=p.parse_args()
    qcfg,qcases,qrows,qverify=load_study(a.qwen_preparation,a.qwen_run,a.qwen_grading)
    ncfg,ncases,nrows,nverify=load_study(a.nanbeige_preparation,a.nanbeige_run,a.nanbeige_grading,a.allow_partial_nanbeige)
    if qcfg['model']!='Qwen3.5-27B' or ncfg['model']!='Nanbeige4.2-3B':raise ValueError('Wrong model identities')
    if ncfg['source_qwen_protocol_sha256']!=digest(a.qwen_preparation/'protocol.json'):raise ValueError('Not derived from this Qwen protocol')
    for field in ('cases_sha256','reported_arms','helper_sources','repair','max_new_tokens','thinking_enabled','seed','temperature','repeat_penalty'):
        if qcfg[field]!=ncfg[field]:raise ValueError(f'Comparison setting differs: {field}')
    if qcases!=ncases:raise ValueError('Puzzle, route or workspace records differ')
    if qverify['source_grade_sidecar_sha256']!=nverify['source_grade_sidecar_sha256']:raise ValueError('Reference sidecars differ')
    qcalls={r['call_id']:r for r in map(json.loads,(a.qwen_preparation/'calls.jsonl').read_text(encoding='utf-8').splitlines())}
    ncalls={r['call_id']:r for r in map(json.loads,(a.nanbeige_preparation/'calls.jsonl').read_text(encoding='utf-8').splitlines())}
    if set(qcalls)!=set(ncalls):raise ValueError('Model call identities differ')
    prefix='<|im_start|>user\n';suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    for key,q in qcalls.items():
        rendered=q['rendered_prompt']
        if not rendered.startswith(prefix) or not rendered.endswith(suffix):raise ValueError('Unexpected Qwen template')
        if ncalls[key]['messages']!=[dict(role='user',content=rendered[len(prefix):-len(suffix)])]:raise ValueError('User message differs')
    table_fn=partial_matched_tables if a.allow_partial_nanbeige else matched_tables
    tables,selected=table_fn(qcases,qrows,nrows,qcfg['reported_arms'])
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'all_pairings.json',tables);save(a.output/'same_procedure_pairs.json',selected)
    lines=['# Matched user-message Campsite comparison','',
        'Each row compares the same procedure on the same public puzzles. Native templates, model architectures, tokenizers and hardware differ. Counts are descriptive for this exposed cohort; they do not isolate recurrence or establish equal compute.','',
        '| Procedure | Matched puzzles | Both correct | Qwen only | Nanbeige only | Both incorrect | Nanbeige minus Qwen | Qwen / Nanbeige capped |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f'| {r["procedure"]} | {r["puzzles"]} | {r["both_correct"]} | {r["qwen_only_correct"]} | {r["nanbeige_only_correct"]} | {r["both_incorrect"]} | {100*r["nanbeige_minus_qwen"]:.1f} percentage points | {r["qwen_capped"]} / {r["nanbeige_capped"]} |')
    if a.allow_partial_nanbeige:lines+=['','Interrupted scheduled prefix only. Each procedure has its own exact matched puzzle intersection. Missing calls are unrun, not incorrect. Differences between procedures may use different puzzles; use within-model paired results for shared denominators. This is not the full planned benchmark.']
    lines+=['','Capped candidates retain their published-verifier grades and are counted separately. Procedures share candidate calls and puzzles; table rows are not independent replications. No interval or general-population inference is supplied for this selected, previously exposed cohort. ARC and math are outside this comparison.']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(a.output/'verification.json',dict(status='partial_matched_campsite_models_compared' if a.allow_partial_nanbeige else 'matched_campsite_models_compared',planned_puzzles=len(qcases),
        qwen_grading_sha256=digest(a.qwen_grading/'verification.json'),nanbeige_grading_sha256=digest(a.nanbeige_grading/'verification.json'),
        same_user_messages_verified=len(qcalls),code_sha256=digest(Path(__file__)),broader_goal_complete=False,
        files={f.name:digest(f) for f in a.output.iterdir() if f.is_file()}))
    print(json.dumps(selected))


if __name__=='__main__':main()
