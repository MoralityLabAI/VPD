"""Answer grading, kept outside the inference process and its uploaded payload."""
import ast
import importlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from .safe_expression import safe_eval

ENV = Path('artifacts/nanbeige_reasoning/sources/research_environments/environments')

def final_text(text):
    if '<think>' in text and '</think>' not in text:
        return ''
    text=text.split('</think>')[-1].strip()
    return text.replace('<|im_end|>','').replace('</s>','').strip()

def arc_grid(text):
    text=final_text(text)
    matches=re.findall(r'<answer>(.*?)</answer>',text,re.S)
    if matches:
        text=matches[-1].strip()
    else:
        text=re.sub(r'^```(?:json)?\s*|\s*```$','',text).strip()
    try:
        grid=json.loads(text)
    except (ValueError,TypeError):
        return None
    if not isinstance(grid,list) or not 1<=len(grid)<=30:
        return None
    if any(not isinstance(row,list) or not 1<=len(row)<=30 for row in grid):
        return None
    if len({len(row) for row in grid})!=1:
        return None
    if any(type(x) is not int or not 0<=x<=9 for row in grid for x in row):
        return None
    return grid

def logic_api():
    # Load only the published pure verifiers, without the environment's task runner.
    # This bypasses __init__'s optional verifiers.v1 import, not any grading code.
    path=(ENV/'reasoning/i3_logic/i3_logic').resolve()
    if 'i3_logic' not in sys.modules:
        package=types.ModuleType('i3_logic')
        package.__path__=[str(path)]
        sys.modules['i3_logic']=package
    data=importlib.import_module('i3_logic.base.data').Data
    verifiers=importlib.import_module('i3_logic.task2verifier').verifier_classes
    # Several published verifiers eval answer strings. Override only their eval
    # lookup with bounded parsing, without modifying their pinned source files.
    for name,module in list(sys.modules.items()):
        if name.startswith('i3_logic.') and getattr(module,'__file__',None):
            if Path(module.__file__).resolve().is_relative_to(path):
                module.eval=safe_eval
    return data,verifiers

def score(grader,text):
    try:
        kind=grader['scorer']
        if kind=='arc_exact_grid':
            parsed=arc_grid(text)
            return {'status':'graded','score':float(parsed==grader['gold']),
                    'format_valid':parsed is not None}
        if kind=='intellect3_logic':
            Data,verifiers=logic_api()
            info=grader['info']
            task=info['task_name']
            if task not in verifiers:
                return {'status':'unsupported_verifier','score':None,'task':task}
            data=Data.from_json_str(info['game_data_str'])
            return {'status':'graded','score':float(verifiers[task]().verify(data,final_text(text))),
                    'execution_guard':'bounded_literal_arithmetic_v1'}
        if kind=='intellect3_math':
            script=ENV/'math/i3_math/i3_math/verify.py'
            # Windows math-verify's nested multiprocessing target is not spawn-safe.
            # Keep published extraction/equivalence logic; enforce an outer process
            # timeout instead of its per-operation signal/multiprocessing timeout.
            inner_timeout='0' if os.name=='nt' else '5'
            result=subprocess.run([sys.executable,'-X','utf8','-m','experiments.nanbeige_reasoning.math_verify_stdin'],
                input=json.dumps(dict(gold=grader['gold'],prediction=text,timeout=int(inner_timeout))),
                capture_output=True,text=True,encoding='utf-8',timeout=30)
            if result.returncode or 'Traceback (most recent call last)' in result.stderr:
                raise RuntimeError(result.stderr[-500:])
            return {'status':'graded','score':float(result.stdout.strip().splitlines()[-1])}
        return {'status':'unsupported_scorer','score':None}
    except Exception as exc:
        return {'status':'grader_error','score':None,'error':f'{type(exc).__name__}: {exc}'}

def paired_outcomes(scored,questions):
    """Pair evaluation outcomes by task, without treating missing grades as zero."""
    q={row['id']:row for row in questions}
    lookup={(row['id'],row['loops']):row for row in scored}
    slices=sorted({(row['lane'],row['split']) for row in scored if row['calibration_split']=='eval'}
                  | {(row['lane'],row['split']) for row in questions if row.get('calibration_split')=='eval'})
    summaries=[]
    for lane,evaluation_slice in slices:
        selected=[row for row in scored if row['lane']==lane and row['split']==evaluation_slice
                  and row['calibration_split']=='eval']
        arc=lane.startswith('arc_')
        units={q[row['id']]['group_id'] if arc else row['id'] for row in selected}
        units.update(row['group_id'] if arc else row['id'] for row in questions
                     if row.get('calibration_split')=='eval' and row['lane']==lane and row['split']==evaluation_slice)
        counts=dict(both_correct=0,native_only_correct=0,one_pass_only_correct=0,both_incorrect=0,
                    missing_conditions=0,ungradable_pairs=0)
        details=[]
        for unit in sorted(units):
            ids=({row['id'] for row in questions if row['lane']==lane and row['group_id']==unit}
                 if arc else {unit})
            required={(key,loops) for key in ids for loops in (1,2)}
            if not required.issubset(lookup):
                state='missing_conditions'
            elif any(lookup[key]['grade']['status']!='graded' for key in required):
                state='ungradable_pairs'
            else:
                if any(lookup[key]['grade']['score'] not in (0,1) for key in required):
                    raise ValueError('Paired success table requires binary scores')
                one=all(lookup[key,1]['grade']['score']==1 for key in ids)
                native=all(lookup[key,2]['grade']['score']==1 for key in ids)
                state=('both_correct' if one and native else 'native_only_correct' if native
                       else 'one_pass_only_correct' if one else 'both_incorrect')
            counts[state]+=1
            details.append(dict(unit=unit,test_inputs=len(ids),outcome=state))
        graded=len(units)-counts['missing_conditions']-counts['ungradable_pairs']
        summaries.append(dict(lane=lane,evaluation_slice=evaluation_slice,
            unit='complete_arc_task' if arc else 'question',units=len(units),paired_graded=graded,**counts,
            native_accuracy=(counts['both_correct']+counts['native_only_correct'])/graded if graded else None,
            one_pass_accuracy=(counts['both_correct']+counts['one_pass_only_correct'])/graded if graded else None,
            native_minus_one_pass=(counts['native_only_correct']-counts['one_pass_only_correct'])/graded if graded else None,
            details=details))
    return summaries


def grade_run(run,questions,graders):
    rows=[json.loads(x) for x in (run/'generations.jsonl').read_text(encoding='utf-8').splitlines()]
    q={x['id']:x for x in questions}
    g={x['id']:x for x in graders}
    if len({(r['id'],r['loops']) for r in rows})!=len(rows):
        raise ValueError('Duplicate generation condition')
    scored=[{**r,'grade':score(g[r['id']],r['scoring_text'])} for r in rows]
    (run/'graded_generations.jsonl').write_text('\n'.join(json.dumps(x) for x in scored)+'\n',encoding='utf-8')
    summaries=[]
    slices=sorted({(r['lane'],r['split']) for r in rows if r['calibration_split']=='eval'})
    for lane,evaluation_slice in slices:
        for loops in sorted({r['loops'] for r in rows}):
            selected=[r for r in scored if r['lane']==lane and r['loops']==loops
                      and r['calibration_split']=='eval' and r['split']==evaluation_slice]
            valid=[r for r in selected if r['grade']['status']=='graded']
            summary=dict(lane=lane,evaluation_slice=evaluation_slice,loops=loops,attempts=len(selected),graded=len(valid),
                grader_errors=len(selected)-len(valid),correct=sum(r['grade']['score'] for r in valid),
                censored=sum(r['max_token_censored'] for r in selected),
                accuracy=sum(r['grade']['score'] for r in valid)/len(valid) if valid else None)
            if lane.startswith('arc_'):
                groups={q[r['id']]['group_id'] for r in selected}
                complete=[]
                for group in groups:
                    required={x['id'] for x in questions if x['lane']==lane and x['group_id']==group}
                    attempts=[r for r in selected if q[r['id']]['group_id']==group]
                    if {r['id'] for r in attempts}!=required:
                        raise ValueError('Incomplete ARC test-input group')
                    complete.append(all(r['grade']['status']=='graded' and r['grade']['score']==1 for r in attempts))
                summary.update(tasks=len(groups),all_test_inputs_correct=sum(complete))
            summaries.append(summary)
    (run/'scores.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    (run/'paired_scores.json').write_text(json.dumps(paired_outcomes(scored,questions),indent=2),encoding='utf-8')
    if rows and all('token_ids' in row for row in rows):
        from .generation_health import write_health
        write_health(run,questions)
    return summaries

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--cohort',type=Path,default=Path('artifacts/nanbeige_reasoning/pilot_v1'))
    args=parser.parse_args()
    root=args.cohort
    print(json.dumps(grade_run(args.run,*[[json.loads(x) for x in (root/name).read_text(encoding='utf-8').splitlines()]
        for name in ('questions.jsonl','graders.jsonl')]),indent=2))
