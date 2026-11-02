"""Recompute the public solver receipt against pinned HF rows and official rules."""
import ast
import importlib.util
import json
import subprocess
from collections import Counter
from pathlib import Path
from .fetch_sources import sha
from .scoring import logic_api

REV='07a177f2751c2adb1cc16dc819c1173c0cd57ad5'
REPO=Path('C:/projects/Hermes-Skills')
ROOT=Path('artifacts/nanbeige_reasoning/hermes_audit')

def copy_committed(path):
    result=subprocess.run(['git','-c','safe.directory=C:/projects/Hermes-Skills','-C',str(REPO),
        'show',f'{REV}:{path}'],capture_output=True,check=True)
    dest=ROOT/'source'/path
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(result.stdout)
    return dest

def public_rule(grid,meta,exactly_one):
    if not isinstance(grid,list) or len(grid)!=meta['n'] or any(len(r)!=meta['m'] for r in grid):
        return False
    if any(v not in ('X','T','C') for r in grid for v in r):
        return False
    tents=[(i,j) for i,r in enumerate(grid) for j,v in enumerate(r) if v=='C']
    trees=[(i,j) for i,r in enumerate(meta['grid']) for j,v in enumerate(r) if v=='T']
    if set(trees)!={(i,j) for i,r in enumerate(grid) for j,v in enumerate(r) if v=='T'} or len(tents)!=len(trees):
        return False
    if [sum(v=='C' for v in r) for r in grid]!=meta['row_constraints']:
        return False
    if [sum(r[j]=='C' for r in grid) for j in range(meta['m'])]!=meta['col_constraints']:
        return False
    for i,j in tents:
        count=sum(abs(i-a)+abs(j-b)==1 for a,b in trees)
        if (count!=1 if exactly_one else count<1):
            return False
        if any((a,b)!=(i,j) and max(abs(i-a),abs(j-b))<=1 for a,b in tents):
            return False
    return True

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    paths=['research/scripts/run_intellect3_logic_dataset_faithful_solver.py',
        'intellect3-logic-hermes/references/receipts.md','intellect3-math-hermes/references/receipts.md',
        'intellect3-logic-hermes/references/contract.md','intellect3-math-hermes/references/contract.md',
        'research/studies/2026-04-22-metta-trm-hermes-pipeline/artifacts/intellect3_logic_dataset_faithful_solver_20260502/intellect3_logic_dataset_faithful_solver.results.json']
    copied=[copy_committed(p) for p in paths]
    spec=importlib.util.spec_from_file_location('hermes_reference_solver',copied[0])
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Data,verifiers=logic_api()
    source=Path('artifacts/nanbeige_reasoning/dataset_v1/graders.jsonl')
    graders=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
    graders=[r for r in graders if r.get('info',{}).get('task_name')=='campsite']
    raw_path=Path('D:/research_engine/trmObservabilityLogs/trm_observability_harness_snapshot/data/qwen27b_intellect3_logic_hybrid_200/predictions.jsonl')
    predictions=[json.loads(x) for x in raw_path.read_text(encoding='utf-8').splitlines()]
    by_target={json.dumps(r['expected_action'],sort_keys=True):{} for r in predictions}
    for r in predictions:
        by_target[json.dumps(r['expected_action'],sort_keys=True)][r['arm']]=r
    results=[]
    for r in graders:
        data=Data.from_json_str(r['info']['game_data_str'])
        meta=data.metadata
        # Neither the reference solver nor the rule predicate receives solution or trace ID.
        public={k:meta[k] for k in ('grid','row_constraints','col_constraints','n','m')}
        solutions=module.solve_dataset_faithful(public['grid'],public['row_constraints'],public['col_constraints'],max_solutions=5001)
        target=meta['solution']
        public_data=Data(question='',answer='',metadata=public)
        official_solutions=[s for s in solutions if verifiers['campsite']().verify(public_data,repr(s))]
        literal_solutions=[s for s in solutions if public_rule(s,public,True)]
        official_first=official_solutions[0] if official_solutions else None
        hist=by_target.get(json.dumps(str(target),sort_keys=True),{})
        if not hist:
            hist=by_target.get(json.dumps(target,sort_keys=True),{})
        first=solutions[0] if solutions else None
        entry=dict(id=r['id'],solution_count=len(solutions),first_solution=first,
            official_filtered_solution=official_first,
            official_filtered_valid=official_first is not None,
            official_filtered_exact=official_first==target,
            literal_contract_has_solution=bool(literal_solutions),
            first_exact=first==target,first_relaxed_valid=public_rule(first,public,False) if first else False,
            first_prompt_literal_valid=public_rule(first,public,True) if first else False,
            first_official_valid=bool(verifiers['campsite']().verify(data,repr(first))),
            target_relaxed_valid=public_rule(target,public,False),
            target_prompt_literal_valid=public_rule(target,public,True),
            target_official_valid=bool(verifiers['campsite']().verify(data,repr(target))),
            historical_arms={})
        for arm,prediction in hist.items():
            candidate=module.parse_grid(prediction['final']['action'])
            entry['historical_arms'][arm]=dict(exact=candidate==target,
                relaxed_valid=public_rule(candidate,public,False) if candidate else False,
                prompt_literal_valid=public_rule(candidate,public,True) if candidate else False,
                official_valid=bool(verifiers['campsite']().verify(data,repr(candidate))),
                model_calls=prediction['model_calls'])
        results.append(entry)
    summary=dict(revision=REV,rows=len(results),solution_counts=dict(Counter(r['solution_count'] for r in results)),
        totals={key:sum(r[key] for r in results) for key in ('official_filtered_valid',
            'official_filtered_exact','literal_contract_has_solution','first_exact','first_relaxed_valid',
            'first_prompt_literal_valid','first_official_valid','target_relaxed_valid',
            'target_prompt_literal_valid','target_official_valid')},
        historical_rows_matched=sum(bool(r['historical_arms']) for r in results),
        historical_arms={arm:{metric:sum(r['historical_arms'].get(arm,{}).get(metric,False) for r in results)
            for metric in ('exact','relaxed_valid','prompt_literal_valid','official_valid')}
            for arm in sorted({a for r in results for a in r['historical_arms']})},
        provenance={'files':{str(p.relative_to(ROOT)):sha(p) for p in copied},
            'historical_predictions_sha256':sha(raw_path),'grader_sidecar_sha256':sha(source)},
        interpretation='Public solver replay is a deterministic control arm, not a neural TRM or fresh 27B run.')
    (ROOT/'rows.jsonl').write_text('\n'.join(json.dumps(x) for x in results)+'\n',encoding='utf-8')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='provenance'},indent=2))

if __name__=='__main__':
    main()
