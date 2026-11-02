"""Observe actual external-controller sweeps without editing its implementation."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from .audit_retrieval_exposure import ROOT,read
from .build_campsite_support import observation
from .fetch_sources import sha


def observed_corrector(module, obs, candidate, exact, cap):
    counts=Counter()
    source=str(Path(module.__file__).resolve())
    exact_result=[]
    def observe(frame,event,arg):
        if frame.f_code.co_filename!=source:
            return
        name=frame.f_code.co_name
        if event=='call' and name in ('search','_score_grid','_solve_exact'):
            counts[name]+=1
        if event=='return' and name=='_solve_exact':
            exact_result.append(arg is not None)
    saved=sys.getprofile()
    try:
        sys.setprofile(observe)
        result=module.TRMLogicCorrector(module.LogicCorrectorConfig(
            exact_search=exact,max_passes=cap,exact_node_limit=200000)).correct(obs,candidate)
    finally:
        sys.setprofile(saved)
    parsed=module.parse_logic_observation(obs)
    mutable=sum(v!='T' for row in parsed['base_grid'] for v in row)
    if counts['_score_grid']:
        trials=counts['_score_grid']-1
        if not mutable or trials%mutable:
            raise ValueError('Score-call count does not correspond to complete repair sweeps')
        sweeps=trials//mutable
    else:
        sweeps=0
    if sweeps>cap or counts['_solve_exact']!=int(exact) or len(exact_result)!=int(exact):
        raise ValueError('Observed controller work differs from configured execution')
    return result,dict(repair_sweeps=sweeps,score_evaluations=counts['_score_grid'],
        recursive_search_calls=counts['search'],exact_search_returned_solution=bool(exact_result and exact_result[0]),
        exact_search_called=bool(counts['_solve_exact']),configured_repair_cap=cap)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--support',type=Path,default=ROOT/'hermes_synthetic_campsite_v1')
    args=p.parse_args()
    verification=json.loads((args.run/'verification.json').read_text())
    if verification['status']!='controller_outputs_independently_verified':
        raise ValueError('Verify controller outputs first')
    manifest=json.loads((args.run/'manifest.json').read_text())
    if sha(args.run/'manifest.json')!=verification['source_manifest_sha256']:
        raise ValueError('Controller source manifest changed')
    source=(args.run/'source/trm_logic_corrector.py').resolve()
    if sha(source)!=manifest['files']['source/trm_logic_corrector.py']:
        raise ValueError('Archived implementation changed')
    spec=importlib.util.spec_from_file_location('vpd_traced_corrector',source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    cases={r['id']:r['public'] for r in read(args.support/'puzzles.jsonl') if r['split']=='dev'}
    for row in read(ROOT/'dataset_v1/graders.jsonl'):
        if row.get('info',{}).get('task_name')=='campsite':
            meta=json.loads(row['info']['game_data_str'])['metadata']
            cases[row['id']]={k:meta[k] for k in ('grid','row_constraints','col_constraints','n','m')}
    bank={hashlib.sha256(r['observation'].encode()).hexdigest():r['target_action'] for r in read(args.support/'logic_support.jsonl')}
    records=read(args.run/'rows.jsonl')
    traced=[]
    for row in records:
        if row['arm']=='retrieval_only':
            continue
        exact=row['arm'].startswith('exact_')
        cap=8 if exact else int(row['arm'].rsplit('_',1)[1])
        candidate='' if row['arm']=='exact_empty' else bank[row['retrieved_observation_sha256']]
        action,counts=observed_corrector(module,observation(cases[row['id']]),candidate,exact,cap)
        if action!=row['action']:
            raise ValueError('Instrumentation changed a saved controller output')
        traced.append(dict(id=row['id'],slice=row['slice'],arm=row['arm'],**counts))
    summary={}
    for subset in ('synthetic_dev','public_campsite'):
        summary[subset]={}
        for arm in sorted({r['arm'] for r in traced}):
            rows=[r for r in traced if r['slice']==subset and r['arm']==arm]
            summary[subset][arm]=dict(cases=len(rows),
                sweep_histogram=dict(sorted(Counter(r['repair_sweeps'] for r in rows).items())),
                exact_solutions=sum(r['exact_search_returned_solution'] for r in rows),
                recursive_search_calls=sum(r['recursive_search_calls'] for r in rows),
                max_recursive_search_calls=max(r['recursive_search_calls'] for r in rows))
    (args.run/'execution_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in traced))
    result=dict(status='controller_execution_observed_with_output_parity',records=len(traced),
        exact_saved_output_matches=len(traced),summary=summary,
        source_manifest_sha256=sha(args.run/'manifest.json'),implementation_sha256=sha(Path(__file__)),
        trace_sha256=sha(args.run/'execution_trace.jsonl'),
        scope='Python call tracing of external symbolic controller; not transformer routing or neural TRM depth. Timing under instrumentation is not used.')
    (args.run/'execution_trace_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],records=len(traced),public=summary['public_campsite']),indent=2))


if __name__=='__main__':
    main()
