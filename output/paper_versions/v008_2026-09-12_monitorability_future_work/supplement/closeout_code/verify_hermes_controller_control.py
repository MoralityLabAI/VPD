"""Verify controller outputs with a separate bit-mask tent/tree matching check."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from .audit_retrieval_exposure import ROOT,read
from .fetch_sources import sha


def validate(public, grid):
    n,m=public['n'],public['m']
    if not isinstance(grid,list) or len(grid)!=n or any(not isinstance(r,list) or len(r)!=m for r in grid):
        return False,False
    if any(v not in ('T','X','C') for row in grid for v in row):
        return False,False
    trees={(r,c) for r in range(n) for c in range(m) if public['grid'][r][c]=='T'}
    if {(r,c) for r in range(n) for c in range(m) if grid[r][c]=='T'}!=trees:
        return False,False
    tents=[(r,c) for r in range(n) for c in range(m) if grid[r][c]=='C']
    if len(tents)!=len(trees) or [sum(v=='C' for v in row) for row in grid]!=public['row_constraints'] or [sum(row[c]=='C' for row in grid) for c in range(m)]!=public['col_constraints']:
        return False,False
    if any(max(abs(r-a),abs(c-b))<=1 for i,(r,c) in enumerate(tents) for a,b in tents[i+1:]):
        return False,False
    ordered=sorted(trees)
    neighbors=[[i for i,(a,b) in enumerate(ordered) if abs(r-a)+abs(c-b)==1] for r,c in tents]
    # Literal per-tent adjacency and official perfect matching are distinct
    # predicates: two tents can each touch exactly one shared tree.
    literal=all(len(v)==1 for v in neighbors)
    states={0}
    for adjacent in neighbors:
        states={mask|(1<<i) for mask in states for i in adjacent if not mask&(1<<i)}
        if not states:
            return False,literal
    accepted=((1<<len(trees))-1) in states
    return accepted,literal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--support',type=Path,default=ROOT/'hermes_synthetic_campsite_v1')
    args=p.parse_args()
    run=args.run.resolve()
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='archived_controller_control_complete' or manifest['model_calls']!=0:
        raise ValueError('Wrong or incomplete controller experiment')
    for name,digest in manifest['files'].items():
        path=(run/name).resolve()
        if not path.is_relative_to(run) or sha(path)!=digest:
            raise ValueError('Controller artifact hash differs')
    if sha(args.support/'manifest.json')!=manifest['support_manifest_sha256']:
        raise ValueError('Support manifest differs')
    support_manifest=json.loads((args.support/'manifest.json').read_text())
    for name,digest in support_manifest['files'].items():
        if sha(args.support/name)!=digest:
            raise ValueError('Support data differ')
    cases={r['id']:dict(public=r['public'],target=r['solution'],slice='synthetic_dev')
        for r in read(args.support/'puzzles.jsonl') if r['split']=='dev'}
    source=ROOT/'dataset_v1/graders.jsonl'
    if sha(source)!=manifest['public_source_sha256']:
        raise ValueError('Public source differs')
    for row in read(source):
        if row.get('info',{}).get('task_name')=='campsite':
            meta=json.loads(row['info']['game_data_str'])['metadata']
            cases[row['id']]=dict(public={k:meta[k] for k in ('grid','row_constraints','col_constraints','n','m')},target=meta['solution'],slice='public_campsite')
    allowed={hashlib.sha256(r['observation'].encode()).hexdigest() for r in read(args.support/'logic_support.jsonl')}
    records=read(run/'rows.jsonl')
    pairs={(r['id'],r['arm']) for r in records}
    if len(records)!=len(pairs) or pairs!={(key,arm) for key in cases for arm in manifest['arms']}:
        raise ValueError('Missing or duplicate controller cases/arms')
    totals={}
    for row in records:
        case=cases[row['id']]
        if row['slice']!=case['slice'] or row['retrieved_observation_sha256'] not in allowed or row['model_calls']!=0:
            raise ValueError('Controller identity or retrieval provenance differs')
        try:
            grid=ast.literal_eval(row['action'])
        except (ValueError,SyntaxError,TypeError):
            grid=None
        official,literal=validate(case['public'],grid)
        actual=dict(official_valid=official,literal_valid=literal,canonical_exact=grid==case['target'])
        if any(row[k]!=v for k,v in actual.items()):
            raise ValueError(f'Independent score mismatch: {row["id"]}, {row["arm"]}')
        summary=totals.setdefault(row['slice'],{}).setdefault(row['arm'],dict(cases=0,official_valid=0,literal_valid=0,canonical_exact=0))
        summary['cases']+=1
        for key,value in actual.items():
            summary[key]+=int(value)
    published=json.loads((run/'summary.json').read_text())
    for subset,arms in totals.items():
        for arm,value in arms.items():
            if any(published[subset][arm][k]!=v for k,v in value.items()):
                raise ValueError('Summary totals differ')
    result=dict(status='controller_outputs_independently_verified',records=len(records),cases=len(cases),
        independent_rule='Bit-mask dynamic programming for perfect tent/tree matching, plus separate grid/count/adjacency checks.',
        source_manifest_sha256=sha(run/'manifest.json'),support_examples=len(allowed),
        implementation_sha256=sha(Path(__file__)),totals=totals,model_calls=0)
    (run/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='totals'},indent=2))


if __name__=='__main__':
    main()
