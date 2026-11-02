"""Construct new Campsite support, separate from every public source puzzle.

This is a seeded synthetic training distribution, not the original publisher's
generator or a reconstruction of historical TRM training.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from .audit_hermes import public_rule
from .audit_retrieval_exposure import puzzle_key, read, ROOT
from .fetch_sources import sha
from .scoring import logic_api, ENV

SHAPES = ((4,4),(5,4),(4,5),(5,5),(6,5),(5,6),(6,6),(7,6))


def symmetry_key(grid, rows, columns):
    keys=[]
    for reflected in (False, True):
        current=np.asarray(grid)
        rr,cc=list(rows),list(columns)
        if reflected:
            current=np.fliplr(current)
            cc=cc[::-1]
        for _ in range(4):
            keys.append(puzzle_key(current.tolist(),rr,cc))
            current=np.rot90(current)
            rr,cc=cc[::-1],rr
    return min(keys)


def construct(rng, n, m):
    cells=[(r,c) for r in range(n) for c in range(m)]
    rng.shuffle(cells)
    count=rng.randint(2,max(2,n*m//5))
    tents=[]
    for cell in cells:
        if all(max(abs(cell[0]-r),abs(cell[1]-c))>1 for r,c in tents):
            tents.append(cell)
            if len(tents)==count:
                break
    if len(tents)!=count:
        return None
    trees=[]
    for r,c in tents:
        adjacent=[(r+dr,c+dc) for dr,dc in ((0,1),(1,0),(0,-1),(-1,0))]
        candidates=[cell for cell in adjacent if 0<=cell[0]<n and 0<=cell[1]<m
            and cell not in tents and cell not in trees
            and sum(abs(cell[0]-a)+abs(cell[1]-b)==1 for a,b in tents)==1]
        if not candidates:
            return None
        trees.append(rng.choice(candidates))
    grid=[['X']*m for _ in range(n)]
    for r,c in trees:
        grid[r][c]='T'
    solution=[row.copy() for row in grid]
    for r,c in tents:
        solution[r][c]='C'
    rows=[sum(v=='C' for v in row) for row in solution]
    columns=[sum(row[c]=='C' for row in solution) for c in range(m)]
    return dict(grid=grid,row_constraints=rows,col_constraints=columns,n=n,m=m),solution


def observation(public):
    n,m=public['n'],public['m']
    grid='\n'.join(' '.join(row) for row in public['grid'])
    return (f'Campsite puzzle ({n}x{m})\n\nGRID:\n{grid}\n\n'
        f'ROW_CONSTRAINTS: {public["row_constraints"]}\nCOL_CONSTRAINTS: {public["col_constraints"]}\n\n'
        "Place C tents in X cells. Preserve trees T. Match the row and column tent counts. "
        "Each tent must touch exactly one tree horizontally or vertically, with a one-to-one tent/tree matching. "
        "Tents must not touch each other, even diagonally. Return only the completed Python-style 2D list.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--train',type=int,default=128)
    p.add_argument('--dev',type=int,default=32)
    p.add_argument('--seed',type=int,default=20260906)
    args=p.parse_args()
    if min(args.train,args.dev)<1:
        p.error('Nonempty training and development sets required')
    source=ROOT/'dataset_v1/graders.jsonl'
    forbidden=set()
    source_count=0
    for row in read(source):
        if row.get('info',{}).get('task_name')=='campsite':
            public=json.loads(row['info']['game_data_str'])['metadata']
            forbidden.add(symmetry_key(public['grid'],public['row_constraints'],public['col_constraints']))
            source_count+=1
    Data,verifiers=logic_api()
    verifier=verifiers['campsite']()
    rng=random.Random(args.seed)
    accepted,seen,rejected=[],set(),Counter()
    for attempt in range(100000):
        if len(accepted)==args.train+args.dev:
            break
        candidate=construct(rng,*SHAPES[len(accepted)%len(SHAPES)])
        if candidate is None:
            rejected['construction']+=1
            continue
        public,solution=candidate
        key=symmetry_key(public['grid'],public['row_constraints'],public['col_constraints'])
        if key in forbidden or key in seen:
            rejected['public_or_generated_symmetry_alias']+=1
            continue
        data=Data(question='',answer='',metadata=public)
        if not public_rule(solution,public,True) or not verifier.verify(data,repr(solution)):
            raise ValueError('Constructed solution violates independent rules')
        seen.add(key)
        identity='synthetic_campsite/'+hashlib.sha256(key.encode()).hexdigest()[:20]
        accepted.append(dict(id=identity,split='train' if len(accepted)<args.train else 'dev',
            public=public,solution=solution,observation=observation(public),
            symmetry_key_sha256=hashlib.sha256(key.encode()).hexdigest(),construction_attempt=attempt))
    if len(accepted)!=args.train+args.dev:
        raise RuntimeError('Construction budget exhausted')
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    support=[]
    for row in accepted:
        if row['split']!='train':
            continue
        public=row['public']
        support.append(dict(id=row['id'],task_family='logic',task='intellect_3_logic',
            observation=row['observation'],target_action=repr(row['solution']),bucket='exact_positive',
            supervision_weight=1.0,reward=1.0,grid_cell_accuracy=1.0,visible_output_emitted=True,
            response_mode='action_only',skill_name='intellect3-logic-hermes',
            support_signature=json.dumps(dict(row_constraints=public['row_constraints'],col_constraints=public['col_constraints']),separators=(',',':')),
            provenance='Seeded synthetic valid solution; no language model call or historical training replay.'))
    for name,rows in [('puzzles.jsonl',accepted),('logic_support.jsonl',support)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    official=ENV/'reasoning/i3_logic/i3_logic/games/tasks/campsite/scripts/campsite_verifier.py'
    result=dict(status='synthetic_campsite_support_constructed',seed=args.seed,train=args.train,dev=args.dev,
        source_puzzles_excluded=source_count,source_symmetry_orbits_excluded=len(forbidden),
        rejected=dict(rejected),shapes=[list(s) for s in SHAPES],all_official_and_literal_valid=True,
        provenance={str(f):sha(f) for f in (source,official,Path(__file__),Path(__file__).with_name('audit_hermes.py'))},
        files={f.name:sha(f) for f in out.iterdir() if f.is_file()},
        limitations=['New constructed distribution; not the original publisher generator or historical TRM training.',
            'Construction enforces literal adjacency and may be easier than official-only public puzzles.',
            'Exact and dihedral symmetry exclusions do not rule out broader semantic similarity.',
            'Only training records enter the support bank; development targets remain local.'])
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','provenance')},indent=2))


if __name__=='__main__':
    main()
