"""Check supplied Futoshiki grids against their own recorded inequalities."""
import json
from collections import Counter
from pathlib import Path
from .fetch_sources import sha
from .scoring import score


def main():
    root=Path('artifacts/nanbeige_reasoning')
    source=root/'dataset_v1/graders.jsonl'
    benchmark={r['id'] for r in map(json.loads,(root/'benchmark_v1/questions.jsonl').read_text(encoding='utf-8').splitlines())}
    rows=[]
    for row in map(json.loads,source.read_text(encoding='utf-8').splitlines()):
        if row['scorer']!='intellect3_logic' or row['info']['task_name']!='futoshiki':
            continue
        game=json.loads(row['info']['game_data_str'])
        grid=game['answer']
        invalid=[]
        for first,second,sign in game['metadata']['constraints']:
            a,b=grid[first[0]][first[1]],grid[second[0]][second[1]]
            if not (a>b if sign=='>' else a<b):
                invalid.append(dict(first=first,second=second,sign=sign,values=[a,b]))
        prediction='[['+','.join(' '.join(map(str,line)) for line in grid)+']]'
        outcome=score(row,prediction)
        rows.append(dict(id=row['id'],in_benchmark=row['id'] in benchmark,
            supplied_grid_grade=outcome,violated_inequalities=invalid))
    out=root/'grader_audit'
    out.mkdir(exist_ok=True)
    (out/'futoshiki_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    summary=dict(source_sha256=sha(source),records=len(rows),
        supplied_grids_with_inequality_violations=sum(bool(r['violated_inequalities']) for r in rows),
        supplied_grids_accepted=sum(r['supplied_grid_grade'].get('score',0)==1 for r in rows),
        benchmark_records=sum(r['in_benchmark'] for r in rows),
        benchmark_supplied_grids_with_violations=sum(r['in_benchmark'] and bool(r['violated_inequalities']) for r in rows),
        interpretation='Reference-grid inconsistency does not prove a puzzle is unsatisfiable. No answers were repaired or used for inference.')
    (out/'futoshiki_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
