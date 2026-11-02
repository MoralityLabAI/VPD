"""Probe official logic graders with source answers, outside model inference."""
import json
import ast
import re
from collections import Counter,defaultdict
from pathlib import Path
from .scoring import score
from .fetch_sources import sha

def candidates(game,family):
    if game.get('answer') not in (None,''):
        yield 'answer',str(game['answer'])
    meta=game.get('metadata') or {}
    for key in ('solution','complete_sudoku','filled_matrix','full_sequence','target_answer','full_grid','current_mines'):
        if key in meta:
            value=meta[key]
            yield 'metadata.'+key,value if isinstance(value,str) else repr(value)
            if not isinstance(value,str):
                yield 'metadata.'+key+'.json',json.dumps(value)
    if meta.get('solutions'):
        yield 'metadata.solutions[0]',str(meta['solutions'][0])
    if family=='cipher' and isinstance(game.get('answer'),str):
        answer=game['answer']
        yield 'answer.nested_string',json.dumps([[answer[2:-2] if len(answer)>=4 else answer]])
    if family=='futoshiki' and isinstance(game.get('answer'),list):
        yield 'answer.space_rows','[['+','.join(' '.join(map(str,row)) for row in game['answer'])+']]'
    if family=='norinori' and meta.get('solution'):
        yield 'metadata.solution.coordinate_tuples',repr([[tuple(cell) for cell in domino] for domino in meta['solution']])
    if family=='mathador':
        for index,solution in enumerate(meta.get('solutions',[])):
            match=re.fullmatch(r'nums: (\(.*?\)), ops: (\(.*?\))',solution)
            if match:
                numbers,ops=map(ast.literal_eval,match.groups())
                expression=str(numbers[0])
                for op,number in zip(ops,numbers[1:]):
                    expression=f'({expression}{op}{number})'
                yield f'metadata.solutions[{index}].left_associated',expression

def main():
    root=Path('artifacts/nanbeige_reasoning')
    source=root/'dataset_v1/graders.jsonl'
    selected=defaultdict(list)
    for line in source.open(encoding='utf-8'):
        row=json.loads(line)
        if row['scorer']=='intellect3_logic' and len(selected[row['info']['task_name']])<3:
            selected[row['info']['task_name']].append(row)
    results=[]
    for family,rows in sorted(selected.items()):
        for row in rows:
            game=json.loads(row['info']['game_data_str'])
            attempts=[]
            accepted=None
            for field,text in candidates(game,family):
                for form,prediction in [('raw',text),('answer_tags','<answer>'+text+'</answer>'),
                        ('boxed','\\boxed{'+text+'}'),('double_brackets','[['+text+']]'),
                        ('python_block','```python\n'+text+'\n```')]:
                    outcome=score(row,prediction)
                    attempts.append({'source_field':field,'form':form,**outcome})
                    if outcome.get('score')==1:
                        accepted={'source_field':field,'form':form}
                        break
                if accepted:
                    break
            results.append(dict(id=row['id'],family=family,accepted=accepted,attempts=attempts,
                empty_answer_probe=score(row,''),
                status='source_answer_accepted' if accepted else 'requires_contract_review'))
    summary={family:dict(Counter(r['status'] for r in results if r['family']==family)) for family in selected}
    destination=root/'grader_audit'
    destination.mkdir(exist_ok=True)
    (destination/'rows.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    (destination/'summary.json').write_text(json.dumps(dict(families=summary,source_sha256=sha(source),
        adapter_sha256=sha(Path(__file__).with_name('safe_expression.py')),
        audit_code_sha256=sha(Path(__file__)),
        positive_probes=len(results),accepted_positive_probes=sum(r['accepted'] is not None for r in results),
        empty_answers_accepted=sum(r['empty_answer_probe'].get('score')==1 for r in results),
        scope='First three source rows per family; positive-path contract probe only',
        interpretation='A failed probe is not a model error and may mean the expected answer format needs reconstruction.'),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
