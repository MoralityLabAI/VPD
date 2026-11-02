import pytest
import json
import sys
from pathlib import Path

from experiments.nanbeige_reasoning.compare_models import outcome_tables
from experiments.nanbeige_reasoning.compare_models import main as compare_main
from experiments.nanbeige_reasoning.fetch_sources import sha


def test_comparison_uses_complete_tasks_and_separates_unknowns():
    questions=[dict(id='a0',group_id='a',lane='arc_agi_1',split='eval',calibration_split='eval'),
        dict(id='a1',group_id='a',lane='arc_agi_1',split='eval',calibration_split='eval'),
        dict(id='b0',group_id='b',lane='arc_agi_1',split='eval',calibration_split='eval'),
        dict(id='unseen',group_id='c',lane='arc_agi_1',split='eval',calibration_split='eval'),
        dict(id='historical',group_id='h',lane='intellect3_logic',split='historical_eval',calibration_split='eval'),
        dict(id='training',group_id='t',lane='arc_agi_1',split='train',calibration_split='train')]
    graded=[]
    for arm,values in [('native',[1,0,1]),('qwen',[1,1,0])]:
        for key,value in zip(['a0','a1','b0'],values):
            graded.append(dict(arm=arm,id=key,grade=dict(status='graded',score=value),max_token_censored=False))
    graded += [dict(arm='native',id='historical',grade=dict(status='grader_error',score=None)),
        dict(arm='qwen',id='historical',grade=dict(status='graded',score=1))]
    tables=outcome_tables(questions,graded,['native','qwen'])
    arc=next(x for x in tables['pairs'] if x['lane']=='arc_agi_1')
    assert arc['units']==3 and arc['paired_graded_units']==2 and arc['missing']==1
    assert arc['left_only_correct']==arc['right_only_correct']==1 and arc['left_minus_right']==0
    native=next(x for x in tables['arms'] if x['lane']=='arc_agi_1' and x['arm']=='native')
    assert native['accuracy']==.5  # Input-level 2/3 would be the wrong result.
    historical=next(x for x in tables['pairs'] if x['evaluation_slice']=='historical_eval')
    assert historical['grader_error']==1 and historical['paired_graded_units']==0
    assert historical['left_minus_right'] is None
    assert all(x['unit']!='t' for x in tables['units'])
    with pytest.raises(ValueError,match='Duplicate graded'):
        outcome_tables(questions,graded+[graded[0]],['native','qwen'])
    invalid=[{**graded[0],'grade':dict(status='graded',score=.5)}]+graded[1:]
    with pytest.raises(ValueError,match='binary grades'):
        outcome_tables(questions,invalid,['native','qwen'])


def test_missing_arc_input_never_counts_as_failed_complete_task():
    questions=[dict(id=str(i),group_id='one_task',lane='arc_agi_2',split='eval',calibration_split='eval') for i in range(2)]
    graded=[dict(arm='native',id='0',grade=dict(status='graded',score=0)),
        dict(arm='qwen',id='0',grade=dict(status='graded',score=1)),
        dict(arm='qwen',id='1',grade=dict(status='graded',score=1))]
    tables=outcome_tables(questions,graded,['native','qwen'])
    assert tables['arms'][0]['missing']==1 and tables['arms'][0]['incorrect']==0
    assert tables['arms'][1]['correct']==1
    assert tables['pairs'][0]['missing']==1 and tables['pairs'][0]['paired_graded_units']==0


def test_comparison_accepts_model_tokenizers_but_rejects_changed_questions(tmp_path,monkeypatch):
    cohort=tmp_path/'cohort';cohort.mkdir()
    question=dict(id='a',group_id='a',lane='arc_agi_1',family='arc_grid',split='eval',
        calibration_split='eval',prompt='Solve this grid.',token_ids=[11])
    (cohort/'questions.jsonl').write_text(json.dumps(question)+'\n')
    (cohort/'graders.jsonl').write_text(json.dumps(dict(id='a',scorer='arc_exact_grid',gold=[[1]]))+'\n')
    arms=[]
    for index,label in enumerate(('native','qwen')):
        run=tmp_path/label;run.mkdir()
        inputs=run/'questions.jsonl'
        # Identical public problem, but valid model-specific token IDs.
        prepared={**question,'token_ids':[11+index]}
        inputs.write_text(json.dumps(prepared)+'\n')
        generation={k:question[k] for k in ('id','lane','family','split','calibration_split')}
        generation.update(condition=label,prompt_tokens=1,generated_tokens=1,token_ids=prepared['token_ids']+[42],
            scoring_text='[[1]]' if index==0 else '[[0]]',max_token_censored=False)
        (run/'generations.jsonl').write_text(json.dumps(generation)+'\n')
        manifest=dict(status='generation_complete',contract=dict(question_sha256=sha(inputs)),
            files={'generations.jsonl':sha(run/'generations.jsonl')})
        (run/'manifest.json').write_text(json.dumps(manifest))
        arms += ['--arm',label,str(run),label,str(inputs)]
    good=tmp_path/'good'
    monkeypatch.setattr(sys,'argv',['compare','--cohort',str(cohort),'--output-dir',str(good),*arms])
    compare_main()
    tables=json.loads((good/'tables.json').read_text())
    assert tables['pairs'][0]['left_only_correct']==1 and tables['pairs'][0]['left_minus_right']==1
    inputs=tmp_path/'qwen/questions.jsonl'
    prepared=json.loads(inputs.read_text());prepared['prompt']='A different problem with the same ID.'
    inputs.write_text(json.dumps(prepared)+'\n')
    manifest_path=tmp_path/'qwen/manifest.json'
    manifest=json.loads(manifest_path.read_text());manifest['contract']['question_sha256']=sha(inputs)
    manifest_path.write_text(json.dumps(manifest))
    bad=tmp_path/'bad'
    monkeypatch.setattr(sys,'argv',['compare','--cohort',str(cohort),'--output-dir',str(bad),*arms])
    with pytest.raises(ValueError,match='Question identity changed'):
        compare_main()
    assert not bad.exists()
