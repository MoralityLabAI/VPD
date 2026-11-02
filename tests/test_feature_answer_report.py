import copy
import pytest
from scripts.nanbeige.report_q4_feature_answers import summarize


def fixture():
    questions=[dict(id=x,group_id=g,lane=lane,split=split,calibration_split='eval') for x,g,lane,split in
        [('a0','a','arc_agi_1','eval'),('a1','a','arc_agi_1','eval'),
         ('m','m','intellect3_math','eval'),('h','h','intellect3_logic','historical_eval')]]
    rows=[]
    for arm,scores in [('baseline',[1,0,1,0]),('second_plus',[1,1,0,0])]:
        for q,s in zip(questions,scores):
            grade=dict(status='graded',score=s)
            rows.append(dict(id=q['id'],arm=arm,grade=grade,native_render_grade=copy.deepcopy(grade),
                max_token_censored=q['id']=='a1',generated_tokens=8192 if q['id']=='a1' else 20))
    return questions,rows,['baseline','second_plus']


def test_complete_task_pairing_and_censor_denominators():
    q,rows,arms=fixture();tables,summary=summarize(q,rows,arms)
    assert summary[0]['correct']==1 and summary[0]['units']==3
    assert summary[1]['fixes']==summary[1]['regressions']==1
    assert summary[1]['capped_inputs']==1 and summary[1]['input_records']==4
    assert any(r['evaluation_slice']=='historical_eval' for r in tables['arms'])
    with pytest.raises(ValueError):summarize(q,rows[:-1],arms)
    with pytest.raises(ValueError):summarize(q,rows+[rows[0]],arms)


def test_grader_errors_and_rendering_changes_are_not_successes():
    q,rows,arms=fixture()
    rows[-1]['grade']=dict(status='grader_error',score=None)
    _,summary=summarize(q,rows,arms)
    assert summary[1]['grader_errors']==1 and summary[1]['paired_units']==2
    assert summary[1]['native_score_differences']==1
