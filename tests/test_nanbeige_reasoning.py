import json
import torch
from experiments.nanbeige_reasoning.build_dataset import build_arc
from experiments.nanbeige_reasoning.run_capture import cache_parity, capture_positions,logit_comparison,require_cache_parity
from experiments.nanbeige_looped.run_multipass import architecture_canary
from pathlib import Path
from experiments.nanbeige_reasoning.scoring import arc_grid, score,grade_run,paired_outcomes
import pytest
from experiments.nanbeige_reasoning.batching import last_logits_only,padded_batch,batch_plan
from experiments.nanbeige_reasoning.run_benchmark import recovered_rows,trim_generated
from experiments.nanbeige_reasoning.safe_expression import safe_eval
from experiments.nanbeige_reasoning.generation_health import token_metrics


def test_generation_health_describes_repetition_without_scoring_answers():
    result=token_metrics([1,1,1,2,2,1])
    assert result['tokens']==6 and result['unique_tokens']==2
    assert result['longest_identical_token_run']==3
    assert result['adjacent_repeat_fraction']==pytest.approx(3/5)
    assert result['unigram_entropy_bits']==pytest.approx(0.9182958340544896)
    assert token_metrics([1])['adjacent_repeat_fraction']==0
    with pytest.raises(ValueError,match='empty'):
        token_metrics([])


def test_verifier_expression_guard_preserves_literals_and_arithmetic(tmp_path):
    assert safe_eval("(('Alice', 'apple'), ('Bob', 'book'))")==(('Alice','apple'),('Bob','book'))
    assert safe_eval('[[1, 2], [3, 4]]')==[[1,2],[3,4]]
    assert safe_eval('(8-7)*5*4 == 20') is True
    assert safe_eval('1/2 + 3/2')==2
    target=tmp_path/'must_not_exist'
    for expression in [f"__import__('pathlib').Path({str(target)!r}).touch()",
                       "[x for x in [1]]", "'a'*1000000000", '2**1000000000',
                       '(1).__class__', '1e999']:
        with pytest.raises(ValueError):
            safe_eval(expression)
    assert not target.exists()

def test_arc_cross_release_eval_alias_quarantined_and_answers_separate(tmp_path):
    task={'train':[{'input':[[1]],'output':[[2]]}],
          'test':[{'input':[[3]],'output':[[9]]},{'input':[[4]],'output':[[8]]}]}
    for lane,split,name in [('arc_agi_1','evaluation','original'),('arc_agi_2','training','renamed')]:
        path=tmp_path/lane/'data'/split/(name+'.json')
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(task))
    questions,graders,excluded=build_arc({x:tmp_path/x for x in ('arc_agi_1','arc_agi_2')})
    assert [r['split'] for r in questions]==['eval','eval','quarantine','quarantine']
    assert len({r['group_id'] for r in questions})==1
    assert all('gold' not in r and '[[9]]' not in r['prompt'] for r in questions)
    assert graders[0]['gold']==[[9]] and len(excluded)==1

def test_actual_architecture_cache_indices_for_one_and_two_passes():
    torch.set_num_threads(2)
    model,_=architecture_canary(Path('models/Nanbeige4.2-3B-tokenizer'))
    for loops in (1,2):
        result=cache_parity(model,torch.tensor([[1,4,5,6,7,8]]),loops)
        assert result['argmax_match']
        require_cache_parity([result])
        assert result['max_abs_error']<1e-5


def test_cache_gate_retains_numerical_and_greedy_requirements():
    low_probability_drift=logit_comparison(torch.tensor([[10.,0.]]),torch.tensor([[10.,.3]]))
    assert low_probability_drift['argmax_match'] and not low_probability_drift['within_tolerance']
    assert low_probability_drift['outside_tolerance']==1
    with pytest.raises(ValueError,match='preflight failed'):
        require_cache_parity([low_probability_drift])
    changed_choice=logit_comparison(torch.tensor([[1.,1.01]]),torch.tensor([[1.01,1.]]))
    assert changed_choice['within_tolerance'] and not changed_choice['argmax_match']
    with pytest.raises(ValueError,match='preflight failed'):
        require_cache_parity([changed_choice])

def test_capture_positions_preserve_prompt_and_generation_regions():
    positions,regions=capture_positions(100,200)
    assert positions==sorted(set(positions))
    assert regions['prompt']==(92,100)
    assert regions['early']==(117,125)
    assert regions['middle']==(142,150)
    assert regions['late']==(192,200)
    assert min(positions)>=0 and max(positions)<200

def test_grid_grader_rejects_partial_reasoning_and_python_boolean_cells():
    assert arc_grid('<think>[[1]]') is None
    assert arc_grid('<answer>[[true]]</answer>') is None
    assert arc_grid('<answer>[[1],[1,2]]</answer>') is None
    assert arc_grid('<think>guess</think><answer>[[1]]</answer>')==[[1]]
    assert score({'scorer':'arc_exact_grid','gold':[[2]]},'[[1]]')['score']==0

def test_math_equivalence_grader_and_unfinished_thinking():
    grader={'scorer':'intellect3_math','gold':'\\frac{1}{2}'}
    assert score(grader,'\\boxed{0.5}')=={'status':'graded','score':1.0}
    assert score(grader,'<think>\\boxed{0.5}')=={'status':'graded','score':0.0}
    assert score(grader,'\\boxed{0.6}')=={'status':'graded','score':0.0}

def test_last_logit_optimization_and_left_padded_generation_match_serial():
    torch.set_num_threads(2)
    model,_=architecture_canary(Path('models/Nanbeige4.2-3B-tokenizer'))
    rows=[{'token_ids':[1,4,5,6,7,8]},{'token_ids':[1,4,5]}]
    for loops in (1,2):
        model.config.num_loops=loops
        ids,mask=padded_batch(rows,0,'cpu')
        with torch.inference_mode():
            original=model(ids,attention_mask=mask,use_cache=False).logits[:,-1]
            with last_logits_only(model):
                optimized=model(ids,attention_mask=mask,use_cache=False).logits[:,-1]
                batched=model.generate(ids,attention_mask=mask,max_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=None)
                for i,row in enumerate(rows):
                    tokens=torch.tensor([row['token_ids']])
                    serial=model.generate(tokens,attention_mask=torch.ones_like(tokens),max_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=None)
                    assert torch.equal(batched[i,-3:],serial[0,-3:])
            torch.testing.assert_close(original,optimized,rtol=0,atol=0)

def test_interrupted_resume_discards_only_unterminated_tail(tmp_path):
    path=tmp_path/'generations.jsonl'
    path.write_bytes(b'{"id":"complete"}\n{"id":"part')
    assert recovered_rows(path)==[{'id':'complete'}]
    assert path.read_bytes()==b'{"id":"complete"}\n'
    assert trim_generated([4,2,0,0],[2])==([4,2],True)

def test_batch_planning_accounts_for_padding_and_continuation():
    rows=[{'id':str(i),'token_ids':[1]*n} for i,n in enumerate([100,101,200,1000])]
    batches=list(batch_plan(rows,8,1024,256,1600))
    assert sum(map(len,batches))==4
    assert all(len(b)*(max(len(r['token_ids']) for r in b)+256)<=1600 for b in batches)

def test_task_level_arc_metric_and_duplicate_rejection(tmp_path):
    questions=[{'id':str(i),'lane':'arc_agi_1','group_id':'a' if i<2 else 'b'} for i in range(3)]
    graders=[{'id':str(i),'scorer':'arc_exact_grid','gold':[[1]]} for i in range(3)]
    rows=[dict(id=str(i),lane='arc_agi_1',loops=2,split='eval',calibration_split='eval',
        scoring_text='[[0]]' if i==1 else '[[1]]',max_token_censored=False) for i in range(3)]
    path=tmp_path/'generations.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    result=grade_run(tmp_path,questions,graders)[0]
    assert result['correct']==2 and result['graded']==3
    assert result['tasks']==2 and result['all_test_inputs_correct']==1
    native=[{**row,'grade':score(graders[i],row['scoring_text'])} for i,row in enumerate(rows)]
    one_pass=[{**row,'loops':1,'grade':{'status':'graded','score':float(i<2)}} for i,row in enumerate(rows)]
    paired=paired_outcomes(native+one_pass,questions)[0]
    assert paired['units']==2 and paired['paired_graded']==2
    assert paired['native_only_correct']==paired['one_pass_only_correct']==1
    assert paired['native_minus_one_pass']==0
    one_pass[0]['grade']={'status':'grader_error','score':None}
    unknown=paired_outcomes(native+one_pass,questions)[0]
    assert unknown['ungradable_pairs']==1 and unknown['paired_graded']==1
    missing=paired_outcomes(native,questions)[0]
    assert missing['missing_conditions']==2 and missing['native_accuracy'] is None
    unseen=[{**q,'calibration_split':'eval','split':'eval'} for q in questions]
    unseen.append(dict(id='unseen',lane='arc_agi_1',group_id='c',calibration_split='eval',split='eval'))
    no_outputs=paired_outcomes([],unseen)[0]
    assert no_outputs['units']==no_outputs['missing_conditions']==3
    path.write_text('\n'.join(json.dumps(r) for r in rows+[rows[0]]))
    with pytest.raises(ValueError,match='Duplicate generation'):
        grade_run(tmp_path,questions,graders)
