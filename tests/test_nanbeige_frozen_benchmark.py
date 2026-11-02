"""Resume controls must preserve batch composition, including partial writes."""
from experiments.nanbeige_reasoning.batching import batch_plan
from experiments.nanbeige_reasoning.run_benchmark import pending_frozen_batches


def test_partial_saved_batch_retains_original_membership_and_padding():
    rows=[{'id':str(i),'token_ids':[1]*length} for i,length in enumerate([10,20,21,30,50])]
    plan=list(batch_plan(rows,2,100,40,160))
    assert [[r['id'] for r in batch] for batch in plan]==[['0','1'],['2','3'],['4']]
    pending=list(pending_frozen_batches(plan,{('0',2)},2))
    assert [[r['id'] for r in batch] for _,batch in pending]==[['0','1'],['2','3'],['4']]
    assert max(len(r['token_ids']) for r in pending[0][1])==20


def test_completed_batches_skip_without_regrouping_and_other_pass_remains():
    rows=[{'id':str(i),'token_ids':[1]*length} for i,length in enumerate([10,20,21,30,50])]
    plan=list(batch_plan(rows,2,100,40,160))
    done={('0',2),('1',2),('2',2)}
    assert [(i,[r['id'] for r in batch]) for i,batch in pending_frozen_batches(plan,done,2)]==[(1,['2','3']),(2,['4'])]
    assert len(list(pending_frozen_batches(plan,done,1)))==3
