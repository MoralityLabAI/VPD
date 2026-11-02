"""Generation-only memory optimization and bounded left-padded batches."""
from contextlib import contextmanager
import torch

@contextmanager
def last_logits_only(model):
    """Generation consumes only final-position logits; preserve all decoder states.

    Do not use for losses, VPD optimization, or any full-logit evaluation.
    """
    if getattr(model.config,'pretraining_tp',1)!=1:
        raise ValueError('LM-head sharding bypasses this hook')
    def select_last(_module,inputs):
        return (inputs[0][:,-1:,:],*inputs[1:])
    handle=model.lm_head.register_forward_pre_hook(select_last)
    try:
        yield
    finally:
        handle.remove()

def padded_batch(rows,pad_id,device):
    width=max(len(r['token_ids']) for r in rows)
    ids=torch.full((len(rows),width),pad_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):
        length=len(row['token_ids'])
        ids[i,-length:]=torch.tensor(row['token_ids'],device=device)
        mask[i,-length:]=1
    return ids,mask

def batch_plan(rows,max_batch,max_prefill_tokens,max_new_tokens,max_total_tokens):
    """Account for both padding and maximum cached continuation length."""
    ordered=sorted(rows,key=lambda r:(len(r['token_ids']),r['id']))
    batch=[]
    for row in ordered:
        width=len(row['token_ids'])
        if width>max_prefill_tokens or width+max_new_tokens>max_total_tokens:
            raise ValueError(f'Single example exceeds explicit resource budget: {row["id"]}')
        size=len(batch)+1
        if batch and (size>max_batch or size*width>max_prefill_tokens or size*(width+max_new_tokens)>max_total_tokens):
            yield batch
            batch=[]
        batch.append(row)
    if batch:
        yield batch
