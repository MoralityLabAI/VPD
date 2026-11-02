"""Resumable paired benchmark generation with exact task/prefix provenance.

Gold answers are never loaded. All native sequences can later be replayed for
activation capture; inference remains independent from SAE/VPD fitting.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.nanbeige_looped.preflight import MODEL_ID,REVISION
from experiments.nanbeige_looped.run_multipass import digest,write_json
from .batching import last_logits_only,padded_batch,batch_plan
from .run_capture import cache_parity,require_cache_parity

def recovered_rows(path):
    """Only a final unterminated write may be discarded after interruption."""
    if not path.exists():
        return []
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        boundary=raw.rfind(b'\n')+1
        path.with_suffix('.interrupted-tail').write_bytes(raw[boundary:])
        raw=raw[:boundary]
        path.write_bytes(raw)
    return [json.loads(line) for line in raw.splitlines()]

def trim_generated(ids,eos_ids):
    for i,token in enumerate(ids):
        if token in eos_ids:
            return ids[:i+1],True
    return ids,False


def pending_frozen_batches(plan,done,loops):
    """Re-execute a partially saved batch at its original shape and membership."""
    for index,batch in enumerate(plan):
        if not all((row['id'],loops) in done for row in batch):
            yield index,batch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True,type=Path)
    p.add_argument('--questions',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--loops',default='2,1')
    p.add_argument('--max-new-tokens',type=int,default=8192)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-prefill-tokens',type=int,default=32768)
    p.add_argument('--max-total-tokens',type=int,default=65536)
    p.add_argument('--max-wall-seconds',type=int,default=21600)
    args=p.parse_args()
    if min(args.max_new_tokens,args.batch_size,args.max_prefill_tokens,args.max_total_tokens,args.max_wall_seconds)<1:
        p.error('Positive resource bounds required')
    loops=[int(x) for x in args.loops.split(',')]
    if not loops or len(set(loops))!=len(loops) or set(loops)-{1,2}:
        p.error('Use one or both of the registered loop conditions 1,2')
    rows=[json.loads(x) for x in args.questions.read_text(encoding='utf-8').splitlines()]
    by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows):
        raise ValueError('Duplicate question IDs')
    plan=list(batch_plan(rows,args.batch_size,args.max_prefill_tokens,args.max_new_tokens,args.max_total_tokens))
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    contract=dict(model_id=MODEL_ID,revision=REVISION,question_sha256=digest(args.questions),
        loops=loops,max_new_tokens=args.max_new_tokens,decoding='greedy',seed=42,
        batch_size_limit=args.batch_size,max_prefill_tokens=args.max_prefill_tokens,
        max_total_tokens=args.max_total_tokens,frozen_batches=[[r['id'] for r in batch] for batch in plan],
        implementation_sha256={name:digest(Path(__file__).with_name(name))
            for name in ('run_benchmark.py','batching.py','run_capture.py')},
        checkpoint_sha256={f.name:digest(f) for f in args.model_dir.iterdir()
                           if f.suffix in ('.json','.py','.safetensors')})
    state=out/'contract.json'
    if state.exists() and json.loads(state.read_text())!=contract:
        raise ValueError('Resume contract differs from recorded model/questions/decoding')
    write_json(state,contract)
    path=out/'generations.jsonl'
    previous=recovered_rows(path)
    expected_order=[(r['id'],n) for n in loops for batch in plan for r in batch]
    if [(r['id'],r['loops']) for r in previous]!=expected_order[:len(previous)]:
        raise ValueError('Resume records differ from the frozen ordered batch schedule')
    previous_by_key={(r['id'],r['loops']):r for r in previous}
    done=set()
    for r in previous:
        key=(r['id'],r['loops'])
        if key in done or r['id'] not in by_id or r['loops'] not in loops:
            raise ValueError('Corrupt or incompatible resumed generation set')
        if r['token_ids'][:r['prompt_tokens']]!=by_id[r['id']]['token_ids']:
            raise ValueError('Resumed generation has a different prompt')
        done.add(key)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    environment=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    environment_path=out/'environment.json'
    if environment_path.exists() and json.loads(environment_path.read_text())!=environment:
        raise ValueError('Resume execution environment differs')
    write_json(environment_path,environment)
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,trust_remote_code=True,use_fast=False)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,trust_remote_code=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda').eval()
    if model.config.num_hidden_layers!=22 or model.config.loop_loss_weights:
        raise ValueError('Unexpected recurrent model configuration')
    sample=torch.tensor([rows[0]['token_ids'][:32]],device='cuda')
    parity,head_controls=[],[]
    control_arrays={'token_ids':sample.cpu().numpy()}
    for n in loops:
        diagnostic,full,cached=cache_parity(model,sample,n,return_logits=True)
        parity.append(diagnostic)
        control_arrays[f'loops_{n}__full']=full
        control_arrays[f'loops_{n}__cached']=cached
        with torch.inference_mode():
            original=model(sample,use_cache=False).logits[:,-1]
            with last_logits_only(model):
                optimized=model(sample,use_cache=False).logits[:,-1]
        head_controls.append(dict(loops=n,max_abs=float((original-optimized).abs().max()),
            argmax_match=bool(torch.equal(original.argmax(-1),optimized.argmax(-1))),
            within_tolerance=bool(torch.allclose(original,optimized,rtol=0.02,atol=0.25))))
        control_arrays[f'loops_{n}__head_original']=original.float().cpu().numpy()
        control_arrays[f'loops_{n}__head_last_only']=optimized.float().cpu().numpy()
    preflight_index=len(list(out.glob('preflight_*.json')))
    write_json(out/f'preflight_{preflight_index:03d}.json',dict(cache_parity=parity,head_controls=head_controls))
    np.savez_compressed(out/f'preflight_{preflight_index:03d}.npz',**control_arrays)
    if not all(r['argmax_match'] and r['within_tolerance'] for r in head_controls):
        raise ValueError('LM-head optimization failed the unchanged preflight tolerance or greedy-choice gate')
    require_cache_parity(parity)
    eos=model.generation_config.eos_token_id
    eos_ids=eos if isinstance(eos,list) else [eos]
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    start=time.monotonic()
    stopped=False
    attempts_path=out/'batch_attempts.jsonl'
    attempts=recovered_rows(attempts_path)
    attempt_id=max((row['attempt_id'] for row in attempts),default=-1)+1
    for n in loops:
        model.config.num_loops=n
        for batch_index,batch in pending_frozen_batches(plan,done,n):
            if time.monotonic()-start>=args.max_wall_seconds:
                stopped=True
                break
            ids,mask=padded_batch(batch,pad,'cuda')
            torch.cuda.synchronize()
            before=time.perf_counter()
            attempt=dict(attempt_id=attempt_id,loops=n,batch_index=batch_index,
                ids=[r['id'] for r in batch],already_saved=[r['id'] for r in batch if (r['id'],n) in done],
                started_unix=time.time(),padded_prompt_length=int(ids.shape[1]))
            with attempts_path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event='started',**attempt))+'\n');stream.flush();os.fsync(stream.fileno())
            with torch.inference_mode(),last_logits_only(model):
                generated=model.generate(ids,attention_mask=mask,max_new_tokens=args.max_new_tokens,
                    do_sample=False,pad_token_id=pad,use_cache=True)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-before
            with attempts_path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event='generated',seconds=elapsed,
                    generated_batch_columns=int(generated.shape[1]-ids.shape[1]),**attempt))+'\n')
                stream.flush();os.fsync(stream.fileno())
            result=[]
            for i,row in enumerate(batch):
                continuation,eos_stop=trim_generated(generated[i,ids.shape[1]:].tolist(),eos_ids)
                text=tokenizer.decode(continuation,skip_special_tokens=False)
                prefix=tokenizer.decode(row['token_ids'][-8:],skip_special_tokens=False)
                scoring=('<think>' if '<think>' in prefix and '</think>' not in prefix else '')+text
                record=dict(id=row['id'],lane=row['lane'],family=row['family'],split=row['split'],
                    calibration_split=row['calibration_split'],loops=n,condition=f'loops_{n}',
                    token_ids=row['token_ids']+continuation,prompt_tokens=len(row['token_ids']),
                    generated_tokens=len(continuation),generated_text=text,scoring_text=scoring,
                    stopped_on_eos=eos_stop,max_token_censored=not eos_stop and len(continuation)==args.max_new_tokens,
                    thinking_closed='</think>' in text,batch_seconds=elapsed,batch_size=len(batch),
                    padded_prompt_length=ids.shape[1],decoding='greedy',seed=42,
                    batch_index=batch_index,attempt_id=attempt_id)
                key=(row['id'],n)
                if key in done:
                    if record['token_ids']!=previous_by_key[key]['token_ids']:
                        write_json(out/f'resume_mismatch_{attempt_id:05d}.json',dict(id=row['id'],loops=n,batch_index=batch_index))
                        raise ValueError('Replayed partial batch differs from saved greedy trajectory')
                else:
                    result.append(record)
            with path.open('a',encoding='utf-8') as stream:
                for r in result:
                    stream.write(json.dumps(r,allow_nan=False)+'\n')
                    done.add((r['id'],n))
                stream.flush()
                os.fsync(stream.fileno())
            write_json(out/'progress.json',dict(completed=len(done),required=len(rows)*len(loops),
                elapsed_seconds=time.monotonic()-start,last_batch_seconds=elapsed))
            print(f'Completed {len(done)}/{len(rows)*len(loops)}; loops={n}; batch={len(batch)}; seconds={elapsed:.1f}',flush=True)
            attempt_id+=1
            del generated,ids,mask
        if stopped:
            break
    write_json(out/'manifest.json',dict(status='paused_time_budget' if stopped else 'generation_complete',
        contract=contract,completed=len(done),required=len(rows)*len(loops),
        checkpoint_sha256={f.name:digest(f) for f in args.model_dir.iterdir() if f.suffix in ('.json','.py','.safetensors')},
        files={f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'},
        limitations=['One-pass is an ablation of a model trained with two passes.',
            'Batch elapsed times cannot be interpreted as independent per-example latency.',
            'Use batch_attempts.jsonl for total generation cost, including replayed and interrupted batches.',
            'Incomplete/censored outcomes must remain visible in reporting.']))

if __name__=='__main__':
    main()
