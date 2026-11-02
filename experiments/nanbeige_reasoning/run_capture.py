"""Generate actual reasoning, then replay exact sequences at both physical passes."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from experiments.nanbeige_looped.capture import RepeatedForwardCapture
from experiments.nanbeige_looped.preflight import REVISION, MODEL_ID
from experiments.nanbeige_looped.run_multipass import digest, write_json
from .positions import capture_positions

def logit_comparison(full,last,*,rtol=0.02,atol=0.25):
    full,last=full.float(),last.float()
    difference=(full-last).abs()
    worst=int(difference.flatten().argmax())
    return dict(max_abs_error=float(difference.max()),
        rms_error=float(difference.square().mean().sqrt()),
        argmax_match=bool(torch.equal(full.argmax(-1),last.argmax(-1))),
        within_tolerance=bool(torch.allclose(full,last,rtol=rtol,atol=atol)),
        outside_tolerance=int((difference>atol+rtol*last.abs()).sum()),
        rtol=rtol,atol=atol,worst_token_id=worst,
        worst_full_logit=float(full.flatten()[worst]),worst_cached_logit=float(last.flatten()[worst]),
        full_argmax=full.argmax(-1).tolist(),cached_argmax=last.argmax(-1).tolist(),
        reference_to_cached_kl=float((full.softmax(-1)*(full.log_softmax(-1)-last.log_softmax(-1))).sum(-1).mean()))


def require_cache_parity(results):
    if not all(r['argmax_match'] and r['within_tolerance'] and r.get('cache_layers_match',True) for r in results):
        raise ValueError('Cached/full-prefix preflight failed; inspect saved numerical diagnostics')


@torch.inference_mode()
def cache_parity(model, tokens, loops, *, return_logits=False, rtol=0.02,atol=0.25):
    model.config.num_loops = loops
    full = model(tokens, use_cache=False).logits[:, -1]
    first = model(tokens[:, :-1], use_cache=True)
    last = model(tokens[:, -1:], past_key_values=first.past_key_values,
                 attention_mask=torch.ones_like(tokens), use_cache=True).logits[:, -1]
    result={'loops':loops,**logit_comparison(full,last,rtol=rtol,atol=atol),
        'cache_layers':len(first.past_key_values),
        'cache_layers_match':len(first.past_key_values)==model.config.num_hidden_layers*loops}
    if return_logits:
        return result,full.float().cpu().numpy(),last.float().cpu().numpy()
    return result

@torch.inference_mode()
def replay(model, token_ids, prompt_length, destination):
    model.config.num_loops = 2
    tokens = torch.tensor([token_ids],device=model.device)
    positions, regions = capture_positions(prompt_length,len(token_ids))
    modules = {f'model.layers.{i}':model.model.layers[i] for i in range(22)}
    modules['model.norm'] = model.model.norm
    def extract(output):
        hidden = output[0] if isinstance(output,tuple) else output
        return hidden[0,positions].float().cpu().numpy().copy()
    with RepeatedForwardCapture(modules, expected_calls=2, extract=extract) as capture:
        model.model(tokens,use_cache=False)
    arrays = {f'{site}__pass_{p}':x for (site,p),x in capture.values.items()}
    arrays.update(token_ids=np.asarray(token_ids),token_positions=np.asarray(positions))
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite activation capture')
    np.savez_compressed(destination,**arrays)
    return {'archive':destination.name,'sha256':digest(destination),'regions':regions,
            'sites':list(modules),'token_positions':positions}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-new-tokens',type=int,default=2048)
    args=parser.parse_args()
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty output directory')
    out.mkdir(parents=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,trust_remote_code=True,use_fast=False)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,trust_remote_code=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda').eval()
    assert model.config.num_hidden_layers==22 and model.config.num_loops==2
    assert not model.config.loop_loss_weights and not getattr(model.config,'enable_double_loop_split',False)
    rows=[json.loads(x) for x in args.questions.read_text(encoding='utf-8').splitlines()]
    parity=[cache_parity(model,torch.tensor([rows[0]['token_ids'][:32]],device='cuda'),n) for n in (1,2)]
    write_json(out/'cache_parity.json',parity)
    require_cache_parity(parity)
    records=[]
    for index,row in enumerate(rows):
        native_tokens=None
        for loops in (2,1):
            model.config.num_loops=loops
            tokens=torch.tensor([row['token_ids']],device='cuda')
            torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.inference_mode():
                sequence=model.generate(tokens,attention_mask=torch.ones_like(tokens),
                    do_sample=False,max_new_tokens=args.max_new_tokens,use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            all_ids=sequence[0].tolist()
            new_ids=all_ids[len(row['token_ids']):]
            # Preserve special reasoning delimiters for the grader and censoring audit.
            text=tokenizer.decode(new_ids,skip_special_tokens=False)
            prefix=tokenizer.decode(row['token_ids'][-8:],skip_special_tokens=False)
            scoring_text=('<think>' if '<think>' in prefix and '</think>' not in prefix else '')+text
            eos=model.generation_config.eos_token_id
            eos_ids=eos if isinstance(eos,list) else [eos]
            record=dict(id=row['id'],lane=row['lane'],family=row['family'],split=row['split'],
                calibration_split=row['calibration_split'],condition=f'loops_{loops}',loops=loops,
                prompt_tokens=len(row['token_ids']),generated_tokens=len(new_ids),elapsed_seconds=elapsed,
                physical_layer_token_executions=22*loops*(len(row['token_ids'])+max(0,len(new_ids)-1)),
                token_ids=all_ids,generated_text=text,scoring_text=scoring_text,
                stopped_on_eos=bool(new_ids and new_ids[-1] in eos_ids),
                max_token_censored=bool(len(new_ids)==args.max_new_tokens and new_ids[-1] not in eos_ids),
                thinking_closed='</think>' in text, seed=42, decoding='greedy')
            records.append(record)
            with (out/'generations.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record,allow_nan=False)+'\n')
            if loops==2:
                native_tokens=all_ids
            print(f'{index+1}/{len(rows)} {row["lane"]} loops={loops} tokens={len(new_ids)} seconds={elapsed:.1f}',flush=True)
        capture=replay(model,native_tokens,len(row['token_ids']),out/f'activation_{index:04d}.npz')
        with (out/'captures.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'id':row['id'],'calibration_split':row['calibration_split'],**capture})+'\n')
    # Full-context native continuations for follow-on VPD/SAE calibration.
    calibration=[]
    for row in rows:
        result=next(r for r in records if r['id']==row['id'] and r['loops']==2)
        calibration.append(dict(id=row['id'],pair_id=row['group_id'],family=row['family'],
            split=row['calibration_split'],token_ids=result['token_ids'],
            prompt_tokens=result['prompt_tokens'],max_token_censored=result['max_token_censored']))
    (out/'calibration_tokens.jsonl').write_text('\n'.join(json.dumps(x) for x in calibration)+'\n',encoding='utf-8')
    write_json(out/'manifest.json',dict(status='capture_complete',model_id=MODEL_ID,revision=REVISION,
        checkpoint_sha256={p.name:digest(p) for p in args.model_dir.iterdir() if p.suffix in ('.json','.py','.safetensors')},
        question_sha256=digest(args.questions),records=len(records),examples=len(rows),
        max_new_tokens=args.max_new_tokens,cache_parity=parity,
        limitations=['Feasibility subset, not full benchmark accuracy.',
            'One-pass condition ablates a two-pass-trained model; it is not a matched trained architecture.',
            'Equal token caps are not equal compute. Layer-token execution counts are a compute proxy, not FLOPs.',
            'Activations replay native two-pass sequences; unfinished reasoning is labeled censored.',
            'No variable routing is used or inferred.'],
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':
    main()
