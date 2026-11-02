"""Full BF16 prompt captures with bounded residency, using existing safetensors.

This fallback records prompt activations, not generated reasoning or task scores.
It does not duplicate the checkpoint into offload files or fit a new VPD/SAE.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
os.environ['HF_MODULES_CACHE']=str(Path('artifacts/nanbeige_reasoning/hf_modules').resolve())
from accelerate import dispatch_model,init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM
from experiments.nanbeige_looped.capture import RepeatedForwardCapture
from experiments.nanbeige_looped.preflight import REVISION
from experiments.nanbeige_looped.run_multipass import digest,write_json

def stream_weights(model,model_dir):
    mapping=json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
    missing=set(dict(model.named_parameters()))-set(mapping)
    if missing:
        raise ValueError(f'Missing checkpoint parameters: {missing}')
    index={name:{'safetensors_file':str((model_dir/file).resolve()),'weight_name':name,'dtype':'bfloat16'}
           for name,file in mapping.items()}
    devices={f'model.layers.{i}':'disk' for i in range(model.config.num_hidden_layers)}
    devices.update({'model.embed_tokens':'disk','model.norm':'disk','lm_head':'disk'})
    return dispatch_model(model,device_map=devices,main_device='cpu',offload_index=index,force_hooks=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=Path('models/Nanbeige4.2-3B-tokenizer'))
    p.add_argument('--questions',type=Path,default=Path('artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--per-lane-split',type=int,default=1)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    provenance=json.loads((args.model_dir/'checkpoint_provenance.json').read_text())
    if provenance['revision']!=REVISION:
        raise ValueError('Wrong checkpoint revision')
    print('Verifying the existing checkpoint and source files',flush=True)
    for name,expected in provenance['files'].items():
        if digest(args.model_dir/name)!=expected:
            raise ValueError(f'Checkpoint integrity failure: {name}')
    config=AutoConfig.from_pretrained(args.model_dir,trust_remote_code=True)
    with init_empty_weights():
        model=AutoModelForCausalLM.from_config(config,trust_remote_code=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model.eval().requires_grad_(False)
    stream_weights(model,args.model_dir)
    rows=[json.loads(x) for x in args.questions.read_text(encoding='utf-8').splitlines()]
    selected=[]
    for lane in sorted({r['lane'] for r in rows}):
        for split in ('train','eval'):
            choices=[r for r in rows if r['lane']==lane and r['calibration_split']==split]
            selected+=sorted(choices,key=lambda r:(len(r['token_ids']),r['id']))[:args.per_lane_split]
    modules={f'model.layers.{i}':model.model.layers[i] for i in range(22)}
    modules['model.norm']=model.model.norm
    records=[]
    for i,row in enumerate(selected):
        tokens=torch.tensor([row['token_ids']],device='cpu')
        positions=sorted(set(range(min(8,len(row['token_ids'])))) |
                         set(range(max(0,len(row['token_ids'])-24),len(row['token_ids']))))
        def extract(output):
            hidden=output[0] if isinstance(output,tuple) else output
            return hidden[0,positions].float().cpu().numpy().copy()
        print(f'Capturing {i+1}/{len(selected)}: {row["lane"]}/{row["calibration_split"]}, {tokens.shape[1]} tokens',flush=True)
        before=time.monotonic()
        with torch.inference_mode(),RepeatedForwardCapture(modules,expected_calls=2,extract=extract) as capture:
            model.model(tokens,use_cache=False)
        archive=out/f'prompt_{i:03d}.npz'
        arrays={f'{site}__pass_{n}':value for (site,n),value in capture.values.items()}
        arrays.update(token_ids=np.asarray(row['token_ids']),token_positions=np.asarray(positions))
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Nonfinite prompt activations')
        np.savez_compressed(archive,**arrays)
        record=dict(id=row['id'],lane=row['lane'],family=row['family'],split=row['calibration_split'],
            prompt_tokens=len(row['token_ids']),seconds=time.monotonic()-before,archive=archive.name,
            sha256=digest(archive),calls=capture.counts)
        records.append(record)
        with (out/'records.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        print(f'Saved {record["seconds"]:.1f}s; all23 sites called twice',flush=True)
    write_json(out/'manifest.json',dict(status='prompt_capture_complete',scope='prompt_only',
        generated_reasoning=False,behavioral_scores=False,new_vpd_fitted=False,new_sae_fitted=False,
        dtype='bfloat16',execution='cpu; one physical block loaded from original safetensors at a time',
        checkpoint=provenance,question_sha256=digest(args.questions),records=records,
        selection='Shortest available prompts within each lane/calibration split; feasibility capture only',
        files={f.name:digest(f) for f in out.iterdir() if f.is_file()}))
    print('Completed prompt captures for all four lanes',flush=True)

if __name__=='__main__':
    main()
