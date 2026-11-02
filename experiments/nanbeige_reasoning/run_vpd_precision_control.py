"""Replay fixed held-out trajectories under precision, dose and zero controls."""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.nanbeige_looped.measure import capture_forward
from experiments.nanbeige_looped.recurrent import RecurrentTarget
from experiments.nanbeige_looped.run_multipass import digest, write_json


class PrecisionTarget(RecurrentTarget):
    def __init__(self, model, counts, *, bf16):
        super().__init__(model, counts, passes=2)
        self.bf16 = bf16

    def autocast(self):
        return torch.autocast(next(self.model.parameters()).device.type,dtype=torch.bfloat16) if self.bf16 else nullcontext()


def condition_plan():
    plan = [dict(name='baseline_before',path=None,alpha=0.,passes=[])]
    for layer in (11,21):
        path = f'model.layers.{layer}.mlp.down_proj'
        plan.append(dict(name=f'layer_{layer+1}_zero_before',path=path,alpha=0.,passes=[0,1]))
        for scope,passes in [('first',[0]),('second',[1]),('both',[0,1])]:
            for dose in (.05,.01,.002):
                for sign in (1,-1):
                    plan.append(dict(name=f'layer_{layer+1}_{scope}_{dose}_{"plus" if sign>0 else "minus"}',
                        path=path,alpha=sign*dose,passes=passes))
        plan.append(dict(name=f'layer_{layer+1}_zero_after',path=path,alpha=0.,passes=[0,1]))
    plan.append(dict(name='baseline_after',path=None,alpha=0.,passes=[]))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-wall-seconds',type=int,default=2700)
    args = p.parse_args()
    read = lambda p:json.loads(p.read_text(encoding='utf-8'))
    contract = read(args.inputs/'contract.json')
    for name,expected in contract['input_files'].items():
        if Path(name).name != name or digest(args.inputs/name) != expected:
            raise ValueError('Changed frozen input')
    for name,expected in contract['checkpoint_files'].items():
        if Path(name).name != name or digest(args.model_dir/name) != expected:
            raise ValueError('Checkpoint mismatch')
    examples = read(args.inputs/'examples.json')
    plan = condition_plan()
    if (len(examples)!=10 or any(r['split']!='eval' for r in examples)
            or len({r['id'] for r in examples})!=10 or plan!=contract['conditions']
            or any(not 1<=len(r['token_ids'])<=4096 for r in examples)):
        raise ValueError('Experiment cohort or condition plan differs')
    if not torch.cuda.is_available():
        raise ValueError('Pretrained precision control requires CUDA')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'contract.json',contract)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    write_json(args.output/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),tf32=False,
        source_sha256={p.name:digest(p) for p in [Path(__file__),
            Path(__file__).parents[1]/'nanbeige_looped/recurrent.py',
            Path(__file__).parents[1]/'nanbeige_looped/measure.py',
            Path(__file__).parents[1]/'nanbeige_looped/vpd_math.py']}))
    from transformers import AutoModelForCausalLM
    sites = ['model.layers.11','model.layers.21','model.norm']
    counts = {s+'.mlp.down_proj':64 for s in sites[:2]}
    start = time.monotonic()
    total = 0
    for precision,dtype in [('bf16',torch.bfloat16),('fp32_upcast',torch.float32)]:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir,trust_remote_code=True,
            torch_dtype=dtype,attn_implementation='sdpa',device_map='cuda').eval()
        if (model.config.num_hidden_layers,model.config.num_loops,model.config.hidden_size)!=(22,2,3072):
            raise ValueError('Unexpected architecture')
        model.config.use_cache=False
        target = PrecisionTarget(model,counts,bf16=precision=='bf16')
        with np.load(args.inputs/'vpd_factors.npz',allow_pickle=False) as factors,torch.no_grad():
            for path,wrapper in target.physical.items():
                wrapper.core.V.copy_(torch.from_numpy(factors[path+'__V']).to('cuda'))
                wrapper.core.U.copy_(torch.from_numpy(factors[path+'__U']).to('cuda'))
        target.eval().requires_grad_(False)
        for example in examples:
            tokens = torch.tensor([example['token_ids']],device='cuda')
            for condition in plan:
                if time.monotonic()-start>args.max_wall_seconds:
                    write_json(args.output/'run_status.json',dict(status='paused_time_bound',records=total,expected=840))
                    raise SystemExit(75)
                context = (target.intervene(condition['path'],0,condition['alpha'],condition['passes'])
                           if condition['path'] is not None else nullcontext())
                before = time.monotonic()
                with context:
                    hidden,logits = capture_forward(target,tokens,sites,rows=8)
                arrays = {f'{site}__pass_{invocation}':x for (site,invocation),x in hidden.items()}
                arrays.update(last_logits=logits,token_positions=np.arange(len(example['token_ids'])-8,len(example['token_ids'])),
                    token_ids=np.asarray(example['token_ids'][-8:]))
                if any(not np.isfinite(x).all() for x in arrays.values()):
                    raise ValueError('Nonfinite saved measurement')
                archive = args.output/f'capture_{total:04d}.npz'
                np.savez_compressed(archive,**arrays)
                record = dict(id=example['id'],pair_id=example['pair_id'],precision=precision,
                    condition=condition,archive=archive.name,sha256=digest(archive),
                    full_context_tokens=len(example['token_ids']),seconds=time.monotonic()-before)
                with (args.output/'records.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record)+'\n');stream.flush()
                total+=1
                write_json(args.output/'run_status.json',dict(status='capturing',records=total,expected=840,
                    precision=precision,last_id=example['id'],condition=condition['name'],elapsed_seconds=time.monotonic()-start))
                if total%10==0 or total==1:
                    print(json.dumps(dict(records=total,precision=precision,elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
        del target,model,tokens
        gc.collect();torch.cuda.empty_cache()
    if total != contract['expected_records']:
        raise ValueError('Incomplete condition grid')
    write_json(args.output/'manifest.json',dict(status='vpd_precision_control_complete',records=total,
        input_contract_sha256=digest(args.inputs/'contract.json'),elapsed_seconds=time.monotonic()-start,
        limitations=contract['limitations'],
        files={p.name:digest(p) for p in args.output.iterdir() if p.is_file() and p.name!='run_status.json'}))
    write_json(args.output/'run_status.json',dict(status='vpd_precision_control_complete',records=total))


if __name__=='__main__':
    main()
