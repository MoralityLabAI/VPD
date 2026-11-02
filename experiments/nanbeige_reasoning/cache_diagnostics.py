"""Compare BF16 and FP32 cache equivalence without changing inference gates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from .run_capture import cache_parity,logit_comparison
from .fetch_sources import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--questions',type=Path,default=Path('artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl'))
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    rows=list(map(json.loads,args.questions.read_text(encoding='utf-8').splitlines()))
    tokens=torch.tensor([rows[0]['token_ids'][:32]],device='cuda')
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,trust_remote_code=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda').eval()
    results,arrays=[],{}
    for precision in ('bf16','fp32'):
        if precision=='fp32':
            model.float()
            torch.cuda.empty_cache()
        for loops in (1,2):
            tolerance=dict(rtol=0.02,atol=0.25) if precision=='bf16' else dict(rtol=1e-4,atol=1e-4)
            result,full,cached=cache_parity(model,tokens,loops,return_logits=True,**tolerance)
            results.append(dict(precision=precision,**result))
            arrays[f'{precision}_loops_{loops}_full']=full
            arrays[f'{precision}_loops_{loops}_cached']=cached
            (args.output_dir/'progress.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            print(json.dumps(results[-1]),flush=True)
    cross=[]
    for loops in (1,2):
        for mode in ('full','cached'):
            b=torch.from_numpy(arrays[f'bf16_loops_{loops}_{mode}'])
            f=torch.from_numpy(arrays[f'fp32_loops_{loops}_{mode}'])
            cross.append(dict(loops=loops,mode=mode,**logit_comparison(b,f)))
    np.savez_compressed(args.output_dir/'logits.npz',**arrays,token_ids=tokens.cpu().numpy())
    result=dict(status='cache_precision_diagnostics_complete',model_id=model.config._name_or_path,
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),tf32=False,
        source_id=rows[0]['id'],prompt_tokens=32,conditions=results,cross_precision=cross,
        original_generation_tolerances_unchanged=True,
        limitation='One prompt prefix diagnoses implementation and precision only; it is not benchmark accuracy or a justification to change tolerances without reviewing the results.',
        checkpoint_sha256={f.name:sha(f) for f in args.model_dir.iterdir() if f.suffix in ('.json','.py','.safetensors')},
        implementation_sha256={str(Path(__file__).name):sha(Path(__file__)),
            'run_capture.py':sha(Path(__file__).with_name('run_capture.py'))},
        question_sha256=sha(args.questions),
        files={f.name:sha(f) for f in args.output_dir.iterdir() if f.is_file()})
    (args.output_dir/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
