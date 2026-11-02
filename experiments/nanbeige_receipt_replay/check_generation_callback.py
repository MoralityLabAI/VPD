"""CPU-only full-logit parity check for the disabled-callback fast path."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from experiments.nanbeige_reasoning.llama_capture import Runtime
from experiments.nanbeige_reasoning.llama_cached_steer import CachedSteeringModel
from .native_generation import GenerationCachedModel
from .io import ROOT,digest,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    request=json.loads(a.request.read_text(encoding='utf-8'));prefix=request['input_ids'][:24]
    if len(prefix)!=24:raise ValueError('Need at least 24 input tokens')
    reference=None;rows=[]
    for label,cls in (('original',CachedSteeringModel),('disabled_callback_fast_path',GenerationCachedModel)):
        start=time.monotonic();model=cls(Runtime(ROOT/'tools/nanbeige/runtime',require_cuda=False),
            ROOT/'models/Nanbeige4.2-3B-Q4_K_M.gguf',context=128,gpu_layers=0,threads=1)
        try:
            model.reset_cache();logits=[];generated=[];decode_start=time.monotonic()
            for offset in range(0,24,8):logits.append(model.evaluate(prefix[offset:offset+8],chunk_size=8))
            for _ in range(8):
                token=int(np.argmax(logits[-1]));generated.append(token)
                logits.append(model.evaluate([token],chunk_size=1))
            elapsed=time.monotonic()-decode_start;array=np.stack(logits)
            if reference is not None:np.testing.assert_array_equal(array,reference)
            reference=array
            np.savez_compressed(a.output/f'{label}.npz',logits=array)
            rows.append(dict(mode=label,evaluated_positions=32,full_logit_rows=len(logits),generated_token_ids=generated,
                             decode_seconds=elapsed,total_seconds=time.monotonic()-start,device='cpu',threads=1))
            print(json.dumps(rows[-1]),flush=True)
        finally:model.close()
    save(a.output/'verification.json',dict(status='bitwise_full_logit_parity_passed',rows=rows,
        model_sha256=digest(ROOT/'models/Nanbeige4.2-3B-Q4_K_M.gguf'),source_request_sha256=digest(a.request),
        code_sha256={n:digest(Path(__file__).with_name(n)) for n in ('native_generation.py','check_generation_callback.py')},
        scope='CPU-only deterministic 24-token prefix and eight greedy continuations, 11 full vocabulary logit vectors. No GPU speed claim; capture and steering paths delegate unchanged.',
        files={p.name:digest(p) for p in a.output.glob('*.npz')}))


if __name__=='__main__':main()
