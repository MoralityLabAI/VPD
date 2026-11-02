"""Actual pretrained Q4 callback and logical-layer checks, without generation."""
import json
import time
from pathlib import Path
import numpy as np
from .llama_capture import Runtime,CaptureModel
from .fetch_sources import sha


def main():
    root=Path('artifacts/nanbeige_reasoning')
    out=root/'q4_callback_preflight_v1'
    out.mkdir(exist_ok=False)
    contract=json.loads((root/'q4_reasoning_pilot_v1/contract.json').read_text())
    runtime_path=Path('tools/nanbeige/runtime')
    for name,expected in contract['runtime_files'].items():
        if sha(runtime_path/name)!=expected:raise ValueError('Runtime hash mismatch')
    model_path=Path('models/Nanbeige4.2-3B-Q4_K_M.gguf')
    if sha(model_path)!=contract['model_sha256']:raise ValueError('Model hash mismatch')
    r=Runtime(runtime_path)
    tokens=json.loads((root/'pilot_v1/questions.jsonl').read_text(encoding='utf-8').splitlines()[0])['token_ids'][:32]
    results=[]
    for loops in (1,2):
        started=time.monotonic()
        model=CaptureModel(r,model_path,loops=loops,context=128,gpu_layers=0)
        try:
            _,plain=model.replay(tokens,[],capture=False)
            arrays,hooked=model.replay(tokens,list(range(32)))
            first,last=plain[31],hooked[31]
            assert np.isfinite(first).all() and all(np.isfinite(a).all() for a in arrays.values())
            assert len(arrays)==23*loops
            error=float(np.abs(first-last).max())
            agreement=bool(first.argmax()==last.argmax())
            assert error<1e-3 and agreement,(error,agreement)
            archive=out/f'loops_{loops}.npz'
            np.savez_compressed(archive,**arrays,token_ids=tokens,plain_logits=first,hooked_logits=last)
            result=dict(loops=loops,logical_layers=r.llama.llama_model_n_layer(model.model),
                activation_arrays=len(arrays),max_abs_logit_error=error,argmax_match=agreement,
                callback_names=sorted(model.observed_names),seconds=time.monotonic()-started,
                archive=archive.name,sha256=sha(archive))
            results.append(result)
            print(json.dumps(result),flush=True)
        finally:model.close()
    (out/'manifest.json').write_text(json.dumps(dict(status='pretrained_q4_callback_verified',
        execution='CPU',weights='Q4_K_M',kv_cache='q8_0',results=results,
        binding_sha256=sha(Path(__file__).with_name('llama_capture.py')),
        contract_sha256=sha(root/'q4_reasoning_pilot_v1/contract.json')),indent=2),encoding='utf-8')


if __name__=='__main__':main()
