"""Check full answer-context allocation and short serial parity without task grading."""
import json
import subprocess
from pathlib import Path

import numpy as np

from .check_llama_steering import read,sha
from .llama_capture import Runtime
from .llama_cached_steer import CachedSteeringModel
from .run_q4_feature_answers import atomic,native_decoder
from .temperature_guard_burst import TemperatureGuard


def main():
    root=Path('artifacts/nanbeige_reasoning')
    gate=root/'q4_cached_steering_burst_v1'
    out=root/'q4_answer_context_v1'
    verification=read(gate/'verification.json')
    assert verification['gate_passed'] and verification['manifest_sha256']==sha(gate/'manifest.json')
    manifest=read(gate/'manifest.json'); gc=read(gate/'contract.json')
    for name,digest in manifest['files'].items():assert sha(gate/name)==digest
    for name,digest in gc['code_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    reference=read(root/'q4_reasoning_pilot_8192_v1/contract.json')
    model_path=Path('models/Nanbeige4.2-3B-Q4_K_M.gguf'); runtime_path=Path('tools/nanbeige/runtime')
    assert sha(model_path)==reference['model_sha256']
    for name,digest in reference['runtime_files'].items():assert sha(runtime_path/name)==digest
    questions_path=root/'pilot_v1/questions.jsonl'
    assert sha(questions_path)==gc['question_sha256']
    questions=[json.loads(s) for s in questions_path.read_text(encoding='utf-8').splitlines()]
    q=next(q for q in questions if q['id']==gc['question_id'])
    assert q['calibration_split']=='train'
    context=max(len(q['token_ids']) for q in questions if q['calibration_split']=='eval')+8192
    source=root/'q4_steering_preflight_v1'
    assert sha(source/'manifest.json')==gc['source_manifest_sha256']
    sm=read(source/'manifest.json')
    assert sha(source/'selected_direction.npz')==sm['files']['selected_direction.npz']
    with np.load(source/'selected_direction.npz') as z:direction=z['direction'].copy()
    out.mkdir(parents=True,exist_ok=True)
    guard=TemperatureGuard(out/'temperature.jsonl',wall_seconds=7200)
    contract=dict(context=context,gate_manifest_sha256=sha(gate/'manifest.json'),model_sha256=reference['model_sha256'],
        question_id=q['id'],question_sha256=sha(questions_path),source_manifest_sha256=sha(source/'manifest.json'),
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['check_q4_answer_context.py','run_q4_feature_answers.py','llama_capture.py','llama_steer.py','llama_cached_steer.py','temperature_guard_burst.py']},
        execution_guard=guard.policy,scope='Full context allocation; 383-token calibration plus 8 greedy steps. Does not test positions near context capacity.')
    if (out/'contract.json').exists():assert read(out/'contract.json')==contract
    else:atomic(out/'contract.json',contract)
    model=None;rows=[]
    try:
        guard.check(force=True)
        model=CachedSteeringModel(Runtime(runtime_path,require_cuda=True),model_path,context=context,gpu_layers=-1)
        guard.attach(model)
        runtime,decode=native_decoder(model)
        atomic(out/'runtime.json',runtime)
        gpu=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.strip()
        atomic(out/'allocation.json',dict(requested_context=context,gpu_after_allocation=gpu))
        for name,intervention in [('baseline',None),('identity',dict(direction=direction,alpha=0.,passes=[0,1],identity_write=True)),('both_minus',dict(direction=direction,alpha=-.1,passes=[0,1]))]:
            archive=out/f'{name}.npz'; receipt=out/f'{name}.json'
            if receipt.exists():
                row=read(receipt);assert sha(archive)==row['sha256'];rows.append(row);continue
            atomic(out/'progress.json',dict(condition=name,phase='cached',completed=len(rows)))
            model.reset_cache(); logits=model.evaluate(q['token_ids'],intervention=intervention,chunk_size=1)
            cached=[];generated=[]
            for step in range(8):
                cached.append(logits.copy()); generated.append(int(logits.argmax()))
                if step<7:logits=model.evaluate([generated[-1]],intervention=intervention,chunk_size=1)
            cached=np.stack(cached);writes=model.total_mutations
            with np.load(gate/f'{name}.npz') as old:
                np.testing.assert_array_equal(cached,old['cached'])
                np.testing.assert_array_equal(generated,old['generated'])
            full=np.empty((0,model.vocab_size),dtype=np.float32)
            if name!='identity':
                atomic(out/'progress.json',dict(condition=name,phase='full_prefix_last_step',completed=len(rows)))
                model.reset_cache()
                full=model.evaluate(q['token_ids']+generated[:7],intervention=intervention,chunk_size=1)[None,:]
                np.testing.assert_array_equal(full[0],cached[-1])
            np.savez_compressed(archive,cached=cached,full_last=full,generated=generated,token_ids=q['token_ids'])
            row=dict(condition=name,archive=archive.name,sha256=sha(archive),mutation_count=writes,
                generated_text=decode(generated),prompt_text=decode(q['token_ids']),
                stop_token_texts={str(t):decode([t]) for t in runtime['eog_token_ids']})
            atomic(receipt,row);rows.append(row)
            print(json.dumps(dict(condition=name,status='context_parity_passed',temperature=guard.last_temperature)),flush=True)
        atomic(out/'results.json',rows)
    except BaseException as exc:
        atomic(out/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',guard=guard.summary()))
        raise
    finally:
        if model:model.close()
        atomic(out/'cleanup.json',dict(model_closed=model is None or model.model is None,context_closed=model is None or model.context is None,guard=guard.summary()))
    atomic(out/'manifest.json',dict(status='answer_context_preflight_passed',context=context,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','verification.json','progress.json','temperature.jsonl']}))


if __name__=='__main__':main()
