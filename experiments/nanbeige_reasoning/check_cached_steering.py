"""Strict calibration-only cache parity gate before answer interventions."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .check_llama_steering import read, write, sha
from .llama_capture import Runtime
from .llama_cached_steer import CachedSteeringModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-size', type=int, default=64, choices=[1,64])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('artifacts/nanbeige_reasoning')
    source = root/'q4_steering_preflight_v1'
    previous = read(source/'manifest.json')
    assert previous['status'] == 'pretrained_q4_residual_steering_preflight_passed'
    for name, digest in previous['files'].items(): assert sha(source/name) == digest
    contract = read(root/'q4_reasoning_pilot_8192_v1/contract.json')
    runtime_path = Path('tools/nanbeige/runtime')
    for name, digest in contract['runtime_files'].items(): assert sha(runtime_path/name) == digest
    model_path = Path('models/Nanbeige4.2-3B-Q4_K_M.gguf')
    assert sha(model_path) == contract['model_sha256']
    question_path = root/'pilot_v1/questions.jsonl'
    assert sha(question_path) == contract['question_sha256']
    questions = [json.loads(s) for s in question_path.read_text(encoding='utf-8').splitlines()]
    # First training input, its full original prompt, no held-out data selection.
    question = next(r for r in questions if r['calibration_split']=='train')
    tokens = question['token_ids']
    with np.load(source/'selected_direction.npz') as z: direction = z['direction'].copy()
    arms = [('baseline',None),('identity',dict(direction=direction,alpha=0.,passes=[0,1],identity_write=True))]
    for name, passes in [('first',[0]),('second',[1]),('both',[0,1])]:
        for label, alpha in [('plus',.1),('minus',-.1)]:
            arms.append((name+'_'+label,dict(direction=direction,alpha=alpha,passes=passes)))
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'contract.json',dict(question_id=question['id'],question_sha256=sha(question_path),
        source_manifest_sha256=sha(source/'manifest.json'),chunk_size=args.chunk_size,
        generated_steps=8,atol=1e-4,rtol=1e-4,greedy_token_identity_required=True,
        scope='Full calibration prompt; all processed positions steered; no task score.',
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['llama_capture.py','llama_steer.py','llama_cached_steer.py','check_cached_steering.py']}))
    model = CachedSteeringModel(Runtime(runtime_path,require_cuda=True),model_path,context=512,gpu_layers=-1)
    results=[]; baseline=None; started=time.monotonic()
    try:
        for name, intervention in arms:
            model.reset_cache()
            logits=model.evaluate(tokens,intervention=intervention,chunk_size=args.chunk_size)
            cached=[]; generated=[]
            for step in range(8):
                cached.append(logits.copy()); generated.append(int(logits.argmax()))
                if step<7: logits=model.evaluate([generated[-1]],intervention=intervention,chunk_size=args.chunk_size)
            writes=model.total_mutations
            full=[]
            for step in range(8):
                model.reset_cache()
                full.append(model.evaluate(tokens+generated[:step],intervention=intervention,chunk_size=args.chunk_size))
            cached=np.stack(cached);full=np.stack(full)
            if name=='baseline': baseline=cached.copy()
            zero_equal=bool(np.array_equal(baseline,cached)) if name=='identity' else None
            difference=np.abs(cached.astype(np.float64)-full.astype(np.float64))
            close=bool(np.allclose(cached,full,atol=1e-4,rtol=1e-4))
            same=bool(np.array_equal(cached.argmax(axis=1),full.argmax(axis=1)))
            np.savez_compressed(args.output/f'{name}.npz',cached=cached,full_prefix=full,token_ids=tokens,generated=generated)
            row=dict(condition=name,max_abs_logit_difference=float(difference.max()),allclose=close,
                greedy_tokens_equal=same,identity_write_equals_baseline=zero_equal,mutation_count=writes,
                archive=f'{name}.npz',sha256=sha(args.output/f'{name}.npz'))
            results.append(row);write(args.output/'results.json',results);print(json.dumps(row),flush=True)
        passed=all(r['allclose'] and r['greedy_tokens_equal'] and r['identity_write_equals_baseline'] is not False for r in results)
        write(args.output/'manifest.json',dict(status='cached_steering_gate_passed' if passed else 'cached_steering_gate_failed',
            elapsed_seconds=time.monotonic()-started,conditions=len(results),
            files={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}))
    finally:
        model.close()
        write(args.output/'cleanup.json',dict(model_closed=model.model is None,context_closed=model.context is None))


if __name__=='__main__': main()
