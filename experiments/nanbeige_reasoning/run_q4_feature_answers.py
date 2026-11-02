"""Resumable local answer experiment gated on verified serial cache parity.

Inference reads public questions only. Grade completed outputs separately.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from .check_llama_steering import read, sha
from .llama_capture import Runtime, P, I, B
from .llama_cached_steer import CachedSteeringModel
from .temperature_guard_burst import TemperatureGuard


def atomic(path, obj):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def record_digest(row):
    payload={k:v for k,v in row.items() if k!='payload_sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save_record(path,row):
    row['payload_sha256']=record_digest(row)
    atomic(path,row)


def read_record(path):
    row=read(path)
    if row.get('payload_sha256')!=record_digest(row): raise ValueError('Answer checkpoint hash mismatch')
    return row


def native_decoder(model):
    vocab=model.lib.llama_model_get_vocab(model.model)
    model.lib.llama_vocab_is_eog.argtypes=[P,I];model.lib.llama_vocab_is_eog.restype=B
    runtime_record=dict(vocab_size=model.vocab_size,eog_token_ids=[i for i in range(model.vocab_size) if model.lib.llama_vocab_is_eog(vocab,i)])
    model.lib.llama_detokenize.argtypes=[P,C.POINTER(I),I,C.c_char_p,I,B,B]
    model.lib.llama_detokenize.restype=I
    def decode(tokens):
        data=(I*len(tokens))(*tokens);buffer=C.create_string_buffer(max(64,len(tokens)*16))
        size=model.lib.llama_detokenize(vocab,data,len(tokens),buffer,len(buffer),False,True)
        if size<0:
            buffer=C.create_string_buffer(-size)
            size=model.lib.llama_detokenize(vocab,data,len(tokens),buffer,len(buffer),False,True)
        if size<0: raise RuntimeError('Token decoding failed')
        return buffer.raw[:size].decode('utf-8',errors='replace')
    return runtime_record,decode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-new-tokens',type=int,default=8192)
    parser.add_argument('--wall-seconds',type=int,default=7200)
    parser.add_argument('--context-check',type=Path,required=True)
    args=parser.parse_args()
    if args.max_new_tokens<1 or args.wall_seconds<1: raise ValueError('Positive limits required')
    root=Path('artifacts/nanbeige_reasoning')
    gate=read(args.gate/'manifest.json');gate_contract=read(args.gate/'contract.json')
    verification=read(args.gate/'verification.json')
    assert verification['status']=='cached_steering_saved_arrays_verified' and verification['gate_passed']
    assert verification['manifest_sha256']==sha(args.gate/'manifest.json')
    assert gate['status']=='cached_steering_gate_passed' and gate_contract['chunk_size']==1
    for name,digest in gate['files'].items(): assert sha(args.gate/name)==digest
    for name,digest in gate_contract['code_sha256'].items(): assert sha(Path(__file__).with_name(name))==digest
    source=root/'q4_steering_preflight_v1'
    assert sha(source/'manifest.json')==gate_contract['source_manifest_sha256']
    source_manifest=read(source/'manifest.json')
    for name,digest in source_manifest['files'].items(): assert sha(source/name)==digest
    with np.load(source/'selected_direction.npz') as z: direction=z['direction'].copy()
    reference=read(root/'q4_reasoning_pilot_8192_v1/contract.json')
    model_path=Path('models/Nanbeige4.2-3B-Q4_K_M.gguf');runtime_path=Path('tools/nanbeige/runtime')
    assert sha(model_path)==reference['model_sha256']
    for name,digest in reference['runtime_files'].items(): assert sha(runtime_path/name)==digest
    qpath=root/'pilot_v1/questions.jsonl'
    assert sha(qpath)==reference['question_sha256']==gate_contract['question_sha256']
    questions=[json.loads(s) for s in qpath.read_text(encoding='utf-8').splitlines()]
    # Round-robin families so partial execution does not spend its entire budget on ARC.
    lanes={}
    for q in questions:
        if q['calibration_split']=='eval': lanes.setdefault(q['lane'],[]).append(q)
    selected=[lane[i] for i in range(max(map(len,lanes.values()))) for lane in lanes.values() if i<len(lane)]
    arms=[dict(name='baseline',alpha=0.,passes=[])]
    for name,passes in [('second',[1]),('first',[0]),('both',[0,1])]:
        for label,alpha in [('plus',.1),('minus',-.1)]: arms.append(dict(name=name+'_'+label,alpha=alpha,passes=passes))
    context=max(len(q['token_ids']) for q in selected)+args.max_new_tokens
    context_check=read(args.context_check/'verification.json')
    assert context_check['status']=='answer_context_and_decoding_verified' and context_check['context']==context
    assert context_check['manifest_sha256']==sha(args.context_check/'manifest.json')
    context_contract=read(args.context_check/'contract.json')
    assert context_contract['gate_manifest_sha256']==sha(args.gate/'manifest.json')
    assert context_contract['model_sha256']==reference['model_sha256']
    for name,digest in context_contract['code_sha256'].items():
        assert sha(Path(__file__).with_name(name))==digest
    for name,digest in read(args.context_check/'manifest.json')['files'].items():
        assert sha(args.context_check/name)==digest
    tokenizer_path=Path('models/Nanbeige4.2-3B-tokenizer')
    tokenizer_files={n:sha(tokenizer_path/n) for n in ['tokenizer.model','tokenizer_config.json','added_tokens.json','special_tokens_map.json','config.json']}
    assert tokenizer_files==context_check['tokenizer_files']
    args.output.mkdir(parents=True,exist_ok=True)
    guard=TemperatureGuard(args.output/'temperature.jsonl',wall_seconds=args.wall_seconds)
    contract=dict(schema='nanbeige.muon_feature_answers.v2',question_sha256=sha(qpath),
        gate_manifest_sha256=sha(args.gate/'manifest.json'),source_manifest_sha256=sha(source/'manifest.json'),
        model_sha256=reference['model_sha256'],evaluation_ids=[q['id'] for q in selected],conditions=arms,
        feature_id=source_manifest['selected_feature'],physical_layer=22,loops=2,chunk_size=1,
        max_new_tokens=args.max_new_tokens,context=context,tokenizer_files=tokenizer_files,
        execution_guard=guard.policy,context_check_verification_sha256=sha(args.context_check/'verification.json'),
        intervention_scope='All prefill and decode positions; frozen decoder direction chosen on training inputs only.',
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['llama_capture.py','llama_steer.py','llama_cached_steer.py','run_q4_feature_answers.py','temperature_guard_burst.py']},
        limits=['Single post-hoc feature and dose magnitude; no semantic attribution.',
                'Fresh serial baseline; earlier batched trajectories are not the paired baseline.',
                'No answer accuracy claim until separate grading and complete ARC aggregation.'])
    cp=args.output/'contract.json'
    if cp.exists(): assert read(cp)==contract
    else: atomic(cp,contract)
    guard.check(force=True)
    model=None;started=time.monotonic();completed=0;stop=False;ready=False
    def save_manifest():
        atomic(args.output/'manifest.json',dict(status='q4_feature_answers_complete' if completed==len(selected)*len(arms) else 'q4_feature_answers_partial',
            completed=completed,expected=len(selected)*len(arms),wall_budget_reached=stop,
            contract_sha256=sha(cp),files={p.name:sha(p) for p in args.output.glob('*.json') if p.name not in ['manifest.json','progress.json','cleanup.json','interrupted.json']}))
    try:
        model=CachedSteeringModel(Runtime(runtime_path,require_cuda=True),model_path,context=context,gpu_layers=-1)
        guard.attach(model)
        runtime_record,decode=native_decoder(model)
        if (args.output/'runtime.json').exists(): assert read(args.output/'runtime.json')==runtime_record
        else: atomic(args.output/'runtime.json',runtime_record)
        eos_ids=set(runtime_record['eog_token_ids'])
        for qi,q in enumerate(selected):
            for arm in arms:
                ready=False
                name=f'{qi:02d}_{arm["name"]}.json';path=args.output/name
                state=read_record(path) if path.exists() else dict(id=q['id'],condition=arm['name'],generated=[],complete=False,
                    prompt_tokens=len(q['token_ids']),contract_sha256=sha(cp),elapsed_seconds=0.)
                assert state['id']==q['id'] and state['condition']==arm['name'] and state['contract_sha256']==sha(cp)
                if state['complete']: completed+=1;continue
                if time.monotonic()-started>=args.wall_seconds: stop=True;break
                atomic(args.output/'progress.json',dict(pid=os.getpid(),id=q['id'],condition=arm['name'],
                    phase='prefill_and_resume',generated_tokens=len(state['generated']),completed=completed,
                    expected=len(selected)*len(arms),elapsed_seconds=time.monotonic()-started,status='running'))
                before=time.monotonic();model.reset_cache()
                intervention=dict(direction=direction,alpha=arm['alpha'],passes=arm['passes']) if arm['passes'] else None
                logits=model.evaluate(q['token_ids'],intervention=intervention,chunk_size=1)
                # Reconstruct the same cache and verify every saved greedy decision on resume.
                generated=state['generated']
                for token in generated:
                    assert int(logits.argmax())==token, 'Resume prefix changed'
                    logits=model.evaluate([token],intervention=intervention,chunk_size=1)
                while len(generated)<args.max_new_tokens:
                    token=int(logits.argmax());generated.append(token)
                    eos=token in eos_ids
                    done=eos or len(generated)==args.max_new_tokens
                    stop=time.monotonic()-started>=args.wall_seconds
                    state.update(generated=generated,complete=done,stopped_on_eog=eos,
                        max_token_censored=not eos and len(generated)==args.max_new_tokens,
                        model_processed_tokens=model.next_position,mutation_count=model.total_mutations)
                    ready=True
                    if done or stop or len(generated)%128==0:
                        state.update(generated_text=decode(generated),elapsed_seconds=state['elapsed_seconds']+time.monotonic()-before)
                        save_record(path,state);before=time.monotonic()
                        atomic(args.output/'progress.json',dict(pid=os.getpid(),id=q['id'],condition=arm['name'],
                            phase='generation',generated_tokens=len(generated),completed=completed+int(done),expected=len(selected)*len(arms),
                            elapsed_seconds=time.monotonic()-started,status='running' if not stop else 'wall_budget_reached'))
                    if done or stop: break
                    logits=model.evaluate([token],intervention=intervention,chunk_size=1)
                if state['complete']:
                    completed+=1;print(json.dumps(dict(id=q['id'],condition=arm['name'],tokens=len(generated),completed=completed)),flush=True)
                if stop: break
            if stop: break
        save_manifest()
    except BaseException as exc:
        if ready:
            state.update(generated_text=decode(state['generated']),elapsed_seconds=state['elapsed_seconds']+time.monotonic()-before)
            save_record(path,state)
        atomic(args.output/'interrupted.json',dict(error=f'{type(exc).__name__}: {exc}',completed=completed,guard=guard.summary()))
        save_manifest()
        raise
    finally:
        if model:model.close()
        atomic(args.output/'cleanup.json',dict(model_closed=model is None or model.model is None,context_closed=model is None or model.context is None,guard=guard.summary()))


if __name__=='__main__': main()
