"""Actual Nanbeige inference: full prompts, fixed output-write doses, CPU only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .llama_capture import Runtime
from .recurrence_state_control import StateHistoryModel
from .intervention_change_fidelity import sha,read,write

BASE=Path('artifacts/nanbeige_reasoning')
HISTORIES=[('one_pass',1,False),('carry',2,False),('reset',2,True)]
DOSES=[-.1,.1,-1.,1.,-10.,10.]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--canary',action='store_true')
    p.add_argument('--wall-seconds',type=int,default=7200)
    args=p.parse_args();out=args.output
    if out.exists():raise FileExistsError('Choose a fresh output directory')
    source=BASE/'q4_reasoning_pilot_8192_v1/contract.json';cfg=read(source)
    runtime_dir=Path('tools/nanbeige/runtime');model_path=Path('models/Nanbeige4.2-3B-Q4_K_M.gguf')
    for name,digest in cfg['runtime_files'].items():assert sha(runtime_dir/name)==digest
    assert sha(model_path)==cfg['model_sha256']
    control=BASE/'muon_sae_control_v1/contract.json';sae=read(control)
    qp=BASE/'pilot_v1/questions.jsonl'
    questions=[json.loads(s) for s in qp.read_text(encoding='utf-8').splitlines()]
    if args.canary:
        questions=[next(q for q in questions if q['id']==sae['training_ids'][0]).copy()]
        questions[0]['token_ids']=questions[0]['token_ids'][:16]
    else:
        questions=[q for q in questions if q['id'] in sae['evaluation_ids']]
        assert len(questions)==10 and all(q['calibration_split']=='eval' for q in questions)
    fit=BASE/'prime-20260906T204058Z-561/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration'
    fm=read(fit/'manifest.json');factors=fit/'vpd_factors.npz'
    assert sha(factors)==fm['artifacts'][factors.name]
    directions={}
    with np.load(factors,allow_pickle=False) as f:
        for layer in (12,22):
            u=f[f'model.layers.{layer-1}.mlp.down_proj__U'][0].astype(np.float64)
            directions[layer]=(u/np.linalg.norm(u)).astype(np.float32)
    out.mkdir(parents=True)
    write(out/'protocol.json',dict(question='Does cross-pass state carry alter frozen-SAE change fidelity at matched residual-write magnitude?',
        histories=HISTORIES,doses=DOSES,layers=[12,22],component=0,seed=42,device='CPU',threads=2,
        conditions='Baseline, identity write, and signed doses at the final token, final executed pass only.',
        intervention='Residual write along normalized fitted VPD component-0 decoder direction. This matches direct-write direction, not the original fixed-coefficient parameter intervention.',
        magnitude='Same absolute L2 write norm for all histories; actual float32 write norm saved. This is not a baseline-relative dose.',
        reset='Replace every token loop-boundary state with its original embedding. Both passes execute; prefix KV states are condition-specific.',
        depth_limit='Carry/reset match executed 44-layer count but not effective dependent depth. This is not a separately trained nonlooped architecture control.',
        parity_gate='Reset final-pass states versus one-pass reference: allclose rtol=2e-5 atol=2e-4; final logits same tolerance and argmax. Mandatory before interpretation.',
        holdout='Existing 10 held-out inputs / 8 task groups, previously inspected. Prompt-end states only, no new generated reasoning or capability scores.',
        questions=[dict(id=q['id'],token_ids=q['token_ids'],calibration_split=q['calibration_split']) for q in questions],
        canary=args.canary,model_sha256=cfg['model_sha256'],runtime_files=cfg['runtime_files'],
        source_questions_sha256=sha(qp),factor_sha256=sha(factors),dictionary_contract_sha256=sha(control),
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ('recurrence_state_control.py','run_recurrence_state_control.py','llama_capture.py','llama_steer.py')}))
    np.savez_compressed(out/'directions.npz',**{str(k):v for k,v in directions.items()})
    started=time.monotonic();runtime=Runtime(runtime_dir);records=[];status='interrupted';m=None
    def checkpoint():
        write(out/'records.json',records)
        write(out/'progress.json',dict(status='running',records=len(records),elapsed_seconds=time.monotonic()-started))
        if time.monotonic()-started>args.wall_seconds:raise TimeoutError('Experiment wall budget exhausted')
    try:
        for history,loops,reset in HISTORIES:
            m=StateHistoryModel(runtime,model_path,loops=loops,context=4096,gpu_layers=0,threads=2,reset_history=reset)
            for index,q in enumerate(questions):
                tokens=q['token_ids'];m.clear_cache();m.append(tokens[:-1],0)
                arrays,logits=m.last_token(tokens)
                file=f'{history}_{index:02d}_baseline.npz'
                np.savez_compressed(out/file,**arrays,logits=logits,token_ids=tokens)
                records.append(dict(history=history,loops=loops,id=q['id'],archive=file,sha256=sha(out/file),condition='baseline'))
                # Mandatory causal-state control; comparing complete final-pass residuals.
                if reset:
                    with np.load(out/f'one_pass_{index:02d}_baseline.npz',allow_pickle=False) as ref:
                        for layer in range(22):np.testing.assert_allclose(arrays[f'model.layers.{layer}__pass_1'],ref[f'model.layers.{layer}__pass_0'],rtol=2e-5,atol=2e-4)
                        np.testing.assert_allclose(arrays['model.norm__pass_1'],ref['model.norm__pass_0'],rtol=2e-5,atol=2e-4)
                        np.testing.assert_allclose(logits,ref['logits'],rtol=2e-5,atol=2e-4)
                        assert logits.argmax()==ref['logits'].argmax()
                for layer in (12,22):
                    for dose in ([0.,.1,-.1] if args.canary else [0.]+DOSES):
                        current,last=m.last_token(tokens,direction=directions[layer],dose=dose,layer=layer)
                        mutation=m.mutations[0];delta=mutation['after'].astype(np.float64)-mutation['before']
                        np.testing.assert_allclose(np.linalg.norm(delta),abs(dose),rtol=1e-3,atol=1e-5)
                        if dose==0:
                            for key in arrays:np.testing.assert_array_equal(arrays[key],current[key])
                            np.testing.assert_array_equal(logits,last)
                        for pidx in range(loops):
                            for il in range(22):
                                if pidx<loops-1 or il<layer-1:
                                    key=f'model.layers.{il}__pass_{pidx}'
                                    np.testing.assert_array_equal(arrays[key],current[key])
                        if reset:
                            with np.load(out/f'one_pass_{index:02d}_L{layer}_{dose:+g}.npz',allow_pickle=False) as ref:
                                for il in range(22):np.testing.assert_allclose(current[f'model.layers.{il}__pass_1'],ref[f'model.layers.{il}__pass_0'],rtol=2e-5,atol=2e-4)
                                np.testing.assert_allclose(last,ref['logits'],rtol=2e-5,atol=2e-4)
                        file=f'{history}_{index:02d}_L{layer}_{dose:+g}.npz'
                        np.savez_compressed(out/file,**current,logits=last,write_before=mutation['before'],write_after=mutation['after'])
                        records.append(dict(history=history,loops=loops,id=q['id'],archive=file,sha256=sha(out/file),condition='write',layer=layer,dose=dose,actual_write_norm=float(np.linalg.norm(delta)),argmax_changed=bool(logits.argmax()!=last.argmax())))
                        checkpoint()
                print(json.dumps(dict(history=history,id=q['id'],records=len(records),elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
            m.close();m=None
        status='completed'
    finally:
        if m is not None:m.close()
        write(out/'records.json',records)
        write(out/'cleanup.json',dict(model_closed=True,device='CPU',new_generated_tokens=0))
        write(out/'progress.json',dict(status=status,records=len(records),elapsed_seconds=time.monotonic()-started))
        write(out/'manifest.json',dict(status=status,records=len(records),elapsed_seconds=time.monotonic()-started,
            protocol_sha256=sha(out/'protocol.json'),files={f.name:sha(f) for f in out.iterdir() if f.is_file()}))


if __name__=='__main__':main()
