"""Capture the native Q4 reasoning trajectories with the same pinned model.

All positions are teacher-forced through both passes. No BF16 weights are used.
Replay records conditional greedy agreement against the original generation.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .llama_capture import Runtime,CaptureModel
from .positions import capture_positions
from .fetch_sources import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text())
    if manifest['status']!='generation_complete' or manifest['records']!=manifest['expected_records']:
        raise ValueError('Require a complete paired generation run')
    contract=manifest['contract']
    runtime_path=Path('tools/nanbeige/runtime')
    model_path=Path('models/Nanbeige4.2-3B-Q4_K_M.gguf')
    if sha(model_path)!=contract['model_sha256']:raise ValueError('Model hash mismatch')
    for name,expected in contract['runtime_files'].items():
        if sha(runtime_path/name)!=expected:raise ValueError('Runtime hash mismatch')
    source=args.run/'generations.jsonl'
    generations=list(map(json.loads,source.read_text(encoding='utf-8').splitlines()))
    native=[r for r in generations if r['loops']==2]
    if len({r['id'] for r in native})!=manifest['examples']:raise ValueError('Missing native examples')
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    runtime=Runtime(runtime_path,require_cuda=args.device=='cuda')
    model=CaptureModel(runtime,model_path,loops=2,context=contract['context'],
        gpu_layers=-1 if args.device=='cuda' else 0,capture_mlp_inputs=True)
    records=[]
    try:
        tokens=native[0]['token_ids'][:64]
        _,plain=model.replay(tokens,[],capture=False)
        _,hooked=model.replay(tokens,list(range(len(tokens))))
        original,observed=plain[len(tokens)-1],hooked[len(tokens)-1]
        error=float(np.max(np.abs(original-observed)))
        agreement=bool(original.argmax()==observed.argmax())
        # Persist diagnostics before enforcing the gate, including on failure.
        np.savez_compressed(out/'callback_preflight_logits.npz',plain_logits=original,
            hooked_logits=observed,token_ids=np.asarray(tokens))
        (out/'callback_preflight.json').write_text(json.dumps(dict(max_abs_logit_error=error,argmax_match=agreement,
            rms_logit_error=float(np.sqrt(np.mean((original-observed)**2))),
            tolerance=1e-3,within_tolerance=error<=1e-3,plain_argmax=int(original.argmax()),hooked_argmax=int(observed.argmax()),
            device=args.device,positions=len(tokens),logits_sha256=sha(out/'callback_preflight_logits.npz')),indent=2))
        if error>1e-3 or not agreement:raise ValueError(f'Callback changes logits: {error}, {agreement}')
        for index,row in enumerate(native):
            before=time.monotonic()
            tokens=row['token_ids']
            positions,regions=capture_positions(row['prompt_tokens'],len(tokens))
            predictions=[pos for pos in positions if row['prompt_tokens']-1<=pos<len(tokens)-1]
            arrays,logits=model.replay(tokens,positions,logit_positions=predictions)
            choices=[]
            for pos in predictions:
                values=logits[pos]
                expected=tokens[pos+1]
                maximum=float(values.max())
                choices.append(dict(position=pos,expected_token=expected,argmax=int(values.argmax()),
                    match=bool(values.argmax()==expected),logit_gap=maximum-float(values[expected])))
            if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite Q4 activation')
            archive=out/f'activation_{index:04d}.npz'
            np.savez_compressed(archive,**arrays,token_ids=np.asarray(tokens),token_positions=np.asarray(positions))
            record=dict(id=row['id'],lane=row['lane'],family=row['family'],split=row['calibration_split'],
                prompt_tokens=row['prompt_tokens'],generated_tokens=row['generated_tokens'],
                max_token_censored=row['max_token_censored'],regions=regions,positions=positions,
                seconds=time.monotonic()-before,archive=archive.name,sha256=sha(archive),
                conditional_greedy_checks=choices,greedy_agreement=sum(x['match'] for x in choices)/len(choices) if choices else None,
                activation_arrays=len(arrays),dtype='float32 outputs from Q4_K_M weights')
            records.append(record)
            with (out/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            print(f'{index+1}/{len(native)} {row["lane"]}: {len(tokens)} tokens, {len(arrays)} arrays, replay agreement {record["greedy_agreement"]}, {record["seconds"]:.1f}s',flush=True)
    finally:model.close()
    result=dict(status='q4_reasoning_capture_complete',records=len(records),device=args.device,
        weights='Q4_K_M',kv_cache=contract['kv_cache'],generated_reasoning=True,
        additional_vpd_input_sites=['physical layer 12 down_proj input','physical layer 22 down_proj input'],
        generation_source_sha256=sha(source),generation_contract=contract,
        binding_sha256=sha(Path(__file__).with_name('llama_capture.py')),
        limitations=['Teacher-forced native trajectories, including explicitly censored traces.',
            'Chunked replay may differ numerically from single-token decoding; conditional agreement is recorded.',
            'This is the quantized model, not a BF16 mechanistic measurement.'],
        files={f.name:sha(f) for f in out.iterdir() if f.is_file()})
    (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    from .transfer_vpd_coordinates import analyze
    analyze(out)


if __name__=='__main__':main()
