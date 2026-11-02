"""Replay ledger decision prefixes through the existing pass-resolved Q4 hooks."""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import time
import numpy as np
from experiments.nanbeige_reasoning.llama_capture import Runtime,CaptureModel
from experiments.nanbeige_reasoning.temperature_guard_burst import TemperatureGuard
from .io import ROOT,digest,save
from .gpu_sensor import gpu_temperature


def decision_prefix(response,tokenizer):
    """Stop before the first generated token containing any action-code character."""
    prompt=response['input_ids'];generated=response['output_ids']
    raw=tokenizer.decode(generated,skip_special_tokens=False)
    closing=raw.find('</think>')
    if closing<0:return prompt,[dict(anchor='prompt_end',token_index=len(prompt)-1)],'no_complete_reasoning'
    start=closing+len('</think>')
    match=re.search(r'"action"\s*:\s*"([^"\\]+)"',raw[start:])
    if not match:return prompt,[dict(anchor='prompt_end',token_index=len(prompt)-1)],'no_action_json'
    code_start=start+match.start(1)
    for count in range(len(generated),-1,-1):
        text=tokenizer.decode(generated[:count],skip_special_tokens=False)
        if len(text)<=code_start and raw.startswith(text):
            tokens=prompt+generated[:count]
            anchors=[dict(anchor='prompt_end',token_index=len(prompt)-1),dict(anchor='pre_action',token_index=len(tokens)-1,
                code_character_offset=code_start,decoded_prefix_characters=len(text),excluded_next_token_id=generated[count] if count<len(generated) else None)]
            assert len(text)<=code_start
            return tokens,anchors,'decision_prefix_found'
    raise ValueError('Could not locate a token boundary before action code')


def attach_guard(model,guard):
    """Use the existing synchronized decode guard with bounded work chunks."""
    guard.policy.update(idle_to_decode_ratio=2.,minimum_idle_seconds=.02,warm_minimum_idle_seconds=.1,
        resume_c=84,cooling_poll_seconds=.05,poll_seconds=.02,check_after_decode=True)
    guard.attach(model)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-seconds',type=int,default=3600);p.add_argument('--limit',type=int)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    reference=json.loads((ROOT/'artifacts/nanbeige_reasoning/q4_reasoning_pilot_8192_v1/contract.json').read_text(encoding='utf-8'))
    runtime_path=ROOT/'tools/nanbeige/runtime';model_path=ROOT/'models/Nanbeige4.2-3B-Q4_K_M.gguf'
    assert digest(model_path)==reference['model_sha256']
    for name,h in reference['runtime_files'].items():assert digest(runtime_path/name)==h
    inputs=json.loads(a.inputs.read_text(encoding='utf-8'))
    if a.limit:inputs=inputs[:a.limit]
    prepared=[]
    for item in inputs:
        path=ROOT/item['source_episode'];assert digest(path)==item['source_episode_sha256']
        r=json.loads(path.read_text(encoding='utf-8'))
        prepared.append((path,r,item['tokens'],item['anchors'],item['status']))
    if not prepared:raise ValueError('No recorded inference prefixes available')
    devices={r['events'][0]['response'].get('generation_context',{}).get('execution_device','cuda') for _,r,_,_,_ in prepared}
    if len(devices)!=1:raise ValueError('Do not mix CPU and GPU episodes in one replay contract')
    device=devices.pop()
    context=max(128,max(len(x[2]) for x in prepared)+8,
        max(x[1]['events'][0]['response'].get('generation_context',{}).get('context_tokens',8192) for x in prepared))
    save(a.output/'contract.json',dict(source_episodes={str(p.relative_to(ROOT)):digest(p) for p,_,_,_,_ in prepared},
        model_sha256=reference['model_sha256'],runtime_files=reference['runtime_files'],context=context,loops=2,execution_device=device,
        weights='Q4_K_M',capture_method='Existing CaptureModel.replay; fixed generated sequence with action-code tokens excluded',
        primary_anchor='prompt_end',secondary_anchor='pre_action after native reasoning; explicit textual plans may already reveal the decision',
        intervention_condition='baseline',source_code_sha256=digest(Path(__file__)),hook_sha256=digest(ROOT/'experiments/nanbeige_reasoning/llama_capture.py')))
    runtime=Runtime(runtime_path,require_cuda=device=='cuda');model=CaptureModel(runtime,model_path,loops=2,context=context,gpu_layers=999 if device=='cuda' else 0,threads=2,capture_mlp_inputs=True)
    guard=TemperatureGuard(a.output/'thermal.jsonl',sensor=gpu_temperature,wall_seconds=a.wall_seconds);attach_guard(model,guard)
    records=[]
    try:
        # Verify the existing hook does not alter logits at an actual ledger prefix.
        tokens=prepared[0][2];positions=sorted({x['token_index'] for x in prepared[0][3]})
        _,plain=model.replay(tokens,[],capture=False,chunk_size=8)
        arrays,hooked=model.replay(tokens,positions,chunk_size=8)
        error=float(np.max(np.abs(plain[len(tokens)-1]-hooked[len(tokens)-1])))
        np.testing.assert_array_equal(plain[len(tokens)-1],hooked[len(tokens)-1])
        save(a.output/'hook_control.json',dict(full_logit_max_abs_error=error,passed=True,scope='One complete ledger decision prefix; capture on/off'))
        for index,(path,r,tokens,anchors,status) in enumerate(prepared):
            start=time.monotonic();positions=sorted({x['token_index'] for x in anchors})
            arrays,_=model.replay(tokens,positions,chunk_size=8)
            assert len([k for k in arrays if re.fullmatch(r'model.layers.\d+__pass_[01]',k)])==44
            assert all(np.isfinite(x).all() for x in arrays.values())
            dest=a.output/f'episode_{index:04d}.npz'
            np.savez_compressed(dest,**arrays,token_ids=np.asarray(tokens,dtype=np.int32),token_positions=np.asarray(positions,dtype=np.int32))
            anchor_records=[]
            for anchor in anchors:
                pos=anchor['token_index'];token=tokens[pos]
                assert anchor['token_id']==token
                anchor_records.append(dict(**anchor,array_row=positions.index(pos)))
            record=dict(episode_id=r['case']['case_id'],world_id=r['case']['world_id'],template_id=r['case']['template_id'],split=r['case']['split'],
                condition_id=f"{r['case']['variant']}-p{int(r['case']['pressure'])}-m{int(r['case']['monitored'])}",
                variant=r['case']['variant'],pressure=r['case']['pressure'],monitored=r['case']['monitored'],turn_id=0,
                intervention_condition='baseline',labels=r['labels'],chosen_action=r['events'][0].get('chosen_action'),
                source_episode=str(path.resolve().relative_to(ROOT)),source_episode_sha256=digest(path),
                generation_context=dict(prompt_tokens=len(r['events'][0]['response']['input_ids']),context_tokens=len(tokens),
                    template_preserves_thinking=True,source_response_location='events[0].response',action_code_tokens_included=False),
                observation_mode='Fixed-prefix replay through the pinned runtime; not an online capture during stochastic generation. Capture on/off parity checks hook neutrality, not cached-generation equivalence.',
                anchor_status=status,anchors=anchor_records,archive=dest.name,sha256=digest(dest),
                array_schema=[dict(key=k,physical_layer=int(m[1])+1,pass_number=int(m[2])+1) for k in arrays if (m:=re.fullmatch(r'model.layers.(\d+)__pass_([01])',k))],
                seconds=time.monotonic()-start)
            records.append(record);save(a.output/'records.json',records)
            print(json.dumps(dict(episode=record['episode_id'],tokens=len(tokens),arrays=len(arrays),seconds=round(record['seconds'],1))),flush=True)
    finally:
        model.close();save(a.output/'thermal_summary.json',guard.summary())
    save(a.output/'manifest.json',dict(status='decision_captures_complete',episodes=len(records),physical_layers=22,passes=2,
        files={p.name:digest(p) for p in a.output.iterdir() if p.is_file()}))


if __name__=='__main__':main()
