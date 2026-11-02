"""Continue six unfinished controls with cooling and unchanged numerical gates."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
from .check_llama_steering import read,write,sha
from .llama_capture import Runtime
from .llama_cached_steer import CachedSteeringModel
from .temperature_guard import TemperatureGuard


def main():
    root=Path('artifacts/nanbeige_reasoning')
    previous=root/'q4_cached_steering_serial_v1'
    out=root/'q4_cached_steering_guarded_v1'
    old=read(previous/'contract.json');records=read(previous/'results.json')
    assert old['chunk_size']==1 and old['atol']==old['rtol']==1e-4
    assert [r['condition'] for r in records]==['baseline','identity']
    for name,digest in old['code_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
    reference=read(root/'q4_reasoning_pilot_8192_v1/contract.json')
    runtime_path=Path('tools/nanbeige/runtime');model_path=Path('models/Nanbeige4.2-3B-Q4_K_M.gguf')
    for name,digest in reference['runtime_files'].items():assert sha(runtime_path/name)==digest
    assert sha(model_path)==reference['model_sha256']
    qpath=root/'pilot_v1/questions.jsonl';assert sha(qpath)==old['question_sha256']
    q=next(json.loads(s) for s in qpath.read_text(encoding='utf-8').splitlines() if json.loads(s)['id']==old['question_id'])
    assert q['calibration_split']=='train';tokens=q['token_ids']
    source=root/'q4_steering_preflight_v1';assert sha(source/'manifest.json')==old['source_manifest_sha256']
    source_manifest=read(source/'manifest.json')
    assert sha(source/'selected_direction.npz')==source_manifest['files']['selected_direction.npz']
    with np.load(source/'selected_direction.npz') as z:direction=z['direction'].copy()
    # Validate both carried controls before creating any new experiment output.
    for row in records:
        assert sha(previous/row['archive'])==row['sha256']
        with np.load(previous/row['archive']) as z:
            np.testing.assert_array_equal(z['cached'],z['full_prefix'])
            np.testing.assert_array_equal(z['token_ids'],tokens)
    out.mkdir(parents=True,exist_ok=False)
    guard=TemperatureGuard(out/'temperature.jsonl',wall_seconds=7200)
    guard.check(force=True)
    contract=dict(old,source_partial_contract_sha256=sha(previous/'contract.json'),
        source_partial_results_sha256=sha(previous/'results.json'),execution_guard=guard.policy,
        continuation_code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['temperature_guard.py','resume_guarded_cached_control.py']},
        carried_conditions=['baseline','identity'],fresh_conditions=['first_plus','first_minus','second_plus','second_minus','both_plus','both_minus'])
    write(out/'contract.json',contract)
    for row in records:shutil.copyfile(previous/row['archive'],out/row['archive'])
    write(out/'results.json',records)
    model=None;started=time.monotonic()
    def progress(condition,phase,step):
        write(out/'progress.json',dict(condition=condition,phase=phase,step=step,completed_conditions=len(records),
            elapsed_seconds=time.monotonic()-started,guard=guard.summary()))
    try:
        model=CachedSteeringModel(Runtime(runtime_path,require_cuda=True),model_path,context=512,gpu_layers=-1)
        guard.attach(model)
        def cached_trajectory(intervention):
            model.reset_cache();logits=model.evaluate(tokens,intervention=intervention,chunk_size=1)
            cached=[];generated=[]
            for step in range(8):
                cached.append(logits.copy());generated.append(int(logits.argmax()))
                if step<7:logits=model.evaluate([generated[-1]],intervention=intervention,chunk_size=1)
            return np.stack(cached),generated,model.total_mutations
        # Verify all saved cached decisions/logits under the added guard after restart.
        restart_arrays={}
        for name,intervention in [('baseline',None),('identity',dict(direction=direction,alpha=0.,passes=[0,1],identity_write=True))]:
            progress(name,'guarded_restart_check',0)
            cached,generated,writes=cached_trajectory(intervention)
            with np.load(previous/f'{name}.npz') as z:np.testing.assert_array_equal(cached,z['cached'])
            restart_arrays[name]=cached
            print(json.dumps(dict(condition=name,guarded_restart_bitwise_equal=True,temperature=guard.last_temperature)),flush=True)
        np.savez_compressed(out/'guarded_restart_check.npz',**restart_arrays)
        write(out/'guarded_restart_check.json',dict(status='guarded_cached_trajectories_bitwise_equal_to_saved_controls',
            conditions=2,steps_per_condition=8,archive_sha256=sha(out/'guarded_restart_check.npz')))
        for label,passes in [('first',[0]),('second',[1]),('both',[0,1])]:
            for sign,alpha in [('plus',.1),('minus',-.1)]:
                name=f'{label}_{sign}';intervention=dict(direction=direction,alpha=alpha,passes=passes)
                progress(name,'cached',0);cached,generated,writes=cached_trajectory(intervention)
                full=[]
                for step in range(8):
                    progress(name,'full_prefix',step);model.reset_cache()
                    full.append(model.evaluate(tokens+generated[:step],intervention=intervention,chunk_size=1))
                full=np.stack(full)
                np.savez_compressed(out/f'{name}.npz',cached=cached,full_prefix=full,token_ids=tokens,generated=generated)
                row=dict(condition=name,max_abs_logit_difference=float(np.abs(cached.astype(np.float64)-full.astype(np.float64)).max()),
                    allclose=bool(np.allclose(cached,full,atol=1e-4,rtol=1e-4)),greedy_tokens_equal=bool(np.array_equal(cached.argmax(1),full.argmax(1))),
                    identity_write_equals_baseline=None,mutation_count=writes,archive=f'{name}.npz',sha256=sha(out/f'{name}.npz'))
                records.append(row);write(out/'results.json',records);print(json.dumps(row),flush=True)
        passed=all(r['allclose'] and r['greedy_tokens_equal'] and r['identity_write_equals_baseline'] is not False for r in records)
        write(out/'thermal_summary.json',guard.summary())
        write(out/'manifest.json',dict(status='cached_steering_gate_passed' if passed else 'cached_steering_gate_failed',conditions=8,
            elapsed_seconds=time.monotonic()-started,carried_conditions=2,fresh_conditions=6,
            files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['temperature.jsonl','progress.json']}))
    except BaseException as exc:
        write(out/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',completed_conditions=len(records),guard=guard.summary()))
        raise
    finally:
        if model:model.close()
        write(out/'cleanup.json',dict(model_closed=model is None or model.model is None,context_closed=model is None or model.context is None))


if __name__=='__main__':main()
