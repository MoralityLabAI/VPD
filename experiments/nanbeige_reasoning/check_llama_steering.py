"""Actual pretrained CUDA control for zero, signed and pass-specific steering."""
import json
import time
from pathlib import Path
import hashlib
import numpy as np

from .llama_capture import Runtime
from .llama_steer import SteeringModel


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    root = Path('artifacts/nanbeige_reasoning')
    out = root/'q4_steering_preflight_v1'
    contract = read(root/'q4_reasoning_pilot_8192_v1/contract.json')
    control = root/'muon_sae_control_v1'
    muon_contract = read(control/'contract.json')
    verified = read(root/'muon_sae_verification/complete.json')
    assert verified['status'] == 'muon_control_independently_verified'
    assert verified['contract_sha256'] == sha(control/'contract.json')
    condition = control/'00_muon_hybrid'
    m = read(condition/'manifest.json')
    assert m['site'] == 'model.layers.21' and m['arm'] == 'muon_hybrid'
    assert sha(condition/'manifest.json') == next(r['condition_manifest_sha256'] for r in verified['conditions'] if r['site']==m['site'] and r['arm']==m['arm'])
    feature_records = read(condition/'records.json')
    assert sha(condition/'records.json') == m['files']['records.json']
    drift = []
    for r in feature_records:
        if r['split'] != 'train': continue
        assert r['id'] in muon_contract['training_ids'] and sha(condition/r['archive']) == r['sha256']
        with np.load(condition/r['archive'], allow_pickle=False) as z:
            drift.append(np.abs(z['pass_1'].astype(np.float64)-z['pass_0'].astype(np.float64)))
    scores = np.concatenate(drift).mean(axis=0)
    feature = int(scores.argmax())
    assert sha(condition/'dictionary.npz') == m['files']['dictionary.npz']
    with np.load(condition/'dictionary.npz', allow_pickle=False) as z:
        direction = z['decoder'][feature].copy()
    np.testing.assert_allclose(np.linalg.norm(direction), 1., atol=1e-6)
    runtime_path = Path('tools/nanbeige/runtime')
    for name, expected in contract['runtime_files'].items():
        assert sha(runtime_path/name) == expected
    model_path = Path('models/Nanbeige4.2-3B-Q4_K_M.gguf')
    assert sha(model_path) == contract['model_sha256']
    questions = [json.loads(s) for s in (root/'pilot_v1/questions.jsonl').read_text(encoding='utf-8').splitlines()]
    question = next(r for r in questions if r['id'] == muon_contract['training_ids'][0])
    assert question['calibration_split'] == 'train' and len(question['token_ids']) >= 96
    tokens = question['token_ids'][:96]
    positions = [0, 63, 94, 95]
    arms = [('identity_write', 0., [0,1], True)]
    for name, passes in [('first',[0]), ('second',[1]), ('both',[0,1])]:
        for sign, alpha in [('plus',.1), ('minus',-.1)]:
            arms.append((f'{name}_{sign}', alpha, passes, False))
    out.mkdir(parents=True, exist_ok=False)
    write(out/'contract.json', dict(schema='nanbeige.q4_residual_steering_preflight.v1',
        question_id=question['id'], split='train', prompt_prefix_tokens=96, token_ids=tokens,
        retained_positions=positions, steering_positions=[95], physical_layer=22,
        dictionary_sha256=sha(condition/'dictionary.npz'), feature_id=feature,
        selection='Largest mean absolute pass-feature difference across training inputs only; calibration control, no feature-semantic selection.',
        source_control_manifest_sha256=sha(control/'manifest.json'),
        source_generation_contract_sha256=sha(root/'q4_reasoning_pilot_8192_v1/contract.json'),
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ['llama_capture.py','llama_steer.py','check_llama_steering.py']},
        header_sha256=sha(root/'sources/llama_cpp/ggml-backend.h'),
        arms=[dict(name=n,alpha=a,passes=s,identity_write=w) for n,a,s,w in arms],
        limitation='Single calibration-prefix implementation test, not a task score, semantic feature interpretation or cached generation test.'))
    np.savez_compressed(out/'selected_direction.npz', direction=direction, training_drift_scores=scores, feature_id=feature)
    runtime = Runtime(runtime_path, require_cuda=True)
    model = SteeringModel(runtime, model_path, loops=2, context=256, gpu_layers=-1)
    started = time.monotonic(); results=[]
    try:
        _, plain = model.replay(tokens, [], capture=False)
        baseline, logits = model.replay(tokens, positions)
        reference = logits[95]
        np.testing.assert_array_equal(reference, plain[95])
        np.savez_compressed(out/'baseline.npz', **baseline, logits=reference, token_ids=tokens, token_positions=positions)
        for name, alpha, passes, identity in arms:
            model.set_steering(direction, alpha=alpha, passes=passes, positions=[95], identity_write=identity)
            arrays, current_logits = model.replay(tokens, positions)
            current = current_logits[95]
            if alpha == 0:
                np.testing.assert_array_equal(reference, current)
                for key in baseline: np.testing.assert_array_equal(baseline[key], arrays[key])
            upstream = 0
            for key in baseline:
                np.testing.assert_array_equal(baseline[key][:-1], arrays[key][:-1])
                upstream += 1
                if passes == [1] and key.endswith('__pass_0'):
                    np.testing.assert_array_equal(baseline[key], arrays[key]); upstream += 1
                if key.startswith('model.layers.') and key.endswith('__pass_0') and int(key.split('.')[2].split('__')[0]) < 21:
                    np.testing.assert_array_equal(baseline[key], arrays[key]); upstream += 1
            mutation_arrays = {}
            for i, change in enumerate(model.mutations):
                before, after = change['before'], change['after']
                expected = before if alpha==0 else before+np.float32(alpha)*direction
                np.testing.assert_array_equal(expected, after)
                mutation_arrays[f'before_{i}']=before; mutation_arrays[f'after_{i}']=after
                np.testing.assert_array_equal(arrays[f"model.layers.21__pass_{change['pass_index']}"][-1],after)
            difference = float(np.max(np.abs(current.astype(np.float64)-reference)))
            assert np.isfinite(current).all() and (difference > 0 if alpha != 0 else difference == 0)
            np.savez_compressed(out/f'{name}.npz', **arrays, **mutation_arrays, logits=current, token_ids=tokens, token_positions=positions)
            result = dict(name=name, alpha=alpha, passes=passes, mutations=len(model.mutations),
                max_abs_logit_change=difference, argmax_changed=bool(current.argmax()!=reference.argmax()),
                upstream_checks=upstream, archive=f'{name}.npz', sha256=sha(out/f'{name}.npz'))
            results.append(result); print(json.dumps(result),flush=True)
        model.clear_steering()
        restored, restored_logits = model.replay(tokens, positions)
        np.testing.assert_array_equal(restored_logits[95], reference)
        for key in baseline: np.testing.assert_array_equal(restored[key], baseline[key])
        write(out/'results.json',results)
        write(out/'manifest.json',dict(status='pretrained_q4_residual_steering_preflight_passed', conditions=len(results),
            selected_feature=feature, zero_write_bitwise_preserved=True, cleared_state_bitwise_restored=True,
            capture_control_logit_error=0., elapsed_seconds=time.monotonic()-started,
            scope='Actual CUDA tensor-write and causal-order checks on one96-token calibration prefix; no task-level or semantic result.',
            files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    finally:
        model.close()
        write(out/'cleanup.json',dict(model_closed=model.model is None, context_closed=model.context is None))


if __name__ == '__main__': main()
