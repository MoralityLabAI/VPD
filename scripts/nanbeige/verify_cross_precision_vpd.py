"""Bind a retrieved BF16 reasoning VPD fit to its frozen Q4 source contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.nanbeige.verify_multipass_results import read, sha


def verify(run):
    run = run.resolve()
    receipt = read(run / 'bundle.receipt.json')
    if sha(run / 'bundle.tar.gz') != receipt['sha256']:
        raise ValueError('Frozen uploaded bundle changed')
    retrieved = run / 'retrieved'
    contract_path = retrieved / receipt.get('input_contract_path',
        'artifacts/nanbeige_reasoning/cross_precision_vpd_inputs_v1/contract.json')
    if not contract_path.resolve().is_relative_to(retrieved):
        raise ValueError('Input contract path escapes retrieval directory')
    contract = read(contract_path)
    if contract != receipt['experiment_contract']:
        raise ValueError('Returned input contract differs from frozen launch')
    dataset = retrieved / contract['dataset']
    if not dataset.resolve().is_relative_to(retrieved) or sha(dataset) != contract['dataset_sha256']:
        raise ValueError('Returned calibration tokens differ from frozen source')
    examples = [json.loads(s) for s in dataset.read_text(encoding='utf-8').splitlines()]
    root = retrieved / 'artifacts/nanbeige_looped/bf16_multipass_calibration'
    manifest = read(root / 'manifest.json')
    verification = read(root / 'independent_verification.json')
    if (manifest['status'] != 'calibration_complete'
            or verification['status'] != 'multipass_saved_effects_independently_verified'
            or verification['manifest_sha256'] != sha(root / 'manifest.json')
            or read(root / 'examples.json') != examples):
        raise ValueError('Fit is incomplete, unverified or uses different trajectories')
    model = manifest['model']
    if (model['kind'] != 'local_pretrained_checkpoint' or model['dtype'] != 'torch.bfloat16'
            or model['expected_revision'] != contract['revision'] or model['model_id'] != contract['model_id']):
        raise ValueError('Measurement model differs from contract')
    expected_assets = {k:v for k,v in contract['checkpoint_files'].items()
                       if Path(k).suffix in ('.json', '.py', '.safetensors')}
    if model['checkpoint_files'] != expected_assets:
        raise ValueError('Recorded checkpoint assets differ from pinned contract')
    expected_sources = {Path(name).name:digest for name,digest in receipt['files'].items()
                        if name.startswith('experiments/nanbeige_looped/')
                        and (name.endswith('.py') or Path(name).name in ('vpd_backend_provenance.json','VPD_BACKEND_LICENSE'))}
    if manifest['implementation_sha256'] != expected_sources:
        raise ValueError('Executed calibration source differs from uploaded bundle')
    train = [r for r in examples if r['split'] == 'train']
    evaluation = [r for r in examples if r['split'] == 'eval']
    if len(train) != contract['train_examples'] or len(evaluation) != contract['eval_examples']:
        raise ValueError('Fit split cardinalities differ')
    prefixes = []
    for row in train:
        prompt, length = row['prompt_tokens'], len(row['token_ids'])
        for end in sorted({prompt+max(1,round((length-prompt)*f)) for f in contract['training_prefix_fractions']}):
            prefixes.append(dict(id=row['id'],start=0,end=end,prompt_tokens=prompt,generated_tokens_in_prefix=end-prompt))
    training = read(root / 'vpd_training.json')
    cfg = training['config']
    if (training['training_window'] != 'generated_prefixes' or training['training_prefixes'] != prefixes
            or manifest['vpd_training_window']['prefixes'] != prefixes
            or manifest['vpd_training_window']['max_tokens'] != max(p['end'] for p in prefixes)
            or cfg['n_steps'] != contract['vpd_steps']
            or cfg['faithfulness_warmup_steps'] != contract['warmup_steps']
            or len(training['history']) != contract['vpd_steps']):
        raise ValueError('Generated-prefix optimization differs from contract')
    for step, row in enumerate(training['history']):
        if (row['step'] != step+1 or row['sequence_length'] != prefixes[step % len(prefixes)]['end']
                or not all(np.isfinite(row[name]) for name in ('faith','importance','stochastic','adversarial'))):
            raise ValueError('Invalid optimization history')
    physical = [f'model.layers.{layer-1}.mlp.down_proj' for layer in contract['physical_layers_one_based']]
    if (training['physical_matrices'] != physical or training['parameter_sets_per_physical_matrix'] != 1
            or set(training['invocation_sites']) != {f'{site}::pass_{n}' for site in physical for n in (0,1)}
            or manifest['passes'] != 2):
        raise ValueError('Physical sharing or invocation scope differs')
    with np.load(root / 'vpd_factors.npz', allow_pickle=False) as factors:
        if set(factors.files) != {f'{site}__{key}' for site in physical for key in ('V','U')}:
            raise ValueError('Missing/extra factor matrix')
        for site in physical:
            if (factors[site+'__V'].shape != (10752, contract['components_per_matrix'])
                    or factors[site+'__U'].shape != (contract['components_per_matrix'],3072)):
                raise ValueError('Factor dimensions differ')
    settings = manifest['calibration_settings']
    if settings['sae_steps'] != contract['sae_steps'] or settings['components_per_matrix'] != contract['components_per_matrix']:
        raise ValueError('Calibration budget differs')
    if settings.get('saved_tensors_on_cpu',False) != contract.get('saved_tensors_on_cpu',False):
        raise ValueError('Recorded autograd memory strategy differs')
    offload_control_sha256 = None
    if contract.get('saved_tensors_on_cpu',False):
        control_path = retrieved / 'artifacts/nanbeige_reasoning/saved_tensor_control.json'
        control = read(control_path)
        if (control['status'] != 'saved_tensor_offload_control_passed' or control['device'] != 'cuda'
                or control['torch_version'] != manifest['torch_version']
                or not control['random_fixture_only'] or control['model_recomputed_for_backward']):
            raise ValueError('Missing or incompatible CUDA offload control')
        offload_control_sha256 = sha(control_path)
    result = dict(status='cross_precision_reasoning_vpd_verified', model_asset_hashes_match=True,
        executed_sources_match_uploaded_bundle=True, exact_source_trajectories_match=True,
        manifest_sha256=sha(root / 'manifest.json'), input_contract_sha256=sha(contract_path),
        source_generation_sha256=contract['generation_sha256'], examples=len(examples),
        training_prefixes=len(prefixes), vpd_steps=len(training['history']),
        offload_control_sha256=offload_control_sha256,
        scope=contract['scope'], limitations=contract['limitations'])
    (root / 'cross_precision_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    print(json.dumps(verify(p.parse_args().run),indent=2))
