"""Independently verify saved Q4 trajectories, VPD coordinates and SAE diffs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.nanbeige_looped.sae import SharedSAE,encode_rows
from .fetch_sources import sha
from .positions import capture_positions
from .transfer_vpd_coordinates import ORIGINAL


def verified_manifest(root,status,only_files=None):
    root=root.resolve()
    value=json.loads((root/'manifest.json').read_text())
    if value['status']!=status:raise ValueError(f'Incomplete stage: {root}')
    selected=value['files'] if only_files is None else {name:value['files'][name] for name in only_files}
    for name,expected in selected.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=expected:raise ValueError(f'Artifact hash mismatch: {path}')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--cohort',type=Path,default=Path('artifacts/nanbeige_reasoning/pilot_v1'))
    args=p.parse_args()
    torch.set_num_threads(2)
    run=args.run
    # The generation manifest also snapshots mutable orchestration/partial
    # analysis logs. Those are not the immutable trajectory payload.
    generation_files=['generations.jsonl','calibration_tokens.jsonl','contract.json']
    generation=verified_manifest(run,'generation_complete',only_files=generation_files)
    qpath=args.cohort/'questions.jsonl'
    if sha(qpath)!=generation['contract']['question_sha256']:raise ValueError('Generation question hash mismatch')
    questions=list(map(json.loads,qpath.read_text(encoding='utf-8').splitlines()))
    q={r['id']:r for r in questions}
    rows=list(map(json.loads,(run/'generations.jsonl').read_text(encoding='utf-8').splitlines()))
    pairs={(r['id'],r['loops']):r for r in rows}
    if len(pairs)!=len(rows) or set(pairs)!={(key,n) for key in q for n in (1,2)}:
        raise ValueError('Incomplete or duplicate generation pairs')
    for row in rows:
        prefix=q[row['id']]['token_ids']
        if row['prompt_tokens']!=len(prefix) or row['token_ids'][:len(prefix)]!=prefix or len(row['token_ids'])!=len(prefix)+row['generated_tokens']:
            raise ValueError('Generation token alignment failed')
    captures=run/'q4_activations'
    capture_manifest=verified_manifest(captures,'q4_reasoning_capture_complete')
    if capture_manifest['generation_source_sha256']!=sha(run/'generations.jsonl'):raise ValueError('Replay generation source mismatch')
    records=list(map(json.loads,(captures/'records.jsonl').read_text().splitlines()))
    if len(records)!=len(q) or {r['id'] for r in records}!=set(q):raise ValueError('Incomplete captures')
    callback=json.loads((captures/'callback_preflight.json').read_text())
    with np.load(captures/'callback_preflight_logits.npz',allow_pickle=False) as control:
        plain,hooked=control['plain_logits'],control['hooked_logits']
        if plain.shape!=(166144,) or hooked.shape!=plain.shape or not np.isfinite(plain).all() or not np.isfinite(hooked).all():
            raise ValueError('Invalid callback-control logits')
        error=float(np.max(np.abs(plain-hooked)))
        if error!=callback['max_abs_logit_error'] or error>1e-3 or plain.argmax()!=hooked.argmax():
            raise ValueError('Callback-control parity failed')
        if not np.array_equal(control['token_ids'],pairs[records[0]['id'],2]['token_ids'][:64]):
            raise ValueError('Callback-control token source mismatch')
    arrays={};position_map={r['id']:r['positions'] for r in records}
    checked_choices=matched_choices=0
    for record in records:
        source=pairs[record['id'],2]
        positions,regions=capture_positions(source['prompt_tokens'],len(source['token_ids']))
        if record['positions']!=positions or record['regions']!={k:list(v) for k,v in regions.items()}:
            raise ValueError('Capture position/region alignment mismatch')
        path=captures/record['archive']
        if sha(path)!=record['sha256']:raise ValueError('Capture record hash mismatch')
        with np.load(path,allow_pickle=False) as a:
            if not np.array_equal(a['token_ids'],source['token_ids']) or not np.array_equal(a['token_positions'],positions):
                raise ValueError('Archive token alignment failed')
            expected={f'model.layers.{i}__pass_{n}' for i in range(22) for n in (0,1)}
            expected|={f'model.norm__pass_{n}' for n in (0,1)}
            expected|={f'model.layers.{i}.mlp.down_proj.input__pass_{n}' for i in (11,21) for n in (0,1)}
            if set(a.files)!=expected|{'token_ids','token_positions'}:raise ValueError('Missing/extra activation site')
            for key in expected:
                width=10752 if '.input__pass_' in key else 3072
                if a[key].shape!=(len(positions),width) or a[key].dtype!=np.float32 or not np.isfinite(a[key]).all():
                    raise ValueError('Invalid activation shape/dtype/values')
            arrays[record['id']]={key:a[key].copy() for key in expected}
        for choice in record['conditional_greedy_checks']:
            if choice['expected_token']!=source['token_ids'][choice['position']+1]:raise ValueError('Greedy reference token mismatch')
            if choice['match']!=(choice['expected_token']==choice['argmax']):raise ValueError('Greedy agreement metadata mismatch')
            checked_choices+=1;matched_choices+=int(choice['match'])
    transferred=captures/'frozen_vpd_transfer'
    transfer_manifest=verified_manifest(transferred,'frozen_vpd_coordinate_transfer_complete')
    basis_path=ORIGINAL/'vpd_factors.npz'
    if sha(basis_path)!=transfer_manifest['source_factors_sha256']:raise ValueError('Frozen basis source mismatch')
    transfer_rows=json.loads((transferred/'summary.json').read_text())
    expected_sites={f'model.layers.{i}.mlp.down_proj' for i in (11,21)}
    if len(transfer_rows)!=2*len(records) or {(r['id'],r['site']) for r in transfer_rows}!={(key,site) for key in q for site in expected_sites}:
        raise ValueError('Incomplete VPD coordinate transfer')
    with np.load(basis_path,allow_pickle=False) as basis:
        for row in transfer_rows:
            site=row['site'];value=arrays[row['id']]
            with np.load(transferred/row['archive'],allow_pickle=False) as saved:
                if not np.array_equal(saved['token_positions'],position_map[row['id']]):raise ValueError('VPD token positions changed')
                c=[value[f'{site}.input__pass_{n}']@basis[site+'__V'] for n in (0,1)]
                y=[v@basis[site+'__U'] for v in c]
                for key,expected in [('coordinates_0',c[0]),('coordinates_1',c[1]),('delta_coordinates',c[1]-c[0]),
                    ('basis_output_0',y[0]),('basis_output_1',y[1]),('delta_basis_output',y[1]-y[0])]:
                    np.testing.assert_allclose(saved[key],expected,rtol=1e-5,atol=1e-4)
    dictionaries=run/'q4_reasoning_saes'
    sae_manifest=verified_manifest(dictionaries,'q4_reasoning_saes_complete')
    training={r['id'] for r in records if r['split']=='train'}
    evaluation={r['id'] for r in records if r['split']=='eval'}
    if training&evaluation or not training or not evaluation:raise ValueError('Invalid SAE splits')
    if set(sae_manifest['training_ids'])!=training or set(sae_manifest['evaluation_ids'])!=evaluation:
        raise ValueError('SAE training/evaluation identity mismatch')
    quality=json.loads((dictionaries/'sae_quality.json').read_text())
    if set(quality)!={'model.layers.11','model.layers.21','model.norm'}:raise ValueError('Missing shared SAE site')
    feature_archives=0;quality_checks=[]
    for site,entry in quality.items():
        path=dictionaries/entry['dictionary']
        if sha(path)!=entry['sha256']:raise ValueError('SAE dictionary hash mismatch')
        with np.load(path,allow_pickle=False) as saved:
            features,width=saved['encoder_weight'].shape
            sae=SharedSAE(width,features,int(saved['top_k']))
            with torch.no_grad():
                sae.encoder.weight.copy_(torch.from_numpy(saved['encoder_weight']))
                sae.encoder.bias.copy_(torch.from_numpy(saved['encoder_bias']))
                sae.decoder.weight.copy_(torch.from_numpy(saved['decoder'].T))
                sae.pre_bias.copy_(torch.from_numpy(saved['pre_bias']))
            sae.eval().requires_grad_(False)
        if len(entry['differences'])!=len(records) or {r['id'] for r in entry['differences']}!=set(q):
            raise ValueError('Incomplete SAE feature records')
        for row in entry['differences']:
            with np.load(dictionaries/row['archive'],allow_pickle=False) as saved:
                if not np.array_equal(saved['token_positions'],position_map[row['id']]):raise ValueError('SAE token positions changed')
                encoded=[encode_rows(sae,arrays[row['id']][f'{site}__pass_{n}'])[0] for n in (0,1)]
                for key,expected in [('pass_0',encoded[0]),('pass_1',encoded[1]),('delta_z',encoded[1]-encoded[0])]:
                    np.testing.assert_allclose(saved[key],expected,rtol=1e-5,atol=1e-4)
            feature_archives+=1
        for n in (0,1):
            heldout=np.concatenate([arrays[r['id']][f'{site}__pass_{n}'] for r in records if r['id'] in evaluation])
            _,measured=encode_rows(sae,heldout)
            observed=measured['explained_variance'];expected=entry['quality']['eval'][f'pass_{n}']['explained_variance']
            if observed is None or expected is None:
                if observed is not expected:raise ValueError('Undefined SAE quality differs')
            elif abs(observed-expected)>1e-5:raise ValueError('Reloaded SAE quality differs')
            quality_checks.append(dict(site=site,invocation=n,heldout_explained_variance=observed))
    result=dict(status='q4_reasoning_mechanistic_artifacts_verified',paired_generation_records=len(rows),
        native_trajectory_captures=len(records),activation_arrays=len(records)*50,
        vpd_coordinate_archives=len(transfer_rows),sae_feature_diff_archives=feature_archives,
        sae_reload_controls=quality_checks,conditional_greedy_checks=checked_choices,
        conditional_greedy_matches=matched_choices,
        callback_control_max_abs_logit_error=callback['max_abs_logit_error'],
        generation_manifest_sha256=sha(run/'manifest.json'),capture_manifest_sha256=sha(captures/'manifest.json'),
        generation_payload_files_verified=generation_files,
        sae_manifest_sha256=sha(dictionaries/'manifest.json'),
        limitations=['Frozen BF16 VPD coordinates on Q4 activations, not newly fitted VPD or causal answer effects.',
            'SAE differences are verified arithmetic; interpretation depends on held-out quality.',
            'Pilot trajectories include explicitly censored reasoning.'])
    (run/'mechanistic_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
