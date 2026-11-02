"""Build a portable, hash-checked index of existing measurements; no inference."""
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/nanbeige_reasoning'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    out=DATA/'dataset_index_v1';out.mkdir(exist_ok=True)
    assets={}
    def asset(path,role,expected=None):
        path=path.resolve()
        if not path.is_relative_to(ROOT): raise ValueError('Index reference escapes project')
        key=path.relative_to(ROOT).as_posix()
        if key not in assets: assets[key]=dict(sha256=sha(path),bytes=path.stat().st_size,roles=[])
        if expected and assets[key]['sha256']!=expected: raise ValueError(f'Hash mismatch: {key}')
        if role not in assets[key]['roles']: assets[key]['roles'].append(role)
        return key
    def catalog_manifest(path,status):
        m=read(path/'manifest.json')
        assert m['status']==status
        asset(path/'manifest.json','source_manifest')
        return m
    run=DATA/'q4_reasoning_pilot_8192_v1';cap=run/'q4_activations'
    cm=catalog_manifest(cap,'q4_reasoning_capture_complete')
    qpath=DATA/'pilot_v1/questions.jsonl'
    asset(qpath,'public_questions',cm['generation_contract']['question_sha256'])
    # Grade metadata is indexed separately and is never copied into model inputs.
    asset(DATA/'pilot_v1/graders.jsonl','grading_sidecar')
    asset(DATA/'pilot_v1/manifest.json','cohort_selection')
    asset(DATA/'sources.lock.json','upstream_source_lock')
    asset(run/'mechanistic_verification.json','independent_verification')
    generation_path=asset(run/'generations.jsonl','paired_trajectories',cm['generation_source_sha256'])
    generation=rows(run/'generations.jsonl')
    lookup={(r['id'],r['loops']):r for r in generation}
    questions=rows(qpath);q={r['id']:r for r in questions}
    assert len(q)==18 and len(lookup)==len(generation)==36
    assert set(lookup)=={(key,loops) for key in q for loops in [1,2]}
    entries={}
    for question in questions:
        key=question['id']
        e={k:question[k] for k in ['id','lane','family','group_id','calibration_split','split','source_split','source_revision','prompt_sha256']}
        e.update(prompt_tokens=len(question['token_ids']),question_file=qpath.relative_to(ROOT).as_posix(),
            trajectories=[],saes=[],vpd_coordinates=[],frozen_sae_vpd_effects=[])
        for loops in [1,2]:
            r=lookup[key,loops]
            assert r['token_ids'][:e['prompt_tokens']]==question['token_ids']
            e['trajectories'].append(dict(file=generation_path,selector=dict(id=key,loops=loops),
                generated_tokens=r['generated_tokens'],max_token_censored=r['max_token_censored']))
        entries[key]=e
    asset(cap/'records.jsonl','capture_metadata',cm['files']['records.jsonl'])
    captures=rows(cap/'records.jsonl')
    assert {r['id'] for r in captures}==set(q) and len(captures)==18
    for r in captures:
        assert r['split']==q[r['id']]['calibration_split']
        entries[r['id']]['activations']=dict(file=asset(cap/r['archive'],'extended_q4_activations',r['sha256']),
            token_positions=r['positions'],regions=r['regions'],physical_layers=22,passes=[0,1],
            variant='extended_8192_native_two_pass',source_weights='Q4_K_M',stored_dtype='float32')
    coordinates=cap/'frozen_vpd_transfer';vm=catalog_manifest(coordinates,'frozen_vpd_coordinate_transfer_complete')
    basis8=ROOT/'artifacts/nanbeige_looped/prime-20260906T034844Z-719/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration/vpd_factors.npz'
    basis8_path=asset(basis8,'vpd8_factors',vm['source_factors_sha256'])
    asset(coordinates/'summary.json','vpd_metadata',vm['files']['summary.json'])
    for r in read(coordinates/'summary.json'):
        entries[r['id']]['vpd_coordinates'].append(dict(file=asset(coordinates/r['archive'],'extended_vpd8_coordinates',r['sha256']),
            site=r['site'],components=8,basis=basis8_path,variant='extended_8192_native_two_pass',source_manifest=(coordinates/'manifest.json').relative_to(ROOT).as_posix()))
    original=run/'q4_reasoning_saes';sm=catalog_manifest(original,'q4_reasoning_saes_complete')
    assert sm['capture_manifest_sha256']==assets[(cap/'manifest.json').relative_to(ROOT).as_posix()]['sha256']
    asset(original/'sae_quality.json','sae_quality',sm['files']['sae_quality.json'])
    for site,m in read(original/'sae_quality.json').items():
        dictionary=asset(original/m['dictionary'],'sae_dictionary',m['sha256'])
        for r in m['differences']:
            entries[r['id']]['saes'].append(dict(site=site,optimizer='adamw_original_cpu',dictionary=dictionary,
                file=asset(original/r['archive'],'extended_original_sae_features',r['sha256']),variant='extended_8192_native_two_pass'))
    control=DATA/'muon_sae_control_v1';mm=catalog_manifest(control,'sae_muon_control_complete')
    asset(DATA/'muon_sae_verification/complete.json','independent_verification')
    muon_contract=read(control/'contract.json');asset(control/'contract.json','sae_training_contract',mm['contract_sha256'])
    assert set(muon_contract['training_ids'])=={key for key,v in q.items() if v['calibration_split']=='train'}
    for directory in sorted(control.glob('*_*/manifest.json')):
        folder=directory.parent;m=read(directory)
        if m.get('status')!='sae_optimizer_condition_complete': continue
        asset(directory,'sae_condition_manifest',mm['files'][directory.relative_to(control).as_posix()])
        dictionary=asset(folder/'dictionary.npz','sae_dictionary',m['files']['dictionary.npz'])
        asset(folder/'records.json','sae_feature_metadata',m['files']['records.json'])
        for r in read(folder/'records.json'):
            assert r['split']==q[r['id']]['calibration_split']
            entries[r['id']]['saes'].append(dict(site=m['site'],optimizer=m['arm']+'_paired_cuda',dictionary=dictionary,
                file=asset(folder/r['archive'],'extended_optimizer_control_features',r['sha256']),variant='extended_8192_native_two_pass'))
    effects=DATA/'muon_vpd_transfer_v1';em=catalog_manifest(effects,'frozen_sae_vpd_transfer_verified')
    fit=DATA/'prime-20260906T204058Z-561/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration'
    fm=catalog_manifest(fit,'calibration_complete')
    assert sha(fit/'manifest.json')==read(effects/'contract.json')['fit_manifest_sha256']
    asset(fit/'examples.json','bf16_token_identity',fm['artifacts']['examples.json'])
    asset(fit/'effects.json','raw_vpd_intervention_metadata',fm['artifacts']['effects.json'])
    asset(fit/'independent_verification.json','independent_verification')
    asset(effects/'records.json','intervention_feature_metadata',em['files']['records.json'])
    asset(effects/'quality.json','transfer_reconstruction_quality',em['files']['quality.json'])
    asset(effects/'contract.json','intervention_transfer_contract',em['files']['contract.json'])
    for r in read(effects/'records.json'):
        key=r['example_id'];assert q[key]['calibration_split']=='eval'
        entries[key]['frozen_sae_vpd_effects'].append(dict(file=asset(effects/r['archive'],'bf16_vpd_frozen_sae_features',r['sha256']),
            variant='initial_2048_budget_bf16_replay_last8',sae_site=r['site'],optimizer=r['arm']+'_paired_cuda',
            dictionary=next(s['dictionary'] for s in entries[key]['saes'] if s['site']==r['site'] and s['optimizer']==r['arm']+'_paired_cuda'),
            intervention_module=r['physical_module'],component=r['component'],alpha=r['alpha'],
            source_archive=asset(fit/r['source_archive'],'raw_bf16_vpd_intervention',r['source_archive_sha256']),source_archive_sha256=r['source_archive_sha256']))
    # This earlier transfer belongs to a different capture, not the extended rows.
    earlier=DATA/'q4_reasoning_vpd64_transfer_v1';early_manifest=catalog_manifest(earlier,'reasoning_vpd_coordinate_transfer_verified')
    initial_cap=DATA/'q4_reasoning_pilot_v1/q4_activations'
    initial_capture_manifest=asset(initial_cap/'manifest.json','initial_capture_manifest',early_manifest['capture_manifest_sha256'])
    initial_capture_metadata=asset(initial_cap/'records.jsonl','initial_capture_metadata',early_manifest['capture_metadata_sha256'])
    basis64_path=asset(fit/'vpd_factors.npz','vpd64_factors',early_manifest['source_factors_sha256'])
    asset(earlier/'summary.json','vpd_metadata',early_manifest['files']['summary.json'])
    assert early_manifest['capture_manifest_sha256']!=sha(cap/'manifest.json')
    for r in read(earlier/'summary.json'):
        entries[r['id']]['vpd_coordinates'].append(dict(file=asset(earlier/r['archive'],'initial_vpd64_coordinates',r['sha256']),
            site=r['site'],components=64,basis=basis64_path,variant='initial_2048_budget_q4',
            source_capture_manifest=initial_capture_manifest,source_capture_metadata=initial_capture_metadata,
            source_manifest=(earlier/'manifest.json').relative_to(ROOT).as_posix()))
    index=list(entries.values())
    assert all(len(r['saes'])==9 and len(r['vpd_coordinates'])==4 for r in index)
    assert sum(len(r['frozen_sae_vpd_effects']) for r in index)==120
    for r in index: assert len(r['frozen_sae_vpd_effects'])==(12 if r['calibration_split']=='eval' else 0)
    for group in {r['group_id'] for r in index}:
        assert len({r['calibration_split'] for r in index if r['group_id']==group})==1
    (out/'examples.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in index),encoding='utf-8')
    write(out/'assets.json',assets)
    summary=dict(status='local_mechinterp_dataset_index_verified',schema='nanbeige.dataset_index.v1',
        examples=len(index),splits=dict(collections.Counter(r['calibration_split'] for r in index)),
        lanes=dict(collections.Counter(r['lane'] for r in index)),assets=len(assets),referenced_bytes=sum(r['bytes'] for r in assets.values()),
        counts=dict(activation_archives=18,paired_trajectories=36,sae_feature_archives=162,vpd_coordinate_archives=72,frozen_sae_vpd_effect_archives=120),
        path_root='project root, three parent directories above this index directory',
        index_sha256=sha(out/'examples.jsonl'),assets_sha256=sha(out/'assets.json'),builder_sha256=sha(Path(__file__)),
        limitations=['Indexes existing measurements; does not rerun their numerical verifiers or model inference.',
            'Initial and extended token-budget variants must not be joined by row index.',
            'A common question ID is not proof of matching token positions or activation precision.',
            'Local unpublished index; no additional redistribution rights are granted.'])
    write(out/'manifest.json',summary)
    print(json.dumps(summary))


if __name__=='__main__': main()
