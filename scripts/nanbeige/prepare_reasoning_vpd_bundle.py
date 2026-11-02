"""Freeze verified Q4 trajectories for a separately labelled BF16 VPD fit."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    source = ROOT / 'artifacts/nanbeige_reasoning/q4_reasoning_pilot_v1'
    questions_path = ROOT / 'artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl'
    manifest = read(source / 'manifest.json')
    verification = read(source / 'mechanistic_verification.json')
    if (manifest['status'] != 'generation_complete'
            or verification['status'] != 'q4_reasoning_mechanistic_artifacts_verified'
            or verification['generation_manifest_sha256'] != digest(source / 'manifest.json')):
        raise ValueError('Source generation has not passed independent verification')
    for name in ('calibration_tokens.jsonl', 'generations.jsonl', 'contract.json'):
        if digest(source / name) != manifest['files'][name]:
            raise ValueError(f'Source payload changed: {name}')
    if digest(questions_path) != manifest['contract']['question_sha256']:
        raise ValueError('Source questions changed')
    examples = rows(source / 'calibration_tokens.jsonl')
    questions = {r['id']: r for r in rows(questions_path)}
    native_rows = [r for r in rows(source / 'generations.jsonl') if r['loops'] == 2]
    native = {r['id']: r for r in native_rows}
    if len(native) != len(native_rows) or len(examples) != 18 or set(native) != {r['id'] for r in examples}:
        raise ValueError('Incomplete or duplicated native source cohort')
    for row in examples:
        generation, question = native[row['id']], questions[row['id']]
        if (row['token_ids'] != generation['token_ids']
                or row['token_ids'][:row['prompt_tokens']] != question['token_ids']
                or row['prompt_tokens'] != len(question['token_ids'])
                or row['split'] != question['calibration_split']
                or row['split'] != generation['calibration_split']
                or row['family'] != question['family']
                or row['max_token_censored'] != generation['max_token_censored']
                or not 0 < row['prompt_tokens'] < len(row['token_ids']) <= 4096):
            raise ValueError(f'Trajectory provenance mismatch: {row["id"]}')
    if sum(r['split'] == 'train' for r in examples) != 8:
        raise ValueError('Unexpected training cohort')
    for field in ('pair_id', 'token_ids'):
        groups = [{json.dumps(r[field]) for r in examples if r['split'] == split}
                  for split in ('train', 'eval')]
        if groups[0] & groups[1]:
            raise ValueError(f'Train/evaluation overlap: {field}')
    output = ROOT / 'artifacts/nanbeige_reasoning/cross_precision_vpd_inputs_v2'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = read(ROOT / 'models/Nanbeige4.2-3B-tokenizer/checkpoint_provenance.json')
    contract = {
        'schema': 'nanbeige.cross_precision_vpd_inputs.v1',
        'source_generation_weights': 'Q4_K_M', 'measurement_weights': 'BF16',
        'model_id': checkpoint['model_id'], 'revision': checkpoint['revision'],
        'checkpoint_files': checkpoint['files'],
        'dataset': 'artifacts/nanbeige_reasoning/q4_reasoning_pilot_v1/calibration_tokens.jsonl',
        'dataset_sha256': digest(source / 'calibration_tokens.jsonl'),
        'generation_manifest_sha256': digest(source / 'manifest.json'),
        'generation_sha256': digest(source / 'generations.jsonl'),
        'source_verification_sha256': digest(source / 'mechanistic_verification.json'),
        'examples': len(examples), 'train_examples': 8, 'eval_examples': 10,
        'total_tokens': sum(len(r['token_ids']) for r in examples),
        'censored_examples': sum(r['max_token_censored'] for r in examples),
        'physical_layers_one_based': [12, 22], 'passes': 2,
        'components_per_matrix': 64, 'warmup_steps': 50, 'vpd_steps': 50,
        'sae_steps': 1000, 'training_prefix_fractions': [0.25, 0.5, 1.0],
        'full_prior_context_retained': True, 'use_cache': False,
        'saved_tensors_on_cpu': True,
        'scope': 'BF16 teacher-forced VPD fitting on fixed Q4-generated trajectories',
        'limitations': ['Not BF16 self-generated reasoning or Q4 parameter fitting.',
                       'Source continuations are token-budget censored.',
                       'Component effects are next-token proxies, not answer accuracy.',
                       'No claim about Astra internals or adaptive neural routing.'],
    }
    contract_path = output / 'contract.json'
    encoded = json.dumps(contract, indent=2, allow_nan=False) + '\n'
    if contract_path.exists() and contract_path.read_text(encoding='utf-8') != encoded:
        raise ValueError('Existing frozen input contract differs; choose a new version')
    contract_path.write_text(encoded, encoding='utf-8')
    files = sorted(p for p in (ROOT / 'experiments/nanbeige_looped').iterdir()
                   if p.is_file() and (p.suffix in ('.py', '.json', '.txt') or p.name == 'VPD_BACKEND_LICENSE'))
    files += [source / 'calibration_tokens.jsonl', contract_path,
              ROOT / 'scripts/nanbeige/run_reasoning_vpd_remote.sh']
    dest = ROOT / 'artifacts/nanbeige_reasoning/reasoning_vpd_bundle.tar.gz'
    temporary = dest.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    temporary.replace(dest)
    receipt = {'sha256': digest(dest), 'files': {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
               'input_contract_path': contract_path.relative_to(ROOT).as_posix(),
               'experiment_contract': contract}
    dest.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(files), 'bytes': dest.stat().st_size, 'sha256': receipt['sha256'],
                      'examples': len(examples), 'total_tokens': contract['total_tokens']}))


if __name__ == '__main__':
    main()
