"""Extend the immutable pilot index with completed state-carry controls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/nanbeige_reasoning'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'dataset_index_v2')
    out = parser.parse_args().output.resolve()
    if out.exists():
        raise FileExistsError('Choose a fresh index directory; previous indices are immutable')
    old = BASE / 'dataset_index_v1'
    original = read(old / 'manifest.json')
    assert sha(old / 'examples.jsonl') == original['index_sha256']
    assert sha(old / 'assets.json') == original['assets_sha256']
    examples = [json.loads(line) for line in (old / 'examples.jsonl').read_text().splitlines()]
    by_id = {q['id']: q for q in examples}
    assets = read(old / 'assets.json')
    for name, item in assets.items():
        assert sha(ROOT / name) == item['sha256'], name

    def asset(path, role, expected=None):
        key = relative(path)
        digest = sha(path)
        if expected is not None:
            assert digest == expected, key
        if key in assets:
            assert assets[key]['sha256'] == digest, key
            assets[key]['roles'] = sorted(set(assets[key]['roles'] + [role]))
        else:
            assets[key] = dict(sha256=digest, bytes=path.stat().st_size, roles=[role])
        return key

    run = BASE / 'recurrence_state_control_v1'
    analysis = BASE / 'recurrence_state_analysis_v1'
    for directory in (run, analysis):
        manifest = read(directory / 'manifest.json')
        assert manifest['status'] == 'completed', directory
        for name, digest in manifest['files'].items():
            asset(directory / name, 'state_carry_source', digest)
        asset(directory / 'manifest.json', 'state_carry_manifest')
    protocol = read(run / 'protocol.json')
    records = read(run / 'records.json')
    assert len(records) == 450 and not protocol['canary']
    questions_path = BASE / 'pilot_v1/questions.jsonl'
    asset(questions_path, 'authoritative_question_source', protocol['source_questions_sha256'])
    questions = {q['id']: q for q in map(json.loads, questions_path.read_text().splitlines())}
    tokenizer_path = ROOT / 'models/Nanbeige4.2-3B-tokenizer/tokenizer.json'
    asset(tokenizer_path, 'token_text_decoder', '1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    contexts = []
    for entry in protocol['questions']:
        q = questions[entry['id']]
        tokens = entry['token_ids']
        assert tokens == q['token_ids'] and q['calibration_split'] == 'eval'
        assert q['id'] in by_id
        contexts.append(dict(id=q['id'], group_id=q['group_id'], lane=q['lane'],
            calibration_split=q['calibration_split'], source_split=q['source_split'],
            source_revision=q['source_revision'], prompt_sha256=q['prompt_sha256'],
            source_prompt=q['prompt'], token_ids=tokens,
            decoded_token_context=tokenizer.decode(tokens, skip_special_tokens=False),
            token_index=len(tokens)-1, token_id=tokens[-1],
            vocabulary_piece=tokenizer.id_to_token(tokens[-1]),
            decoded_token_piece=tokenizer.decode(tokens[-1:], skip_special_tokens=False),
            tokenizer=relative(tokenizer_path), generated_tokens=0))
    context_by_id = {q['id']: q for q in contexts}
    dictionary_sources = read(analysis / 'dictionary_sources.json')
    for name, digest in dictionary_sources.items():
        asset(ROOT / name, 'frozen_state_carry_dictionary', digest)
    feature_links = []
    for site, number in [('model.layers.11', 1), ('model.layers.21', 0), ('model.norm', 2)]:
        for optimizer in ('adamw', 'muon_hybrid'):
            path = analysis / f'encodings_{site}_{optimizer}.npz'
            with np.load(path, allow_pickle=False) as encoded:
                assert encoded['source_archives'].tolist() == [r['archive'] for r in records]
                for key in ('full_top_k', 'frozen_baseline_support', 'full_top_k_candidate_support'):
                    assert encoded[key].shape == (450, 6144)
            feature_links.append(dict(file=relative(path), site=site, optimizer=optimizer,
                dictionary=relative(BASE / 'muon_sae_control_v1' / f'{number:02d}_{optimizer}' / 'dictionary.npz'),
                encoding_keys=['full_top_k', 'frozen_baseline_support'],
                support_key='full_top_k_candidate_support'))
    baselines = {(r['id'], r['history']): r['archive'] for r in records if r['condition'] == 'baseline'}
    conditions = []
    state_count = 0
    for i, record in enumerate(records):
        context = context_by_id[record['id']]
        states = []
        with np.load(run / record['archive'], allow_pickle=False) as native:
            for invocation in range(record['loops']):
                for layer in range(22):
                    key = f'model.layers.{layer}__pass_{invocation}'
                    assert native[key].shape == (1, 3072) and np.isfinite(native[key]).all()
                    states.append(dict(key=key, physical_layer=layer+1, pass_number=invocation+1, row=0))
                key = f'model.norm__pass_{invocation}'
                assert native[key].shape == (1, 3072) and np.isfinite(native[key]).all()
                states.append(dict(key=key, physical_layer=None, site='loop_normalization', pass_number=invocation+1, row=0))
            if record['condition'] == 'baseline':
                assert native['token_ids'].tolist() == context['token_ids']
            else:
                assert native['write_before'].shape == native['write_after'].shape
        state_count += len(states)
        condition_id = Path(record['archive']).stem
        conditions.append(dict(condition_id=condition_id, input_id=record['id'],
            group_id=context['group_id'], lane=context['lane'], seed=protocol['seed'],
            history=record['history'], executed_passes=record['loops'],
            execution_variant='cpu_q4_prompt_end_state_carry_control',
            token_index=context['token_index'], token_id=context['token_id'],
            context_id=record['id'], native_archive=relative(run / record['archive']),
            baseline_archive=relative(run / baselines[record['id'], record['history']]),
            intervention=dict(kind=record['condition'], physical_layer=record.get('layer'),
                pass_number=record['loops'] if record['condition'] == 'write' else None,
                component=0 if record['condition'] == 'write' else None,
                signed_l2_dose=record.get('dose'), actual_write_norm=record.get('actual_write_norm'),
                direction_archive=relative(run / 'directions.npz')),
            states=states, sae_encodings=[dict(link, row=i, pass_number=record['loops']) for link in feature_links]))
    assert len(contexts) == 10 and len({q['group_id'] for q in contexts}) == 8
    assert len({r['condition_id'] for r in conditions}) == 450
    assert state_count == 17250
    for example in examples:
        example['state_carry_conditions'] = [r['condition_id'] for r in conditions if r['input_id'] == example['id']]
    assert sorted(Counter(r['input_id'] for r in conditions).values()) == [45]*10
    out.mkdir(parents=True)
    jsonl(out / 'examples.jsonl', examples)
    jsonl(out / 'state_carry_conditions.jsonl', conditions)
    jsonl(out / 'state_carry_contexts.jsonl', contexts)
    save(out / 'assets.json', assets)
    (out / 'README.md').write_text('''# Nanbeige mechanistic dataset, index v2

This index retains all 18 inputs and measurements from index v1 and adds the completed state-carry experiment on ten of those inputs (eight task groups). It adds 450 conditions, not 450 independent examples. The 18-input cohort remains five ARC-AGI-1, five ARC-AGI-2, four INTELLECT-3 logic and four INTELLECT-3 math inputs. This is a local measured pilot, not the full benchmark.

`examples.jsonl` retains earlier links and adds condition IDs. Join these to `state_carry_conditions.jsonl`; join each context ID to `state_carry_contexts.jsonl`. Assets resolve relative to the project root and carry SHA256 checksums in `assets.json`. No model loading is needed to inspect saved arrays.

Each condition specifies its authoritative prompt-end token position, native archive, same-history baseline, signed residual write, physical layer and pass. Archive keys use zero-based layer/pass indices; index fields use one-based numbers. Loop normalization has no physical-layer number. There are 17,250 layer/normalization vectors in total. Six dictionaries provide 2,700 linked encoding rows, each available with ordinary top-k and frozen-baseline-support encoding. These encodings cover the final executed pass at layers 12, 22 and loop normalization; native states cover every executed layer invocation.

The stored token IDs are the authoritative model input. Source prompt text and a tokenizer-decoded context are both retained; re-tokenizing source text is not asserted to reproduce the native input. Decoded token pieces can contain whitespace or byte fragments and are not semantic units. These are prompt-end states with no generated continuation, and must not be joined by row index to earlier Q4/BF16 continuation captures.

One-pass executes 22 layers. Carry and reset execute 44; reset replaces every token's boundary state with its original embedding. This controls state carry within one checkpoint, not the effect of training a looped architecture. Writes follow a normalized fitted VPD direction at the final token/final pass; they are not the earlier fixed-coefficient weight edits. Frozen feature IDs are dictionary-specific. See the linked analysis for state and change reconstruction quality before interpreting feature differences.

The native run and numerical validation are complete. The full ARC/logic/math benchmark comparison and ledger semantic experiment remain incomplete. This index establishes neither hidden chain-of-thought nor semantic localization. Existing source rights and the v1 execution-variant limitations remain applicable. No weights or activation data are copied or redistributed by this index.
''', encoding='utf-8')
    save(out / 'manifest.json', dict(status='verified', schema='nanbeige.dataset_index.v2',
        examples=18, new_unique_inputs=0, state_carry_inputs=10, state_carry_task_groups=8,
        conditions=450, native_state_vectors=state_count, sae_encoding_row_links=2700,
        inherited_assets=original['assets'], assets=len(assets), referenced_bytes=sum(a['bytes'] for a in assets.values()),
        source_index=relative(old), source_manifest_sha256=sha(old / 'manifest.json'),
        builder=relative(Path(__file__)), builder_sha256=sha(Path(__file__)),
        model_sha256=protocol['model_sha256'], runtime_files=protocol['runtime_files'],
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        full_benchmark_completed=False, ledger_episodes=0))
    print(json.dumps(dict(status='dataset_index_built', directory=str(out), examples=18, conditions=450, state_vectors=state_count, assets=len(assets))))


if __name__ == '__main__':
    main()
