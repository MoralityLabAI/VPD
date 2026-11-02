"""Independently resolve every v2 condition/encoding link against saved sources."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, default=ROOT / 'artifacts/nanbeige_reasoning/dataset_index_v2')
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    if a.receipt.exists():
        raise FileExistsError('Preserve previous verification receipts')
    manifest = read(a.index / 'manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(a.index / name) == digest, name
    assets = read(a.index / 'assets.json')
    for name, entry in assets.items():
        assert sha(ROOT / name) == entry['sha256'], name
    examples = lines(a.index / 'examples.jsonl')
    contexts = {r['id']: r for r in lines(a.index / 'state_carry_contexts.jsonl')}
    conditions = lines(a.index / 'state_carry_conditions.jsonl')
    by_condition = {r['condition_id']: r for r in conditions}
    assert len(by_condition) == len(conditions) == 450
    source = ROOT / 'artifacts/nanbeige_reasoning/recurrence_state_control_v1'
    protocol = read(source / 'protocol.json')
    native = {r['archive']: r for r in read(source / 'records.json')}
    authoritative_tokens = {q['id']: q['token_ids'] for q in protocol['questions']}
    baselines = {(r['id'], r['history']): name for name, r in native.items() if r['condition'] == 'baseline'}
    feature_selectors = {}
    feature_checks = state_checks = 0
    for example in examples:
        assert set(example['state_carry_conditions']) == {r['condition_id'] for r in conditions if r['input_id'] == example['id']}
    for r in conditions:
        actual = native[Path(r['native_archive']).name]
        assert r['input_id'] == actual['id'] and r['history'] == actual['history']
        assert Path(r['baseline_archive']).name == baselines[actual['id'], actual['history']]
        context = contexts[r['context_id']]
        assert context['token_ids'] == authoritative_tokens[actual['id']]
        assert r['token_index'] == len(context['token_ids'])-1
        assert r['token_id'] == context['token_ids'][r['token_index']]
        assert r['intervention']['physical_layer'] == actual.get('layer')
        assert r['intervention']['signed_l2_dose'] == actual.get('dose')
        expected = {(layer, invocation) for layer in range(1, 23) for invocation in range(1, actual['loops']+1)}
        assert {(s['physical_layer'], s['pass_number']) for s in r['states'] if s['physical_layer']} == expected
        with np.load(ROOT / r['native_archive'], allow_pickle=False) as arrays:
            for s in r['states']:
                site = f"model.layers.{s['physical_layer']-1}" if s['physical_layer'] else 'model.norm'
                assert s['key'] == f"{site}__pass_{s['pass_number']-1}"
                assert arrays[s['key']][s['row']].shape == (3072,)
                state_checks += 1
        for link in r['sae_encodings']:
            if link['file'] not in feature_selectors:
                with np.load(ROOT / link['file'], allow_pickle=False) as encoded:
                    feature_selectors[link['file']] = encoded['source_archives'].tolist()
                    for key in link['encoding_keys']:
                        assert encoded[key].shape == (450, 6144)
            assert feature_selectors[link['file']][link['row']] == Path(r['native_archive']).name
            assert link['pass_number'] == actual['loops']
            assert link['dictionary'] in assets
            feature_checks += 1
    # An executable paired read: same prompt/site/dose, reset final pass vs one-pass.
    pairs = 0
    for r in conditions:
        if r['history'] != 'reset':
            continue
        ref = by_condition[r['condition_id'].replace('reset_', 'one_pass_', 1)]
        assert r['input_id'] == ref['input_id'] and r['token_index'] == ref['token_index']
        with np.load(ROOT / r['native_archive'], allow_pickle=False) as x, np.load(ROOT / ref['native_archive'], allow_pickle=False) as y:
            for layer in range(22):
                np.testing.assert_array_equal(x[f'model.layers.{layer}__pass_1'], y[f'model.layers.{layer}__pass_0'])
            np.testing.assert_array_equal(x['model.norm__pass_1'], y['model.norm__pass_0'])
            np.testing.assert_array_equal(x['logits'], y['logits'])
        pairs += 1
    assert (state_checks, feature_checks, pairs) == (17250, 2700, 150)
    receipt = dict(status='verified', index_manifest_sha256=sha(a.index / 'manifest.json'),
        verifier_sha256=sha(Path(__file__)), assets_checked=len(assets), condition_links=len(conditions),
        native_state_links=state_checks, feature_row_links=feature_checks, exact_reset_pairs=pairs,
        examples=len(examples), independent_state_carry_inputs=len(contexts),
        lanes=dict(Counter(q['lane'] for q in contexts.values())),
        groups=len({q['group_id'] for q in contexts.values()}), inference_performed=False)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
