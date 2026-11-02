"""Grade complete causal answer arms locally, after exact token re-decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare_models import outcome_tables
from .fetch_sources import sha
from .scoring import score


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--fit-input-contract', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, default=Path('models/Nanbeige4.2-3B-tokenizer'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.run / 'manifest.json')
    if manifest['status'] != 'answer_interventions_complete':
        raise ValueError('Answer intervention cohort is incomplete')
    for name, expected in manifest['files'].items():
        path = (args.run / name).resolve()
        if path.parent != args.run.resolve() or sha(path) != expected:
            raise ValueError(f'Generation payload changed: {name}')
    contract = manifest['contract']
    qpath, gpath = args.cohort / 'questions.jsonl', args.cohort / 'graders.jsonl'
    if sha(qpath) != contract['question_sha256'] or sha(args.fit_input_contract) != contract['fit_input_contract_sha256']:
        raise ValueError('Source contract or questions differ')
    source = read(args.fit_input_contract)
    tokenizer_assets = ('tokenizer.model', 'tokenizer.json', 'tokenizer_config.json',
                        'added_tokens.json', 'special_tokens_map.json', 'config.json')
    for name in tokenizer_assets:
        if sha(args.tokenizer_dir / name) != source['checkpoint_files'][name]:
            raise ValueError(f'Tokenizer asset changed: {name}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False, trust_remote_code=True)
    questions = [json.loads(s) for s in qpath.read_text(encoding='utf-8').splitlines()]
    graders = [json.loads(s) for s in gpath.read_text(encoding='utf-8').splitlines()]
    q, g = {r['id']: r for r in questions}, {r['id']: r for r in graders}
    if len(q) != len(questions) or len(g) != len(graders) or set(q) != set(g):
        raise ValueError('Duplicate or missing cohort identity')
    selected = [r['id'] for r in questions if r['calibration_split'] == 'eval']
    if selected != contract['evaluation_ids']:
        raise ValueError('Held-out cohort selection changed')
    arms = {a['name']: a for a in contract['conditions']}
    if len(arms) != len(contract['conditions']) or 'baseline' not in arms:
        raise ValueError('Invalid condition list')
    records = [json.loads(s) for s in (args.run / 'generations.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = [(qid, arm) for qid in selected for arm in arms]
    if ([(r['id'], r['condition']) for r in records] != expected
            or manifest['records'] != len(expected) or manifest['expected_records'] != len(expected)):
        raise ValueError('Incomplete, duplicated or reordered generation arms')
    for row in records:
        question = q[row['id']]
        prefix = question['token_ids']
        if (row['intervention'] != arms[row['condition']] or row['component'] != contract['component']
                or row['loops'] != 2 or row['prompt_tokens'] != len(prefix)
                or row['token_ids'][:len(prefix)] != prefix
                or len(row['token_ids']) != len(prefix) + row['generated_tokens']
                or row['model_forward_calls'] != row['generated_tokens']
                or any(row[key] != question[key] for key in ('lane', 'family', 'split', 'calibration_split'))):
            raise ValueError('Answer generation provenance mismatch')
        text = tokenizer.decode(row['token_ids'][len(prefix):], skip_special_tokens=False)
        prompt_tail = tokenizer.decode(prefix[-8:], skip_special_tokens=False)
        scoring = ('<think>' if '<think>' in prompt_tail and '</think>' not in prompt_tail else '') + text
        if text != row['generated_text'] or scoring != row['scoring_text']:
            raise ValueError('Saved scoring text differs from exact token decoding')
    if args.output_dir.exists():
        raise ValueError('Choose a new grading output directory')
    args.output_dir.mkdir(parents=True)
    graded = []
    with (args.output_dir / 'graded.jsonl').open('w', encoding='utf-8') as stream:
        for row in records:
            item = dict(id=row['id'], arm=row['condition'], grade=score(g[row['id']], row['scoring_text']),
                max_token_censored=row['max_token_censored'], generated_tokens=row['generated_tokens'],
                elapsed_seconds=row['elapsed_seconds'])
            graded.append(item)
            stream.write(json.dumps(item, allow_nan=False) + '\n')
            stream.flush()
    tables = outcome_tables(questions, graded, list(arms))
    tables['baseline_pairs'] = [r for r in tables['pairs'] if r['left'] == 'baseline' or r['right'] == 'baseline']
    (args.output_dir / 'tables.json').write_text(json.dumps(tables, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    result = dict(status='answer_interventions_graded', records=len(records), conditions=len(arms),
        generation_manifest_sha256=sha(args.run / 'manifest.json'), graders_sha256=sha(gpath),
        question_sha256=sha(qpath), implementation_sha256={name:sha(Path(__file__).with_name(name))
            for name in ('grade_answer_interventions.py', 'compare_models.py', 'scoring.py', 'safe_expression.py')},
        limits=['Pilot causal effects on answers; single component and coefficient only.',
                'ARC success requires every held-out test input of each selected task.',
                'Censored answers and grader errors remain explicit.',
                'No model-scale, feature-semantic or Astra-routing conclusion.'],
        files={f.name:sha(f) for f in args.output_dir.iterdir() if f.is_file()})
    (args.output_dir / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','records','conditions')}))


if __name__ == '__main__':
    main()
