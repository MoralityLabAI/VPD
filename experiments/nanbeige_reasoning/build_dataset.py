"""Build a question-only corpus and separate private-to-grader sidecars.

INTELLECT partitions are local holdouts of public RL training data, not official
test sets. Historical Campsite rows and embedded ARC rows are quarantined.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from .fetch_sources import ROOT, HF_REV, REVISIONS, sha

OUT = ROOT.parent / 'dataset_v1'

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def local_split(group):
    bucket = int(digest(group)[:8], 16) % 10
    return 'eval' if bucket == 0 else 'dev' if bucket == 1 else 'train'

def build_arc(sources):
    raw = []
    for lane, directory in sources.items():
        for split in ('training', 'evaluation'):
            for path in sorted((directory / 'data' / split).glob('*.json')):
                task = json.loads(path.read_text())
                public = {'train': task['train'], 'test': [{'input': x['input']} for x in task['test']]}
                raw.append((lane, split, path.stem, task, digest(public)))
    # Connected groups join IDs AND identical public tasks across both releases.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a,b)] = min(a,b)
    for _, _, task_id, _, signature in raw:
        union('id:' + task_id, 'public:' + signature)
    heldout = {find('id:' + task_id) for _, split, task_id, _, _ in raw if split == 'evaluation'}
    records, graders, exclusions = [], [], []
    for lane, split, task_id, task, signature in raw:
        group = find('id:' + task_id)
        partition = 'eval' if split == 'evaluation' else 'quarantine' if group in heldout else 'train'
        if partition == 'quarantine':
            exclusions.append({'lane': lane, 'task_id': task_id, 'reason': 'ARC evaluation alias in another release'})
        for test_index, test in enumerate(task['test']):
            ident = f'{lane}/{task_id}/{test_index}'
            prompt = ('Infer the grid transformation from the examples. Solve the test input. '
                      'Return the final grid as a JSON array of arrays inside <answer>...</answer>.\n'
                      + canonical({'examples': task['train'], 'test_input': test['input']}))
            records.append(dict(id=ident, lane=lane, family='arc_grid', split=partition,
                group_id=group, task_id=task_id, test_index=test_index, prompt=prompt,
                prompt_sha256=digest(prompt), source_split=split,
                source_revision=REVISIONS[lane][1], public_task_sha256=signature))
            graders.append(dict(id=ident, scorer='arc_exact_grid', gold=test['output']))
    return records, graders, exclusions

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, graders, exclusions = build_arc({lane: ROOT / lane for lane in ('arc_agi_1', 'arc_agi_2')})
    for subset in ('logic', 'math'):
        for index, row in enumerate(pq.read_table(ROOT / 'intellect3' / (subset + '.parquet')).to_pylist()):
            question = row['question']
            signature = digest(question)
            info = json.loads(row['info']) if subset == 'logic' else {}
            family = info.get('task_name', 'math')
            partition = local_split(signature)
            if family == 'campsite':
                partition = 'historical_eval'
            elif family in ('arc_agi', 'arc_agi_2'):
                partition = 'quarantine'
            ident = f'intellect3_{subset}/{index:05d}/{signature[:12]}'
            records.append(dict(id=ident, lane='intellect3_' + subset, family=family,
                split=partition, group_id='question:' + signature, source_row=index,
                prompt=question, prompt_sha256=signature, source_split='train', source_revision=HF_REV))
            graders.append(dict(id=ident, scorer='intellect3_' + subset, **(
                {'info': info} if subset == 'logic' else {'gold': row['answer']})))
    # Identical questions are never allowed to span training and any heldout set.
    heldout_hashes = {r['prompt_sha256'] for r in records if r['split'] in ('eval','dev','historical_eval','quarantine')}
    for r in records:
        if r['split'] == 'train' and r['prompt_sha256'] in heldout_hashes:
            r['split'] = 'quarantine'
    for name, rows in [('questions.jsonl', records), ('graders.jsonl', graders)]:
        with (OUT / name).open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(canonical(row) + '\n')
    counts = Counter((r['lane'], r['split']) for r in records)
    manifest = dict(schema='nanbeige.reasoning.dataset.v1', records=len(records),
        counts={f'{a}/{b}': n for (a,b), n in sorted(counts.items())},
        logic_families=dict(Counter(r['family'] for r in records if r['lane']=='intellect3_logic')),
        exclusions=exclusions, question_only_fields=True,
        files={name: sha(OUT/name) for name in ('questions.jsonl','graders.jsonl')},
        source_lock_sha256=sha(ROOT.parent/'sources.lock.json'),
        limitations=['INTELLECT local holdouts are from public RL training data.',
            'Exact question/ARC alias de-duplication does not establish pretraining decontamination.',
            'Historical math exposure is not yet mapped; no fresh TRM generalization claim.',
            'All 109 Campsite rows reserved as historical evaluation; embedded ARC rows quarantined.',
            'ARC evaluation is public only. All test inputs of a task share a partition.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k in ('records','counts','logic_families')},indent=2))

if __name__ == '__main__':
    main()
