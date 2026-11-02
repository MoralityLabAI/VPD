"""Deterministic, length-bounded feasibility subset; not a full benchmark score."""
import json
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from .build_dataset import OUT, digest, canonical
from .fetch_sources import sha

def model_prompt(row):
    # Logic formats are task-specific: adding universal answer tags would break
    # the official direct-string graders (for example, Dyck language).
    if row['lane']=='intellect3_math':
        return row['prompt'] + '\nPut your final answer in \\boxed{...}.'
    return row['prompt']

def main():
    tokenizer = AutoTokenizer.from_pretrained('models/Nanbeige4.2-3B-tokenizer',
        trust_remote_code=True, use_fast=False)
    rows = [json.loads(x) for x in (OUT/'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    selected, rejected = [], []
    used_training_groups = set()
    for lane in ('arc_agi_1','arc_agi_2','intellect3_logic','intellect3_math'):
        for split in ('train','eval'):
            candidates = [r for r in rows if r['lane']==lane and r['split']==split]
            if lane=='intellect3_logic' and split=='eval':
                candidates += [r for r in rows if r['lane']==lane and r['split']=='historical_eval']
            groups = defaultdict(list)
            for r in candidates:
                groups[r['group_id']].append(r)
            ordered = sorted(groups, key=lambda g: digest('pilot-v1:' + g))
            # Include one historical Campsite case to check compatibility with Hermes.
            if lane=='intellect3_logic' and split=='eval':
                ordered.sort(key=lambda g: groups[g][0]['family'] != 'campsite')
            chosen = 0
            families = set()
            for group in ordered:
                if split=='train' and group in used_training_groups:
                    continue
                items = groups[group]
                if lane=='intellect3_logic' and items[0]['family'] in families:
                    continue
                tokenized = []
                for r in items:
                    ids = tokenizer.apply_chat_template([{'role':'user','content':model_prompt(r)}],
                        tokenize=True, add_generation_prompt=True, enable_thinking=True)
                    tokenized.append({**r, 'token_ids':ids, 'calibration_split':split})
                if max(len(r['token_ids']) for r in tokenized) > 2048:
                    rejected.append({'lane':lane,'group':group,'reason':'prompt exceeds 2048 tokens'})
                    continue
                selected.extend(tokenized)
                if split=='train':
                    used_training_groups.add(group)
                chosen += 1
                families.add(items[0]['family'])
                if chosen == 2:
                    break
            if chosen != 2:
                raise ValueError(f'Not enough eligible groups: {lane}/{split}')
    ids = {r['id'] for r in selected}
    graders = [json.loads(x) for x in (OUT/'graders.jsonl').read_text(encoding='utf-8').splitlines()
               if json.loads(x)['id'] in ids]
    destination = OUT.parent/'pilot_v1'
    destination.mkdir(exist_ok=True)
    for name, values in [('questions.jsonl',selected),('graders.jsonl',graders)]:
        (destination/name).write_text('\n'.join(map(canonical,values))+'\n',encoding='utf-8')
    manifest = dict(kind='feasibility_subset', seed_label='pilot-v1', requested_groups_per_lane_split=2,
        records=len(selected), max_prompt_tokens=2048, max_new_tokens=2048,
        selection_uses_gold=False, rejected_groups=rejected,
        dataset_manifest_sha256=sha(OUT/'manifest.json'),
        ids=[r['id'] for r in selected], files={n:sha(destination/n) for n in ('questions.jsonl','graders.jsonl')})
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'records':len(selected),'rejected_groups':len(rejected),
        'examples':[(r['lane'],r['family'],r['calibration_split'],len(r['token_ids'])) for r in selected]},indent=2))

if __name__=='__main__':
    main()
