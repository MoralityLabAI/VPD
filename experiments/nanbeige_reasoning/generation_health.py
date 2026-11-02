"""Descriptive token-repetition diagnostics, separate from task scoring."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def token_metrics(tokens):
    if not tokens:
        raise ValueError('Cannot characterize an empty continuation')
    counts = Counter(tokens)
    length = len(tokens)
    longest = run = 1
    for previous,current in zip(tokens,tokens[1:]):
        run = run+1 if current==previous else 1
        longest = max(longest,run)
    return dict(tokens=length,unique_tokens=len(counts),unique_fraction=len(counts)/length,
        unigram_entropy_bits=-sum((n/length)*math.log2(n/length) for n in counts.values()),
        largest_token_fraction=max(counts.values())/length,
        adjacent_repeat_fraction=sum(a==b for a,b in zip(tokens,tokens[1:]))/max(1,length-1),
        longest_identical_token_run=longest)


def write_health(run,questions):
    run = Path(run)
    source = run/'generations.jsonl'
    raw = source.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n')+1]
    generations = list(map(json.loads,raw.splitlines()))
    by_id = {q['id']:q for q in questions}
    observed = {(r['id'],r['loops']) for r in generations}
    if len(observed)!=len(generations):
        raise ValueError('Duplicate generation conditions')
    rows = []
    for row in generations:
        if row['id'] not in by_id or row['loops'] not in (1,2):
            raise ValueError('Unexpected generation condition')
        if row['token_ids'][:row['prompt_tokens']]!=by_id[row['id']]['token_ids']:
            raise ValueError('Generation prompt differs from source')
        tokens = row['token_ids'][row['prompt_tokens']:]
        if len(tokens)!=row['generated_tokens']:
            raise ValueError('Continuation length differs from receipt')
        rows.append(dict(id=row['id'],lane=row['lane'],loops=row['loops'],
            calibration_split=row['calibration_split'],censored=row['max_token_censored'],
            thinking_closed=row['thinking_closed'],metrics=token_metrics(tokens)))
    by_pair = {(r['id'],r['loops']):r for r in rows}
    pairs = []
    for key in by_id:
        if all((key,n) in by_pair for n in (1,2)):
            one,two = [by_pair[key,n] for n in (1,2)]
            pairs.append(dict(id=key,lane=one['lane'],calibration_split=one['calibration_split'],
                equal_generated_length=one['metrics']['tokens']==two['metrics']['tokens'],
                loops_1=one['metrics'],loops_2=two['metrics']))
    complete = observed=={(key,n) for key in by_id for n in (1,2)}
    result = dict(status='paired_generation_health_complete' if complete else 'partial_generation_health',
        observed_records=len(rows),expected_records=2*len(by_id),paired_inputs=len(pairs),
        source_complete_line_prefix_sha256=hashlib.sha256(raw).hexdigest(),rows=rows,pairs=pairs,
        interpretation='Continuous repetition diagnostics, not accuracy, semantic quality, reasoning compression, or a causal mechanism. Structured task outputs can naturally have low token diversity. Incomplete pairs cannot support cohort-wide comparisons.')
    (run/'generation_health.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--cohort',type=Path,default=Path('artifacts/nanbeige_reasoning/pilot_v1'))
    args=p.parse_args()
    result=write_health(args.run,list(map(json.loads,(args.cohort/'questions.jsonl').read_text(encoding='utf-8').splitlines())))
    print(json.dumps({key:result[key] for key in ('status','observed_records','expected_records','paired_inputs')},indent=2))
