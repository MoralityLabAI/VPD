"""Prepare archived support for fresh fixed cohorts using public-input exclusion.

An empty filtered bank is recorded explicitly; it is not a working retrieval arm.
Original archived rows and historical scores are never changed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from .audit_retrieval_exposure import ROOT, ARCHIVE, read, normalize, puzzle_key, observation_key
from .fetch_sources import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args=p.parse_args()
    config_path=ROOT/'hermes_audit/historical_config_audit.json'
    config=json.loads(config_path.read_text())
    question_path=ROOT/'dataset_v1/questions.jsonl'
    grader_path=ROOT/'dataset_v1/graders.jsonl'
    cohort_paths=[ROOT/f'{name}/questions.jsonl' for name in ('pilot_v1','benchmark_v1')]
    excluded={r['id'] for path in cohort_paths for r in read(path)}
    math_keys,logic_keys=defaultdict(set),defaultdict(set)
    for row in read(question_path):
        if row['lane']=='intellect3_math':
            math_keys[normalize(row['prompt'])].add(row['id'])
    # Only the public grid and constraints are used for identity; never solutions.
    for row in read(grader_path):
        if row.get('info',{}).get('task_name')=='campsite':
            meta=json.loads(row['info']['game_data_str'])['metadata']
            logic_keys[puzzle_key(meta['grid'],meta['row_constraints'],meta['col_constraints'])].add(row['id'])
    provenance={str(path):sha(path) for path in [config_path,question_path,grader_path,*cohort_paths,Path(__file__)]}
    summaries,exclusions={},[]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for lane in ('logic','math'):
        source=ARCHIVE/config[lane]['trm']['input_path']
        provenance[str(source)]=sha(source)
        bank=read(source)
        selected=[]
        for index,row in enumerate(bank):
            if row.get('task_family')!=lane:
                continue
            ids=(logic_keys.get(observation_key(row['observation']),set()) if lane=='logic'
                else math_keys.get(normalize(row['observation']),set()))
            overlap=ids&excluded
            if overlap:
                exclusions.append(dict(lane=lane,source_row=index,matched_cohort_ids=sorted(overlap)))
            else:
                selected.append(row)
        def eligible(row):
            return row.get('bucket') in ('exact_positive','near_miss','weak_positive') and bool(row.get('target_action')) and float(row.get('supervision_weight') or 0)>=config[lane]['trm']['min_supervision_weight']
        output=args.output_dir/f'{lane}.jsonl'
        output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
        # Reload the written bank and verify the public-input exclusion again.
        loaded=read(output)
        assert loaded==selected
        for row in loaded:
            ids=(logic_keys.get(observation_key(row['observation']),set()) if lane=='logic'
                else math_keys.get(normalize(row['observation']),set()))
            if ids&excluded:
                raise ValueError('Evaluation input retained in support')
        summaries[lane]=dict(source_rows=len(bank),source_family_rows=sum(r.get('task_family')==lane for r in bank),
            retained_critic_rows=len(loaded),retained_retriever_rows=sum(eligible(r) for r in loaded),
            excluded_rows=sum(r['lane']==lane for r in exclusions),
            family_counts=dict(Counter(r['task_family'] for r in loaded)),
            bank=output.name,sha256=sha(output),has_retrieval_support=any(eligible(r) for r in loaded))
    (args.output_dir/'exclusions.json').write_text(json.dumps(exclusions,indent=2))
    result=dict(status='fresh_cohort_support_filtered',cohort_public_ids=len(excluded),banks=summaries,
        provenance=provenance,files={f.name:sha(f) for f in args.output_dir.iterdir() if f.is_file()},
        limitations=['Exact public-input/alias exclusion only; semantic and pretraining exposure not ruled out.',
            'Empty logic support requires separate training data before claiming a fresh retrieval-augmented comparison.',
            'Retained archived math support is suitable only for these fresh cohorts; its historical evaluation overlap remains.',
            'No inference or neural controller training performed.'])
    (args.output_dir/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
