"""Recover the evidence supporting historical math task identity.

Matching ordered answer labels alone is reported as provisional, not proof of
identical prompts. Available support-bank observations provide direct checks.
"""
import json
from pathlib import Path
from .fetch_sources import sha


def main():
    root=Path('artifacts/nanbeige_reasoning')
    archive=Path('D:/research_engine/trmObservabilityLogs/trm_observability_harness_snapshot/data')
    pred=archive/'qwen27b_intellect3_math_hybrid_200/predictions.jsonl'
    historical=[r for r in map(json.loads,pred.read_text(encoding='utf-8').splitlines()) if r['arm']=='vanilla']
    source=[r for r in map(json.loads,(root/'dataset_v1/questions.jsonl').read_text(encoding='utf-8').splitlines())
            if r['lane']=='intellect3_math']
    by_index={r['source_row']:r for r in source}
    graders={r['id']:r for r in map(json.loads,(root/'dataset_v1/graders.jsonl').read_text(encoding='utf-8').splitlines())
             if r['scorer']=='intellect3_math'}
    support_paths=[archive/'trm_math_intellect3_math_hybrid_support_25.jsonl',
                   archive/'trm_math_intellect3_math_hybrid_support_posonly.jsonl']
    observations={}
    for path in support_paths:
        for row in map(json.loads,path.read_text(encoding='utf-8').splitlines()):
            observations.setdefault(row['row_id'],[]).append(row['observation'])
    calibration={r['id'] for r in map(json.loads,(root/'benchmark_v1/questions.jsonl').read_text(encoding='utf-8').splitlines())
                 if r['calibration_split']=='train'}
    rows=[]
    for row in historical:
        index=int(row['row_id'].rsplit('_',1)[1])
        candidate=by_index[index]
        checks=[value==candidate['prompt'] for value in observations.get(row['row_id'],[])]
        rows.append(dict(historical_id=row['row_id'],candidate_id=candidate['id'],source_row=index,
            ordered_label_match=str(row['expected_action']).strip()==str(graders[candidate['id']]['gold']).strip(),
            available_prompt_checks=checks,direct_prompt_match=bool(checks) and all(checks),
            planned_calibration_overlap=candidate['id'] in calibration,
            identity_status='direct_support_prompt_match' if checks and all(checks) else 'provisional_ordered_id_and_label'))
    result=dict(rows=len(rows),ordered_labels_matched=sum(r['ordered_label_match'] for r in rows),
        directly_matched_prompts=sum(r['direct_prompt_match'] for r in rows),
        contradictory_prompt_checks=sum(not all(r['available_prompt_checks']) for r in rows),
        planned_calibration_overlaps=sum(r['planned_calibration_overlap'] for r in rows),
        missing_original_trajectory='C:/projects/Tesseract/Tesseract/data/normalized_trajectories/intellect_3_math.jsonl',
        provenance={str(p):sha(p) for p in [pred,*support_paths]},
        interpretation='Task order and labels support a first-200-source-row mapping, but unobserved historical prompts remain provisional. Output budgets remain unmatched. No model benchmark has been rerun here.')
    out=root/'hermes_audit'
    (out/'math_identity_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'math_identity_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2))


if __name__=='__main__':main()
