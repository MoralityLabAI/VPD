"""Identify historical comparator coverage without treating labels as prompts."""
import ast
import json
from pathlib import Path
from experiments.nanbeige_reasoning.check_llama_steering import read, sha, write
from experiments.nanbeige_reasoning.audit_retrieval_exposure import observation_key,puzzle_key


def main():
    root=Path('artifacts/nanbeige_reasoning');audit=root/'hermes_audit'
    qpath=root/'pilot_v1/questions.jsonl';gpath=root/'pilot_v1/graders.jsonl'
    questions=list(map(json.loads,qpath.read_text(encoding='utf-8').splitlines()))
    graders={r['id']:r for r in map(json.loads,gpath.read_text(encoding='utf-8').splitlines())}
    historical={r['id']:r for r in map(json.loads,(audit/'rows.jsonl').read_text(encoding='utf-8').splitlines())}
    config=read(audit/'historical_config_audit.json')
    archive=Path('D:/research_engine/trmObservabilityLogs/trm_observability_harness_snapshot')
    raw=archive/'data/qwen27b_intellect3_logic_hybrid_200/predictions.jsonl'
    assert sha(raw)==config['logic']['predictions_sha256']==read(audit/'summary.json')['provenance']['historical_predictions_sha256']
    predictions=list(map(json.loads,raw.read_text(encoding='utf-8').splitlines()))
    bank_path=archive/config['logic']['trm']['input_path']
    bank=list(map(json.loads,bank_path.read_text(encoding='utf-8').splitlines()))
    math={r['candidate_id']:r for r in read(audit/'math_identity_rows.json')}
    coverage=[]
    for q in questions:
        if q['calibration_split']!='eval':continue
        item=dict(id=q['id'],lane=q['lane'],split=q['split'],status='no_verified_historical_prompt_match')
        if q['id'] in historical:
            meta=json.loads(graders[q['id']]['info']['game_data_str'])['metadata']
            target=meta['solution']
            matches=[r for r in predictions if (ast.literal_eval(r['expected_action']) if isinstance(r['expected_action'],str) else r['expected_action'])==target]
            assert len(matches)==4 and len({r['row_id'] for r in matches})==1
            assert {r['arm'] for r in matches}==set(historical[q['id']]['historical_arms'])
            assert all(not {'observation','prompt','question'} & set(r) for r in matches)
            key=puzzle_key(meta['grid'],meta['row_constraints'],meta['col_constraints'])
            support=[i for i,r in enumerate(bank) if observation_key(r.get('observation',''))==key]
            item.update(status='candidate_match_by_reference_target_only',
                historical_row_id=matches[0]['row_id'],original_prediction_prompts_available=False,
                archived_support_public_input_matches=support,
                historical_arms=historical[q['id']]['historical_arms'],
                limits='Public puzzle recovered in support bank does not prove which prompt reached each historical model call; support exposure also prevents a clean holdout interpretation.')
        elif q['id'] in math:
            item.update(status=math[q['id']]['identity_status'],historical_math_identity=math[q['id']])
        coverage.append(item)
    output=root/'pilot_historical_coverage_v1';output.mkdir(exist_ok=False)
    write(output/'coverage.json',coverage)
    sources=[qpath,gpath,audit/'rows.jsonl',audit/'summary.json',audit/'historical_config_audit.json',
        audit/'math_identity_rows.json',raw,bank_path]
    write(output/'manifest.json',dict(status='historical_pilot_coverage_audited',evaluation_inputs=len(coverage),
        target_only_candidates=sum(r['status']=='candidate_match_by_reference_target_only' for r in coverage),
        verified_original_prediction_prompt_matches=0,
        source_sha256={str(p):sha(p) for p in sources},coverage_sha256=sha(output/'coverage.json'),
        auditor_sha256=sha(Path(__file__)),
        scope='No model run or new scoring. No join by answer value alone is labeled a matched benchmark.'))
    print(json.dumps(dict(status='historical_pilot_coverage_audited',records=len(coverage),
        target_only_candidates=sum(r['status']=='candidate_match_by_reference_target_only' for r in coverage))))


if __name__=='__main__':main()
