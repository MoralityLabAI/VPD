"""Audit archived retrieval-bank overlap without invoking any model or router.

Bank membership is distinguished from a recorded retrieval. Historical receipts
omit exemplar IDs, so equal answer strings cannot prove which example was used.
"""
import ast
import json
import re
from collections import Counter
from pathlib import Path
from .fetch_sources import sha


ROOT = Path('artifacts/nanbeige_reasoning')
ARCHIVE = Path('D:/research_engine/trmObservabilityLogs/trm_observability_harness_snapshot')


def read(path):
    return list(map(json.loads, path.read_text(encoding='utf-8').splitlines()))


def normalize(text):
    return re.sub(r'\s+', ' ', str(text).strip().lower())


def puzzle_key(grid, rows, columns):
    return json.dumps([grid, rows, columns], separators=(',', ':'))


def observation_key(text):
    # Same public fields as the archived Campsite parser; no target is involved.
    row = re.search(r'ROW_CONSTRAINTS:\s*(\[[^\n]+\])', text)
    col = re.search(r'COL_CONSTRAINTS:\s*(\[[^\n]+\])', text)
    grid = re.search(r'GRID:\n(.*?)\n\nROW_CONSTRAINTS:', text, re.S)
    if not all((row, col, grid)):
        return None
    return puzzle_key([line.split() for line in grid[1].strip().splitlines()],
                      ast.literal_eval(row[1]), ast.literal_eval(col[1]))


def main():
    config_path = ROOT/'hermes_audit/historical_config_audit.json'
    config = json.loads(config_path.read_text())
    question_path = ROOT/'dataset_v1/questions.jsonl'
    grader_path = ROOT/'dataset_v1/graders.jsonl'
    questions = {r['id']: r for r in read(question_path)}
    graders = {r['id']: r for r in read(grader_path)}
    cohort_path = ROOT/'benchmark_v1/questions.jsonl'
    cohort = {r['id']: r for r in read(cohort_path)}
    identity_path = ROOT/'hermes_audit/math_identity_rows.json'
    identity = json.loads(identity_path.read_text())
    history_math = {r['candidate_id']: r for r in identity}
    logic_audit_path = ROOT/'hermes_audit/rows.jsonl'
    logic_audit = read(logic_audit_path)
    history_logic = {r['id'] for r in logic_audit}
    math_by_prompt = {}
    for key, q in questions.items():
        if q['lane']=='intellect3_math':
            math_by_prompt.setdefault(normalize(q['prompt']), []).append(key)
    logic_by_puzzle = {}
    for key, g in graders.items():
        if g.get('info', {}).get('task_name')=='campsite':
            data = json.loads(g['info']['game_data_str'])
            meta = data['metadata']
            public = puzzle_key(meta['grid'], meta['row_constraints'], meta['col_constraints'])
            logic_by_puzzle.setdefault(public, []).append(key)
    records, summaries, exclusions = [], {}, []
    paths = [config_path, question_path, grader_path, cohort_path, identity_path, logic_audit_path]
    for lane in ('logic', 'math'):
        bank_path = ARCHIVE/config[lane]['trm']['input_path']
        paths.append(bank_path)
        bank = read(bank_path)
        minimum = config[lane]['trm']['min_supervision_weight']
        def match_ids(row):
            return (math_by_prompt.get(normalize(row['observation']), []) if lane=='math'
                    else logic_by_puzzle.get(observation_key(row['observation']), []))
        family_rows = [r for r in bank if lane!='logic' or r.get('task_family')=='logic']
        critic_ids = {key for row in family_rows for key in match_ids(row)}
        exclusions.extend(dict(lane=lane,id=key,calibration_split=cohort[key]['calibration_split'],
            reason='Public input appears in archived critic/source support bank')
            for key in sorted(critic_ids & cohort.keys()))
        eligible = [(i, r) for i, r in enumerate(bank)
                    if (lane!='logic' or r.get('task_family')=='logic')
                    and r.get('bucket') in ('exact_positive','near_miss','weak_positive')
                    and float(r.get('supervision_weight') or 0)>=minimum and r.get('target_action')]
        matched, unmatched = [], 0
        for index, row in eligible:
            ids = match_ids(row)
            unmatched += not bool(ids)
            for key in ids:
                entry = dict(lane=lane, bank_row=index, source_id=key,
                    historical_evaluation_member=key in (history_math if lane=='math' else history_logic),
                    benchmark_member=key in cohort,
                    benchmark_calibration_split=cohort.get(key,{}).get('calibration_split'),
                    public_input_match='normalized_prompt' if lane=='math' else 'grid_and_constraints',
                    target_equal=None)
                if lane=='math':
                    entry['target_equal']=str(row['target_action']).strip()==str(graders[key]['gold']).strip()
                matched.append(entry)
        records.extend(matched)
        summaries[lane] = dict(bank_rows=len(bank), eligible_rows=len(eligible), unmatched_eligible_rows=unmatched,
            critic_source_rows=len(family_rows),
            critic_historical_overlap_ids=len(critic_ids & (history_math.keys() if lane=='math' else history_logic)),
            matched_source_ids=len({r['source_id'] for r in matched}),
            historical_overlap_ids=len({r['source_id'] for r in matched if r['historical_evaluation_member']}),
            benchmark_overlap_ids=len({r['source_id'] for r in matched if r['benchmark_member']}))
    prediction_path = ARCHIVE/'data/qwen27b_intellect3_math_hybrid_200/predictions.jsonl'
    if sha(prediction_path)!=config['math']['predictions_sha256']:
        raise ValueError('Historical math predictions changed since the configuration audit')
    paths.append(prediction_path)
    all_predictions = read(prediction_path)
    predictions = [r for r in all_predictions if r['arm']=='math_skill_trm']
    exposed = {history_math[r['source_id']]['historical_id'] for r in records
               if r['lane']=='math' and r['historical_evaluation_member']}
    receipt_groups = {}
    for member in (True, False):
        selected = [r for r in predictions if (r['row_id'] in exposed)==member]
        receipt_groups['retriever_eligible_member' if member else 'not_matched_in_retriever_bank'] = dict(rows=len(selected),
            retrieved_action_present=sum(bool(r['trm'].get('retrieved_action')) for r in selected),
            final_exact=sum(str(r['final']['action']).strip()==str(r['expected_action']).strip() for r in selected),
            retrieved_action_equals_reference=sum(str(r['trm'].get('retrieved_action')).strip()==str(r['expected_action']).strip() for r in selected),
            all_arm_exact={arm:sum(str(r['final']['action']).strip()==str(r['expected_action']).strip()
                for r in all_predictions if r['arm']==arm and (r['row_id'] in exposed)==member)
                for arm in sorted({r['arm'] for r in all_predictions})},
            routes=dict(Counter(r['trm'].get('route_source') for r in selected)))
    logic_exposed = {r['source_id'] for r in records if r['lane']=='logic'}
    logic_groups = {}
    for member in (True, False):
        selected = [r for r in logic_audit if (r['id'] in logic_exposed)==member]
        logic_groups['retriever_eligible_member' if member else 'not_matched_in_retriever_bank'] = dict(rows=len(selected),
            arms={arm:{metric:sum(r['historical_arms'][arm][metric] for r in selected)
                       for metric in ('exact','official_valid')}
                  for arm in sorted(logic_audit[0]['historical_arms'])})
    paths.extend([Path(__file__), ARCHIVE/'scripts/bench_intellect3_math_hybrid.py',
        ARCHIVE/'scripts/bench_intellect3_logic_hybrid.py', ARCHIVE/'harness/trm_retrieval.py',
        ARCHIVE/'harness/trm_critic.py'])
    output = dict(status='bank_overlap_audited', banks=summaries, historical_math_receipts=receipt_groups,
        historical_logic_receipts=logic_groups,
        limitations=['Bank overlap alone is not evidence of direct target retrieval.',
            'Archived runners contain normalized-observation exclusion filters, but their execution for these receipts is not established; receipt exemplar IDs are absent.',
            'Positive support-bank membership is selected, so these partitions are descriptive rather than unbiased treatment effects.',
            'Historical math prompt identity is direct for 25 rows and provisional for the remaining 175.',
            'An unmatched exact public input does not establish absence of semantic or pretraining exposure.'],
        provenance={str(p):sha(p) for p in paths})
    dest = ROOT/'hermes_audit'
    (dest/'retrieval_exposure_rows.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (dest/'benchmark_support_exclusions.json').write_text(json.dumps(dict(
        benchmark_sha256=sha(cohort_path), records=exclusions,
        instruction='Keep the benchmark fixed. Exclude all matching public inputs from any fresh retrieval and critic support banks, including aliases; this file does not modify either bank.'),indent=2),encoding='utf-8')
    (dest/'retrieval_exposure_summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='provenance'},indent=2))


if __name__=='__main__':
    main()
