"""Freeze the user-selected Hermes skill and audit its public-input routing."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from experiments.nanbeige_reasoning.build_dataset import digest

ROOT = Path('artifacts/nanbeige_reasoning')
REV = '07a177f2751c2adb1cc16dc819c1173c0cd57ad5'
SKILL = 'intellect3-logic-hermes'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    out = ROOT / 'selected_logic_skill_v1'
    if (out/'verification.json').exists():
        raise ValueError('Completed selection already exists')
    out.mkdir(exist_ok=True)
    git = ['git', '-c', 'safe.directory=C:/projects/Hermes-Skills',
           '-C', 'C:/projects/Hermes-Skills']
    names = ['SKILL.md', 'references/contract.md', 'references/receipts.md',
             'references/learnings-3-27.md', 'references/logic_router_200.summary.json',
             'references/logic_hybrid_200.summary.json',
             'references/primehub_skill_imprint.latest.json',
             'scripts/build_skill_prompt.py', 'scripts/check_signature_gate.py']
    sources = {}
    for name in names:
        data = subprocess.run(git + ['show', f'{REV}:{SKILL}/{name}'],
                              capture_output=True, check=True).stdout
        target = out / 'source' / SKILL / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        sources[f'{SKILL}/{name}'] = sha(target)
    skill = out / 'source' / SKILL
    # Both scripts were reviewed before execution. These run without inference.
    prompt = subprocess.run([sys.executable, str(skill/'scripts/build_skill_prompt.py')],
                            capture_output=True, check=True).stdout
    (out/'skill_prompt.txt').write_bytes(prompt)
    selected = set(json.loads((skill/'references/logic_router_200.summary.json').read_text())
                   ['signature_gate']['selected_signatures'])
    qpath = ROOT/'dataset_v1/questions.jsonl'
    auditpath = ROOT/'hermes_audit/rows.jsonl'
    wanted = {json.loads(line)['id'] for line in auditpath.read_text().splitlines()}
    questions = [json.loads(line) for line in qpath.read_text(encoding='utf-8').splitlines()]
    pilot = {json.loads(line)['id']: json.loads(line)['calibration_split']
             for line in (ROOT/'pilot_v1/questions.jsonl').read_text(encoding='utf-8').splitlines()}
    records = []
    for q in questions:
        if q['id'] not in wanted:
            continue
        assert q['family'] == 'campsite'
        assert digest(q['prompt']) == q['prompt_sha256']
        vectors = {}
        for name in ('row_constraints', 'col_constraints'):
            found = re.findall(r'^'+name+r'\s*=\s*(\[[^\n]+\])\s*$', q['prompt'], re.M)
            assert len(found) == 1
            vectors[name] = ast.literal_eval(found[0])
            assert all(type(v) is int and v >= 0 for v in vectors[name])
        signature = json.dumps(vectors, separators=(',', ':'))
        result = subprocess.run([sys.executable, str(skill/'scripts/check_signature_gate.py'),
                                 '--rows', json.dumps(vectors['row_constraints']),
                                 '--cols', json.dumps(vectors['col_constraints'])],
                                capture_output=True, text=True, check=True)
        gate = dict(line.split('=', 1) for line in result.stdout.splitlines())
        expected = 'logic_skill_trm' if signature in selected else 'logic_skill'
        assert gate == dict(signature=signature, matched=str(signature in selected).lower(), route=expected)
        records.append(dict(id=q['id'], prompt=q['prompt'], prompt_sha256=q['prompt_sha256'],
                            signature=signature, route=expected, pilot_split=pilot.get(q['id']),
                            source_revision=q['source_revision']))
    assert len(records) == len(wanted) == 109
    assert len({r['id'] for r in records}) == 109
    (out/'public_inputs_and_routes.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    counts = dict(Counter(r['route'] for r in records))
    receipt = dict(status='selected_skill_frozen_routing_verified', selected_skill=SKILL,
                   revision=REV, user_selection='I guess the int-3 logic skill within',
                   updated_unix=time.time(), source_sha256=sources,
                   prompt_sha256=sha(out/'skill_prompt.txt'),
                   input_routes_sha256=sha(out/'public_inputs_and_routes.jsonl'),
                   input_source_sha256=sha(qpath), membership_source_sha256=sha(auditpath),
                   records=len(records), route_counts=counts,
                   pilot_routes=[{k:v for k,v in r.items() if k in ('id','route','pilot_split')}
                                 for r in records if r['pilot_split']],
                   selected_signatures=len(selected), gate_cli_checks=len(records),
                   grader_data_used_for_routing=False, model_calls=0,
                   inference_complete=False, new_paid_allocation=False,
                   pending_user_selection=None,
                   limitations=['Historical public cohort is exposure-audited, not a clean held-out benchmark.',
                                'Frozen skill scripts generate prompts and routes; they do not execute the TRM helpers.',
                                'New comparison requires helper integration and model inference; no new accuracy is reported.'],
                   preparer_sha256=sha(Path(__file__)))
    write(out/'verification.json', receipt)
    (out/'README.md').write_text(
        '# Selected INTELLECT-3 logic skill\n\n'
        f'User selected `{SKILL}` at pinned Hermes-Skills revision `{REV}`. '
        'The default prompt includes the pinned latest imprint. The gate uses the '
        'packaged router summary because no route-policy override is present.\n\n'
        f'All 109 public Campsite inputs passed the packaged gate CLI and independent signature check: {counts}. '
        'Routing reads only constraints parsed from the original public prompt. '
        'Reference answers and grader metadata are absent from inference inputs.\n\n'
        'Selection is resolved. This is preparation, with zero new model calls. '
        'The remaining work is wiring the contractual helpers into the matched model run, '
        'then inference and grading. Report this exposure-audited historical cohort separately '
        'from any fresh holdout. The skill targets Campsite, so this configuration does not establish ARC or math performance.\n',
        encoding='utf-8')
    ready = ROOT/'comparison_readiness_v1'
    write(ready/'selection_resolved.json', dict(selected_skill=SKILL, revision=REV,
          status='selection_resolved_preparation_complete', selection_receipt=str(out/'verification.json'),
          selection_receipt_sha256=sha(out/'verification.json'), supersedes='blocker_audit.json selection blocker',
          historical_original_prompt_gap_remains=True, pending_user_selection=None))
    status_path = ready/'verification.json'
    previous = json.loads(status_path.read_text())
    write(ready/'verification_before_skill_selection.json', previous)
    previous.update(status='skill_selected_fresh_comparison_prepared', pending_user_input=None,
                    selected_skill=SKILL, selection_receipt=str(out/'verification.json'),
                    updated_unix=time.time())
    write(status_path, previous)
    (ready/'STATUS.md').write_text(
        '# Comparator status\n\nThe user selected `intellect3-logic-hermes`. '
        'The pinned prompt and 109 public-input gate decisions are frozen in '
        '`../selected_logic_skill_v1/`. No further configuration choice is pending.\n\n'
        'Original historical model-facing prompts remain unavailable; the selected skill '
        'anchors a fresh run. Helper integration and inference remain outstanding. '
        'The previous blocker audit is retained as historical evidence and superseded by '
        '`selection_resolved.json`.\n', encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('status','records','route_counts','pilot_routes','model_calls')}))


if __name__ == '__main__':
    main()
