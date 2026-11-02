"""Independently check saved partial grades without model inference or repair."""
import json
from pathlib import Path
from .verify_hermes_controller_control import validate
from .scoring import final_text,logic_api
from experiments.nanbeige_receipt_replay.io import digest,save


def main():
    root=Path('artifacts/nanbeige_reasoning')
    grade=root/'nanbeige_skill_partial_grading_v1'
    prep=root/'nanbeige_skill_comparison_inputs_v1'
    out=root/'partial_closeout_validation_v1'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    v=read(grade/'verification.json')
    for name,sha in v['files'].items():assert digest(grade/name)==sha
    assert digest(prep/'protocol.json')==v['protocol_sha256']
    cases={c['id']:c for c in read(prep/'cases.json')}
    _,verifiers=logic_api();extract=verifiers['campsite']().extract_answer
    rows=read(grade/'rows.json');checks=[]
    for r in rows:
        grid=extract(final_text(r['action']))
        official,literal=validate(cases[r['id']]['public'],grid)
        assert official==bool(r['official_score']),(r['id'],r['arm'],'official')
        assert literal==r['literal_valid'],(r['id'],r['arm'],'literal')
        checks.append(dict(id=r['id'],arm=r['arm'],official=official,literal=literal))
    summary=read(grade/'summary.json')
    for arm,t in summary['totals'].items():
        subset=[r for r in rows if r['arm']==arm]
        assert len(subset)==t['puzzles'] and sum(r['official_score'] for r in subset)==t['official_correct']
    assert len(rows)==96
    out.mkdir(parents=True,exist_ok=False)
    save(out/'rows.json',checks)
    save(out/'verification.json',dict(status='partial_grades_independently_verified',rows=len(rows),
        official_and_literal_checks=2*len(rows),new_model_calls=0,
        rule='Independent bit-mask tent/tree matching plus grid, counts and adjacency checks; existing published answer extraction.',
        grading_sha256=digest(grade/'verification.json'),validator_sha256=digest(Path(__file__)),
        rule_implementation_sha256=digest(Path('experiments/nanbeige_reasoning/verify_hermes_controller_control.py')),
        files={'rows.json':digest(out/'rows.json')}))
    print(json.dumps(dict(status='independent_checks_passed',graded_rows=len(rows),predicate_checks=2*len(rows))))


if __name__=='__main__':main()
