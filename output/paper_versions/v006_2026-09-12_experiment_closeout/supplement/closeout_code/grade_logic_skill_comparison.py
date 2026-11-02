"""Grade fresh calls and apply frozen, reference-free commit-stage correction."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from functools import lru_cache
from transformers import AutoTokenizer
from .scoring import score,final_text,logic_api
from .audit_hermes import public_rule

ROOT=Path('artifacts/nanbeige_reasoning')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def paired(rows,a,b,allow_partial=False):
    by={(r['id'],r['arm']):r for r in rows}
    assert len(by)==len(rows), 'Duplicate graded rows'
    ids={r['id'] for r in rows}
    if allow_partial:ids={i for i in ids if (i,a) in by and (i,b) in by}
    else:assert all((i,a) in by and (i,b) in by for i in ids)
    states=dict(both_correct=0,a_only=0,b_only=0,both_incorrect=0)
    for i in ids:
        x,y=(by[i,arm]['official_score']==1 for arm in (a,b))
        key='both_correct' if x and y else 'a_only' if x else 'b_only' if y else 'both_incorrect'
        states[key]+=1
    return dict(a=a,b=b,pairs=len(ids),**states)


def source_key(case,arm):
    return case['id']+'::'+(case['route'] if arm in ('routed_candidate','routed_final') else
                           'logic_skill' if arm=='plain_candidate_routed_repair' else arm)


def validate_coverage(generations,calls,allow_partial):
    keys=list(generations)
    if allow_partial:
        if not keys or keys!=list(calls)[:len(keys)]:
            raise ValueError('Partial closeout requires a nonempty exact scheduled prefix')
    elif set(keys)!=set(calls):raise ValueError('All planned calls are required')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preparation',type=Path,default=ROOT/'logic_skill_comparison_v1')
    p.add_argument('--allow-partial',action='store_true',help='Explicit closeout of an interrupted scheduled prefix; missing calls remain unrun')
    args=p.parse_args();run=args.run
    preparation=args.preparation
    cfg=json.loads((preparation/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if args.allow_partial:assert manifest['status']=='interrupted'
    else:assert manifest['status']=='logic_skill_generation_complete' and manifest['records']==cfg['unique_model_calls']
    for name,h in manifest['files'].items(): assert sha(run/name)==h,name
    assert sha(preparation/'calls.jsonl')==cfg['calls_sha256']
    assert sha(preparation/'cases.json')==cfg['cases_sha256']
    contract=json.loads((run/'contract.json').read_text(encoding='utf-8'))
    assert contract['protocol_sha256']==sha(preparation/'protocol.json')
    assert json.loads((run/'cleanup.json').read_text(encoding='utf-8'))['server_stopped']
    calls={r['call_id']:r for r in map(json.loads,(preparation/'calls.jsonl').read_text(encoding='utf-8').splitlines())}
    cases=json.loads((preparation/'cases.json').read_text(encoding='utf-8'))
    tokenizer_root=Path(cfg['tokenizer_root']) if 'tokenizer_root' in cfg else ROOT/'comparator_sources/qwen35_27b/official'
    for name,checksum in cfg['tokenizer_assets'].items():assert sha(tokenizer_root/name)==checksum
    if cfg.get('output_decoder'):
        from tokenizers import Tokenizer
        decoder=Tokenizer.from_file(str(tokenizer_root/'tokenizer.json'))
        decode_output=lambda ids:decoder.decode(ids,skip_special_tokens=False)
    else:
        tok=AutoTokenizer.from_pretrained(tokenizer_root,local_files_only=True,use_fast=False)
        decode_output=lambda ids:tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    generations={}
    for f in sorted(run.glob('[0-9][0-9][0-9].json')):
        r=json.loads(f.read_text(encoding='utf-8'));c=calls[r['call_id']]
        assert r['call_id'] not in generations and r['id']==c['id'] and r['arm']==c['arm']
        assert r['contract_sha256']==sha(run/'contract.json')
        assert r['prompt_token_ids_sha256']==hashlib.sha256(json.dumps(c['token_ids']).encode()).hexdigest()
        assert r['prompt_tokens']==len(c['token_ids']) and r['generated_tokens']==len(r['generated_token_ids'])
        if cfg.get('output_decoder'):
            from experiments.nanbeige_receipt_replay.native_text import verify_native_text
            verify_native_text(decode_output(r['generated_token_ids']),r['generated_text'])
        else:assert decode_output(r['generated_token_ids'])==r['generated_text']
        r['source_file']=str(f);generations[r['call_id']]=r
    assert len(generations)==manifest['records']
    validate_coverage(generations,calls,args.allow_partial)
    helper=preparation/'helper_source/trm_logic_corrector.py'
    assert sha(helper)==cfg['helper_sources'][helper.name]
    spec=importlib.util.spec_from_file_location('frozen_logic_corrector',helper)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    corrector=module.TRMLogicCorrector(module.LogicCorrectorConfig(**cfg['repair']))
    # Produce every corrected answer before loading any reference sidecar.
    interventions=[]
    Data,verifiers=logic_api();verifier=verifiers['campsite']()
    for c in cases:
        for arm in ('routed_final','plain_candidate_routed_repair'):
            g=generations.get(source_key(c,arm))
            if g is None:continue
            candidate=verifier.extract_answer(final_text(g['generated_text']))
            start=time.perf_counter()
            action=(corrector.correct(c['helper_observation'],repr(candidate) if candidate else '')
                    if c['route']=='logic_skill_trm' else None)
            action=action or g['generated_text']
            interventions.append(dict(id=c['id'],arm=arm,action=action,
                source_call_id=g['call_id'],helper_applied=c['route']=='logic_skill_trm',
                helper_seconds=time.perf_counter()-start,
                changed_grid=verifier.extract_answer(final_text(action))!=candidate))
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    write(out/'corrections.json',interventions)
    grader_path=ROOT/'dataset_v1/graders.jsonl'
    graders={r['id']:r for r in map(json.loads,grader_path.read_text(encoding='utf-8').splitlines())}
    grade_data={c['id']:Data.from_json_str(graders[c['id']]['info']['game_data_str']) for c in cases}
    @lru_cache(maxsize=None)
    def grade_text(question_id,text):
        # Identical published verifier and final-text adapter as scoring.score.
        # logic_api installed its parsing guard above; avoid rewalking every
        # imported module for each shared candidate and rendering view.
        return dict(status='graded',score=float(verifier.verify(grade_data[question_id],final_text(text))))
    interventions={(r['id'],r['arm']):r for r in interventions}
    results=[]
    for c in cases:
        for arm in cfg['reported_arms']:
            key=source_key(c,arm)
            if key not in generations:continue
            g=generations[key]
            correction=interventions.get((c['id'],arm))
            text=correction['action'] if correction else g['generated_text']
            grade=grade_text(c['id'],text)
            assert grade['status']=='graded',grade
            native_grade=grade_text(c['id'],g['native_text'])
            assert native_grade['status']=='graded'
            parsed=verifier.extract_answer(final_text(text))
            meta=json.loads(graders[c['id']]['info']['game_data_str'])['metadata']
            valid_shape=bool(isinstance(parsed,list) and len(parsed)==c['public']['n']
                and all(isinstance(row,list) and len(row)==c['public']['m']
                        and all(v in ('T','X','C') for v in row) for row in parsed))
            literal=bool(valid_shape and public_rule(parsed,c['public'],True))
            results.append(dict(id=c['id'],arm=arm,route=c['route'],pilot_split=c['pilot_split'],
                official_score=grade['score'],canonical_exact=parsed==meta['solution'],literal_valid=literal,
                source_call_id=key,generated_tokens=g['generated_tokens'],max_token_censored=g['max_token_censored'],
                original_candidate_official_score=grade_text(c['id'],g['generated_text'])['score'],
                native_candidate_official_score=native_grade['score'],
                correction=correction,action=text))
    expected=sum(source_key(c,arm) in generations for c in cases for arm in cfg['reported_arms'])
    assert len(results)==expected and len({(r['id'],r['arm']) for r in results})==expected
    totals={}
    for arm in cfg['reported_arms']:
        rows=[r for r in results if r['arm']==arm]
        totals[arm]=dict(puzzles=len(rows),official_correct=sum(r['official_score'] for r in rows),
            canonical_exact=sum(r['canonical_exact'] for r in rows),literal_valid=sum(r['literal_valid'] for r in rows),
            capped_candidates=sum(r['max_token_censored'] for r in rows),
            source_generated_tokens=sum(r['generated_tokens'] for r in rows))
    pairs=[paired(results,a,b,args.allow_partial) for a,b in [('routed_final','vanilla'),('logic_skill','vanilla'),
        ('routed_final','routed_candidate'),('routed_final','plain_candidate_routed_repair')]]
    summary=dict(status='partial_logic_skill_comparison_graded' if args.allow_partial else 'fresh_logic_skill_comparison_graded',
        unique_model_calls=len(generations),puzzles=len({r['id'] for r in results}),
        planned_calls=len(calls),unrun_calls=len(calls)-len(generations),planned_puzzles=len(cases),
        missing_call_ids=[key for key in calls if key not in generations],
        actual_generated_tokens=sum(g['generated_tokens'] for g in generations.values()),
        total_capped_calls=sum(g['max_token_censored'] for g in generations.values()),
        candidate_native_grading_disagreements=sum(
            grade_text(g['id'],g['generated_text'])['score']!=grade_text(g['id'],g['native_text'])['score']
            for g in generations.values()),
        totals=totals,paired=pairs,pilot=[r for r in results if r['pilot_split']],
        limitations=cfg['limitations'])
    write(out/'rows.json',results);write(out/'summary.json',summary)
    report=['# Fresh Hermes INTELLECT-3 logic comparison','',
        f'{len(generations)} / {len(calls)} scheduled model calls; {cfg["model"]} Q4_K_M; answer-only; 1,024-token output cap. '
        'Unrouted skill outputs are reused exactly across routed arms. Missing calls are unrun, not incorrect.','',
        '| Arm | Official correct / observed puzzles | Canonical exact | Literal rule valid | Capped candidates |',
        '|---|---:|---:|---:|---:|']
    for arm,t in totals.items():report.append(f'| {arm} | {int(t["official_correct"])} / {t["puzzles"]} | {t["canonical_exact"]} | {t["literal_valid"]} | {t["capped_candidates"]} |')
    report+=['','The model candidate is preserved before correction. The plain-candidate repair arm isolates '
        'the commit-stage helper from the additional retrieved workspace. Reported arms share some calls; '
        'their token totals must not be added to estimate actual compute.','',*['- '+s for s in cfg['limitations']]]
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    write(out/'verification.json',dict(status='partial_grading_and_correction_complete' if args.allow_partial else 'grading_and_correction_complete',
        protocol_sha256=sha(preparation/'protocol.json'),generation_manifest_sha256=sha(run/'manifest.json'),
        source_grade_sidecar_sha256=sha(grader_path),helper_source_sha256=sha(helper),
        grader_implementation_sha256=sha(Path(__file__)),
        files={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pilot','limitations')},indent=2))


if __name__=='__main__':main()
