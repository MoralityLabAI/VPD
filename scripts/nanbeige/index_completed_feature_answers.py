"""Index completed answer steering without conflating it with activation capture."""
import json
from pathlib import Path
from experiments.nanbeige_reasoning.check_llama_steering import read,sha,write
from experiments.nanbeige_reasoning.run_q4_feature_answers import read_record


def main():
    project=Path.cwd().resolve();root=project/'artifacts/nanbeige_reasoning'
    run=root/'prime-20260908T200850Z-601'
    source=run/'retrieved/artifacts/nanbeige_reasoning/prime_q4_feature_answers_v1'
    contract=read(source/'contract.json');manifest=read(source/'manifest.json')
    assert manifest['status']=='q4_feature_answers_complete' and manifest['completed']==70
    assert manifest['contract_sha256']==sha(source/'contract.json')
    for name,digest in manifest['files'].items():assert sha(source/name)==digest
    grading_manifest=read(run/'grading_v2/manifest.json')
    for name,digest in grading_manifest['files'].items():assert sha(run/'grading_v2'/name)==digest
    assert grading_manifest['generation_manifest_sha256']==sha(source/'manifest.json')
    graded=read(run/'grading_v2/graded.json');lookup={(r['id'],r['arm']):r for r in graded}
    parents={r['id'] for r in map(json.loads,(root/'dataset_index_v1/examples.jsonl').read_text(encoding='utf-8').splitlines())}
    assert set(contract['evaluation_ids'])<=parents and len(lookup)==70
    records=[];assets={}
    def asset(path,role):
        name=path.relative_to(project).as_posix()
        assets[name]=dict(sha256=sha(path),bytes=path.stat().st_size,role=role)
        return name
    for index,qid in enumerate(contract['evaluation_ids']):
        for arm in contract['conditions']:
            path=source/f'{index:02d}_{arm["name"]}.json';row=read_record(path)
            assert row['complete'] and row['id']==qid and row['condition']==arm['name']
            grade=lookup[qid,arm['name']]
            assert grade['generated_tokens']==len(row['generated']) and grade['grade']['status']=='graded'
            records.append(dict(id=qid,condition=arm['name'],execution_variant='fresh_serial_q4_muon_feature_574',
                physical_layer=22,passes_zero_based=arm['passes'],alpha=arm['alpha'],
                generated_tokens=len(row['generated']),stopped_on_eog=row['stopped_on_eog'],
                max_token_censored=row['max_token_censored'],grade=grade['grade'],
                native_render_grade=grade['native_render_grade'],
                answer_asset=asset(path,'immutable_complete_answer_tokens_and_native_text'),
                generation_payload_sha256=row['payload_sha256']))
    receipts=[source/'contract.json',source/'manifest.json',source/'runtime.json',
        run/'grading_v2/manifest.json',run/'grading_v2/graded.json',run/'grading_v2/tables.json',
        run/'grading_transport_revision.json',run/'native_decode_audit_v3/verification.json',
        run/'completion_verification.json',run/'termination_verification.json',
        root/'dataset_index_v1/manifest.json']
    for p in receipts:asset(p,'provenance_and_verification')
    output=root/'answer_steering_index_v1';output.mkdir(exist_ok=False)
    write(output/'records.json',records);write(output/'assets.json',assets)
    (output/'README.md').write_text('# Completed Muon-feature answer steering\n\n'
        'This supplement indexes 70 complete fresh continuations on ten held-out inputs under seven conditions. '
        'Asset paths resolve from the project root. It links the original token IDs/native text, independent raw-decoding grades, '
        'native-rendering sensitivity grades, intervention settings and provenance receipts.\n\n'
        'These continuations are a distinct execution variant. Matching question IDs to `dataset_index_v1` does not align token positions '
        'or make the older activation/VPD/SAE arrays measurements of these new continuations. This supplement contains answer trajectories '
        'and grades; it adds no hidden-state captures. Pass indices 0 and 1 name repeated invocations of the same physical layer.\n\n'
        'All seven conditions score zero on eight complete evaluation units. Three conditions solve the same individual ARC test input, '
        'but each fails its companion input. There are 561,231 generated tokens and 62 token-capped outputs. '
        'The first Windows grading attempt is retained in `grading`; `grading_v2` fixes stdin transport and contains the final grades. '
        'The math verifier itself is unchanged.\n',encoding='utf-8')
    write(output/'manifest.json',dict(status='completed_answer_steering_index_verified',records=70,inputs=10,conditions=7,
        execution_variant='fresh_serial_q4_muon_feature_574',hidden_state_captures_added=0,
        builder_sha256=sha(Path(__file__)),generation_manifest_sha256=sha(source/'manifest.json'),
        grading_manifest_sha256=sha(run/'grading_v2/manifest.json'),
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='completed_answer_steering_index_verified',records=len(records),assets=len(assets))))


if __name__=='__main__':main()
