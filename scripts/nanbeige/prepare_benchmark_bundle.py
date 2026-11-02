"""Freeze all 1,223 public model inputs for the full paired Nanbeige benchmark."""
import json
from pathlib import Path
import tarfile

from experiments.nanbeige_reasoning.batching import batch_plan
from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest, read, rows


def main():
    root=Path(__file__).resolve().parents[2]
    cohort=root/'artifacts/nanbeige_reasoning/benchmark_v1'
    manifest=read(cohort/'manifest.json')
    questions=rows(cohort/'questions.jsonl')
    if (digest(cohort/'questions.jsonl')!=manifest['files']['questions.jsonl']
            or len(questions)!=manifest['records'] or len(questions)!=1223
            or len({q['id'] for q in questions})!=len(questions)):
        raise ValueError('Full benchmark question cohort changed')
    plan=list(batch_plan(questions,8,32768,8192,65536))
    checkpoint=read(root/'models/Nanbeige4.2-3B-tokenizer/checkpoint_provenance.json')
    output=root/'artifacts/nanbeige_reasoning/full_benchmark_inputs_v1'
    output.mkdir(parents=True,exist_ok=True)
    contract=dict(schema='nanbeige.full_benchmark_inputs.v1',model_id=checkpoint['model_id'],
        revision=checkpoint['revision'],checkpoint_files=checkpoint['files'],
        questions='artifacts/nanbeige_reasoning/benchmark_v1/questions.jsonl',
        questions_sha256=digest(cohort/'questions.jsonl'),cohort_manifest_sha256=digest(cohort/'manifest.json'),
        examples=len(questions),expected_generation_records=2*len(questions),counts=manifest['counts'],
        loops=[2,1],max_new_tokens=8192,batch_size_limit=8,max_prefill_tokens=32768,max_total_tokens=65536,
        frozen_batches=[[q['id'] for q in batch] for batch in plan],
        full_prompt_tokens=sum(len(q['token_ids']) for q in questions),
        max_prompt_tokens=max(len(q['token_ids']) for q in questions),
        total_padded_token_upper_bound_per_pass=sum(len(batch)*(max(len(q['token_ids']) for q in batch)+8192) for batch in plan),
        limitations=['Public source holdouts, not pretraining-clean or private test data.',
                    'One-pass ablates a model trained with two passes.',
                    'Equal output caps do not imply equal compute.',
                    'No inference is completed by preparing this bundle.'])
    encoded=json.dumps(contract,indent=2)+'\n'
    path=output/'contract.json'
    if path.exists() and path.read_text(encoding='utf-8')!=encoded:
        raise ValueError('Existing full-benchmark contract differs')
    path.write_text(encoded,encoding='utf-8')
    files=sorted(p for p in (root/'experiments/nanbeige_looped').iterdir()
        if p.is_file() and (p.suffix in ('.py','.json','.txt') or p.name=='VPD_BACKEND_LICENSE'))
    files += [root/'experiments/nanbeige_reasoning'/name for name in
              ('__init__.py','batching.py','run_benchmark.py','run_capture.py','positions.py')]
    files += [root/'scripts/nanbeige/run_benchmark_remote.sh',cohort/'questions.jsonl',path]
    destination=root/'artifacts/nanbeige_reasoning/benchmark_bundle.tar.gz'
    temporary=destination.with_suffix('.tmp')
    with tarfile.open(temporary,'w:gz') as archive:
        for source in files:
            archive.add(source,arcname=source.relative_to(root).as_posix(),recursive=False)
    temporary.replace(destination)
    receipt=dict(sha256=digest(destination),files={p.relative_to(root).as_posix():digest(p) for p in files},
                 input_contract_path=path.relative_to(root).as_posix(),experiment_contract=contract)
    destination.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(files),bytes=destination.stat().st_size,sha256=receipt['sha256'],
        examples=len(questions),batches_per_pass=len(plan),expected_records=2*len(questions))))


if __name__=='__main__':
    main()
