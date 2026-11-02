"""Pin Qwen tokenizer assets and prepare identical public questions for 27B."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from transformers import AutoTokenizer
from .fetch_sources import sha
from .prepare_pilot import model_prompt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,default=Path('artifacts/nanbeige_reasoning/comparator_sources/qwen35_27b'))
    p.add_argument('--cohort',type=Path,default=Path('artifacts/nanbeige_reasoning/pilot_v1'))
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    lock=json.loads((args.sources/'model_lock.json').read_text())
    official=lock['official']
    assets=args.sources/'official'
    for name,entry in official['metadata'].items():
        if sha(assets/name)!=entry['sha256']:
            raise ValueError(f'Pinned official metadata changed: {name}')
    api=HfApi(token=False)
    files=set(api.list_repo_files(official['repo'],revision=official['revision']))
    selected=[name for name in ('vocab.json','merges.txt','tokenizer.json','special_tokens_map.json','added_tokens.json') if name in files]
    if not {'vocab.json','merges.txt'}.issubset(selected):
        raise ValueError('Missing assets for the pinned Qwen2 tokenizer')
    for name in selected:
        hf_hub_download(official['repo'],name,revision=official['revision'],local_dir=assets,token=False)
    tokenizer=AutoTokenizer.from_pretrained(assets,use_fast=False,local_files_only=True)
    source=args.cohort/'questions.jsonl'
    rows=list(map(json.loads,source.read_text(encoding='utf-8').splitlines()))
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate input IDs')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Choose an empty output directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in rows:
        content=model_prompt(row)
        messages=[dict(role='user',content=content)]
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        # The publisher template applies Jinja |trim to each message. Preserve
        # the original question and account for that boundary-only difference.
        expected='<|im_start|>user\n'+content.strip()+'<|im_end|>\n<|im_start|>assistant\n<think>\n'
        if rendered!=expected:
            raise ValueError(f'Unexpected template change to model-facing content: {row["id"]}')
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=True)
        decoded=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if decoded!=rendered:
            raise ValueError('Qwen tokenizer roundtrip differs from rendered prompt')
        record={k:v for k,v in row.items() if k!='token_ids'}
        record.update(token_ids=ids,model_prompt=content,rendered_prompt=rendered,
            nanbeige_prompt_tokens=len(row['token_ids']),comparator_model='Qwen3.5-27B',thinking_enabled=True,
            template_boundary_whitespace_trim=dict(leading=len(content)-len(content.lstrip()),trailing=len(content)-len(content.rstrip())))
        records.append(record)
        if len(records)%128==0:print(f'Tokenized {len(records)}/{len(rows)} unchanged question records',flush=True)
    target=args.output_dir/'questions.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    import transformers
    manifest=dict(status='comparator_inputs_prepared',records=len(records),
        source_questions_sha256=sha(source),source_cohort=str(args.cohort),questions_sha256=sha(target),
        official_model=official['repo'],official_revision=official['revision'],
        model_lock_sha256=sha(args.sources/'model_lock.json'),tokenizer_class=type(tokenizer).__name__,
        transformers_version=transformers.__version__,thinking_enabled=True,
        tokenizer_assets={name:sha(assets/name) for name in selected+['tokenizer_config.json','chat_template.jinja']},
        max_prompt_tokens=max(len(r['token_ids']) for r in records),
        total_prompt_tokens=sum(len(r['token_ids']) for r in records),
        identical_question_records=True,publisher_template_boundary_trim_records=sum(r['model_prompt']!=r['model_prompt'].strip() for r in records),
        no_nonboundary_content_changes=True,roundtrip_verified=True,grader_sidecar_copied=False,
        weights_downloaded=False,inference_run=False,
        limitation='Fresh 27B comparator. GGUF tokenizer/template agreement must be checked on the actual runtime before inference; this is not a historical reproduction.')
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
