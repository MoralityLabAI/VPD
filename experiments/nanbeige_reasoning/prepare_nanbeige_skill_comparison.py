"""Prepare the same Campsite user messages for Nanbeige's native template."""
import argparse
import json
from pathlib import Path
import shutil

from transformers import AutoTokenizer
from experiments.nanbeige_receipt_replay.io import ROOT, digest, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=ROOT/'artifacts/nanbeige_reasoning/logic_skill_comparison_v1'
    cfg=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    assert digest(source/'calls.jsonl')==cfg['calls_sha256']
    assert digest(source/'cases.json')==cfg['cases_sha256']
    qroot=ROOT/'artifacts/nanbeige_reasoning/comparator_sources/qwen35_27b/official'
    for name,checksum in cfg['tokenizer_assets'].items():assert digest(qroot/name)==checksum
    qtok=AutoTokenizer.from_pretrained(qroot,local_files_only=True,use_fast=False)
    nroot=ROOT/'models/Nanbeige4.2-3B-tokenizer'
    ntok=AutoTokenizer.from_pretrained(nroot,local_files_only=True,use_fast=False)
    from tokenizers import Tokenizer
    decoder=Tokenizer.from_file(str(nroot/'tokenizer.json'))
    calls=[json.loads(line) for line in (source/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
    prepared=[]
    prefix='<|im_start|>user\n'
    suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    for call in calls:
        rendered=call['rendered_prompt']
        assert rendered.startswith(prefix) and rendered.endswith(suffix)
        assert qtok.decode(call['token_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==rendered
        content=rendered[len(prefix):-len(suffix)]
        messages=[dict(role='user',content=content)]
        assert qtok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)==rendered
        target=ntok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False,preserve_thinking=True)
        ids=ntok.encode(target,add_special_tokens=False)
        assert decoder.decode(ids,skip_special_tokens=False)==target
        assert target.endswith(suffix) and content in target
        prepared.append(dict(call_id=call['call_id'],id=call['id'],arm=call['arm'],messages=messages,
            user_content_sha256=__import__('hashlib').sha256(content.encode('utf-8')).hexdigest(),
            rendered_prompt=target,token_ids=ids,qwen_prompt_tokens=len(call['token_ids'])))
    assert len(prepared)==len({r['call_id'] for r in prepared})==259
    assert max(len(r['token_ids']) for r in prepared)+cfg['max_new_tokens']<=cfg['context']
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'calls.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prepared),encoding='utf-8')
    shutil.copy2(source/'cases.json',args.output/'cases.json')
    shutil.copytree(source/'helper_source',args.output/'helper_source')
    for name,checksum in cfg['helper_sources'].items():assert digest(args.output/'helper_source'/name)==checksum
    cfg.update(status='matched_nanbeige_inputs_prepared',model='Nanbeige4.2-3B',
        tokenizer_root=str(nroot.relative_to(ROOT)).replace('\\','/'),
        tokenizer_assets={p.name:digest(p) for p in nroot.iterdir() if p.is_file()},
        calls_sha256=digest(args.output/'calls.jsonl'),source_qwen_protocol_sha256=digest(source/'protocol.json'),
        source_qwen_calls_sha256=digest(source/'calls.jsonl'),max_prompt_tokens=max(len(r['token_ids']) for r in prepared),
        model_lock_sha256=None,loops=2,preparer_sha256=digest(Path(__file__)),
        input_message_identity='Exact rendered Qwen user contents, re-rendered with Nanbeige native template',
        inference_run=False,
        output_decoder='pinned tokenizer.json; exact special-token spacing',
        limitations=cfg['limitations']+[
            'Nanbeige native template adds its publisher default system message; full rendered prompts differ.',
            'Same response token cap is not equal compute; model tokenizers, size, runtime and recurrence differ.',
            'This matched user-message experiment covers Campsite only; ARC/math comparison remains incomplete.',
            'Answer-only condition is a comparator control, not the native-thinking ledger protocol.'])
    save(args.output/'protocol.json',cfg)
    print(json.dumps(dict(status=cfg['status'],calls=259,puzzles=109,max_prompt_tokens=cfg['max_prompt_tokens'],model_calls=0)))


if __name__=='__main__':main()
