"""Resolve exact decision-token boundaries separately from the ggml process."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from .capture_decisions import decision_prefix
from .io import ROOT,digest,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new capture-input path')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Nanbeige4.2-3B-tokenizer',use_fast=True,local_files_only=True)
    rows=[]
    for path in sorted(a.episodes.resolve().glob('*.json')):
        r=json.loads(path.read_text(encoding='utf-8'));event=r['events'][0]
        if not event.get('response'):continue
        tokens,anchors,status=decision_prefix(event['response'],tokenizer)
        for anchor in anchors:
            token=tokens[anchor['token_index']]
            anchor.update(token_id=token,token_piece=tokenizer.convert_ids_to_tokens(token),decoded_token_text=tokenizer.decode([token],skip_special_tokens=False))
        rows.append(dict(source_episode=str(path.relative_to(ROOT)),source_episode_sha256=digest(path),tokens=tokens,anchors=anchors,status=status))
    save(a.output,rows);print('Prepared',len(rows),'decision prefixes')


if __name__=='__main__':main()
