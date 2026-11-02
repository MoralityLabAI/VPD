"""Snapshot outputs, independently verify native cleanup, and seal evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from experiments.nanbeige_reasoning.answer_decoding import AnswerDecoder
from experiments.nanbeige_reasoning.native_text_cleanup import native_cleanup, SOURCE_REVISION
from experiments.nanbeige_reasoning.run_q4_feature_answers import read_record
from experiments.nanbeige_reasoning.check_llama_steering import sha, read, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--output-name',default='native_decode_audit_v2');args=parser.parse_args()
    output=args.run/args.output_name;output.mkdir(exist_ok=False)
    generated=args.run/'retrieved/artifacts/nanbeige_reasoning/prime_q4_feature_answers_v1'
    decoder=AnswerDecoder(Path('models/Nanbeige4.2-3B-tokenizer'))
    records=[]
    for p in sorted(generated.glob('[0-9][0-9]_*.json')):
        row=read_record(p)
        records.append(dict(name=p.name,tokens=row['generated'],saved_text=row['generated_text'],
            payload_sha256=row['payload_sha256'],complete=row['complete']))
    for i,text in enumerate(['', 'a . b ! c ? d ,', "I 'm he 's we 're they 've do n't I 'll I 'd",
        "a ' b ' c", '  .  !  ?  ,', 'row0: ..........\nrow1: 6 . 6', '你好 . café !',
        '<think>a .</think>\n[[1, 2], [3, 4]]</s>', " ' ' '  .!?", 'a  . , ! ?  b']):
        records.append(dict(name=f'synthetic_{i}',tokens=decoder.tokenizer.encode(text,add_special_tokens=False)))
    snapshot=output/'snapshot.json';write(snapshot,dict(records=records))
    with (output/'native.log').open('wb') as log:
        subprocess.run([sys.executable,'-m','scripts.nanbeige.audit_native_decoding',str(snapshot)],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    native=read(output/'native_decoding.json');assert native['snapshot_sha256']==sha(snapshot)
    assert len(native['records'])==len(records)
    comparisons=[]
    for row,actual in zip(records,native['records']):
        assert row['name']==actual['name']
        raw=decoder.decode(row['tokens'])
        assert raw==actual['raw_pieces'], row['name']+' raw token-piece disagreement'
        assert native_cleanup(raw)==actual['native'], row['name']+' cleanup mismatch'
        if 'saved_text' in row:assert actual['native']==row['saved_text'], row['name']+' saved-text mismatch'
        comparisons.append(dict(name=row['name'],tokens=len(row['tokens']),cleanup_changed_text=raw!=actual['native']))
    write(output/'comparisons.json',comparisons)
    source=Path('artifacts/nanbeige_reasoning/sources/llama_cpp/llama-vocab.cpp')
    write(output/'verification.json',dict(status='native_cleanup_independently_verified',
        source_revision=SOURCE_REVISION,source_sha256=sha(source),
        raw_decoder_sha256=sha(Path('experiments/nanbeige_reasoning/answer_decoding.py')),
        cleanup_sha256=sha(Path('experiments/nanbeige_reasoning/native_text_cleanup.py')),
        verifier_sha256=sha(Path(__file__)),records=len(records),
        policy='Preserve generated IDs and text. Verify native cleanup exactly; score original raw HF decoding and native text separately.',
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='native_cleanup_independently_verified',records=len(records),
        changed=sum(r['cleanup_changed_text'] for r in comparisons))))


if __name__=='__main__':main()
