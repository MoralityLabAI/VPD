"""Locate native/HF rendering differences without modifying generation artifacts."""
import argparse
import json
from pathlib import Path
from experiments.nanbeige_reasoning.answer_decoding import AnswerDecoder
from experiments.nanbeige_reasoning.run_q4_feature_answers import read_record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('cloud_run',type=Path);args=parser.parse_args()
    run=args.cloud_run/'retrieved/artifacts/nanbeige_reasoning/prime_q4_feature_answers_v1'
    decoder=AnswerDecoder(Path('models/Nanbeige4.2-3B-tokenizer'));rows=[]
    for p in sorted(run.glob('[0-9][0-9]_*.json')):
        row=read_record(p);native=row['generated_text'];independent=decoder.decode(row['generated'])
        if native==independent:continue
        first=next((i for i,(a,b) in enumerate(zip(native,independent)) if a!=b),min(len(native),len(independent)))
        rows.append(dict(file=p.name,complete=row['complete'],tokens=len(row['generated']),
            native_length=len(native),independent_length=len(independent),first_difference=first,
            native_context=repr(native[max(0,first-70):first+100]),
            independent_context=repr(independent[max(0,first-70):first+100]),
            last_tokens=row['generated'][-16:],native_tail=repr(native[-80:]),independent_tail=repr(independent[-80:])))
    out=args.cloud_run/'long_decode_diagnostic.json';out.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows))

if __name__=='__main__':main()
