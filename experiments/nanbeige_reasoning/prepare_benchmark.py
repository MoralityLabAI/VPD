"""Prepare the full public ARC evaluation and fixed INTELLECT comparison cohorts.

Token lengths are measured without truncating or excluding long tasks. The runner
must report resource/length failures explicitly, not silently select easier rows.
"""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from .build_dataset import OUT,canonical,digest
from .fetch_sources import sha
from .prepare_pilot import model_prompt

def choose(rows,limit):
    return sorted(rows,key=lambda r:digest('benchmark-v1:'+r['group_id']))[:limit]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intellect-eval',type=int,default=200)
    parser.add_argument('--calibration-per-lane',type=int,default=32)
    parser.add_argument('--output-dir',type=Path,default=OUT.parent/'benchmark_v1')
    args=parser.parse_args()
    if args.intellect_eval<29 or args.calibration_per_lane<1:
        raise ValueError('Select enough logic evaluation examples to cover every family')
    all_rows=[json.loads(x) for x in (OUT/'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=[]
    # No prompt-length filtering or gold-outcome filtering in this cohort.
    selected += [r for r in all_rows if r['lane'].startswith('arc_') and r['split']=='eval']
    selected += choose([r for r in all_rows if r['lane']=='intellect3_math' and r['split']=='eval'],args.intellect_eval)
    logic=[r for r in all_rows if r['lane']=='intellect3_logic' and r['split']=='eval']
    families=defaultdict(list)
    for r in logic:
        families[r['family']].append(r)
    for family in families:
        families[family]=choose(families[family],len(families[family]))
    chosen=[]
    while len(chosen)<args.intellect_eval:
        advanced=False
        for family in sorted(families):
            if families[family]:
                chosen.append(families[family].pop(0))
                advanced=True
                if len(chosen)==args.intellect_eval:
                    break
        if not advanced:
            raise ValueError('Not enough logic examples')
    selected+=chosen
    # Historical Campsite comparison remains an additional, separately named slice.
    selected += [r for r in all_rows if r['split']=='historical_eval']
    used_groups=set()
    for lane in ('arc_agi_1','arc_agi_2','intellect3_logic','intellect3_math'):
        available=[r for r in all_rows if r['lane']==lane and r['split']=='train' and r['group_id'] not in used_groups]
        groups={r['group_id'] for r in choose(available,args.calibration_per_lane)}
        # Keep all test inputs of selected ARC training groups, too.
        selected += [r for r in available if r['group_id'] in groups]
        used_groups.update(groups)
    tokenizer=AutoTokenizer.from_pretrained('models/Nanbeige4.2-3B-tokenizer',trust_remote_code=True,use_fast=False)
    for index,r in enumerate(selected):
        r['token_ids']=tokenizer.apply_chat_template([{'role':'user','content':model_prompt(r)}],
            tokenize=True,add_generation_prompt=True,enable_thinking=True)
        r['calibration_split']='train' if r['split']=='train' else 'eval'
        if (index+1)%200==0:
            print('Tokenized',index+1,'/',len(selected),flush=True)
    assert len({r['id'] for r in selected})==len(selected)
    assert not ({r['group_id'] for r in selected if r['calibration_split']=='train'} &
                {r['group_id'] for r in selected if r['calibration_split']=='eval'})
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    (out/'questions.jsonl').write_text('\n'.join(map(canonical,selected))+'\n',encoding='utf-8')
    ids={r['id'] for r in selected}
    graders=[json.loads(x) for x in (OUT/'graders.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(x)['id'] in ids]
    (out/'graders.jsonl').write_text('\n'.join(map(canonical,graders))+'\n',encoding='utf-8')
    counts=Counter((r['lane'],r['split']) for r in selected)
    lengths=sorted(len(r['token_ids']) for r in selected)
    manifest=dict(kind='public_arc_and_fixed_intellect_cohort',records=len(selected),
        counts={f'{a}/{b}':v for (a,b),v in counts.items()},
        arc_eval_tasks={lane:len({r['group_id'] for r in selected if r['lane']==lane and r['split']=='eval'})
                        for lane in ('arc_agi_1','arc_agi_2')},
        eval_logic_families=dict(Counter(r['family'] for r in selected if r['lane']=='intellect3_logic' and r['split']=='eval')),
        length_filter=False,token_length_quantiles={str(q):lengths[round((len(lengths)-1)*q)] for q in (0,.5,.9,.99,1)},
        prompt_tokens=sum(lengths),seed_label='benchmark-v1',
        limits=['Family-stratified logic sample is not a source-frequency-weighted score.',
                'INTELLECT samples are local holdouts of public training data.',
                'Campsite historical evaluation remains separate from fresh logic sampling.',
                'No assertion that all selected examples have been run.'],
        source_manifest_sha256=sha(OUT/'manifest.json'),
        files={n:sha(out/n) for n in ('questions.jsonl','graders.jsonl')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    main()
