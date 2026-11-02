"""Run archived Campsite components with fresh synthetic retrieval support.

These are controller-only ablations, with zero language-model calls. They do
not reproduce a full 27B+skill prompt or assert neural TRM performance.
"""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import types
from .audit_hermes import public_rule
from .audit_retrieval_exposure import ROOT, ARCHIVE, read
from .build_campsite_support import symmetry_key, observation
from .fetch_sources import sha
from .scoring import logic_api


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--support',type=Path,default=ROOT/'hermes_synthetic_campsite_v1')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.support/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(args.support/name)!=digest:
            raise ValueError('Synthetic support hash mismatch')
    puzzles=read(args.support/'puzzles.jsonl')
    bank=read(args.support/'logic_support.jsonl')
    training={r['id'] for r in puzzles if r['split']=='train'}
    if {r['id'] for r in bank}!=training or len(bank)!=len(training):
        raise ValueError('Only the complete training split may enter support')
    seen=set()
    Data,verifiers=logic_api()
    official=verifiers['campsite']()
    for row in puzzles:
        public=row['public']
        key=symmetry_key(public['grid'],public['row_constraints'],public['col_constraints'])
        if key in seen or not official.verify(Data(question='',answer='',metadata=public),repr(row['solution'])) or not public_rule(row['solution'],public,True):
            raise ValueError('Duplicate or invalid synthetic puzzle')
        seen.add(key)
    cases=[dict(id=r['id'],slice='synthetic_dev',public=r['public'],target=r['solution'])
        for r in puzzles if r['split']=='dev']
    source_path=ROOT/'dataset_v1/graders.jsonl'
    for row in read(source_path):
        if row.get('info',{}).get('task_name')!='campsite':
            continue
        metadata=json.loads(row['info']['game_data_str'])['metadata']
        public={k:metadata[k] for k in ('grid','row_constraints','col_constraints','n','m')}
        if symmetry_key(public['grid'],public['row_constraints'],public['col_constraints']) in seen:
            raise ValueError('Public puzzle or symmetry alias entered synthetic support/dev')
        cases.append(dict(id=row['id'],slice='public_campsite',public=public,target=metadata['solution']))
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    module_path=out/'source'
    module_path.mkdir()
    audit=json.loads((ROOT/'hermes_audit/controller_implementation_audit.json').read_text())
    for name in ('trm_retrieval.py','trm_logic_corrector.py'):
        source=ARCHIVE/'harness'/name
        if sha(source)!=audit['sources']['harness/'+name]['sha256']:
            raise ValueError('Archived controller source changed')
        shutil.copy2(source,module_path/name)
    package=types.ModuleType('vpd_hermes_control')
    package.__path__=[str(module_path.resolve())]
    sys.modules[package.__name__]=package
    retrieval=importlib.import_module('vpd_hermes_control.trm_retrieval')
    repair=importlib.import_module('vpd_hermes_control.trm_logic_corrector')
    retriever=retrieval.TRMRetriever.from_rows(bank,min_supervision_weight=.4)
    if len(retriever.examples)!=len(bank):
        raise ValueError('Generated support did not enter the actual archived retriever')
    arms={'retrieval_only':None,'exact_empty':(True,8,False),'exact_retrieval':(True,8,True)}
    arms.update({f'local_retrieval_passes_{n}':(False,n,True) for n in (0,1,2,4,8)})
    records=[]
    for index,case in enumerate(cases):
        public=case['public']
        obs=observation(public)
        query=dict(task_family='logic',task='intellect_3_logic',observation=obs,
            support_signature=json.dumps(dict(row_constraints=public['row_constraints'],col_constraints=public['col_constraints']),separators=(',',':')),
            response_mode='action_only',skill_name='intellect3-logic-hermes')
        start=time.perf_counter()
        retrieved,example,similarity=retriever.predict(query)
        retrieval_seconds=time.perf_counter()-start
        candidates={}
        for arm,config in arms.items():
            start=time.perf_counter()
            if config is None:
                action=retrieved
            else:
                exact,passes,use_retrieval=config
                corrector=repair.TRMLogicCorrector(repair.LogicCorrectorConfig(
                    exact_search=exact,max_passes=passes,exact_node_limit=200000))
                action=corrector.correct(obs,retrieved if use_retrieval else '')
            candidates[arm]=(action,time.perf_counter()-start)
        # Targets appear only after all candidates have been produced.
        data=Data(question='',answer='',metadata=public)
        for arm,(action,elapsed) in candidates.items():
            parsed=retrieval.parse_grid(action)
            records.append(dict(id=case['id'],slice=case['slice'],arm=arm,action=action,
                official_valid=bool(official.verify(data,str(action))),
                literal_valid=public_rule(parsed,public,True) if parsed is not None else False,
                canonical_exact=parsed==case['target'],controller_seconds=elapsed,
                retrieval_seconds=retrieval_seconds,retrieved_row_id=example.row_id,
                retrieved_observation_sha256=__import__('hashlib').sha256(example.observation.encode()).hexdigest(),
                retrieval_similarity=similarity,model_calls=0))
        with (out/'rows.jsonl').open('a',encoding='utf-8') as f:
            f.write(''.join(json.dumps(r)+'\n' for r in records[-len(arms):]))
        if (index+1)%20==0:
            print(f'Completed {index+1}/{len(cases)} controller cases',flush=True)
    summaries={}
    for subset in ('synthetic_dev','public_campsite'):
        summaries[subset]={}
        for arm in arms:
            selected=[r for r in records if r['slice']==subset and r['arm']==arm]
            summaries[subset][arm]=dict(cases=len(selected),official_valid=sum(r['official_valid'] for r in selected),
                literal_valid=sum(r['literal_valid'] for r in selected),canonical_exact=sum(r['canonical_exact'] for r in selected),
                controller_seconds=sum(r['controller_seconds'] for r in selected))
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(status='archived_controller_control_complete',
        cases=len(cases),records=len(records),arms=list(arms),model_calls=0,
        support_manifest_sha256=sha(args.support/'manifest.json'),public_source_sha256=sha(source_path),
        implementation_sha256=sha(Path(__file__)),
        files={f.relative_to(out).as_posix():sha(f) for f in out.rglob('*') if f.is_file() and '__pycache__' not in f.parts},
        limitations=['Controller-only ablations; not a full skill+27B run or neural TRM benchmark.',
            'Configured local-search pass caps are not measured neural router passes.',
            'Synthetic support enforces stricter adjacency than most public reference solutions.',
            'Exact search retains the archived 200000-node cap; local search and nearest-neighbor ranking are unmodified.']),indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
