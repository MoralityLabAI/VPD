"""Verify retrieved reasoning artifacts, source hashes, and intervention payloads."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.nanbeige.collect_multipass import collect,sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    result=collect(args.run)
    root=args.run/'retrieved/artifacts/nanbeige_reasoning/reasoning_capture'
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='capture_complete'
    question_path=Path('artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl')
    assert manifest['question_sha256']==sha256(question_path)
    for name,expected in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path)!=expected:
            raise ValueError(f'Capture hash mismatch: {name}')
        if path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as arrays:
                for key in arrays.files:
                    if not np.isfinite(arrays[key]).all():
                        raise ValueError(f'Nonfinite array: {name}:{key}')
                if len([k for k in arrays.files if '__pass_' in k])!=46:
                    raise ValueError('Missing physical-layer/pass captures')
        result['verified_files']+=1
    local=Path('models/Nanbeige4.2-3B-tokenizer')
    # Dynamic snapshots may include extra cache metadata; all recorded model files
    # must agree with the independently downloaded local snapshot when present.
    matches={name:sha256(local/name)==expected for name,expected in manifest['checkpoint_sha256'].items()
             if (local/name).exists()}
    if not matches or not all(matches.values()) or not any(n.endswith('.safetensors') for n in matches):
        raise ValueError('Independent checkpoint hash verification failed')
    generations=[json.loads(x) for x in (root/'generations.jsonl').read_text(encoding='utf-8').splitlines()]
    questions=[json.loads(x) for x in question_path.read_text(encoding='utf-8').splitlines()]
    if len(generations)!=2*len(questions):
        raise ValueError('Incomplete paired generation set')
    for q in questions:
        rows=[r for r in generations if r['id']==q['id']]
        if {r['loops'] for r in rows}!={1,2} or len(rows)!=2:
            raise ValueError('Duplicate or missing experimental condition')
        if any(r['token_ids'][:len(q['token_ids'])]!=q['token_ids'] for r in rows):
            raise ValueError('Generation prefix changed')
    result.update(reasoning_capture_complete=True,generations=len(generations),
        independent_checkpoint_matches=matches,
        termination=json.loads((args.run/'termination.json').read_text()))
    (args.run/'reasoning_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
