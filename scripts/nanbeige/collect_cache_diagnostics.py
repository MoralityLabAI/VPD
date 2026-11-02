"""Verify remote cache diagnostics against raw logits and independent sources."""
import argparse
import json
from pathlib import Path
import tarfile

import numpy as np

from scripts.nanbeige.collect_multipass import collect, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run.resolve()
    if not (run/'retrieved').exists():
        collect(run)
    root=run/'retrieved/artifacts/nanbeige_reasoning/cache_diagnostics'
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='cache_precision_diagnostics_complete':
        raise ValueError('Incomplete diagnostic manifest')
    for name,expected in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha256(path)!=expected:
            raise ValueError(f'Diagnostic file hash mismatch: {name}')
    question=Path('artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl')
    if sha256(question)!=manifest['question_sha256']:
        raise ValueError('Question source mismatch')
    source=next(r for r in map(json.loads,question.read_text().splitlines()) if r['id']==manifest['source_id'])
    checkpoint=Path('models/Nanbeige4.2-3B-tokenizer')
    matches={name:sha256(checkpoint/name)==expected for name,expected in manifest['checkpoint_sha256'].items() if (checkpoint/name).is_file()}
    if not all(matches.values()) or not any(name.endswith('.safetensors') for name in matches):
        raise ValueError('Independent pretrained checkpoint verification failed')
    receipt=json.loads((run/'bundle.receipt.json').read_text())
    if sha256(run/'bundle.tar.gz')!=receipt['sha256']:
        raise ValueError('Frozen source bundle mismatch')
    import hashlib
    with tarfile.open(run/'bundle.tar.gz','r:gz') as bundle:
        for name,expected in manifest['implementation_sha256'].items():
            member=bundle.extractfile('experiments/nanbeige_reasoning/'+name)
            if member is None or hashlib.sha256(member.read()).hexdigest()!=expected:
                raise ValueError('Executed implementation does not match frozen bundle')
    verified=[]
    with np.load(root/'logits.npz',allow_pickle=False) as arrays:
        if not np.array_equal(arrays['token_ids'],[source['token_ids'][:32]]):
            raise ValueError('Diagnostic token prefix mismatch')
        conditions=manifest['conditions']
        if {(r['precision'],r['loops']) for r in conditions}!={('bf16',1),('bf16',2),('fp32',1),('fp32',2)} or len(conditions)!=4:
            raise ValueError('Missing or duplicated precision/loop condition')
        for row in conditions:
            key=f"{row['precision']}_loops_{row['loops']}"
            full,cached=arrays[key+'_full'],arrays[key+'_cached']
            if full.shape!=(1,166144) or cached.shape!=full.shape or not np.isfinite(full).all() or not np.isfinite(cached).all():
                raise ValueError('Invalid logit shape or values')
            diff=np.abs(full-cached)
            outside=int((diff>(row['atol']+row['rtol']*np.abs(cached))).sum())
            expected=dict(max_abs_error=float(diff.max()),argmax_match=bool(full.argmax()==cached.argmax()),
                outside_tolerance=outside,within_tolerance=outside==0)
            for field,value in expected.items():
                if row[field]!=value:
                    raise ValueError(f'Raw logits disagree with {key}:{field}: {row[field]} vs {value}')
            if row['cache_layers']!=22*row['loops'] or not row['cache_layers_match']:
                raise ValueError('Cache cardinality mismatch')
            # This margin is computed from the full-prefix distribution only.
            top=np.partition(full.ravel(),-2)[-2:]
            verified.append(dict(precision=row['precision'],loops=row['loops'],**expected,
                full_argmax=int(full.argmax()),cached_argmax=int(cached.argmax()),
                full_top_two_margin=float(top.max()-top.min()),cache_layers=row['cache_layers']))
    termination=json.loads((run/'termination.json').read_text())
    if not termination['termination_command_succeeded']:
        raise ValueError('Pod cleanup needs attention')
    result=dict(status='cache_diagnostics_verified',archive_sha256=sha256(run/'results.tar.gz'),
        manifest_sha256=sha256(root/'manifest.json'),independent_checkpoint_matches=matches,
        frozen_implementation_matches=True,conditions=verified,termination=termination,
        generation_gate_changed=False,scope='Single 32-token prefix; precision and cache diagnostic only')
    (run/'cache_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
