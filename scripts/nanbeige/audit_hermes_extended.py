"""Recompute the 40-row math ledger and classify two failed ARC-AGI traces."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

from .inspect_hermes_extended import ROOT, REV, PATHS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    original=read(ROOT/'source_manifest.json')
    for name,meta in original['files'].items():assert sha(ROOT/'source'/name)==meta['sha256']
    combined=read(ROOT/'source'/PATHS[0]);provenance=dict(original['files']);rows=[]
    for shard in combined['shards']:
        name=shard['path'].replace('\\','/')+'/patch_bank_benchmark.rows.jsonl'
        assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
        data=subprocess.check_output(['git','-c','safe.directory=C:/projects/Hermes-Skills','-C',
            'C:/projects/Hermes-Skills','show',f'{REV}:{name}'])
        path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
        provenance[name]=dict(sha256=sha(path),bytes=len(data))
        subset=[json.loads(s) for s in data.decode().splitlines()]
        assert len(subset)==shard['calls'];rows.extend(subset)
    pairs={(r['row_id'],r['arm']) for r in rows}
    assert len(rows)==len(pairs)==combined['calls']==280
    ids={r['row_id'] for r in rows};assert len(ids)==combined['unique_rows']==40
    by_arm=defaultdict(list);by_id=defaultdict(dict)
    for row in rows:
        exact=str(row['action']).strip()==str(row['expected']).strip()
        assert exact==row['exact'],row['row_id']
        by_arm[row['arm']].append(row);by_id[row['row_id']][row['arm']]=row
    arms={}
    for name,values in by_arm.items():
        assert {r['row_id'] for r in values}==ids
        value=dict(rows=len(values),exact=sum(r['exact'] for r in values),errors=sum(bool(r['error']) for r in values))
        expected=combined['summary']['arms'][name]
        assert all(value[k]==expected[k] for k in value),(name,value,expected)
        arms[name]=value
    for name,gate in combined['summary']['adoption_gates'].items():
        fixes=sum(r[name]['exact'] and not r['incumbent_current_skill']['exact'] for r in by_id.values())
        regressions=sum(not r[name]['exact'] and r['incumbent_current_skill']['exact'] for r in by_id.values())
        assert fixes==gate['fixes_vs_incumbent'] and regressions==gate['regressions_vs_incumbent']
    oracle=sum(any(v['exact'] for v in r.values()) for r in by_id.values())
    traces=[]
    for name in PATHS[2:]:
        values=[json.loads(s) for s in (ROOT/'source'/name).read_text().splitlines()]
        assert len(values)==1
        r=values[0]
        assert r['failure_type']=='bridge_failure' and 'invoke failure' in r['observation']
        traces.append(dict(source=name,environment=r['env_name'],logged_score=r['score'],
            failure_type=r['failure_type'],completion_tokens=r['usage']['completion_tokens'],
            interpretation='Adapter reset failed; no valid task evaluation; exclude from reasoning-accuracy denominators.'))
    result=dict(status='historical_ledger_recomputed_and_adapter_failures_classified',repository_revision=REV,
        math=dict(rows=40,calls=280,arms=arms,oracle_rows=oracle,paired_adoption_counts_verified=True,
            limits=['String-exact receipt recomputation, not symbolic regrading or fresh inference.',
                    'Ledger has row IDs and expected answers but no original prompts; identity with Nanbeige questions is not established.']),
        arc_agi=traces,source_files=provenance,verifier_sha256=sha(Path(__file__)))
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    table='\n'.join(f'| {name} | {r["exact"]}/40 | {r["errors"]} |' for name,r in arms.items())
    report=f'''# Additional historical Hermes evidence

The pinned four-shard math ledger contains 280 calls on 40 row IDs. Independent string-exact recomputation reproduces every arm, error count, and paired fix/regression count.

| Arm | Exact answers | Recorded errors |
|---|---:|---:|
{table}

The answer-selection oracle across all seven arms is {oracle}/40. It uses outcome knowledge and is not an executable policy score. These records lack the original prompts, so they cannot be joined to the Nanbeige cohort by answer value or row order alone. This audit does not rerun the 27B model or use symbolic-equivalence scoring.

The two inspected 27B traces labeled `arc_agi` and `arc_agi_tool` both report an adapter reset failure. Their logged zero scores are infrastructure outcomes, not valid ARC-AGI reasoning measurements. Both record a 64-token model response. They provide no evidence of an ARC-AGI-1/2 benchmark clear or a measured 27B failure rate on valid tasks.

All source bytes are preserved from Hermes-Skills revision `{REV}` and hashed in `verification.json`. This extends the historical comparison; it does not replace the requested matched comparison.
'''
    (ROOT/'REPORT.md').write_text(report)
    print(json.dumps(dict(status=result['status'],calls=len(rows),math_arms=arms,oracle_rows=oracle,invalid_arc_agi_traces=len(traces))))

if __name__=='__main__':main()
