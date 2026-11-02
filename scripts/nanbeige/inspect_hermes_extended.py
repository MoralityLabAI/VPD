"""Preserve selected historical comparator receipts from the pinned repository."""
import hashlib
import json
from pathlib import Path
import subprocess

REV='07a177f2751c2adb1cc16dc819c1173c0cd57ad5'
ROOT=Path('artifacts/nanbeige_reasoning/hermes_extended_audit_v1')
BASE='research/studies/2026-04-22-metta-trm-hermes-pipeline/artifacts/'
PATHS=[
 BASE+'intellect3_math_patch_bank_benchmark_27b_20260502_combined40/combined_patch_bank_benchmark.results.json',
 BASE+'intellect3_math_optimized_trm_router/math_skill_trm_optimized_router.results.json',
 'data/primehub_overnight_all/qwen35_27b/qwen35_27b_arc_agi_q0017.jsonl',
 'data/primehub_overnight_all/qwen35_27b/qwen35_27b_arc_agi_tool_q0018.jsonl',
]

def main():
    ROOT.mkdir(exist_ok=False)
    files={}
    for name in PATHS:
        data=subprocess.check_output(['git','-c','safe.directory=C:/projects/Hermes-Skills','-C',
            'C:/projects/Hermes-Skills','show',f'{REV}:{name}'])
        out=ROOT/'source'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
        files[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        value=[json.loads(s) for s in data.decode().splitlines()] if name.endswith('.jsonl') else json.loads(data)
        summary={}
        if isinstance(value,dict):
            for k,v in value.items():
                summary[k]={'type':type(v).__name__,'length':len(v) if isinstance(v,(list,dict,str)) else None,
                    'keys':list(v)[:20] if isinstance(v,dict) else (list(v[0]) if isinstance(v,list) and v and isinstance(v[0],dict) else None)}
        else:summary={'rows':len(value),'row_keys':list(value[0]) if value else []}
        print(json.dumps(dict(path=name,structure=summary)))
    (ROOT/'source_manifest.json').write_text(json.dumps(dict(repository_revision=REV,files=files),indent=2)+'\n')

if __name__=='__main__':main()
