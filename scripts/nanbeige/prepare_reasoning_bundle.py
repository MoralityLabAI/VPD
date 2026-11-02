"""Allowlisted public experiment payload; grading data stays on the local host."""
import json
import tarfile
from pathlib import Path
from experiments.nanbeige_reasoning.fetch_sources import sha

root=Path(__file__).resolve().parents[2]
files=[]
for package in ('nanbeige_looped','nanbeige_reasoning'):
    files.extend(p for p in (root/'experiments'/package).iterdir()
        if p.is_file() and (p.suffix in ('.py','.json','.txt') or p.name=='VPD_BACKEND_LICENSE'))
files += [root/'scripts/nanbeige/run_reasoning_remote.sh',
          root/'artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl',
          root/'artifacts/nanbeige_reasoning/pilot_v1/manifest.json']
dest=root/'artifacts/nanbeige_reasoning/reasoning_bundle.tar.gz'
temporary=dest.with_suffix('.tmp')
with tarfile.open(temporary,'w:gz') as archive:
    for p in files:
        archive.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
temporary.replace(dest)
dest.with_suffix('.receipt.json').write_text(json.dumps({'sha256':sha(dest),
    'files':{p.relative_to(root).as_posix():sha(p) for p in files}},indent=2),encoding='utf-8')
print('Bundled',len(files),'files;',dest.stat().st_size,'bytes')
