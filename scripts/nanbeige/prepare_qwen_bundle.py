"""Package public comparator inputs and code, excluding answer sidecars/weights."""
import json
from pathlib import Path
import tarfile
from experiments.nanbeige_reasoning.fetch_sources import sha

root=Path(__file__).resolve().parents[2]
files=[]
for package in ('nanbeige_looped','nanbeige_reasoning'):
    files.extend(p for p in (root/'experiments'/package).iterdir()
                 if p.is_file() and (p.suffix in ('.py','.json','.txt') or p.name=='VPD_BACKEND_LICENSE'))
files.extend(root/'scripts/nanbeige'/name for name in ('run_qwen_remote.sh','download_file.py'))
base=root/'artifacts/nanbeige_reasoning'
for cohort in ('pilot','benchmark'):
    directory=base/f'qwen_comparator_{cohort}_inputs_v1'
    files.extend(directory/name for name in ('questions.jsonl','manifest.json'))
sources=base/'comparator_sources/qwen35_27b'
files.append(sources/'model_lock.json')
pilot=json.loads((base/'qwen_comparator_pilot_inputs_v1/manifest.json').read_text())
files.extend(sources/'official'/name for name in pilot['tokenizer_assets'])
files.append(sources/'official/config.json')
files.append(sources/'official/generation_config.json')
names=[p.relative_to(root).as_posix() for p in files]
assert len(names)==len(set(names)) and not any(Path(name).name=='graders.jsonl' for name in names)
destination=base/'qwen_comparator_bundle.tar.gz'
temporary=destination.with_suffix('.tmp')
with tarfile.open(temporary,'w:gz') as archive:
    for path in files:archive.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
temporary.replace(destination)
receipt=dict(sha256=sha(destination),bytes=destination.stat().st_size,
    files={p.relative_to(root).as_posix():sha(p) for p in files},grader_sidecar_included=False,model_weights_included=False)
destination.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
# Read every member back to verify the exact upload payload.
import hashlib
with tarfile.open(destination,'r:gz') as archive:
    assert set(archive.getnames())==set(receipt['files'])
    for member in archive:
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==receipt['files'][member.name]
print(json.dumps(dict(status='comparator_bundle_verified',files=len(files),bytes=receipt['bytes'],sha256=receipt['sha256'],remote_inference_run=False),indent=2))
