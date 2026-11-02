"""Apply an audited sensor portability repair strictly before any gate starts."""
import hashlib
import json
import tarfile
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

root = Path('/home/ubuntu/nanbeige-reasoning')
assert not (root/'artifacts/nanbeige_reasoning/prime_q4_cache_gate_v1').exists()
original = '58cfe0ede10c5ab256a138597e2d222c131d9cfebd694f351c794ac0292b3e5a'
replacement = '66797dc31dc0c11b5bd9b54d36e9a9f3b58efdd7d5568631fb1020181d6e0ea6'
assert sha(root/'bundle.tar.gz') == original
assert sha(root/'sensor_repair.tar.gz') == replacement
manifest = json.loads((root/'bundle_manifest.json').read_text())
for name, digest in manifest['files'].items(): assert sha(root/name) == digest, name
out = root/'artifacts/nanbeige_reasoning/prime_q4_port_repair'
out.mkdir(exist_ok=False)
names = ['experiments/nanbeige_reasoning/temperature_guard_burst.py',
         'artifacts/nanbeige_reasoning/prime_q4_source.json', 'bundle_manifest.json']
with tarfile.open(root/'sensor_repair.tar.gz') as archive:
    for name in names:
        target = root/name
        (out/target.name).write_bytes(target.read_bytes())
        member = archive.getmember(name)
        assert member.isfile()
        data = archive.extractfile(member).read()
        temp = target.with_suffix(target.suffix+'.repair')
        temp.write_bytes(data)
        temp.replace(target)
for name, digest in json.loads((root/'bundle_manifest.json').read_text())['files'].items():
    assert sha(root/name) == digest, name
receipt = dict(status='pre_inference_port_repair_verified', original_bundle_sha256=original,
    repair_patch_sha256=replacement, changed_files=names,
    resulting_bundle_manifest_sha256=sha(root/'bundle_manifest.json'),
    reason='Replace Windows-only CREATE_NO_WINDOW with Linux creationflags=0; sensor and thresholds unchanged.')
(out/'receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
