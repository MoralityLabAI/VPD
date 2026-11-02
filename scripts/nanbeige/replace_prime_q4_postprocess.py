"""Replace only the owned local postprocessor after an audited decoder change."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path('/mnt/c/projects/VPD')
run=root/'artifacts/nanbeige_reasoning/prime-20260908T200850Z-601'
receipt=run/'postprocess_controller.json'
old=json.loads(receipt.read_text());pid=old['pid']
expected=['python3','-m','scripts.nanbeige.finish_prime_q4',str(run)]
actual=Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
assert actual==expected, f'Owned watcher identity differs: {actual}'
assert not (run/'completion_verification.json').exists(), 'Watcher may already be grading'
archive=run/'postprocess_controller_v1.json'
assert not archive.exists()
archive.write_bytes(receipt.read_bytes())
os.kill(pid,signal.SIGTERM)
for _ in range(50):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists() or p.read_text().split()[2]=='Z':break
    time.sleep(.1)
else:raise RuntimeError('Old watcher did not stop; no replacement started')
audit=run/'native_decode_audit_v3/verification.json'
assert json.loads(audit.read_text())['status']=='native_cleanup_independently_verified'
command=expected+['--native-cleanup-audit',str(audit)]
with (run/'postprocess_controller_v2.log').open('wb') as log:
    child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,
        stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=child.pid,started_unix=time.time(),detached=True,command=command,
    replaced_pid=pid,old_watcher_stopped=True,
    script_sha256=hashlib.sha256((root/'scripts/nanbeige/finish_prime_q4.py').read_bytes()).hexdigest(),
    cleanup_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
