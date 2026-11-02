"""Detach the local verifier/grader watcher; never allocate or alter cloud state."""
import json
from pathlib import Path
import subprocess
import time

root=Path('/mnt/c/projects/VPD')
run=root/'artifacts/nanbeige_reasoning/prime-20260908T200850Z-601'
receipt=run/'postprocess_controller.json'
assert not receipt.exists(), 'Postprocess watcher already dispatched'
with (run/'postprocess_controller.log').open('wb') as log:
    child=subprocess.Popen(['python3','-m','scripts.nanbeige.finish_prime_q4',str(run)],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=child.pid,started_unix=time.time(),detached=True)
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
