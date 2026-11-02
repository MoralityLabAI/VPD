"""Start the experiment independently of the provisioning SSH connection."""
import json
import pathlib
import subprocess
import time

root=pathlib.Path('/home/ubuntu/nanbeige-reasoning')
job=root/'artifacts/nanbeige_reasoning/cloud_job'
job.mkdir(parents=True,exist_ok=True)
assert not (job/'pid.json').exists(), 'Refusing duplicate worker launch'
with (job/'worker.log').open('wb') as log:
    child=subprocess.Popen(['bash','scripts/nanbeige/prime_q4_worker.sh'],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=child.pid,started_unix=time.time(),detached=True)
(job/'pid.json').write_text(json.dumps(receipt)+'\n')
print(json.dumps(receipt))
