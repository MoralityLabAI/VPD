"""Save read-only GPU telemetry from this experiment's live SSH target."""
import json
from pathlib import Path
import subprocess
import sys
import time

run=Path(sys.argv[1]).resolve()
assert not (run/'termination.json').exists(), 'Do not contact a terminated pod'
assert (run/'pod_id.txt').read_text().strip()=='2bf3b09bb42643b08c093944aadb6d09'
host,port=(run/'ssh_target.txt').read_text().split()
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',
         '-o',f'UserKnownHostsFile={run}/known_hosts','-i','/home/duganist/.ssh/id_rsa','-p',port,host,
         'nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv']
result=subprocess.run(command,capture_output=True,text=True,timeout=30,check=True)
receipt=dict(observed_unix=time.time(),pod_id=(run/'pod_id.txt').read_text().strip(),gpu_csv=result.stdout)
(run/'gpu_during_generation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
