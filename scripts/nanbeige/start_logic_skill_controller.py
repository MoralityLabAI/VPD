"""Detach the bounded WSL controller from the tool's terminal and Windows client."""
import json
import os
from pathlib import Path
import subprocess
import time

assert os.name=='posix'
root=Path('/mnt/c/projects/VPD')
stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
log=root/f'artifacts/nanbeige_reasoning/logic_durable_wsl_{stamp}.log'
env=os.environ.copy();env.pop('NANBEIGE_LAUNCHER_FROZEN',None)
env['PRIME_DISABLE_VERSION_CHECK']='1'
with log.open('wb') as stream:
    child=subprocess.Popen(['bash',str(root/'scripts/nanbeige/launch_logic_skill_bounded.sh')],
        cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=child.pid,started_unix=time.time(),stdout=str(log),detached_wsl_session=True,
    max_hourly=1.99,max_hours=5.5,provider='lambdalabs',gpu='A100 40GB',region='US')
(root/'artifacts/nanbeige_reasoning/logic_controller_current.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
