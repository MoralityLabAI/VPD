"""Offline launcher regression: fake cloud/SSH, including a mid-run source edit.

Run with WSL Python. No network client or real credential helper is invoked.
Fixtures and command receipts remain under artifacts for inspection.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def write(path,text,executable=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8',newline='\n')
    if executable:path.chmod(0o755)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runpod',action='store_true')
    parser.add_argument('--resource-ranges',action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--qwen',action='store_true')
    mode.add_argument('--reasoning-vpd',action='store_true')
    mode.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if os.name!='posix':raise SystemExit('Run this offline test through WSL Python')
    project=Path(__file__).resolve().parents[2]
    fixture=project/'artifacts/nanbeige_reasoning/launcher_tests'/str(time.time_ns())
    fixture.mkdir(parents=True)
    binary=fixture/'bin'
    state=fixture/'fake_cloud.json'
    state.write_text(json.dumps(dict(created=False,terminated=False)))
    fake_prime='''#!/usr/bin/python3
import json,os,sys
from pathlib import Path
root=Path(os.environ['VPD_LAUNCH_TEST_ROOT'])
state=root/'fake_cloud.json'; data=json.loads(state.read_text()); args=sys.argv[1:]
if 'availability' in args:
 print(json.dumps({'gpu_resources':[{'id':'fixture-offer','provider':'massedcompute','stock_status':'Available','gpu_count':1,'gpu_memory':48,'price_value':0.75,'gpu_type':'RTX6000Ada 48GB','location':'US','is_spot':False}, {'id':'fixture-runpod','provider':'runpod','stock_status':'Low','gpu_count':1,'gpu_memory':48,'price_value':0.5068,'gpu_type':'A40 48GB','location':'CA','is_spot':False,'vcpus':'9','memory_gb':'50'}]}))
elif 'create' in args:
 if args[args.index('--id')+1]=='fixture-runpod':
  assert args[args.index('--disk-size')+1]=='120'
  assert args[args.index('--vcpus')+1]=='9'
  assert args[args.index('--memory')+1]=='50'
 data.update(created=True,name=args[args.index('--name')+1]); state.write_text(json.dumps(data))
 source=root/'launcher_input.sh'; source.write_text('# EDITED DURING PROVISIONING\\n'*20+source.read_text())
 print('Created fixture-only pod')
elif 'list' in args:
 print(json.dumps({'pods':[{'id':'fixture-pod','name':data['name']}] if data['created'] else []}))
elif 'status' in args:
 print(json.dumps({'ssh':'fixture@127.0.0.1 -p 1234','status':'ACTIVE'}))
elif 'terminate' in args:
 data['terminated']=True; state.write_text(json.dumps(data)); print('Terminated fixture-only pod')
else: raise SystemExit('Unexpected fake cloud command: '+repr(args))
'''
    if args.resource_ranges:
        fake_prime=fake_prime.replace("'vcpus':'9','memory_gb':'50'", "'vcpus':'9-12','memory_gb':'50-64'")
    fake_transport='''#!/usr/bin/python3
import json,os,sys
from pathlib import Path
root=Path(os.environ['VPD_LAUNCH_TEST_ROOT'])
with (root/'transport.jsonl').open('a') as f:f.write(json.dumps({'program':Path(sys.argv[0]).name,'args':sys.argv[1:]})+'\\n')
if Path(sys.argv[0]).name=='scp' and ':' not in sys.argv[-1]:
 Path(sys.argv[-1]).write_bytes(b'OFFLINE FIXTURE, NOT MODEL RESULTS')
'''
    write(binary/'prime',fake_prime,True)
    write(binary/'ssh',fake_transport,True)
    write(binary/'scp',fake_transport,True)
    write(fixture/'keys/id_rsa','OFFLINE DUMMY KEY\n')
    write(fixture/'scripts/nanbeige/prime_ssh_selection.py','''import json,sys
from pathlib import Path
receipt=Path(sys.argv[2])
if sys.argv[1]=='select':receipt.write_text(json.dumps({'fixture':True}))
elif sys.argv[1]=='restore':
 assert json.loads(receipt.read_text())['fixture']
 receipt.with_suffix('.restored').write_text('fixture restoration complete')
else:raise SystemExit('Unexpected key-helper operation')
''')
    artifact=fixture/'artifacts/nanbeige_reasoning'
    artifact.mkdir(parents=True)
    for name in ('prime_q4_answers_bundle.tar.gz','prime_q4_answers_bundle.tar.receipt.json'):
        shutil.copy2(project/'artifacts/nanbeige_reasoning'/name,artifact/name)
    if args.qwen:
        for name in ('qwen_comparator_bundle.tar.gz','qwen_comparator_bundle.tar.receipt.json'):
            shutil.copy2(project/'artifacts/nanbeige_reasoning'/name,artifact/name)
    if args.reasoning_vpd:
        for name in ('reasoning_vpd_bundle.tar.gz','reasoning_vpd_bundle.tar.receipt.json'):
            shutil.copy2(project/'artifacts/nanbeige_reasoning'/name,artifact/name)
    if args.benchmark:
        for name in ('benchmark_bundle.tar.gz','benchmark_bundle.tar.receipt.json'):
            shutil.copy2(project/'artifacts/nanbeige_reasoning'/name,artifact/name)
    source=(project/'scripts/nanbeige/launch_prime_q4_answers.sh').read_text()
    source=source.replace('/mnt/c/projects/VPD',str(fixture))
    source=source.replace('/home/duganist/.local/bin/prime',str(binary/'prime'))
    source=source.replace('/home/duganist/.local/share/uv/tools/prime/bin/python3','/usr/bin/python3')
    source=source.replace('/home/duganist/.ssh/id_rsa',str(fixture/'keys/id_rsa'))
    script=fixture/'launcher_input.sh'
    write(script,source)
    expected=hashlib.sha256(script.read_bytes()).hexdigest()
    subprocess.run(['bash','-n',str(script)],check=True)
    env=os.environ.copy()
    env.update(PATH=str(binary)+':'+env['PATH'],VPD_LAUNCH_TEST_ROOT=str(fixture),
               NANBEIGE_REASONING_REGION='CA' if args.runpod else 'US',
               NANBEIGE_REASONING_MODE='qwen-pilot' if args.qwen else ('reasoning-vpd' if args.reasoning_vpd else ('benchmark' if args.benchmark else 'cache-diagnostics')),
               NANBEIGE_COMPUTE_PROFILE='runpod_a40' if args.runpod else 'massed_ada')
    env.pop('NANBEIGE_LAUNCHER_FROZEN',None)
    with (fixture/'launcher.log').open('w') as log:
        process=subprocess.Popen(['bash',str(script)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();raise
    assert code==0, (fixture/'launcher.log').read_text()
    runs=list(artifact.glob('prime-*'))
    assert len(runs)==1
    run=runs[0]
    assert json.loads((run/'choice.json').read_text())['provider']==('runpod' if args.runpod else 'massedcompute')
    inputs=json.loads((run/'launch_inputs.json').read_text())
    assert inputs['launcher_sha256']==expected
    assert hashlib.sha256(script.read_bytes()).hexdigest()!=expected
    assert json.loads(state.read_text())['terminated']
    assert (run/'ssh_selection.restored').exists()
    transport=list(map(json.loads,(fixture/'transport.jsonl').read_text().splitlines()))
    entry=('run_qwen_remote.sh pilot' if args.qwen else ('run_reasoning_vpd_remote.sh' if args.reasoning_vpd else ('run_benchmark_remote.sh' if args.benchmark else 'run_prime_q4_answers_remote.sh')))
    assert any(r['program']=='ssh' and r['args'][-1].endswith(entry) for r in transport)
    original_bundle=artifact/('qwen_comparator_bundle.tar.gz' if args.qwen else ('reasoning_vpd_bundle.tar.gz' if args.reasoning_vpd else ('benchmark_bundle.tar.gz' if args.benchmark else 'prime_q4_answers_bundle.tar.gz')))
    assert (run/'bundle.tar.gz').read_bytes()==original_bundle.read_bytes()
    assert any(r['program']=='scp' and str(run/'bundle.tar.gz') in r['args'] for r in transport)
    assert (run/'results.tar.gz').read_bytes()==b'OFFLINE FIXTURE, NOT MODEL RESULTS'
    result=dict(status='offline_lifecycle_passed',fixture=str(fixture),real_cloud_calls=0,
        frozen_launcher_survived_source_edit=True,requested_mode_forwarded=True,
        bundle_receipt_verified=True,artifact_retrieval_called=True,ssh_restored=True,pod_terminated=True)
    (fixture/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
