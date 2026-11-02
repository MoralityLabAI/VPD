"""Exercise retrieval ordering on normal completion and controller interruption."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.nanbeige.monitor_prime_q4 import verify_finished


class CompletionTests(unittest.TestCase):
    def make_run(self,root,count=70):
        from experiments.nanbeige_reasoning.run_q4_feature_answers import save_record
        run=root/'nanbeige_reasoning/prime_q4_feature_answers_v1';run.mkdir(parents=True)
        (run/'contract.json').write_text('{}')
        digest=hashlib.sha256((run/'contract.json').read_bytes()).hexdigest()
        for i in range(count):
            save_record(run/f'{i:02d}_arm.json',dict(id=str(i),condition='test',complete=True,
                generated_text='数学',contract_sha256=digest))
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
        (run/'manifest.json').write_text(json.dumps(dict(status='q4_feature_answers_complete',files=files)))
        (run/'cleanup.json').write_text(json.dumps(dict(model_closed=True,context_closed=True)))
        return run

    def test_complete_unicode_records(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_run(Path(d));self.assertEqual(verify_finished(Path(d))['outputs'],70)

    def test_38_outputs_rejected_even_with_success_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_run(Path(d),38)
            with self.assertRaises(AssertionError):verify_finished(Path(d))

    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.make_run(Path(d));p=run/'00_arm.json'
            row=json.loads(p.read_text());row['generated_text']='changed';p.write_text(json.dumps(row))
            with self.assertRaises(AssertionError):verify_finished(Path(d))


def lifecycle(interrupt):
    assert os.name=='posix'
    project=Path(__file__).resolve().parents[2]
    fixture=Path(tempfile.mkdtemp(prefix='durable-',dir=project/'artifacts/nanbeige_reasoning/launcher_tests'))
    def put(name,text,executable=False):
        p=fixture/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
        if executable:p.chmod(0o755)
    fake='''#!/usr/bin/python3
import json,os,sys
from pathlib import Path
r=Path(os.environ['FIXTURE']);a=sys.argv[1:];name=Path(sys.argv[0]).name
def event(s):
 with (r/'events').open('a') as f:f.write(s+'\\n')
if name=='prime':
 if 'availability' in a:print(json.dumps({'gpu_resources':[dict(id='offer',provider='massedcompute',stock_status='Available',gpu_count=1,gpu_memory=48,price_value=.75,gpu_type='RTX6000Ada 48GB',location='US',is_spot=False)]}))
 elif 'create' in a:(r/'name').write_text(a[a.index('--name')+1]);event('create')
 elif 'list' in a:print(json.dumps({'pods':[dict(id='pod',name=(r/'name').read_text())]}))
 elif 'status' in a:print(json.dumps(dict(status='ACTIVE',ssh='fixture@127.0.0.1')))
 elif 'terminate' in a:event('terminate')
 else:raise SystemExit(a)
elif name in ('ssh','scp'):event(name)
'''
    for name in ['prime','ssh','scp']:put('bin/'+name,fake,True)
    put('key','fixture only')
    put('scripts/nanbeige/prime_ssh_selection.py',"import sys,json;from pathlib import Path\np=Path(sys.argv[2]);p.write_text(json.dumps({'fixture':True}))\n")
    put('scripts/nanbeige/monitor_prime_q4.py','''import os,signal,sys
from pathlib import Path
r=Path(os.environ['FIXTURE'])
with (r/'events').open('a') as f:f.write(('final_backup' if '--backup-only' in sys.argv else 'periodic_backup')+'\\n')
if '--backup-only' not in sys.argv and os.environ['INTERRUPT']=='1':os.kill(os.getppid(),signal.SIGTERM)
''')
    bundle='artifacts/nanbeige_reasoning/prime_q4_answers_bundle.tar.gz'
    put(bundle,'offline fake bundle')
    digest=hashlib.sha256((fixture/bundle).read_bytes()).hexdigest()
    put('artifacts/nanbeige_reasoning/prime_q4_answers_bundle.tar.receipt.json',json.dumps(dict(sha256=digest)))
    source=(project/'scripts/nanbeige/launch_prime_q4_durable.sh').read_text()
    for a,b in [('/mnt/c/projects/VPD',str(fixture)),('/home/duganist/.local/bin/prime',str(fixture/'bin/prime')),
        ('/home/duganist/.local/share/uv/tools/prime/bin/python3','/usr/bin/python3'),('/home/duganist/.ssh/id_rsa',str(fixture/'key'))]:source=source.replace(a,b)
    put('launcher.sh',source)
    env=os.environ.copy();env.update(PATH=str(fixture/'bin')+':'+env['PATH'],FIXTURE=str(fixture),INTERRUPT=str(int(interrupt)))
    env.pop('NANBEIGE_LAUNCHER_FROZEN',None)
    with (fixture/'log').open('w') as log:
        result=subprocess.run(['bash',str(fixture/'launcher.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=25)
    assert result.returncode==(143 if interrupt else 0),(fixture/'log').read_text()
    events=(fixture/'events').read_text().splitlines()
    assert events.index('periodic_backup')<events.index('final_backup')<events.index('terminate'),events
    print(json.dumps(dict(status='durable_lifecycle_passed',interrupted=interrupt,fixture=str(fixture),events=events)))


if __name__=='__main__':
    if os.name=='posix':lifecycle(False);lifecycle(True)
    else:unittest.main()
