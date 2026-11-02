import json
from pathlib import Path
from types import SimpleNamespace

from scripts.nanbeige import wait_for_authorized_capacity as capacity


def prepare(monkeypatch,tmp_path):
    base=tmp_path/'artifacts/nanbeige_reasoning'
    base.mkdir(parents=True)
    (base/'paid_launch_upload_review_block.json').write_text(json.dumps(dict(status='upload_authorized',compute_authorized=True)))
    monkeypatch.setattr(capacity,'__file__',str(tmp_path/'scripts/nanbeige/wait_for_authorized_capacity.py'))
    monkeypatch.setattr(capacity,'os',SimpleNamespace(name='posix'))
    monkeypatch.setattr(capacity.argparse.ArgumentParser,'parse_args',lambda _:SimpleNamespace(mode='benchmark'))
    clock=[0.]
    monkeypatch.setattr(capacity.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(capacity.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    return base


def test_no_offer_retries_but_creation_failure_stops(monkeypatch,tmp_path):
    base=prepare(monkeypatch,tmp_path)
    calls=[]
    def launch(command,*,cwd,stdout,stderr):
        calls.append(command)
        run=base/f'prime-fixture-{len(calls)}';run.mkdir()
        stdout.write(f'Multipass launch directory: {run}\n')
        stdout.write('No available US runpod RTX6000Ada 48GB below the experimental hourly ceiling\n')
        if len(calls)==2:(run/'create.log').write_text('Creation was attempted; do not retry.')
        return SimpleNamespace(pid=123,wait=lambda:1)
    monkeypatch.setattr(capacity.subprocess,'Popen',launch)
    capacity.main()
    assert len(calls)==2
    report=json.loads(next((base/'capacity_waits').glob('*/status.json')).read_text())
    assert report['status']=='launch_needs_attention'
    assert [a['creation_attempted'] for a in report['attempts']]==[False,True]


def test_unexpected_prelaunch_failure_is_not_retried(monkeypatch,tmp_path):
    base=prepare(monkeypatch,tmp_path)
    calls=[]
    def launch(command,*,cwd,stdout,stderr):
        calls.append(command)
        run=base/'prime-fixture';run.mkdir()
        stdout.write(f'Multipass launch directory: {run}\nBundle hash mismatch\n')
        return SimpleNamespace(pid=123,wait=lambda:1)
    monkeypatch.setattr(capacity.subprocess,'Popen',launch)
    capacity.main()
    assert len(calls)==1
    report=json.loads(next((base/'capacity_waits').glob('*/status.json')).read_text())
    assert report['status']=='unexpected_prelaunch_failure'
