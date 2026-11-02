"""Pull incremental local backups while a detached remote experiment runs."""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import time


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def verify_finished(root):
    run=root/'nanbeige_reasoning/logic_skill_comparison_outputs_v1'
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='logic_skill_generation_complete' and manifest['records']==259
    for name,h in manifest['files'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h,name
    rows=[json.loads(p.read_text()) for p in sorted(run.glob('[0-9][0-9][0-9].json'))]
    assert len(rows)==len({r['call_id'] for r in rows})==259
    assert json.loads((run/'cleanup.json').read_text())['server_stopped']
    return dict(status='all_259_outputs_retrieved_and_hash_verified',outputs=259)


def backup(run, host, port, *, runtime=False):
    remote = '/home/ubuntu/nanbeige-reasoning/'
    opts = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',
            '-o',f'UserKnownHostsFile={run}/known_hosts','-i','/home/duganist/.ssh/id_rsa','-p',port]
    folder = 'tools/nanbeige/runtime/' if runtime else 'artifacts/'
    target = run/('runtime_backup' if runtime else 'retrieved/artifacts')
    target.mkdir(parents=True,exist_ok=True)
    command = ['rsync','-az','--partial','--delay-updates','--timeout=60',
               '--exclude=*.tmp','--exclude=*.repair', '-e',shlex.join(opts),
               f'{host}:{remote}{folder}',str(target)+'/']
    with (run/'backup.log').open('a') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=pathlib.Path)
    parser.add_argument('--backup-only',action='store_true')
    args = parser.parse_args(); run = args.run.resolve()
    host, port = (run/'ssh_target.txt').read_text().split()
    if args.backup_only:
        raise SystemExit(backup(run,host,port))
    failures = 0; started = time.monotonic(); runtime_saved = False
    while time.monotonic()-started < 28000:
        try:
            code = backup(run,host,port)
        except subprocess.TimeoutExpired:
            code = 124
        failures = failures+1 if code else 0
        artifacts = run/'retrieved/artifacts'
        write(run/'monitor_status.json',dict(status='monitoring',backup_exit_code=code,
            consecutive_backup_failures=failures,updated_unix=time.time()))
        if not code:
            if not runtime_saved and (artifacts/'nanbeige_reasoning/logic_skill_comparison_outputs_v1/contract.json').exists():
                runtime_saved = backup(run,host,port,runtime=True) == 0
            exit_file = artifacts/'nanbeige_reasoning/cloud_job/exit.json'
            if exit_file.exists():
                remote = json.loads(exit_file.read_text())
                if remote['exit_code'] != 0:
                    raise RuntimeError(f'Remote worker failed: {remote}')
                receipt = verify_finished(artifacts)
                write(run/'completion_verification.json',receipt)
                write(run/'monitor_status.json',dict(status='completed_and_retrieved',updated_unix=time.time()))
                return
        if failures >= 5 and time.monotonic()-started > 600:
            raise RuntimeError('Five consecutive backup failures; stopping instead of generating unbacked results')
        time.sleep(30)
    raise RuntimeError('Monitor wall limit reached')


if __name__ == '__main__': main()
