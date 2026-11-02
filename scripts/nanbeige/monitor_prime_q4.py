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
    run = root/'nanbeige_reasoning/prime_q4_feature_answers_v1'
    contract = run/'contract.json'
    contract_hash = hashlib.sha256(contract.read_bytes()).hexdigest()
    rows = []
    for p in sorted(run.glob('[0-9][0-9]_*.json')):
        row = json.loads(p.read_text())
        payload = {k:v for k,v in row.items() if k != 'payload_sha256'}
        encoded = json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
        assert hashlib.sha256(encoded).hexdigest() == row['payload_sha256'], p
        assert row['contract_sha256'] == contract_hash and row['complete'], p
        rows.append((row['id'],row['condition']))
    assert len(rows) == len(set(rows)) == 70, f'Expected 70 complete outputs, found {len(rows)}'
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['status'] == 'q4_feature_answers_complete'
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest() == digest, name
    clean = json.loads((run/'cleanup.json').read_text())
    assert clean['model_closed'] and clean['context_closed']
    return {'status':'all_70_outputs_retrieved_and_hash_verified','outputs':len(rows),
            'generation_contract_sha256':contract_hash}


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
            if not runtime_saved and (artifacts/'nanbeige_reasoning/prime_q4_runtime/contract.json').exists():
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
