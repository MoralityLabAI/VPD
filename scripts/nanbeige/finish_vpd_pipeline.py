"""Finish local analysis after one verified live VPD launcher exits.

This finite pipeline creates no compute resources and does not alter inference.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--controller-pid',type=int,required=True)
    p.add_argument('--controller-created',type=float,required=True)
    p.add_argument('--max-wait-seconds',type=int,default=30000)
    args=p.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to(ROOT/'artifacts/nanbeige_reasoning'):
        raise ValueError('Run must belong to the local Nanbeige experiment workspace')
    state_path=run/'analysis_pipeline_state.json'
    if state_path.exists():
        raise ValueError('An analysis pipeline already has state for this run; inspect it before restarting')
    controller=psutil.Process(args.controller_pid)
    if (abs(controller.create_time()-args.controller_created)>.01
            or controller.name().lower()!='wsl.exe'
            or not any('launch_reasoning_vpd_india.sh' in arg for arg in controller.cmdline())):
        raise ValueError('Observed process is not the expected live VPD launcher')
    def state(status,**extra):
        data=dict(status=status,updated_unix=time.time(),controller_pid=args.controller_pid,
                  controller_created=args.controller_created,**extra)
        temporary=state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        temporary.replace(state_path)
    state('waiting_for_live_launcher')
    deadline=time.monotonic()+args.max_wait_seconds
    while True:
        try:
            if abs(controller.create_time()-args.controller_created)>.01:
                break
            controller.wait(timeout=5)
            break
        except psutil.TimeoutExpired:
            if time.monotonic()>deadline:
                state('wait_bound_reached',launcher_not_terminated_by_pipeline=True)
                raise SystemExit(75)
        except psutil.NoSuchProcess:
            break
    if not (run/'termination.json').exists() or not read(run/'termination.json')['termination_command_succeeded']:
        state('requires_attention',reason='Launcher ended without a successful termination receipt')
        raise SystemExit(1)
    if not (run/'results.tar.gz').exists():
        state('failed_before_retrieval',termination=read(run/'termination.json'))
        raise SystemExit(1)
    def stage(name,module,arguments):
        command=[sys.executable,'-X','utf8','-m',module,*map(str,arguments)]
        state('running_stage',stage=name)
        with (run/f'analysis_{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        if result.returncode:
            state('analysis_failed',stage=name,exit_code=result.returncode)
            raise SystemExit(result.returncode)
    try:
        if not (run/'retrieval_verification.json').exists():
            stage('collection','scripts.nanbeige.collect_multipass',[run])
        else:
            from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest
            if read(run/'retrieval_verification.json')['archive_sha256']!=digest(run/'results.tar.gz'):
                raise ValueError('Existing retrieval receipt belongs to a different archive')
        if not read(run/'retrieval_verification.json')['calibration_complete']:
            state('calibration_incomplete',termination=read(run/'termination.json'))
            raise SystemExit(1)
        calibration=run/'retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration'
        stage('saved_effect_verification','scripts.nanbeige.verify_multipass_results',[calibration])
        stage('source_verification','scripts.nanbeige.verify_cross_precision_vpd',[run])
        stage('factor_coverage','scripts.nanbeige.check_factor_coverage',
              [calibration,ROOT/'models/Nanbeige4.2-3B-tokenizer'])
        stage('effect_summary','scripts.nanbeige.summarize_multipass',[calibration])
        state('analysis_complete',calibration_dir=str(calibration),
              mechanism_or_full_benchmark_claim=False)
        print(json.dumps(dict(status='analysis_complete',calibration_dir=str(calibration))))
    except Exception as error:
        state('analysis_failed',exception_type=type(error).__name__,message=str(error))
        raise


if __name__=='__main__':
    main()
