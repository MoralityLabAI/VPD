"""Bounded, logged cooling pauses for this experiment's own inference server."""
import argparse
import json
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
import psutil


@contextmanager
def cooling(server_pid,out,*,enable_pauses=False):
    # The observed 30-second pause did not recover sustained throughput.
    # Keep interruption opt-in; normal experiment runs use driver-managed cooling.
    if not enable_pauses:
        yield
        return
    server=psutil.Process(server_pid)
    created=server.create_time()
    expected=Path('tools/nanbeige/runtime/llama-server.exe').resolve()
    if Path(server.exe()).resolve()!=expected:raise ValueError('Not the owned inference executable')
    stop=threading.Event()
    def worker():
        pauses=0
        while not stop.wait(15) and pauses<64:
            suspended=False
            try:
                if server.create_time()!=created or not server.is_running():return
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,clocks_event_reasons.sw_thermal_slowdown',
                    '--format=csv,noheader,nounits'],text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW)
                if len(raw.strip().splitlines())!=1:continue
                temperature_text,clock_text,thermal_text=[x.strip() for x in raw.strip().split(',')]
                temperature,clock=int(temperature_text),int(clock_text)
                thermal_active=thermal_text.lower()=='active'
                if not ((temperature>=85 and thermal_active) or (temperature>=87 and clock<=600)):continue
                before=time.monotonic()
                server.suspend();suspended=True
                stop.wait(30)
                server.resume();suspended=False
                pauses+=1
                with (Path(out)/'thermal_pauses.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(pid=server_pid,created=created,unix=time.time(),
                        temperature_c=temperature,clock_mhz=clock,sw_thermal_slowdown=thermal_active,
                        policy='driver_thermal_flag_v2',paused_seconds=time.monotonic()-before,resumed=True))+'\n')
                print(f'Cooling pause {pauses}: owned server {server_pid} resumed',flush=True)
                if stop.wait(30):return
            except (psutil.NoSuchProcess,psutil.ZombieProcess):return
            except (psutil.AccessDenied,subprocess.SubprocessError,ValueError):
                if stop.wait(30):return
            finally:
                if suspended:
                    try:server.resume()
                    except psutil.NoSuchProcess:pass
    thread=threading.Thread(target=worker,name='experiment-cooling',daemon=True)
    thread.start()
    try:yield
    finally:stop.set();thread.join(timeout=10)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller-pid',type=int,required=True)
    p.add_argument('--controller-created',type=float,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--enable-pauses',action='store_true',help='Explicitly opt into experimental cooling interruptions')
    args=p.parse_args()
    controller=psutil.Process(args.controller_pid)
    if controller.create_time()!=args.controller_created or 'experiments.nanbeige_reasoning.run_q4_reasoning' not in controller.cmdline():
        raise ValueError('Controller identity mismatch')
    while controller.is_running():
        candidates=[c for c in controller.children() if c.name().lower()=='llama-server.exe']
        if not candidates:
            time.sleep(2);continue
        if len(candidates)!=1:raise ValueError('Ambiguous owned server')
        server=candidates[0]
        print('Cooling guard attached to server',server.pid,flush=True)
        with cooling(server.pid,args.out,enable_pauses=args.enable_pauses):
            try:server.wait()
            except psutil.NoSuchProcess:pass


if __name__=='__main__':main()
