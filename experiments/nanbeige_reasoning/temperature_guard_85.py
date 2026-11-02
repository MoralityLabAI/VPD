"""User-requested 85C cooling threshold for owned C-API decoding."""
import ctypes as C
import json
import subprocess
import time
from pathlib import Path


def gpu_temperature():
    result=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=5,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    lines=result.stdout.strip().splitlines()
    if len(lines)!=1: raise RuntimeError('Expected one GPU temperature reading')
    value=float(lines[0])
    if not 0<=value<=120: raise RuntimeError('Invalid GPU temperature reading')
    return value


class TemperatureGuard:
    def __init__(self,log,*,sensor=gpu_temperature,clock=time.monotonic,sleep=time.sleep,wall_seconds=7200):
        self.log=Path(log);self.sensor=sensor;self.clock=clock;self.sleep=sleep
        self.started=clock();self.last_check=-float('inf');self.last_decode_seconds=0.
        self.wall_seconds=wall_seconds;self.last_temperature=None;self.peak_temperature=None
        self.calls=0;self.cooling_pauses=0;self.idle_seconds=0.
        self.policy=dict(pause_c=85,resume_c=78,stop_c=87,poll_seconds=2,cooling_timeout_seconds=180,
            idle_to_decode_ratio=1.,wall_seconds=wall_seconds,synchronize_after_decode=True)

    def event(self,kind,**values):
        with self.log.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(event=kind,elapsed_seconds=self.clock()-self.started,**values))+'\n')

    def check(self,force=False):
        if self.clock()-self.started>=self.wall_seconds: raise RuntimeError('Guard wall budget reached')
        if not force and self.clock()-self.last_check<2: return
        try: temperature=self.sensor()
        except Exception as exc:
            self.event('sensor_failure',error=f'{type(exc).__name__}: {exc}')
            raise RuntimeError('Temperature unavailable; stopping decoding') from exc
        self.last_check=self.clock();self.last_temperature=temperature
        self.peak_temperature=max(temperature,self.peak_temperature or temperature)
        self.event('temperature',temperature_c=temperature,calls=self.calls)
        if temperature>=87: raise RuntimeError('Temperature hard-stop threshold reached')
        if temperature>=85:
            self.cooling_pauses+=1;start=self.clock();self.event('cooling_start',temperature_c=temperature)
            while temperature>78:
                if self.clock()-start>=180 or self.clock()-self.started>=self.wall_seconds:
                    raise RuntimeError('Cooling or wall budget exhausted; stopping decoding')
                self.sleep(2);self.idle_seconds+=2
                try: temperature=self.sensor()
                except Exception as exc: raise RuntimeError('Temperature unavailable during cooling') from exc
                self.last_check=self.clock();self.last_temperature=temperature
                self.peak_temperature=max(temperature,self.peak_temperature)
                self.event('cooling_temperature',temperature_c=temperature)
                if temperature>=87: raise RuntimeError('Temperature hard-stop threshold reached during cooling')
            self.event('cooling_end',temperature_c=temperature,paused_seconds=self.clock()-start)

    def attach(self,model):
        self.check(force=True)
        native=model.lib.llama_decode
        synchronize=model.lib.llama_synchronize
        synchronize.argtypes=[C.c_void_p];synchronize.restype=None
        def guarded_decode(context,batch):
            self.sleep(self.last_decode_seconds);self.idle_seconds+=self.last_decode_seconds
            self.check()
            started=self.clock()
            result=native(context,batch)
            if result==0:synchronize(context)
            self.last_decode_seconds=self.clock()-started;self.calls+=1
            return result
        model.lib.llama_decode=guarded_decode
        self.event('attached',policy=self.policy)

    def summary(self):
        return dict(policy=self.policy,calls=self.calls,peak_temperature_c=self.peak_temperature,
            cooling_pauses=self.cooling_pauses,idle_seconds=self.idle_seconds,elapsed_seconds=self.clock()-self.started)
