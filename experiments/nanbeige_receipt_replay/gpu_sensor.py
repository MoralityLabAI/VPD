"""Read GPU temperature through NVML without repeatedly launching nvidia-smi.

Read-only binding to NVIDIA's installed library; never changes clock/power state.
API: https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
"""
import atexit
import ctypes as C
import os
from pathlib import Path
import threading

_sensor=None
_lock=threading.Lock()


class NvidiaTemperature:
    def __init__(self):
        path=Path(os.environ['SystemRoot'])/'System32/nvml.dll'
        self.lib=C.CDLL(str(path));self.closed=True
        declarations={
            'nvmlInit_v2':[], 'nvmlShutdown':[],
            'nvmlDeviceGetCount_v2':[C.POINTER(C.c_uint)],
            'nvmlDeviceGetHandleByIndex_v2':[C.c_uint,C.POINTER(C.c_void_p)],
            'nvmlDeviceGetTemperature':[C.c_void_p,C.c_int,C.POINTER(C.c_uint)]}
        for name,args in declarations.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=C.c_int
        self.check(self.lib.nvmlInit_v2());self.closed=False
        try:
            count=C.c_uint();self.check(self.lib.nvmlDeviceGetCount_v2(C.byref(count)))
            if count.value!=1:raise RuntimeError('Exactly one GPU is required')
            self.device=C.c_void_p();self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0,C.byref(self.device)))
        except BaseException:self.close();raise

    @staticmethod
    def check(result):
        if result!=0:raise RuntimeError(f'NVML temperature query failed with code {result}')

    def __call__(self):
        if self.closed:raise RuntimeError('NVML sensor closed')
        value=C.c_uint();self.check(self.lib.nvmlDeviceGetTemperature(self.device,0,C.byref(value)))
        if not 0<=value.value<=120:raise RuntimeError('Invalid GPU temperature')
        return float(value.value)

    def close(self):
        if not self.closed:self.lib.nvmlShutdown();self.closed=True


def gpu_temperature():
    global _sensor
    with _lock:
        if _sensor is None:
            _sensor=NvidiaTemperature();atexit.register(_sensor.close)
        return _sensor()
