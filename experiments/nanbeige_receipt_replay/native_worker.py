"""Existing cached Nanbeige inference, with safe pauses after GPU synchronization.

Pure ctypes/NumPy subprocess: no Torch or tokenizer library enters this process.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import time
from experiments.nanbeige_reasoning.llama_capture import Runtime,P,I,F,U,B
from experiments.nanbeige_reasoning.llama_cached_steer import CachedSteeringModel
from experiments.nanbeige_reasoning.run_q4_feature_answers import native_decoder
from .adaptive_pacing import AdaptivePacingGuard
from .gpu_sensor import gpu_temperature
from .io import ROOT,digest,save,read_json


class ChainParams(C.Structure):_fields_=[('no_perf',B)]


class NativeSampler:
    """ABI and ownership follow the repository's pinned llama.h, lines 1238–1540."""
    def __init__(self,lib,seed,temperature):
        self.lib=lib
        specs={
            'llama_sampler_chain_default_params':(ChainParams,[]),
            'llama_sampler_chain_init':(P,[ChainParams]),'llama_sampler_chain_add':(None,[P,P]),
            'llama_sampler_init_top_k':(P,[I]),'llama_sampler_init_top_p':(P,[F,C.c_size_t]),
            'llama_sampler_init_min_p':(P,[F,C.c_size_t]),'llama_sampler_init_temp':(P,[F]),
            'llama_sampler_init_dist':(P,[U]),'llama_sampler_sample':(I,[P,P,I]),'llama_sampler_free':(None,[P])}
        for name,(result,args) in specs.items():
            fn=getattr(lib,name);fn.restype=result;fn.argtypes=args
        self.pointer=lib.llama_sampler_chain_init(lib.llama_sampler_chain_default_params())
        if not self.pointer:raise RuntimeError('Sampler allocation failed')
        for fn,args in [('top_k',(20,)),('top_p',(.95,0)),('min_p',(.05,0)),('temp',(temperature,)),('dist',(seed,))]:
            child=getattr(lib,'llama_sampler_init_'+fn)(*args)
            if not child:self.close();raise RuntimeError('Sampler component allocation failed')
            lib.llama_sampler_chain_add(self.pointer,child)
    def sample(self,context):return int(self.lib.llama_sampler_sample(self.pointer,context,-1))
    def close(self):
        if self.pointer:self.lib.llama_sampler_free(self.pointer);self.pointer=None


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--context',type=int,required=True)
    p.add_argument('--wall-seconds',type=int,required=True);a=p.parse_args();out=a.output
    guard=AdaptivePacingGuard(out/'thermal.jsonl',sensor=gpu_temperature,wall_seconds=a.wall_seconds)
    guard.policy.update(resume_c=80,cooling_poll_seconds=.05,idle_to_decode_ratio=2.,minimum_idle_seconds=.02,warm_minimum_idle_seconds=.3,poll_seconds=.02,check_after_decode=True)
    model=None
    try:
        guard.check(force=True)
        model=CachedSteeringModel(Runtime(ROOT/'tools/nanbeige/runtime',require_cuda=True),ROOT/'models/Nanbeige4.2-3B-Q4_K_M.gguf',
            context=a.context,gpu_layers=-1,threads=2)
        guard.attach(model);vocab,decode=native_decoder(model);eog=set(vocab['eog_token_ids'])
        save(out/'ready.json',dict(status='ready',vocab=vocab,thermal_policy=guard.policy,prefill_chunk_size=8,
            sampler=dict(chain=['top_k','top_p','min_p','temperature','dist'],top_k=20,top_p=.95,min_p=.05,min_keep=0),
            worker_sha256=digest(Path(__file__)),pacing_guard_sha256=digest(Path(__file__).with_name('adaptive_pacing.py'))))
        index=1
        while not (out/'stop_requested.json').exists():
            guard.check()
            path=out/'requests'/f'{index:05d}.json'
            if not path.exists():time.sleep(.05);continue
            request=read_json(path);tokens=request['input_ids'];generated=[];sampler=None;start=time.monotonic()
            result=dict(completed=False,output_ids=generated,request_sha256=digest(path))
            dest=out/'results'/f'{index:05d}.json'
            try:
                if len(tokens)+request['max_new_tokens']>a.context:raise ValueError('Full history exceeds context budget')
                model.reset_cache();model.evaluate(tokens,chunk_size=8)
                sampler=NativeSampler(model.lib,request['seed'],request['temperature'])
                finish='length'
                for step in range(request['max_new_tokens']):
                    token=sampler.sample(model.context);generated.append(token)
                    if token in eog:finish='stop';break
                    if step%16==0:save(dest,result)
                    if step+1<request['max_new_tokens']:model.evaluate([token],chunk_size=1)
                result.update(completed=True,finish_reason=finish,raw_decoded_output=decode(generated),elapsed_seconds=time.monotonic()-start)
                save(dest,result)
                print(json.dumps(dict(call=index,tokens=len(generated),finish=finish,seconds=result['elapsed_seconds'])),flush=True)
            except BaseException as exc:
                result.update(error=f'{type(exc).__name__}: {exc}',raw_decoded_output=decode(generated),elapsed_seconds=time.monotonic()-start)
                save(dest,result);raise
            finally:
                if sampler:sampler.close()
            index+=1
    finally:
        if model:model.close()
        save(out/'worker_stopped.json',dict(stopped=True,thermal=guard.summary()))


if __name__=='__main__':main()
