"""Ledger adapter for the repository's pinned Nanbeige raw-token server."""
from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
import urllib.error
import psutil
from transformers import AutoTokenizer
from experiments.nanbeige_looped.q4_smoke import ROOT, Q4_SHA256, digest
from experiments.nanbeige_reasoning.run_q4_reasoning import RAW_TOKEN_PARSER, request
from .gpu_sensor import gpu_temperature
from .io import save


class NanbeigeBackend(AbstractContextManager):
    def __init__(self, output, *, context=8192, port=8097, wall_seconds=3600, gpu_layers='all', cpu_threads=2):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.context=context;self.port=port;self.wall_seconds=wall_seconds
        if gpu_layers not in ('all','0'):raise ValueError('Use full GPU offload or explicit CPU execution')
        self.gpu_layers=gpu_layers;self.cpu_threads=cpu_threads
        self.process=None;self.log=None;self.thread=None;self.stop=threading.Event();self.thermal_error=None
        self.calls=0;self.started=time.monotonic();self.key=secrets.token_hex(32)
        self.base=f'http://127.0.0.1:{port}'

    def _guard(self):
        proc=psutil.Process(self.process.pid);created=proc.create_time();suspended=False
        previous=None
        try:
            while not self.stop.wait(.02):
                if proc.create_time()!=created or not proc.is_running():return
                temp=gpu_temperature()
                now=time.monotonic()
                rate=max(0.,(temp-previous[0])/(now-previous[1])) if previous else 0.
                anticipate=temp>=84 and temp+rate*.04>=85
                previous=(temp,now)
                with (self.output/'thermal.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(unix=time.time(),temperature_c=temp,suspended=suspended,rise_c_per_second=rate,anticipatory_pause=anticipate))+'\n')
                if temp>=87:raise RuntimeError('GPU reached 87C hard stop')
                if (temp>=85 or anticipate) and not suspended:proc.suspend();suspended=True
                elif suspended and temp<=84:proc.resume();suspended=False
                if time.monotonic()-self.started>=self.wall_seconds:raise TimeoutError('Inference wall budget reached')
        except Exception as exc:
            self.thermal_error=f'{type(exc).__name__}: {exc}'
            save(self.output/'guard_failure.json',dict(error=self.thermal_error))
            if self.process.poll() is None:self.process.terminate()
        finally:
            if suspended:
                try:proc.resume()
                except psutil.Error:pass

    def __enter__(self):
        model=ROOT/'models/Nanbeige4.2-3B-Q4_K_M.gguf'
        runtime=ROOT/'tools/nanbeige/runtime';executable=runtime/'llama-server.exe'
        reference=json.loads((ROOT/'artifacts/nanbeige_reasoning/q4_reasoning_pilot_8192_v1/contract.json').read_text(encoding='utf-8'))
        assert digest(model)==Q4_SHA256==reference['model_sha256']
        for name,h in reference['runtime_files'].items():assert digest(runtime/name)==h,name
        tdir=ROOT/'models/Nanbeige4.2-3B-tokenizer'
        self.tokenizer=AutoTokenizer.from_pretrained(tdir,use_fast=False,local_files_only=True)
        fixture=[{'role':'user','content':'First.'},{'role':'assistant','content':'done','reasoning_content':'HISTORY_REASONING_SENTINEL'}, {'role':'user','content':'Next.'}]
        rendered=self.render(fixture)
        assert 'HISTORY_REASONING_SENTINEL' in rendered and rendered.endswith('<think>\n')
        command=[str(executable),'-m',str(model),'-c',str(self.context),'-b','64','-ub','64',
            '-t',str(self.cpu_threads),'-tb',str(self.cpu_threads),'--poll','0','--poll-batch','0',
            '-ngl',self.gpu_layers,'--fit','off','-fa','on','-np','1','-ctk','q8_0','-ctv','q8_0',
            '--no-context-shift','--cache-ram','0','--host','127.0.0.1','--port',str(self.port),
            '--override-kv','nanbeige.num_loops=int:2','-lv','4']
        if self.gpu_layers=='0':command+=['--no-op-offload','--no-kv-offload']
        save(self.output/'runtime_contract.json',dict(model_sha256=Q4_SHA256,runtime_files=reference['runtime_files'],
            revision='451ed48c3273ecef7ea8faaa43c31ce529763bb1',loops=2,physical_layers=22,context=self.context,
            quantization='Q4_K_M',kv_cache='q8_0',template_sha256=digest(tdir/'tokenizer_config.json'),
            execution_device='cpu' if self.gpu_layers=='0' else 'cuda',cpu_threads=self.cpu_threads,
            enable_thinking=True,preserve_thinking=True,history_reasoning_fixture_passed=True,
            command=command,adapter_sha256=digest(Path(__file__)),wall_seconds=self.wall_seconds,
            thermal_policy=dict(pause_c=85,resume_c=84,hard_stop_c=87,poll_seconds=.02,sensor='read-only native NVML',anticipatory_pause='At 84C or above when linear 0.04-second rise projection reaches 85C'),token_ids_authoritative=True))
        env=os.environ.copy();env['LLAMA_API_KEY']=self.key
        self.log=(self.output/'server.log').open('w',encoding='utf-8')
        self.process=subprocess.Popen(command,env=env,stdout=self.log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        self.thread=threading.Thread(target=self._guard,daemon=True);self.thread.start()
        try:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError('Server exited; inspect server.log')
                try:
                    if request(self.base,self.key,'/health',timeout=2).get('status')=='ok':break
                except (urllib.error.URLError,TimeoutError):pass
                time.sleep(.5)
            else:raise TimeoutError('Server startup timed out')
            props=request(self.base,self.key,'/props',timeout=10);save(self.output/'runtime_props.json',props)
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def render(self,messages):
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
            enable_thinking=True,preserve_thinking=True)

    def __call__(self, *, messages, seed, metadata, temperature, max_new_tokens):
        if self.thermal_error:raise RuntimeError(self.thermal_error)
        remaining=self.wall_seconds-(time.monotonic()-self.started)
        if remaining<=0:raise TimeoutError('Inference wall budget exhausted')
        rendered=self.render(messages)
        ids=self.tokenizer.encode(rendered,add_special_tokens=False)
        if len(ids)+max_new_tokens>self.context:raise ValueError('Full history plus output cap exceeds context; refusing truncation')
        self.calls+=1;path=self.output/'calls'/f'{self.calls:05d}.json'
        row=dict(messages=messages,metadata=metadata,seed=seed,temperature=temperature,max_new_tokens=max_new_tokens,
            rendered_prompt=rendered,input_ids=ids,started_unix=time.time())
        save(path,row)
        native=request(self.base,self.key,'/tokenize',dict(content=rendered,add_special=False,parse_special=True),timeout=30)['tokens']
        row['native_tokenizer_match']=native==ids
        # Existing experiments supply pinned HF token IDs directly even if GGUF
        # text tokenization differs. Record any drift; never silently retokenize.
        row['native_tokenizer_ids']=native
        before=time.monotonic()
        try:
            completion=request(self.base,self.key,'/completion',dict(prompt=ids,n_predict=max_new_tokens,
                temperature=temperature,seed=seed,repeat_penalty=1.0,cache_prompt=False,return_tokens=True,
                stream=False,ignore_eos=False,chat_parser=RAW_TOKEN_PARSER),timeout=max(1,remaining))
            row['server_response']=completion
            save(path,row)
            output=completion.get('tokens')
            if not output or completion.get('truncated'):raise RuntimeError('Empty generation or truncated context')
            raw=self.tokenizer.decode(output,skip_special_tokens=False)
            clean=raw
            eos=self.tokenizer.eos_token
            if eos and clean.endswith(eos):clean=clean[:-len(eos)]
            if '</think>' in clean:
                thought,visible=clean.split('</think>',1)
                if thought.startswith('<think>'):thought=thought[len('<think>'):]
                status='complete'
            else:
                thought=clean;visible='';status='partial_unclosed'
            reply=dict(content=visible,reasoning_content=thought,raw_decoded_output=raw,
                reasoning_status=status,finish_reason='length' if completion.get('stop_type')=='limit' else 'stop',
                input_ids=ids,output_ids=output,rendered_prompt=rendered,
                generation_context=dict(enable_thinking=True,preserve_thinking=True,loops=2,quantization='Q4_K_M',context_tokens=self.context,
                    execution_device='cpu' if self.gpu_layers=='0' else 'cuda'),
                backend_stop_type=completion.get('stop_type'),tokenizer_match=native==ids,
                timings=completion.get('timings'),generation_settings=completion.get('generation_settings'),
                elapsed_seconds=time.monotonic()-before,call_record=str(path.relative_to(ROOT)))
            row.update(response=reply,server_response=completion,completed=True);save(path,row)
            print(json.dumps(dict(call=self.calls,phase=metadata.get('phase'),tokens=len(output),finish=reply['finish_reason'],seconds=round(reply['elapsed_seconds'],1))),flush=True)
            return reply
        except BaseException as exc:
            row.update(completed=False,error=f'{type(exc).__name__}: {exc}');save(path,row);raise

    def __exit__(self,*args):
        self.stop.set()
        if self.thread:self.thread.join(timeout=8)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.log:self.log.close()
        save(self.output/'server_stopped.json',dict(stopped=self.process is None or self.process.poll() is not None,
            pid=self.process.pid if self.process else None,calls=self.calls,guard_error=self.thermal_error))
