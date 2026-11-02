"""Tokenizer/episode adapter for the repository's existing cached inference class."""
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from transformers import AutoTokenizer
from .backend import NanbeigeBackend,ROOT,Q4_SHA256,digest,save
from .native_text import verify_native_text
from .io import read_json


class NativeNanbeigeBackend(NanbeigeBackend):
    def __init__(self,*args,enable_thinking=True,**kwargs):
        if type(enable_thinking) is not bool:raise ValueError('enable_thinking must be boolean')
        self.enable_thinking=enable_thinking
        self.output_decoder=None
        super().__init__(*args,**kwargs)

    def render(self,messages):
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
            enable_thinking=self.enable_thinking,preserve_thinking=True)

    def __enter__(self):
        if self.gpu_layers!='all':raise ValueError('This cached worker configuration requires CUDA')
        if self.cpu_threads!=2:raise ValueError('The cached worker uses two CPU threads')
        if any(self.output.iterdir()):raise ValueError('Choose a fresh native runtime directory')
        reference=json.loads((ROOT/'artifacts/nanbeige_reasoning/q4_reasoning_pilot_8192_v1/contract.json').read_text(encoding='utf-8'))
        assert digest(ROOT/'models/Nanbeige4.2-3B-Q4_K_M.gguf')==Q4_SHA256==reference['model_sha256']
        for name,sha in reference['runtime_files'].items():assert digest(ROOT/'tools/nanbeige/runtime'/name)==sha
        tdir=ROOT/'models/Nanbeige4.2-3B-tokenizer'
        self.tokenizer=AutoTokenizer.from_pretrained(tdir,use_fast=False,local_files_only=True)
        from tokenizers import Tokenizer
        self.output_decoder=Tokenizer.from_file(str(tdir/'tokenizer.json'))
        fixture=[dict(role='user',content='First'),dict(role='assistant',content='done',reasoning_content='HISTORY_REASONING_SENTINEL'),dict(role='user',content='Next')]
        rendered=self.render(fixture)
        suffix='<think>\n' if self.enable_thinking else '<think>\n\n</think>\n\n'
        assert 'HISTORY_REASONING_SENTINEL' in rendered and rendered.endswith(suffix)
        command=[sys.executable,'-m','experiments.nanbeige_receipt_replay.native_worker','--output',str(self.output),'--context',str(self.context),'--wall-seconds',str(self.wall_seconds)]
        save(self.output/'runtime_contract.json',dict(transport='existing_cached_c_api_worker',model_sha256=Q4_SHA256,runtime_files=reference['runtime_files'],
            loops=2,physical_layers=22,context=self.context,quantization='Q4_K_M',kv_cache='q8_0',execution_device='cuda',
            template_sha256=digest(tdir/'tokenizer_config.json'),enable_thinking=self.enable_thinking,preserve_thinking=True,history_reasoning_fixture_passed=True,
            output_decoder='pinned tokenizer.json; avoids legacy SentencePiece special-token spaces',output_decoder_sha256=digest(tdir/'tokenizer.json'),
            command=command,adapter_sha256=digest(Path(__file__)),code_sha256={name:digest(ROOT/'experiments/nanbeige_reasoning'/name) for name in ['llama_capture.py','llama_steer.py','llama_cached_steer.py','temperature_guard_burst.py']},
            execution_note='No OS process suspension. Existing guard waits between synchronized decode calls.',wall_seconds=self.wall_seconds))
        self.log=(self.output/'worker.log').open('w',encoding='utf-8')
        self.process=subprocess.Popen(command,cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline=time.monotonic()+180
            while not (self.output/'ready.json').exists():
                if self.process.poll() is not None:raise RuntimeError('Cached worker exited during startup; inspect worker.log')
                if time.monotonic()>deadline:raise TimeoutError('Cached worker startup timeout')
                time.sleep(.2)
            return self
        except BaseException:self.__exit__(None,None,None);raise

    def __call__(self,*,messages,seed,metadata,temperature,max_new_tokens):
        if self.thermal_error:raise RuntimeError(self.thermal_error)
        prompt=self.render(messages);ids=self.tokenizer.encode(prompt,add_special_tokens=False)
        if len(ids)+max_new_tokens>self.context:raise ValueError('Full history plus output cap exceeds context; refusing truncation')
        self.calls+=1;request_path=self.output/'requests'/f'{self.calls:05d}.json';result_path=self.output/'results'/request_path.name
        save(request_path,dict(messages=messages,metadata=metadata,seed=seed,temperature=temperature,max_new_tokens=max_new_tokens,input_ids=ids,rendered_prompt=prompt))
        request_sha256=digest(request_path)
        try:
            while True:
                if result_path.exists():
                    result=read_json(result_path)
                    if result['request_sha256']!=request_sha256:raise RuntimeError('Worker response belongs to a different request')
                    if result.get('error'):raise RuntimeError(result['error'])
                    if result['completed']:break
                if self.process.poll() is not None:raise RuntimeError('Cached worker stopped before completing response')
                if time.monotonic()-self.started>self.wall_seconds+180:raise TimeoutError('Worker wall budget expired')
                time.sleep(.1)
            if self.output_decoder is None:raise RuntimeError('Native output decoder was not initialized')
            tokenizer_text=self.output_decoder.decode(result['output_ids'],skip_special_tokens=False)
            raw=result['raw_decoded_output']
            text_match=verify_native_text(tokenizer_text,raw)
            clean=raw;eos=self.tokenizer.eos_token
            if eos and clean.endswith(eos):clean=clean[:-len(eos)]
            if '</think>' in clean:
                thought,visible=clean.split('</think>',1);thought=thought.removeprefix('<think>');status='complete'
            elif self.enable_thinking or clean.startswith('<think>'):
                thought=clean.removeprefix('<think>');visible='';status='partial_unclosed'
            else:thought=None;visible=clean;status='not_requested'
            reply=dict(content=visible,reasoning_content=thought,raw_decoded_output=raw,reasoning_status=status,
                finish_reason=result['finish_reason'],input_ids=ids,output_ids=result['output_ids'],rendered_prompt=prompt,
                generation_context=dict(enable_thinking=self.enable_thinking,preserve_thinking=True,loops=2,quantization='Q4_K_M',context_tokens=self.context,execution_device='cuda',transport='existing_cached_c_api_worker',prefill_chunk_size=8),
                elapsed_seconds=result['elapsed_seconds'],tokenizer_match=None,native_hf_decoded_text_match=text_match=='exact',
                native_tokenizer_text_relation=text_match,tokenizer_decoded_output=tokenizer_text,
                output_decoder='pinned tokenizer.json',
                generation_settings=dict(temperature=temperature,seed=seed,top_k=20,top_p=.95,min_p=.05,min_keep=0,repeat_penalty=1.,sampler='native pinned llama.cpp chain'),
                call_record=str(result_path.relative_to(ROOT)))
            save(self.output/'calls'/request_path.name,dict(request=json.loads(request_path.read_text(encoding='utf-8')),response=reply,completed=True))
            print(json.dumps(dict(call=self.calls,phase=metadata.get('phase'),tokens=len(result['output_ids']),finish=result['finish_reason'],seconds=round(result['elapsed_seconds'],1))),flush=True)
            return reply
        except BaseException as exc:self.thermal_error=f'{type(exc).__name__}: {exc}';raise

    def __exit__(self,*args):
        save(self.output/'stop_requested.json',dict(stop=True))
        if self.process and self.process.poll() is None:
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Windows venv launchers have a separate Python worker child.
                # Stop that owned child as well as the launcher on forced exit.
                try:children=psutil.Process(self.process.pid).children(recursive=True)
                except psutil.NoSuchProcess:children=[]
                for child in children:
                    try:
                        args=child.cmdline()
                        if 'experiments.nanbeige_receipt_replay.native_worker' in args and str(self.output) in args:
                            child.terminate()
                            try:child.wait(timeout=10)
                            except psutil.TimeoutExpired:child.kill();child.wait(timeout=10)
                    except psutil.NoSuchProcess:pass
                self.process.terminate();self.process.wait(timeout=10)
        if self.log:self.log.close()
        save(self.output/'server_stopped.json',dict(stopped=self.process is None or self.process.poll() is not None,transport='existing_cached_c_api_worker',
            pid=self.process.pid if self.process else None,calls=self.calls,guard_error=self.thermal_error))
