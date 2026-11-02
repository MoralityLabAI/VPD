"""Run frozen answer-only calls; no grading or reference sidecars in this process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import time
import urllib.request
import urllib.error
from transformers import AutoTokenizer


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def request(base,token,route,data=None,timeout=60):
    req=urllib.request.Request(base+route,data=json.dumps(data).encode() if data is not None else None,
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response: return json.load(response)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--server',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=Path('artifacts/nanbeige_reasoning/logic_skill_comparison_v1'))
    p.add_argument('--sources',type=Path,default=Path('artifacts/nanbeige_reasoning/comparator_sources/qwen35_27b'))
    p.add_argument('--output',type=Path,default=Path('artifacts/nanbeige_reasoning/logic_skill_comparison_outputs_v1'))
    p.add_argument('--wall-seconds',type=int,default=18000)
    args=p.parse_args()
    cfg=json.loads((args.inputs/'protocol.json').read_text())
    assert sha(args.inputs/'calls.jsonl')==cfg['calls_sha256']
    assert sha(args.sources/'model_lock.json')==cfg['model_lock_sha256']
    for name,h in cfg['tokenizer_assets'].items(): assert sha(args.sources/'official'/name)==h
    lock=json.loads((args.sources/'model_lock.json').read_text())
    weight=lock['quantized']['weights'][0]
    assert args.model.stat().st_size==weight['bytes'] and sha(args.model)==weight['sha256']
    rows=[json.loads(x) for x in (args.inputs/'calls.jsonl').read_text().splitlines()]
    assert len(rows)==len({x['call_id'] for x in rows})==259
    assert max(len(x['token_ids']) for x in rows)+cfg['max_new_tokens']<=cfg['context']
    tok=AutoTokenizer.from_pretrained(args.sources/'official',local_files_only=True,use_fast=False)
    version=subprocess.run([str(args.server.resolve()),'--version'],capture_output=True,text=True,check=True)
    assert '0eadefe' in version.stdout+version.stderr
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    assert not (out/'contract.json').exists(), 'Fresh run only; preserve existing outputs'
    contract=dict(protocol_sha256=sha(args.inputs/'protocol.json'),model_sha256=sha(args.model),
        calls_sha256=sha(args.inputs/'calls.jsonl'),runner_sha256=sha(Path(__file__)),
        runtime_version=version.stdout+version.stderr,
        runtime_files={f.name:sha(f) for f in args.server.parent.iterdir() if f.is_file() and ('.so' in f.name or f==args.server)},
        expected_records=len(rows),generation_settings={k:cfg[k] for k in
        ('max_new_tokens','context','thinking_enabled','seed','temperature','repeat_penalty','kv_cache','weights')})
    write(out/'contract.json',contract)
    token=secrets.token_hex(32); env=dict(os.environ,LLAMA_API_KEY=token)
    base='http://127.0.0.1:8098'
    command=[str(args.server.resolve()),'-m',str(args.model.resolve()),'-c',str(cfg['context']),
        '-b','512','-ub','512','-ngl','all','--fit','off','-fa','on','-np','1',
        '-ctk','q8_0','-ctv','q8_0','--no-context-shift','--cache-ram','0',
        '--host','127.0.0.1','--port','8098','-lv','4']
    write(out/'command.json',command)
    started=time.monotonic(); completed=0
    with (out/'server.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+300
            while True:
                if proc.poll() is not None: raise RuntimeError('Server exited during loading')
                try:
                    if request(base,token,'/health',timeout=2).get('status')=='ok': break
                except (urllib.error.URLError,TimeoutError): pass
                if time.monotonic()>deadline: raise TimeoutError('Server loading')
                time.sleep(1)
            offload=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',(out/'server.log').read_text(errors='replace'))
            assert offload and all(int(a)==int(b)>0 for a,b in offload)
            assert 'CUDA0' in (out/'server.log').read_text(errors='replace')
            gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw',
                                '--format=csv'],capture_output=True,text=True,check=True)
            (out/'gpu_after_load.txt').write_text(gpu.stdout)
            try: request(base,'invalid','/props')
            except urllib.error.HTTPError as exc: assert exc.code==401
            else: raise ValueError('Server authentication is disabled')
            write(out/'runtime_props.json',request(base,token,'/props'))
            for row in rows:
                native=request(base,token,'/tokenize',dict(content=row['rendered_prompt'],add_special=False,parse_special=True))
                assert native['tokens']==row['token_ids'],row['call_id']
            write(out/'tokenizer_preflight.json',dict(status='all_inputs_match',records=len(rows),full_cuda_offload=True))
            # Parser bypass preserves the actual output token stream.
            RAW_TOKEN_PARSER=json.dumps(dict(parsers=[dict(type='epsilon')],rules={},root=0))
            for index,row in enumerate(rows):
                if time.monotonic()-started>args.wall_seconds: raise TimeoutError('Generation wall bound')
                before=time.monotonic()
                response=request(base,token,'/completion',dict(prompt=row['token_ids'],
                    n_predict=cfg['max_new_tokens'],temperature=0,seed=42,repeat_penalty=1.,
                    return_tokens=True,cache_prompt=False,stream=False,ignore_eos=False,
                    chat_parser=RAW_TOKEN_PARSER),timeout=1200)
                ids=response.get('tokens')
                assert ids and not response.get('truncated')
                decoded=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                native=request(base,token,'/detokenize',dict(tokens=ids),timeout=30)
                result=dict(call_id=row['call_id'],id=row['id'],arm=row['arm'],
                    prompt_tokens=len(row['token_ids']),generated_tokens=len(ids),generated_token_ids=ids,
                    generated_text=decoded,native_text=native['content'],response_content=response.get('content'),
                    stop_type=response.get('stop_type'),max_token_censored=response.get('stop_type')=='limit',
                    elapsed_seconds=time.monotonic()-before,timings=response.get('timings'),
                    generation_settings=response.get('generation_settings'),contract_sha256=sha(out/'contract.json'),
                    prompt_token_ids_sha256=hashlib.sha256(json.dumps(row['token_ids']).encode()).hexdigest())
                write(out/f'{index:03d}.json',result)
                completed+=1
                write(out/'progress.json',dict(completed=completed,expected=len(rows),updated_unix=time.time()))
                print(f'{completed}/{len(rows)} {row["arm"]} {len(ids)} tokens {result["elapsed_seconds"]:.1f}s',flush=True)
        finally:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            write(out/'cleanup.json',dict(server_stopped=proc.poll() is not None,returncode=proc.returncode))
    assert completed==259
    write(out/'manifest.json',dict(status='logic_skill_generation_complete',records=completed,
          elapsed_seconds=time.monotonic()-started,
          files={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}))


if __name__=='__main__':main()
