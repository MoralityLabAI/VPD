"""Run the pinned fresh 27B baseline on the identical public-question cohort.

The runner owns a loopback-only GPU server. It does not implement TRM retrieval
or call itself a historical reproduction. Model tokens and all costs are saved.
"""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import time
import urllib.error

from transformers import AutoTokenizer
from .fetch_sources import sha
from .run_benchmark import recovered_rows
from .run_q4_reasoning import request,write_json,RAW_TOKEN_PARSER


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--server',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--context',type=int,default=32768)
    p.add_argument('--max-new-tokens',type=int,default=8192)
    p.add_argument('--max-wall-seconds',type=int,default=43200)
    p.add_argument('--port',type=int,default=8098)
    args=p.parse_args()
    if min(args.context,args.max_new_tokens,args.max_wall_seconds)<1 or not 1024<=args.port<=65535:
        p.error('Invalid context, generation or wall bound, or port')
    lockpath=args.sources/'model_lock.json'
    lock=json.loads(lockpath.read_text())
    prepared=json.loads((args.inputs/'manifest.json').read_text())
    if prepared['model_lock_sha256']!=sha(lockpath):raise ValueError('Comparator source lock changed')
    questionpath=args.inputs/'questions.jsonl'
    if prepared['questions_sha256']!=sha(questionpath):raise ValueError('Prepared input hash mismatch')
    for name,expected in prepared['tokenizer_assets'].items():
        if sha(args.sources/'official'/name)!=expected:raise ValueError('Tokenizer asset changed')
    weights=lock['quantized']['weights']
    if len(weights)!=1 or args.model.stat().st_size!=weights[0]['bytes'] or sha(args.model)!=weights[0]['sha256']:
        raise ValueError('Pinned Qwen Q4 checkpoint size/hash mismatch')
    rows=list(map(json.loads,questionpath.read_text(encoding='utf-8').splitlines()))
    by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Duplicate question ID')
    if any(len(r['token_ids'])+args.max_new_tokens>args.context for r in rows):
        raise ValueError('Context must retain every prompt and the entire output cap')
    version=subprocess.run([str(args.server.resolve()),'--version'],capture_output=True,text=True,check=True)
    version_text=version.stdout+version.stderr
    if '0eadefe' not in version_text:raise ValueError('Runtime is not the pinned llama.cpp commit')
    tokenizer=AutoTokenizer.from_pretrained(args.sources/'official',local_files_only=True,use_fast=False)
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    binaries=[f for f in args.server.parent.iterdir() if f.is_file() and ('.so' in f.name or f.suffix in ('.exe','.dll') or f==args.server)]
    contract=dict(model='Qwen3.5-27B',condition='qwen35_27b_q4_thinking',question_sha256=sha(questionpath),
        model_sha256=weights[0]['sha256'],source_lock_sha256=sha(lockpath),runtime_commit='0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0',
        runtime_files={f.name:sha(f) for f in binaries},context=args.context,max_new_tokens=args.max_new_tokens,
        seed=42,temperature=0,repeat_penalty=1.,kv_cache='q8_0',weights='Q4_K_M',thinking_enabled=True,
        raw_response_parser=json.loads(RAW_TOKEN_PARSER))
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:
        raise ValueError('Resume contract mismatch')
    write_json(out/'contract.json',contract)
    previous=recovered_rows(out/'generations.jsonl')
    done=set()
    for row in previous:
        if row['id'] in done or row['id'] not in by_id or row['condition']!=contract['condition']:
            raise ValueError('Invalid resumed record')
        if row['token_ids'][:row['prompt_tokens']]!=by_id[row['id']]['token_ids']:
            raise ValueError('Resumed prompt changed')
        done.add(row['id'])
    token=secrets.token_hex(32)
    env=dict(os.environ,LLAMA_API_KEY=token)
    base=f'http://127.0.0.1:{args.port}'
    command=[str(args.server.resolve()),'-m',str(args.model.resolve()),'-c',str(args.context),'-b','512','-ub','512',
        '-ngl','all','--fit','off','-fa','on','-np','1','-ctk','q8_0','-ctv','q8_0',
        '--no-context-shift','--cache-ram','0','--host','127.0.0.1','--port',str(args.port),'-lv','3']
    write_json(out/f'command_{len(done):04d}.json',command)
    started=time.monotonic()
    paused=False
    logpath=out/f'server_{len(done):04d}.log'
    with logpath.open('w') as log:
        server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            deadline=time.monotonic()+300
            while True:
                if server.poll() is not None:raise RuntimeError('Qwen server exited during startup')
                try:
                    if request(base,token,'/health',timeout=2).get('status')=='ok':break
                except (urllib.error.URLError,TimeoutError):pass
                if time.monotonic()>deadline:raise TimeoutError('Qwen server startup')
                time.sleep(1)
            logs=logpath.read_text(errors='replace')
            offload=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',logs)
            if not offload or any(int(a)!=int(b) or int(a)<1 for a,b in offload) or 'CUDA0' not in logs:
                raise ValueError('Full CUDA model offload was not verified')
            try:request(base,'invalid','/props',timeout=5)
            except urllib.error.HTTPError as exc:
                if exc.code!=401:raise
            else:raise ValueError('Loopback server authentication is disabled')
            write_json(out/'runtime_props.json',request(base,token,'/props',timeout=10))
            # Check the actual GGUF vocabulary against every prepared sequence.
            for row in rows:
                observed=request(base,token,'/tokenize',dict(content=row['rendered_prompt'],add_special=False,parse_special=True),timeout=30)
                if observed['tokens']!=row['token_ids']:raise ValueError(f'GGUF tokenizer differs: {row["id"]}')
            write_json(out/'tokenizer_preflight.json',dict(status='all_inputs_match',records=len(rows),full_cuda_offload=True))
            def complete(row,count):
                return request(base,token,'/completion',dict(prompt=row['token_ids'],n_predict=count,
                    temperature=0,seed=42,repeat_penalty=1.,return_tokens=True,cache_prompt=False,
                    stream=False,ignore_eos=False,chat_parser=RAW_TOKEN_PARSER),timeout=max(1800,count))
            if previous:
                first=previous[0];count=min(16,first['generated_tokens'])
                matched=complete(by_id[first['id']],count)['tokens']==first['token_ids'][first['prompt_tokens']:first['prompt_tokens']+count]
                write_json(out/'resume_preflight.json',dict(id=first['id'],tokens=count,match=matched))
                if not matched:raise ValueError('Restarted comparator changed saved greedy prefix')
            for row in rows:
                if row['id'] in done:continue
                if time.monotonic()-started>args.max_wall_seconds:paused=True;break
                before=time.monotonic();response=complete(row,args.max_new_tokens)
                new=response.get('tokens')
                if not new or response.get('truncated'):raise ValueError('Empty/truncated comparator output')
                text=tokenizer.decode(new,skip_special_tokens=False)
                result={k:row[k] for k in ('id','lane','family','split','calibration_split','group_id')}
                result.update(condition=contract['condition'],model=contract['model'],prompt_tokens=len(row['token_ids']),
                    generated_tokens=len(new),token_ids=row['token_ids']+new,generated_text=text,scoring_text='<think>'+text,
                    elapsed_seconds=time.monotonic()-before,stop_type=response.get('stop_type'),
                    max_token_censored=response.get('stop_type')=='limit',thinking_closed='</think>' in text,
                    decoded_replacement_characters=text.count('\ufffd'),timings=response.get('timings'),
                    generation_settings=response.get('generation_settings'))
                with (out/'generations.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(result)+'\n');stream.flush();os.fsync(stream.fileno())
                done.add(row['id'])
                print(f'{len(done)}/{len(rows)} {row["lane"]}: {len(new)} tokens, {result["elapsed_seconds"]:.1f}s',flush=True)
        finally:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            write_json(out/'server_stopped.json',dict(pid=server.pid,returncode=server.returncode))
    write_json(out/'manifest.json',dict(status='paused_time_budget' if paused else 'generation_complete',
        records=len(done),expected_records=len(rows),contract=contract,elapsed_seconds=time.monotonic()-started,
        limitations=['Fresh quantized comparator, not verified as the historical checkpoint.',
            'Different trained architectures and tokenizers; no causal scale claim.',
            'No TRM retrieval, repair, or solver assistance in this baseline arm.'],
        files={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}))


if __name__=='__main__':main()
