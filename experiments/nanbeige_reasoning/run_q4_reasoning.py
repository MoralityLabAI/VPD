"""Paired Q4 reasoning on the pinned local runtime; never a BF16 parity claim.

Loads only model-facing questions. Each loop condition owns and stops its server.
The same quantized physical weights and KV precision are used in both conditions.
"""
from __future__ import annotations
import argparse
import json
import os
import secrets
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from transformers import AutoTokenizer
from experiments.nanbeige_looped.q4_smoke import Q4_SHA256,digest,ROOT
from .run_benchmark import recovered_rows
from .thermal_pause import cooling

# Raw /completion still invokes a chat parser in this pinned runtime. An empty
# parser leaves native content/tokens intact and accepts invalid UTF-8 byte-token
# sequences. Sampling and stop rules are independent of this response parser.
RAW_TOKEN_PARSER=json.dumps(dict(parsers=[dict(type='epsilon')],rules={},root=0))


def request(base,key,endpoint,payload=None,timeout=900):
    req=urllib.request.Request(base+endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions',type=Path,default=Path('artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--context',type=int,default=4096)
    p.add_argument('--loops',default='2,1')
    p.add_argument('--port',type=int,default=8096)
    p.add_argument('--max-wall-seconds',type=int,default=7200)
    p.add_argument('--completion-timeout-seconds',type=int,
        help='HTTP response wait only; sampling and output-token caps are unchanged')
    args=p.parse_args()
    if args.max_wall_seconds <= 0 or (args.completion_timeout_seconds is not None
                                    and args.completion_timeout_seconds <= 0):
        p.error('Wall and completion timeout budgets must be positive')
    completion_timeout=(args.completion_timeout_seconds if args.completion_timeout_seconds is not None
                        else max(900,args.max_new_tokens//2))
    loops=list(map(int,args.loops.split(',')))
    if not loops or len(set(loops))!=len(loops) or set(loops)-{1,2}:
        p.error('Loop conditions must be 1 and/or 2')
    rows=list(map(json.loads,args.questions.read_text(encoding='utf-8').splitlines()))
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate input IDs')
    if any(len(r['token_ids'])+args.max_new_tokens>args.context for r in rows):
        raise ValueError('Context cannot retain every prompt and the full output cap')
    model=ROOT/'models/Nanbeige4.2-3B-Q4_K_M.gguf'
    executable=ROOT/'tools/nanbeige/runtime/llama-server.exe'
    if digest(model)!=Q4_SHA256:raise ValueError('Q4 checkpoint hash mismatch')
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    contract=dict(question_sha256=digest(args.questions),model_sha256=Q4_SHA256,
        runtime_files={f.name:digest(f) for f in executable.parent.iterdir() if f.suffix in ('.exe','.dll')},
        runtime_commit='0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0',
        loops=loops,context=args.context,max_new_tokens=args.max_new_tokens,
        weights='Q4_K_M',kv_cache='q8_0',seed=42,temperature=0,
        repeat_penalty=1.0,context_shift=False,bf16_parity_validated=False)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:
        raise ValueError('Resume contract mismatch')
    write_json(out/'contract.json',contract)
    write_json(out/'transport.json',dict(endpoint='/completion',chat_parser=json.loads(RAW_TOKEN_PARSER),
        token_ids_authoritative=True,text_decoder='pinned HF tokenizer; invalid UTF-8 may become U+FFFD',
        reason='Bypass UTF-8-validating chat response parser without changing sampling',
        actual_runtime_regression='q4_reasoning_pilot_v1/raw_token_transport_test/verification.json'))
    path=out/'generations.jsonl'
    previous=recovered_rows(path)
    # Record transport-only changes separately from the immutable model contract.
    # Unique receipts preserve earlier attempts, including failed requests.
    write_json(out/f'execution_attempt_{time.time_ns()}.json',dict(
        started_unix=time.time(),source_sha256=digest(Path(__file__)),
        completed_records_at_start=len(previous),completion_timeout_seconds=completion_timeout,
        max_wall_seconds=args.max_wall_seconds,
        existing_generations_sha256=digest(path) if path.exists() else None,
        scope='Transport wait budget; no decoding or model-contract change'))
    by_id={r['id']:r for r in rows}
    done=set()
    for r in previous:
        pair=(r['id'],r['loops'])
        if pair in done or r['id'] not in by_id or r['loops'] not in loops:
            raise ValueError('Corrupt resumed conditions')
        if r['token_ids'][:r['prompt_tokens']]!=by_id[r['id']]['token_ids']:
            raise ValueError('Resumed prompt mismatch')
        done.add(pair)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Nanbeige4.2-3B-tokenizer',use_fast=False,local_files_only=True)
    started=time.monotonic()
    interrupted=False
    for count in loops:
        pending=[r for r in rows if (r['id'],count) not in done]
        if not pending:continue
        key=secrets.token_hex(32)
        command=[str(executable),'-m',str(model),'-c',str(args.context),'-b','64','-ub','64',
            '-ngl','all','--fit','off','-fa','on','-np','1','-ctk','q8_0','-ctv','q8_0',
            '--no-context-shift','--cache-ram','0','--host','127.0.0.1','--port',str(args.port),
            '--override-kv',f'nanbeige.num_loops=int:{count}','-lv','4']
        # The random localhost-only token is never persisted in command receipts.
        env=os.environ.copy()
        env['LLAMA_API_KEY']=key
        base=f'http://127.0.0.1:{args.port}'
        logpath=out/f'server_loops_{count}_{len(done):03d}.log'
        print(f'Starting Q4 loops={count}, {len(pending)} remaining examples',flush=True)
        with logpath.open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            cooler=cooling(process.pid,out)
            cooling_entered=False
            try:
                cooler.__enter__();cooling_entered=True
                deadline=time.monotonic()+180
                while True:
                    if process.poll() is not None:raise RuntimeError(f'Server exited; inspect {logpath}')
                    try:
                        if request(base,key,'/health',timeout=2).get('status')=='ok':break
                    except (urllib.error.URLError,TimeoutError):pass
                    if time.monotonic()>deadline:raise TimeoutError('Server startup timeout')
                    time.sleep(1)
                props=request(base,key,'/props',timeout=10)
                write_json(out/f'runtime_props_{count}.json',props)
                write_json(out/f'command_{count}.json',command)
                # Confirm that API authentication is active before processing inputs.
                try:
                    request(base,'invalid','/props',timeout=5)
                except urllib.error.HTTPError as exc:
                    if exc.code!=401:raise
                else:raise RuntimeError('Local API authentication was not enabled')
                completed=[r for r in previous if r['loops']==count]
                if completed:
                    control=completed[0];sample=by_id[control['id']]
                    length=min(16,control['generated_tokens'])
                    check=request(base,key,'/completion',dict(prompt=sample['token_ids'],n_predict=length,
                        temperature=0,seed=42,repeat_penalty=1.0,cache_prompt=False,return_tokens=True,stream=False,
                        chat_parser=RAW_TOKEN_PARSER))
                    expected=control['token_ids'][control['prompt_tokens']:control['prompt_tokens']+length]
                    match=check.get('tokens')==expected
                    write_json(out/f'resume_preflight_{count}.json',dict(id=control['id'],tokens=length,match=match,
                        reason='Confirm cached completed prefix under restarted runtime with prompt-cache RAM disabled'))
                    if not match:raise RuntimeError('Restarted runtime changed the saved greedy prefix')
                for index,row in enumerate(pending):
                    remaining=args.max_wall_seconds-(time.monotonic()-started)
                    if remaining<=0:
                        interrupted=True
                        break
                    before=time.monotonic()
                    completion=request(base,key,'/completion',dict(prompt=row['token_ids'],
                        n_predict=args.max_new_tokens,temperature=0,seed=42,repeat_penalty=1.0,
                        cache_prompt=False,return_tokens=True,stream=False,ignore_eos=False,chat_parser=RAW_TOKEN_PARSER),
                        timeout=min(completion_timeout,remaining))
                    new=completion.get('tokens')
                    if not new or completion.get('truncated'):
                        raise RuntimeError('Empty generation or silently truncated context')
                    text=tokenizer.decode(new,skip_special_tokens=False)
                    prefix=tokenizer.decode(row['token_ids'][-8:],skip_special_tokens=False)
                    scoring_text=('<think>' if '<think>' in prefix and '</think>' not in prefix else '')+text
                    stop=completion.get('stop_type')
                    record=dict(id=row['id'],lane=row['lane'],family=row['family'],split=row['split'],
                        calibration_split=row['calibration_split'],condition=f'q4_loops_{count}',loops=count,
                        prompt_tokens=len(row['token_ids']),generated_tokens=len(new),
                        token_ids=row['token_ids']+new,generated_text=text,scoring_text=scoring_text,
                        elapsed_seconds=time.monotonic()-before,stop_type=stop,
                        stopped_on_eos=stop=='eos',max_token_censored=stop=='limit',
                        thinking_closed='</think>' in text,quantization='Q4_K_M',kv_cache='q8_0',
                        timings=completion.get('timings'),generation_settings=completion.get('generation_settings'),
                        response_parser='epsilon_raw_tokens',decoded_replacement_characters=text.count('\ufffd'),
                        physical_layer_token_executions=22*count*(len(row['token_ids'])+max(0,len(new)-1)))
                    with path.open('a',encoding='utf-8') as f:
                        f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                    done.add((row['id'],count))
                    print(f'{index+1}/{len(pending)} loops={count} {row["lane"]} tokens={len(new)} stop={stop} seconds={record["elapsed_seconds"]:.1f}',flush=True)
            finally:
                if cooling_entered:cooler.__exit__(None,None,None)
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                write_json(out/f'server_stopped_{count}.json',dict(pid=process.pid,returncode=process.returncode,stopped=True))
        if interrupted:break
    records=recovered_rows(path)
    calibration=[dict(id=r['id'],pair_id=by_id[r['id']]['group_id'],family=r['family'],
        split=r['calibration_split'],token_ids=r['token_ids'],prompt_tokens=r['prompt_tokens'],
        max_token_censored=r['max_token_censored'],generation_weights='Q4_K_M') for r in records if r['loops']==2]
    (out/'calibration_tokens.jsonl').write_text('\n'.join(json.dumps(r) for r in calibration)+'\n',encoding='utf-8')
    write_json(out/'manifest.json',dict(status='paused_time_budget' if interrupted else 'generation_complete',
        examples=len(rows),records=len(records),expected_records=len(rows)*len(loops),
        elapsed_seconds=time.monotonic()-started,contract=contract,activation_capture_complete=False,
        limitations=['Quantized behavior; BF16 parity not established.','One-pass ablates a two-pass-trained model.',
            'Pilot subset is not full benchmark accuracy.','Equal output caps are not equal compute.'],
        files={f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}))


if __name__=='__main__':main()
