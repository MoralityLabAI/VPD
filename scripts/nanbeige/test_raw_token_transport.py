"""Actual pinned-runtime regression for malformed byte-token output transport."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.nanbeige_reasoning.run_q4_reasoning import request, write_json
from transformers import AutoTokenizer

PARSER = json.dumps(dict(parsers=[dict(type='epsilon')], rules={}, root=0))


def main():
    run = ROOT/'artifacts/nanbeige_reasoning/q4_reasoning_pilot_v1'
    out = run/'raw_token_transport_test'
    out.mkdir(exist_ok=True)
    command = json.loads((run/'command_1.json').read_text())
    command[command.index('--port')+1] = '8097'
    token = secrets.token_hex(32)
    env = dict(os.environ, LLAMA_API_KEY=token)
    base = 'http://127.0.0.1:8097'
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/Nanbeige4.2-3B-tokenizer', use_fast=False, local_files_only=True)
    vocab = tokenizer.get_vocab()
    bad = next(vocab[s] for s in ('<0x9D>', 'Ŀ') if s in vocab and '\ufffd' in tokenizer.decode([vocab[s]]))
    rows = list(map(json.loads, (run/'generations.jsonl').read_text().splitlines()))
    control = next(r for r in rows if r['loops'] == 1)
    prompt = control['token_ids'][:control['prompt_tokens']]
    expected = control['token_ids'][control['prompt_tokens']:control['prompt_tokens']+16]
    with (out/'server.log').open('w') as log:
        process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic()+180
            while True:
                if process.poll() is not None: raise RuntimeError('Probe server exited')
                try:
                    if request(base,token,'/health',timeout=2).get('status') == 'ok': break
                except (urllib.error.URLError, TimeoutError): pass
                if time.monotonic()>deadline: raise TimeoutError('Probe startup')
                time.sleep(1)
            payload = dict(prompt=prompt,n_predict=16,temperature=0,seed=42,repeat_penalty=1.,cache_prompt=False,return_tokens=True,stream=False)
            ordinary = request(base,token,'/completion',payload)
            bypass = request(base,token,'/completion',dict(payload,chat_parser=PARSER))
            assert ordinary['tokens'] == bypass['tokens'] == expected
            forced = dict(payload,n_predict=3,logit_bias=[[bad,1000.]])
            try:
                request(base,token,'/completion',forced)
            except urllib.error.HTTPError as exc:
                error = json.load(exc)
                assert exc.code==500 and 'Content-only format' in error['error']['message'], error
            else: raise AssertionError('Invalid-byte probe did not reproduce parser failure')
            recovered = request(base,token,'/completion',dict(forced,chat_parser=PARSER))
            assert recovered['tokens'] == [bad]*3, recovered
            assert recovered['stop_type']=='limit' and not recovered['truncated']
            result = dict(status='actual_runtime_transport_passed',saved_prefix_exact_match=True,
                malformed_utf8_default_parser_rejected=True,raw_token_parser_preserved_ids=True,
                diagnostic_only_logit_bias=True,forced_token_id=bad,parser=json.loads(PARSER),
                model_sampling_changed_in_experiment=False,control_id=control['id'],
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'artifacts/nanbeige_reasoning/sources/llama_cpp').glob('*.cpp')})
            write_json(out/'verification.json',result)
            print(json.dumps(result,indent=2),flush=True)
        finally:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill();process.wait()
            write_json(out/'server_stopped.json',dict(pid=process.pid,returncode=process.returncode))


if __name__=='__main__': main()
