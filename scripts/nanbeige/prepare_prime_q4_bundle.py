"""Build an isolated Linux port; never edit the currently running Windows experiment."""
import hashlib
import json
import py_compile
import shutil
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/nanbeige_reasoning'
DEST=BASE/'prime_q4_answers_bundle_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,text):
    p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8',newline='\n')
def copy(name):
    p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
def replace(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    originals={}
    modules=['llama_capture.py','llama_steer.py','llama_cached_steer.py','check_llama_steering.py',
        'check_cached_steering.py','run_q4_feature_answers.py','temperature_guard_burst.py','answer_decoding.py']
    for name in modules:
        rel=f'experiments/nanbeige_reasoning/{name}';originals[rel]=sha(ROOT/rel);copy(rel)
    rel='experiments/nanbeige_reasoning/llama_capture.py'
    text=(DEST/rel).read_text()
    start=text.index('        self.dll_directory=os.add_dll_directory')
    end=text.index('        specs={',start)
    text=text[:start]+'''        self.dll_directory=None
        self.base=C.CDLL(str(self.path/'libggml-base.so'),mode=C.RTLD_GLOBAL)
        self.backend=C.CDLL(str(self.path/'libggml.so'),mode=C.RTLD_GLOBAL)
        self.llama=C.CDLL(str(self.path/'libllama.so'),mode=C.RTLD_GLOBAL)
        self.cuda_dependencies=[]
'''+text[end:]
    put(rel,text)
    for name in ['check_cached_steering.py','run_q4_feature_answers.py']:
        rel=f'experiments/nanbeige_reasoning/{name}';text=(DEST/rel).read_text()
        text=replace(text,"root/'q4_reasoning_pilot_8192_v1/contract.json'","root/'prime_q4_runtime/contract.json'")
        if name=='check_cached_steering.py':
            text=replace(text,'from .llama_cached_steer import CachedSteeringModel',
                'from .llama_cached_steer import CachedSteeringModel\nfrom .run_q4_feature_answers import native_decoder\nfrom .temperature_guard_burst import TemperatureGuard')
            text=replace(text,'generated_steps=8,atol=1e-4','context=10153,generated_steps=8,atol=1e-4')
            text=replace(text,'context=512,gpu_layers=-1','context=10153,gpu_layers=-1')
            text=replace(text,'    try:\n        for name, intervention in arms:',
                "    try:\n        guard=TemperatureGuard(args.output/'temperature.jsonl',wall_seconds=3600)\n        guard.attach(model)\n        runtime,decode=native_decoder(model)\n        write(args.output/'runtime.json',runtime)\n        for name, intervention in arms:")
            text=replace(text,"archive=f'{name}.npz',sha256=sha(args.output/f'{name}.npz'))",
                "archive=f'{name}.npz',sha256=sha(args.output/f'{name}.npz'),generated_text=decode(generated),prompt_text=decode(tokens),stop_token_texts={str(t):decode([t]) for t in runtime['eog_token_ids']})")
            text=replace(text,"if p.is_file()}))","if p.is_file() and p.name!='temperature.jsonl'}))")
        put(rel,text)
    rel='experiments/nanbeige_reasoning/temperature_guard_burst.py';text=(DEST/rel).read_text()
    text=replace(text,'creationflags=subprocess.CREATE_NO_WINDOW','creationflags=0')
    text=replace(text,'idle_to_decode_ratio=4.,minimum_idle_seconds=.1,warm_minimum_idle_seconds=.3',
        'idle_to_decode_ratio=0.,minimum_idle_seconds=0.,warm_minimum_idle_seconds=0.')
    text=replace(text,"idle=max(4*self.last_decode_seconds,.3 if self.last_temperature is not None and self.last_temperature>=80 else .1)","idle=0.0")
    put(rel,text)
    for name in ['verify_cached_steering.py','prepare_prime_q4_runtime.py','verify_prime_q4_gate.py','run_prime_q4_answers_remote.sh','prime_q4_worker.sh','start_prime_q4_worker.py']:
        copy(f'scripts/nanbeige/{name}')
    copy('artifacts/nanbeige_reasoning/pilot_v1/questions.jsonl')
    source=BASE/'q4_steering_preflight_v1';manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert sha(source/name)==digest
        copy(f'artifacts/nanbeige_reasoning/q4_steering_preflight_v1/{name}')
    copy('artifacts/nanbeige_reasoning/q4_steering_preflight_v1/manifest.json')
    for name in ['tokenizer.model','tokenizer_config.json','added_tokens.json','special_tokens_map.json','config.json']:
        copy(f'models/Nanbeige4.2-3B-tokenizer/{name}')
    put('artifacts/nanbeige_reasoning/prime_q4_source.json',json.dumps(dict(
        local_source_sha256=originals,source_gate_manifest_sha256=sha(BASE/'q4_cached_steering_burst_v1/manifest.json'),
        changes=['Linux shared-library loader only; same C ABI bindings and activation writes.',
            'Linux subprocess creation flags for the unchanged nvidia-smi temperature sensor.',
            'Zero artificial duty-cycle idle; 85/78/87 C cooling and emergency thresholds retained.',
            'Fresh full 10153-context calibration gate and fresh 70-output remote cohort.'],
        model_sha256='01739fad13be318c5c8620f8de52731608bb750ae887e9445cc81cb4a189e45d',
        llama_cpp_revision='0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0'),indent=2))
    for p in DEST.rglob('*.py'):py_compile.compile(str(p),doraise=True)
    files=[p for p in DEST.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='bundle_manifest.json']
    receipt=dict(builder_sha256=sha(Path(__file__)),files={p.relative_to(DEST).as_posix():sha(p) for p in files})
    put('bundle_manifest.json',json.dumps(receipt,indent=2));files.append(DEST/'bundle_manifest.json')
    target=BASE/'prime_q4_answers_bundle.tar.gz'
    with tarfile.open(target,'w:gz') as archive:
        for p in files:archive.add(p,arcname=p.relative_to(DEST).as_posix(),recursive=False)
    with tarfile.open(target) as archive:
        for member in archive.getmembers():
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
    (BASE/'prime_q4_answers_bundle.tar.receipt.json').write_text(json.dumps(dict(sha256=sha(target),files=receipt['files']),indent=2))
    assert all(sha(ROOT/name)==digest for name,digest in originals.items()),'Running sources changed'
    print(json.dumps(dict(status='isolated_linux_bundle_verified',files=len(files),bytes=target.stat().st_size,sha256=sha(target))))


if __name__=='__main__':main()
