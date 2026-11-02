"""Lock the compiled Linux runtime, model and actual dependency provenance before inference."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root=Path('artifacts/nanbeige_reasoning');out=root/'prime_q4_runtime';out.mkdir(parents=True,exist_ok=True)
    source=json.loads((root/'prime_q4_source.json').read_text())
    revision=subprocess.check_output(['git','-C','tools/nanbeige/llama-source','rev-parse','HEAD'],text=True).strip()
    assert revision==source['llama_cpp_revision']
    model=Path('models/Nanbeige4.2-3B-Q4_K_M.gguf');assert sha(model)==source['model_sha256']
    runtime=Path('tools/nanbeige/runtime')
    libraries={p.name:sha(p) for p in runtime.glob('*.so*') if p.is_file()}
    assert all(n in libraries for n in ['libllama.so','libggml.so','libggml-base.so','libggml-cuda.so','libggml-cpu.so'])
    contract=dict(model_sha256=sha(model),question_sha256=sha(root/'pilot_v1/questions.jsonl'),runtime_files=libraries,
        source_revision=revision,platform=platform.platform(),python=platform.python_version(),
        packages={n:importlib.metadata.version(n) for n in ['numpy','transformers','sentencepiece','protobuf']},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (out/'dependencies.txt').write_text(subprocess.check_output(['ldd',str(runtime/'libggml-cuda.so')],text=True))
    print(json.dumps(contract))


if __name__=='__main__':main()
