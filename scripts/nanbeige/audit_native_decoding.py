"""Check a frozen checkpoint snapshot with vocabulary-only native decoding.

Run in a separate process from HF/Torch. No model tensors, inference context,
or GPU work are loaded.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
from types import SimpleNamespace
from experiments.nanbeige_reasoning.llama_capture import Runtime, P, I, B
from experiments.nanbeige_reasoning.run_q4_feature_answers import native_decoder
from experiments.nanbeige_reasoning.check_llama_steering import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot',type=Path);args=parser.parse_args()
    data=json.loads(args.snapshot.read_text(encoding='utf-8'))
    runtime=Runtime(Path('tools/nanbeige/runtime'))
    lib=runtime.llama;params=lib.llama_model_default_params()
    params.vocab_only=True;params.n_gpu_layers=0
    model=lib.llama_model_load_from_file(b'models/Nanbeige4.2-3B-Q4_K_M.gguf',params)
    if not model:raise RuntimeError('Vocabulary load failed')
    try:
        vocab=lib.llama_model_get_vocab(model)
        _,decode=native_decoder(SimpleNamespace(lib=lib,model=model,vocab_size=lib.llama_vocab_n_tokens(vocab)))
        lib.llama_token_to_piece.argtypes=[P,I,C.c_char_p,I,I,B]
        lib.llama_token_to_piece.restype=I
        pieces={}
        for token in {t for row in data['records'] for t in row['tokens']}:
            buf=C.create_string_buffer(1024)
            n=lib.llama_token_to_piece(vocab,token,buf,len(buf),0,True)
            if n<0:raise ValueError('Token piece buffer too small')
            pieces[token]=buf.raw[:n]
        output=[]
        for row in data['records']:
            output.append(dict(name=row['name'],native=decode(row['tokens']),
                raw_pieces=b''.join(pieces[t] for t in row['tokens']).decode('utf-8',errors='replace')))
        path=args.snapshot.with_name('native_decoding.json')
        path.write_text(json.dumps(dict(snapshot_sha256=sha(args.snapshot),records=output,
            runtime_files={p.name:sha(p) for p in Path('tools/nanbeige/runtime').glob('*.dll')},
            verifier_sha256=sha(Path(__file__))),indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(status='vocabulary_only_decoded',records=len(output))))
    finally:
        lib.llama_model_free(model);lib.llama_backend_free()


if __name__=='__main__':main()
