"""Pin public sources for a fresh 27B comparison; do not download model weights."""
import json
import time
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from .fetch_sources import sha


def main():
    out=Path('artifacts/nanbeige_reasoning/comparator_sources/qwen35_27b')
    out.mkdir(parents=True,exist_ok=True)
    lock=out/'model_lock.json'
    if lock.exists():
        print('Existing source lock retained:',lock)
        return
    api=HfApi(token=False)
    official=api.model_info('Qwen/Qwen3.5-27B',files_metadata=True,timeout=30)
    quant=api.model_info('unsloth/Qwen3.5-27B-GGUF',files_metadata=True,timeout=30)
    weights=[]
    for f in quant.siblings:
        if 'Q4_K_M' in f.rfilename and f.rfilename.endswith('.gguf'):
            digest=f.lfs.get('sha256') if isinstance(f.lfs,dict) else getattr(f.lfs,'sha256',None)
            if not digest or len(digest)!=64:
                raise ValueError('Missing public weight SHA256')
            weights.append(dict(filename=f.rfilename,bytes=f.size,sha256=digest))
    if not weights:
        raise ValueError('Requested Q4_K_M variant absent')
    metadata={}
    names={f.rfilename for f in official.siblings}
    for name in ('config.json','tokenizer_config.json','chat_template.jinja','generation_config.json','README.md'):
        if name not in names:continue
        path=Path(hf_hub_download(official.id,name,revision=official.sha,
            local_dir=out/'official',token=False))
        metadata[name]=dict(sha256=sha(path),bytes=path.stat().st_size)
    config=json.loads((out/'official/config.json').read_text(encoding='utf-8'))
    text=config.get('text_config',config)
    result=dict(status='fresh_comparator_sources_pinned',unix=time.time(),
        official=dict(repo=official.id,revision=official.sha,metadata=metadata),
        quantized=dict(repo=quant.id,revision=quant.sha,weights=weights),
        architecture={name:text.get(name) for name in ('model_type','hidden_size','num_hidden_layers',
            'intermediate_size','num_attention_heads','num_key_value_heads','layer_types')},
        model_weights_downloaded=False,inference_run=False,
        limitations=['Fresh comparison source, not proven identical to the historical Q4_K_M checkpoint.',
            'Exact relationship of the community quantization to the current official revision is unverified.',
            'A different trained architecture is a practical comparator, not a controlled causal scale intervention.'])
    lock.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','quantized','model_weights_downloaded','inference_run')},indent=2))


if __name__=='__main__':
    main()
