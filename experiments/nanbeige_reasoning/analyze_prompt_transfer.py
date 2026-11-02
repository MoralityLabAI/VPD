"""Measure transfer of the already-frozen calibration SAEs to benchmark prompts.

This reports feature differences and reconstruction; it does not refit on the new
examples or assert that the old narrow dictionaries generalize successfully.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.nanbeige_looped.sae import SharedSAE,encode_rows
from .fetch_sources import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures',type=Path)
    p.add_argument('--dictionaries',type=Path,default=Path(
        'artifacts/nanbeige_looped/prime-20260906T034844Z-719/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration'))
    args=p.parse_args()
    torch.set_num_threads(2)
    records=[json.loads(x) for x in (args.captures/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    quality=json.loads((args.dictionaries/'sae_quality.json').read_text())
    original_manifest=json.loads((args.dictionaries/'manifest.json').read_text())
    out=args.captures/'frozen_sae_transfer'
    out.mkdir(exist_ok=True)
    hidden_rows=[]
    for record in records:
        path=args.captures/record['archive']
        if sha(path)!=record['sha256']:
            raise ValueError('Capture hash mismatch')
        with np.load(path,allow_pickle=False) as archive:
            for site in record['calls']:
                h0,h1=[archive[f'{site}__pass_{n}'] for n in (0,1)]
                norms=[float(np.linalg.norm(h)) for h in (h0,h1)]
                hidden_rows.append(dict(id=record['id'],lane=record['lane'],split=record['split'],site=site,
                    rows=len(h0),pass_0_norm=norms[0],pass_1_norm=norms[1],
                    cosine=float(np.sum(h0*h1)/(norms[0]*norms[1])),
                    difference_norm=float(np.linalg.norm(h1-h0))))
    (out/'all_layer_hidden_differences.json').write_text(json.dumps(hidden_rows,indent=2),encoding='utf-8')
    rows=[]
    controls=[]
    for site,entry in quality.items():
        path=args.dictionaries/entry['dictionary']
        if sha(path)!=entry['sha256']:
            raise ValueError('Frozen SAE hash mismatch')
        with np.load(path,allow_pickle=False) as a:
            features,width=a['encoder_weight'].shape
            sae=SharedSAE(width,features,int(a['top_k']))
            with torch.no_grad():
                sae.encoder.weight.copy_(torch.from_numpy(a['encoder_weight']))
                sae.encoder.bias.copy_(torch.from_numpy(a['encoder_bias']))
                sae.decoder.weight.copy_(torch.from_numpy(a['decoder'].T))
                sae.pre_bias.copy_(torch.from_numpy(a['pre_bias']))
            sae.eval().requires_grad_(False)
        heldout=[filename for example,filename in original_manifest['baseline_archives'].items()
                 if example not in entry['train_example_ids']]
        for invocation in (0,1):
            original=[]
            for filename in heldout:
                with np.load(args.dictionaries/filename,allow_pickle=False) as old:
                    original.append(old[f'{site}__pass_{invocation}'])
            _,measured=encode_rows(sae,np.concatenate(original))
            reference=entry['held_out_quality'][f'pass_{invocation}']
            error=abs(measured['explained_variance']-reference['explained_variance'])
            if error>1e-5:
                raise ValueError('Reloaded SAE does not reproduce original held-out quality')
            controls.append(dict(site=site,invocation=invocation,measured=measured,
                original=reference,explained_variance_abs_error=error))
        for i,record in enumerate(records):
            path=args.captures/record['archive']
            if sha(path)!=record['sha256']:
                raise ValueError('Capture hash mismatch')
            with np.load(path,allow_pickle=False) as a:
                h=[a[f'{site}__pass_{n}'] for n in (0,1)]
                encoded=[encode_rows(sae,value) for value in h]
                z=[v[0] for v in encoded]
                destination=out/f'{site}_{i:03d}.npz'
                np.savez_compressed(destination,pass_0=z[0],pass_1=z[1],delta_z=z[1]-z[0],
                    token_positions=a['token_positions'],token_ids=a['token_ids'][a['token_positions']])
                norms=[float(np.linalg.norm(value)) for value in h]
                rows.append(dict(id=record['id'],lane=record['lane'],split=record['split'],site=site,
                    pass_0_quality=encoded[0][1],pass_1_quality=encoded[1][1],
                    pass_cosine=float(np.sum(h[0]*h[1])/(norms[0]*norms[1])),
                    second_first_norm_ratio=norms[1]/norms[0],
                    hidden_difference_norm=float(np.linalg.norm(h[1]-h[0])),
                    feature_difference_norm=float(np.linalg.norm(z[1]-z[0])),
                    archive=destination.name,sha256=sha(destination)))
    (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'dictionary_reload_controls.json').write_text(json.dumps(controls,indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(status='transfer_analysis_complete',
        capture_run_complete=(args.captures/'manifest.json').exists(),examples=len(records),
        no_new_fitting=True,scope='prompt-only benchmark activation transfer',
        dictionaries={k:v['sha256'] for k,v in quality.items()},
        original_calibration_manifest_sha256=sha(args.dictionaries/'manifest.json'),
        interpretation='Reconstruction quality must justify feature interpretation. No generated reasoning or task scores.',
        files={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}),indent=2),encoding='utf-8')
    print('Saved',len(rows),'frozen-dictionary transfer records')

if __name__=='__main__':
    main()
