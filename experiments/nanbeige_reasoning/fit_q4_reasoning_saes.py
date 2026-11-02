"""Fit shared SAEs on training Q4 trajectories and evaluate held-out trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.nanbeige_looped.sae import fit_shared_sae,encode_rows
from .fetch_sources import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    args=p.parse_args()
    manifest=json.loads((args.captures/'manifest.json').read_text())
    if manifest['status']!='q4_reasoning_capture_complete':raise ValueError('Incomplete activation capture')
    if sha(args.captures/'records.jsonl')!=manifest['files']['records.jsonl']:
        raise ValueError('Capture metadata hash mismatch')
    records=list(map(json.loads,(args.captures/'records.jsonl').read_text(encoding='utf-8').splitlines()))
    if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate capture IDs')
    training=[r for r in records if r['split']=='train']
    evaluation=[r for r in records if r['split']=='eval']
    if not training or not evaluation:raise ValueError('Both splits are required')
    if set(r['id'] for r in training)&set(r['id'] for r in evaluation):raise ValueError('Split overlap')
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    sites=['model.layers.11','model.layers.21','model.norm']
    results={}
    for site_index,site in enumerate(sites):
        values={}
        for record in records:
            path=args.captures/record['archive']
            if sha(path)!=record['sha256']:raise ValueError('Capture hash mismatch')
            with np.load(path,allow_pickle=False) as a:
                values[record['id']]=[a[f'{site}__pass_{invocation}'] for invocation in (0,1)]
        train=[np.concatenate([values[r['id']][invocation] for r in training]) for invocation in (0,1)]
        print(f'Fitting {site}: {len(train[0])} training token positions per pass, {args.steps} updates',flush=True)
        sae=fit_shared_sae(train,steps=args.steps,expansion=2,top_k=32,seed=42,device='cpu')
        dictionary=out/f'sae_{site_index:02d}.npz'
        sae.export(dictionary,site=site)
        quality={}
        for split,selected in [('train',training),('eval',evaluation)]:
            quality[split]={}
            for invocation in (0,1):
                rows=np.concatenate([values[r['id']][invocation] for r in selected])
                _,quality[split][f'pass_{invocation}']=encode_rows(sae,rows)
        differences=[]
        for index,record in enumerate(records):
            encoded=[encode_rows(sae,h) for h in values[record['id']]]
            z0,z1=[item[0] for item in encoded]
            destination=out/f'features_{site_index:02d}_{index:03d}.npz'
            np.savez_compressed(destination,pass_0=z0,pass_1=z1,delta_z=z1-z0,
                token_positions=np.asarray(record['positions']))
            differences.append(dict(id=record['id'],lane=record['lane'],split=record['split'],
                max_token_censored=record['max_token_censored'],
                quality={f'pass_{i}':encoded[i][1] for i in (0,1)},
                feature_difference_norm=float(np.linalg.norm(z1-z0)),archive=destination.name,sha256=sha(destination)))
        results[site]=dict(dictionary=dictionary.name,sha256=sha(dictionary),quality=quality,differences=differences)
        (out/'sae_quality.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(f'Completed {site}: held-out EV {quality["eval"]["pass_0"]["explained_variance"]:.4f} / {quality["eval"]["pass_1"]["explained_variance"]:.4f}',flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(status='q4_reasoning_saes_complete',steps=args.steps,
        weights='Q4_K_M',training_ids=[r['id'] for r in training],evaluation_ids=[r['id'] for r in evaluation],
        capture_manifest_sha256=sha(args.captures/'manifest.json'),seed=42,
        limitation='Pilot trajectories, including censored reasoning; no validated feature semantics or causal intervention results.',
        files={f.name:sha(f) for f in out.iterdir() if f.is_file()}),indent=2),encoding='utf-8')


if __name__=='__main__':main()
