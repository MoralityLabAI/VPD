"""Apply the frozen common dictionaries to captured prompt/reasoning regions."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.nanbeige_looped.sae import SharedSAE,encode_rows
from experiments.nanbeige_reasoning.fetch_sources import sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('calibration',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    captures=[json.loads(x) for x in (args.capture/'captures.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (args.capture/'generations.jsonl').read_text().splitlines()]
    native={r['id']:r for r in rows if r['loops']==2}
    summaries=json.loads((args.calibration/'sae_quality.json').read_text())
    out=args.capture/'phase_analysis'
    out.mkdir(exist_ok=True)
    hidden_metrics=[]
    for record in captures:
        with np.load(args.capture/record['archive']) as data:
            for site in record['sites']:
                for phase,(start,end) in record['regions'].items():
                    positions=data['token_positions']
                    mask=(positions>=start)&(positions<end)
                    if not mask.any():
                        continue
                    first,second=[data[f'{site}__pass_{p}'][mask] for p in (0,1)]
                    n0,n1=np.linalg.norm(first),np.linalg.norm(second)
                    hidden_metrics.append(dict(id=record['id'],lane=native[record['id']]['lane'],
                        split=record['calibration_split'],site=site,phase=phase,rows=int(mask.sum()),
                        first_norm=float(n0),second_norm=float(n1),delta_norm=float(np.linalg.norm(second-first)),
                        cosine=float(np.sum(first*second)/(n0*n1)) if n0*n1 else None,
                        second_first_norm_ratio=float(n1/n0) if n0 else None))
    (out/'hidden_phase_metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in hidden_metrics)+'\n')
    feature_quality=[]
    for site,entry in summaries.items():
        with np.load(args.calibration/entry['dictionary']) as arrays:
            features,width=arrays['encoder_weight'].shape
            sae=SharedSAE(width,features,int(arrays['top_k']))
            with torch.no_grad():
                sae.encoder.weight.copy_(torch.from_numpy(arrays['encoder_weight']))
                sae.encoder.bias.copy_(torch.from_numpy(arrays['encoder_bias']))
                sae.decoder.weight.copy_(torch.from_numpy(arrays['decoder'].T))
                sae.pre_bias.copy_(torch.from_numpy(arrays['pre_bias']))
            sae.eval().requires_grad_(False)
        pooled=defaultdict(list)
        for i,record in enumerate(captures):
            with np.load(args.capture/record['archive']) as data:
                hidden=[data[f'{site}__pass_{p}'] for p in (0,1)]
                features=[encode_rows(sae,h)[0] for h in hidden]
                np.savez_compressed(out/f'features_{site}_{i:04d}.npz',
                    pass_0=features[0],pass_1=features[1],delta=features[1]-features[0],
                    token_positions=data['token_positions'])
                if record['calibration_split']=='eval':
                    for phase,(start,end) in record['regions'].items():
                        mask=(data['token_positions']>=start)&(data['token_positions']<end)
                        if mask.any():
                            for p in (0,1):
                                pooled[(native[record['id']]['lane'],phase,p)].append(hidden[p][mask])
        for (lane,phase,p),values in pooled.items():
            _,quality=encode_rows(sae,np.concatenate(values))
            feature_quality.append(dict(site=site,lane=lane,phase=phase,pass_index=p,**quality))
    (out/'sae_phase_quality.json').write_text(json.dumps(feature_quality,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(
        status='analysis_complete',feature_basis='frozen calibration SAE; no evaluation refit',
        calibration_dictionary_training_scope='late native-continuation rows only',
        interpretation='Negative or poor phase reconstruction prevents mechanistic feature interpretation.',
        dictionaries={site:entry['sha256'] for site,entry in summaries.items()},
        capture_manifest_sha256=sha(args.capture/'manifest.json'),
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}),indent=2))
    print('Analyzed',len(hidden_metrics),'layer/phase records and',len(feature_quality),'SAE quality groups')

if __name__=='__main__':
    main()
