"""Check the frozen ARC-trained dictionaries on ledger states before feature interpretation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
import json
from pathlib import Path
import numpy as np
from .io import ROOT,digest,save
from .capture_contract import load_captures
from experiments.nanbeige_reasoning.transfer_muon_to_vpd import numpy_encoder
from experiments.nanbeige_reasoning.intervention_change_fidelity import decode,totals,scores


def describe_effect(dictionary,h0,h1):
    """Keep true changes, feature changes, and reconstruction fidelity distinct."""
    z0=numpy_encoder(dictionary,h0);z1=numpy_encoder(dictionary,h1)
    actual=np.asarray(h1,dtype=np.float64)-h0
    reconstructed=decode(z1-z0,dictionary['decoder'])
    t=totals(actual,reconstructed)
    return dict(delta_h=actual,delta_z=z1-z0,delta_reconstruction=reconstructed,delta_error=actual-reconstructed),{**t,**scores(t)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest,records,states=load_captures(a.captures)
    a.output.mkdir(parents=True,exist_ok=False)
    control=ROOT/'artifacts/nanbeige_reasoning/muon_sae_control_v1';contract=json.loads((control/'contract.json').read_text(encoding='utf-8'))
    collected={}
    for r in records:
        z=states[r['episode_id']]
        for anchor in r['anchors']:
            for site in contract['sites']:
                for inv in [0,1]:
                    collected.setdefault((site,inv,anchor['anchor'],r['split']),[]).append(z[f'{site}__pass_{inv}'][anchor['array_row']])
    results=[]
    for index,site in enumerate(contract['sites']):
        for arm in ['adamw','muon_hybrid']:
            folder=control/f'{index:02d}_{arm}';dm=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));dp=folder/'dictionary.npz'
            assert digest(dp)==dm['files']['dictionary.npz']
            with np.load(dp,allow_pickle=False) as z:d={k:z[k].astype(np.float64) for k in ['encoder_weight','encoder_bias','pre_bias','decoder','top_k']}
            for (s,inv,anchor,split),rows in collected.items():
                if s!=site:continue
                x=np.stack(rows).astype(np.float64);code=numpy_encoder(d,x);recon=decode(code,d['decoder'])+d['pre_bias']
                sse=float(np.sum((x-recon)**2));sst=float(np.sum((x-x.mean(0))**2))
                results.append(dict(site=site,pass_number=inv+1,anchor=anchor,split=split,arm=arm,rows=len(rows),
                    mse=sse/x.size,explained_variance=1-sse/sst if sst else None,mean_l0=float(np.mean(np.sum(code!=0,axis=1))),
                    dictionary_sha256=digest(dp),dictionary_refit=False,feature_semantics_validated=False))
            print(site,arm,'frozen reconstruction checked',flush=True)
    save(a.output/'quality.json',results)
    save(a.output/'manifest.json',dict(status='ledger_frozen_sae_quality_complete',capture_manifest_sha256=digest(a.captures/'manifest.json'),
        code_sha256=digest(Path(__file__)),capture_validator_sha256=digest(Path(__file__).with_name('capture_contract.py')),
        new_dictionary_fits=0,quality_sha256=digest(a.output/'quality.json'),
        interpretation='Poor reconstruction prevents interpreting feature absence; even positive state EV does not establish intervention-change fidelity.'))


if __name__=='__main__':main()
