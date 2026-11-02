"""Read a matched activation/SAE pair from the dataset index, without inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.nanbeige.build_mechinterp_index import ROOT,DATA,read,rows,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--example')
    parser.add_argument('--site',default='model.layers.21',choices=['model.layers.11','model.layers.21','model.norm'])
    parser.add_argument('--optimizer',default='muon_hybrid_paired_cuda',choices=['adamw_original_cpu','adamw_paired_cuda','muon_hybrid_paired_cuda'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    directory=DATA/'dataset_index_v1';manifest=read(directory/'manifest.json')
    assert sha(directory/'examples.jsonl')==manifest['index_sha256']
    assert sha(directory/'assets.json')==manifest['assets_sha256']
    examples=rows(directory/'examples.jsonl');assets=read(directory/'assets.json')
    example=next(r for r in examples if r['id']==args.example) if args.example else examples[0]
    features=next(r for r in example['saes'] if r['site']==args.site and r['optimizer']==args.optimizer)
    assert features['variant']==example['activations']['variant']
    def checked(name):
        p=(ROOT/name).resolve()
        assert p.is_relative_to(ROOT) and sha(p)==assets[name]['sha256']
        return p
    checked(features['dictionary'])
    with np.load(checked(example['activations']['file']),allow_pickle=False) as a,np.load(checked(features['file']),allow_pickle=False) as z:
        np.testing.assert_array_equal(a['token_positions'],z['token_positions'])
        np.testing.assert_array_equal(z['token_positions'],example['activations']['token_positions'])
        np.testing.assert_array_equal(z['pass_1']-z['pass_0'],z['delta_z'])
        h0,h1=a[args.site+'__pass_0'],a[args.site+'__pass_1']
        assert h0.shape==h1.shape==(32,3072) and z['delta_z'].shape==(32,6144)
        scores=np.abs(z['delta_z'].astype(np.float64)).mean(axis=0)
        top=np.argsort(-scores,kind='stable')[:5]
        result=dict(id=example['id'],calibration_split=example['calibration_split'],site=args.site,optimizer=args.optimizer,
            dictionary=features['dictionary'],dictionary_sha256=assets[features['dictionary']]['sha256'],
            matched_positions=32,activation_shape=list(h0.shape),feature_shape=list(z['delta_z'].shape),
            activation_pass_difference_l2=float(np.linalg.norm(h1.astype(np.float64)-h0)),
            largest_mean_absolute_feature_differences=[dict(feature_id=int(i),mean_absolute_delta=float(scores[i])) for i in top],
            limitation='Descriptive coordinates in this frozen dictionary only; feature IDs carry no validated semantic label.',
            model_inference_performed=False,index_sha256=manifest['index_sha256'])
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__': main()
