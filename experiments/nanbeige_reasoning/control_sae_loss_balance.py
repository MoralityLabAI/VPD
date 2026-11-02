"""Paired CPU SAE loss-weighting control on fixed layer-22 activations.

Post-hoc exploratory control: three fixed seeds, both objectives, no selection.
No language-model inference or cloud resources are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.nanbeige_looped.sae import SharedSAE, encode_rows
from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest, read
from scripts.nanbeige.verify_multipass_results import load_sae


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    manifest = read(args.fit/'manifest.json')
    verification = read(args.fit/'independent_verification.json')
    lineage = read(args.fit/'cross_precision_verification.json')
    if (verification['status'] != 'multipass_saved_effects_independently_verified'
            or lineage['status'] != 'cross_precision_reasoning_vpd_verified'
            or verification['manifest_sha256'] != digest(args.fit/'manifest.json')
            or lineage['manifest_sha256'] != digest(args.fit/'manifest.json')):
        raise ValueError('Verified generated-prefix fit required')
    examples = read(args.fit/'examples.json')
    if digest(args.fit/'examples.json') != manifest['artifacts']['examples.json']:
        raise ValueError('Changed examples')
    training = [r for r in examples if r['split']=='train']
    evaluation = [r for r in examples if r['split']=='eval']
    if len(training) != 8 or len(evaluation) != 10:
        raise ValueError('Unexpected pilot split')
    site = 'model.layers.21'
    values, positions, sources = {}, {}, {}
    for r in examples:
        name = manifest['baseline_archives'][r['id']]
        if digest(args.fit/name) != manifest['artifacts'][name]:
            raise ValueError('Changed activation archive')
        sources[name] = manifest['artifacts'][name]
        with np.load(args.fit/name, allow_pickle=False) as a:
            positions[r['id']] = a['token_positions']
            np.testing.assert_array_equal(a['token_ids'], np.asarray(r['token_ids'])[positions[r['id']]])
            values[r['id']] = [a[f'{site}__pass_{i}'].astype(np.float32) for i in (0,1)]
    train = [np.concatenate([values[r['id']][i] for r in training]) for i in (0,1)]
    if any(x.shape != (64,3072) for x in train):
        raise ValueError('Expected original 64 training rows per pass')
    variance = np.asarray([np.square(x.astype(np.float64)-x.mean(0,dtype=np.float64)).mean() for x in train])
    if not np.isfinite(variance).all() or np.any(variance<=0):
        raise ValueError('Invalid training variances')
    inverse = 1/variance
    balanced_weights = inverse/inverse.mean()
    arms = [('raw_mse', [1.,1.]), ('inverse_variance_mse', balanced_weights.tolist())]
    seeds = [42,43,44]
    steps = 1000
    args.output.mkdir(parents=True, exist_ok=False)
    contract = dict(schema='nanbeige.sae_loss_balance.v1', site=site, seeds=seeds,
        arms={name: weights for name,weights in arms}, steps=steps, features=6144, top_k=32,
        batch_size=128, learning_rate=.001, weight_decay=0., device='cpu', torch_threads=2,
        torch_version=torch.__version__, training_rows_per_pass=64,
        training_variance=variance.tolist(), weighting='Inverse per-pass training variance, normalized to mean weight one',
        initial_pre_bias='Equal-pass pooled training mean', exploratory_posthoc=True,
        training_ids=[r['id'] for r in training], evaluation_ids=[r['id'] for r in evaluation],
        source_manifest_sha256=digest(args.fit/'manifest.json'), source_files=sources,
        source_code_sha256=digest(Path(__file__)), shared_sae_code_sha256=digest(Path(__file__).parents[1]/'nanbeige_looped/sae.py'),
        claim_scope='Paired loss-weighting control; no new evaluation cohort, feature semantics, or behavioral outcomes.')
    write(args.output/'contract.json', contract)
    summaries = []
    started = time.monotonic()
    for seed in seeds:
        generator = np.random.default_rng(seed)
        batches = generator.integers(len(train[0]), size=(steps,64))
        batch_sha = hashlib.sha256(batches.tobytes()).hexdigest()
        np.save(args.output/f'batches_{seed}.npy', batches, allow_pickle=False)
        initial_hashes = []
        for arm, weights in arms:
            out = args.output/f'{seed}_{arm}'
            out.mkdir()
            torch.manual_seed(seed)
            model = SharedSAE(3072,6144,32)
            with torch.no_grad():
                model.pre_bias.copy_(torch.from_numpy(np.mean([x.mean(0) for x in train],axis=0)))
            initial = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            initial_hashes.append(initial)
            if len(set(initial_hashes)) != 1:
                raise ValueError('Paired model initializations differ')
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
            weights_tensor = torch.tensor(weights,dtype=torch.float32)
            history = []
            for step, indices in enumerate(batches):
                batch = torch.from_numpy(np.concatenate([x[indices] for x in train]))
                reconstruction, _ = model(batch)
                losses = (reconstruction-batch).square().reshape(2,64,3072).mean((1,2))
                loss = (losses*weights_tensor).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite objective')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                model.normalize_decoder()
                history.append(dict(step=step+1, objective=float(loss.detach()),
                    pass_mse=losses.detach().tolist()))
                if step==0 or (step+1)%50==0:
                    write(args.output/'run_status.json',dict(status='training',seed=seed,arm=arm,step=step+1,
                        completed_conditions=len(summaries),elapsed_seconds=time.monotonic()-started))
                    print(json.dumps(dict(seed=seed,arm=arm,step=step+1,loss=history[-1]['objective'],
                        elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
            model.eval().requires_grad_(False)
            dictionary = out/'dictionary.npz'
            model.export(dictionary,site=site)
            reloaded = load_sae(dictionary)
            quality = {}
            for split, selected in [('train',training),('eval',evaluation)]:
                quality[split] = {}
                for i in (0,1):
                    x = np.concatenate([values[r['id']][i] for r in selected])
                    z, measure = encode_rows(model,x)
                    zr, mr = encode_rows(reloaded,x)
                    np.testing.assert_array_equal(z,zr)
                    if measure != mr:
                        raise ValueError('Reloaded dictionary quality differs')
                    quality[split][f'pass_{i}'] = measure
            records = []
            for index,r in enumerate(examples):
                encoded = [encode_rows(reloaded,x) for x in values[r['id']]]
                z0,z1 = [v[0] for v in encoded]
                path = out/f'features_{index:02d}.npz'
                np.savez_compressed(path,pass_0=z0,pass_1=z1,delta_z=z1-z0,token_positions=positions[r['id']])
                records.append(dict(id=r['id'],split=r['split'],archive=path.name,sha256=digest(path),
                    quality={f'pass_{i}':v[1] for i,v in enumerate(encoded)}))
            write(out/'history.json', history)
            write(out/'records.json', records)
            result = dict(seed=seed, arm=arm, weights=weights, quality=quality,
                initial_parameters_sha256=initial, batches_sha256=batch_sha, dictionary_sha256=digest(dictionary),
                reloaded_quality_exact=True, completed_steps=len(history), output=out.name)
            write(out/'manifest.json',dict(status='sae_loss_condition_complete',**result,
                files={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
            summaries.append(result)
            write(args.output/'summary.json', summaries)
            print(json.dumps(dict(completed_condition=out.name,quality=quality)),flush=True)
            del model,reloaded,optimizer
    write(args.output/'manifest.json',dict(status='sae_loss_balance_control_complete',conditions=len(summaries),
        contract_sha256=digest(args.output/'contract.json'),elapsed_seconds=time.monotonic()-started,
        files={p.relative_to(args.output).as_posix():digest(p) for p in args.output.rglob('*')
               if p.is_file() and p != args.output/'manifest.json' and p != args.output/'run_status.json'}))
    write(args.output/'run_status.json',dict(status='sae_loss_balance_control_complete',conditions=len(summaries)))


if __name__=='__main__':
    main()
