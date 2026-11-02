"""Paired local SAE optimizer pilot on verified, cached recurrent activations.

No language model is loaded. Muon updates the two SAE matrices; AdamW updates
biases. The control uses AdamW for every parameter on the same runtime/device.
"""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.nanbeige_looped.sae import SharedSAE, encode_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def optimizers(model, arm):
    named = dict(model.named_parameters())
    if arm == 'adamw':
        groups = {'adamw': list(named)}
        opts = [torch.optim.AdamW(list(named.values()), lr=.001, weight_decay=0., foreach=False)]
    elif arm == 'muon_hybrid':
        matrix = [n for n, p in named.items() if p.ndim == 2]
        bias = [n for n, p in named.items() if p.ndim != 2]
        if set(matrix) != {'encoder.weight', 'decoder.weight'}:
            raise ValueError('Unexpected SAE matrix parameters')
        groups = {'muon': matrix, 'adamw': bias}
        opts = [torch.optim.Muon([named[n] for n in matrix], lr=.001, weight_decay=0.,
                    momentum=.95, nesterov=True, ns_steps=5,
                    ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-7,
                    adjust_lr_fn='match_rms_adamw'),
                torch.optim.AdamW([named[n] for n in bias], lr=.001, weight_decay=0., foreach=False)]
    else:
        raise ValueError('Unknown optimizer arm')
    assigned = [id(p) for opt in opts for group in opt.param_groups for p in group['params']]
    if len(assigned) != len(set(assigned)) or set(assigned) != {id(p) for p in named.values()}:
        raise ValueError('Optimizer partition omits or duplicates a parameter')
    return opts, groups


def initialize(train, seed, features=6144, top_k=32):
    torch.manual_seed(seed)
    model = SharedSAE(train[0].shape[1], features, top_k)
    with torch.no_grad():
        model.pre_bias.copy_(torch.from_numpy(np.mean([x.mean(0) for x in train], axis=0)))
    initial = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    return model, initial


def usage(z):
    frequency = (z > 0).sum(axis=0)
    return {'active_features': int((frequency > 0).sum()),
            'inactive_features_on_this_split': int((frequency == 0).sum()),
            'mean_l0': float((z > 0).sum(axis=1).mean())}


def load_data(run, sites):
    cap = run / 'q4_activations'
    checked = read(run / 'mechanistic_verification.json')
    if checked['status'] != 'q4_reasoning_mechanistic_artifacts_verified':
        raise ValueError('Verified source run required')
    if checked['capture_manifest_sha256'] != digest(cap / 'manifest.json'):
        raise ValueError('Changed source capture manifest')
    manifest = read(cap / 'manifest.json')
    if manifest['files']['records.jsonl'] != digest(cap / 'records.jsonl'):
        raise ValueError('Changed capture records')
    records = [json.loads(s) for s in (cap / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records) != 18 or len({r['id'] for r in records}) != 18:
        raise ValueError('Expected eighteen distinct source inputs')
    if sum(r['split'] == 'train' for r in records) != 8 or sum(r['split'] == 'eval' for r in records) != 10:
        raise ValueError('Changed train/evaluation split')
    values = {site: {} for site in sites}
    for r in records:
        path = cap / r['archive']
        if digest(path) != r['sha256'] or manifest['files'][r['archive']] != r['sha256']:
            raise ValueError('Changed source activation archive')
        with np.load(path, allow_pickle=False) as saved:
            for site in sites:
                pair = [saved[f'{site}__pass_{i}'].astype(np.float32) for i in (0, 1)]
                if any(x.shape != (32, 3072) or not np.isfinite(x).all() for x in pair):
                    raise ValueError('Invalid aligned source activations')
                values[site][r['id']] = pair
    return records, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--sites', nargs='+', default=['model.layers.21', 'model.layers.11', 'model.norm'])
    p.add_argument('--max-wall-seconds', type=int, default=7200)
    args = p.parse_args()
    if args.steps < 1 or args.max_wall_seconds < 1 or len(set(args.sites)) != len(args.sites):
        p.error('Positive budgets and unique sites required')
    if any(s not in ['model.layers.21', 'model.layers.11', 'model.norm'] for s in args.sites):
        p.error('Unsupported site')
    if not torch.cuda.is_available():
        raise RuntimeError('This protocol requires local CUDA; no silent CPU fallback')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, values = load_data(args.run, args.sites)
    training = [r['id'] for r in records if r['split'] == 'train']
    evaluation = [r['id'] for r in records if r['split'] == 'eval']
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seed = 42
    batches = np.random.default_rng(seed).integers(256, size=(args.steps, 64))
    np.save(args.output / 'batches.npy', batches, allow_pickle=False)
    muon_source = Path(inspect.getfile(torch.optim.Muon))
    contract = dict(schema='nanbeige.sae_muon_control.v1', source_run=str(args.run.resolve()),
        source_verification_sha256=digest(args.run / 'mechanistic_verification.json'),
        capture_manifest_sha256=digest(args.run / 'q4_activations/manifest.json'),
        source_archives={r['archive']: r['sha256'] for r in records},
        training_ids=training, evaluation_ids=evaluation, sites=args.sites, seed=seed,
        steps=args.steps, features=6144, top_k=32, batch_size=128, training_rows_per_pass=256,
        objective='Raw pooled MSE; same selected row indices from each pass',
        decoder_projection='Unit column norm after every optimizer step',
        learning_rate=.001, weight_decay=0., bias_optimizer='AdamW', muon_momentum=.95,
        muon_nesterov=True, muon_ns_steps=5, muon_ns_coefficients=[3.4445, -4.775, 2.0315],
        muon_eps=1e-7, muon_adjust_lr_fn='match_rms_adamw',
        device=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        tf32=False, parameter_dtype='float32', muon_ns_dtype='bfloat16',
        batches_sha256=digest(args.output / 'batches.npy'),
        source_code_sha256=digest(__file__), shared_sae_sha256=digest(Path(__file__).parents[1] / 'nanbeige_looped/sae.py'),
        muon_implementation_sha256=digest(muon_source),
        prior_paper_sha256=digest('artifacts/nanbeige_reasoning/muon_sources/constitutional_adapter.pdf'),
        prior_paper_url='https://www.moralitylab.xyz/papers/Late_Layer_Feature_Reweighting_by_a_Constitutional_Adapter_An_Industrial_Sparse_Autoencoder_Analysis.pdf',
        provenance_note='Paper-inspired implementation; original optimizer code/hyperparameters not supplied in PDF.',
        exploratory_posthoc=True, max_wall_seconds=args.max_wall_seconds,
        scope='Single-seed paired optimizer pilot on cached Q4 activations; no new language-model inference, VPD fit or behavioral intervention.')
    write(args.output / 'contract.json', contract)
    summaries = []
    try:
        for site_index, site in enumerate(args.sites):
            train = [np.concatenate([values[site][key][i] for key in training]) for i in (0, 1)]
            initial_hashes = []
            arms = ['adamw', 'muon_hybrid'] if site_index % 2 == 0 else ['muon_hybrid', 'adamw']
            for arm in arms:
                out = args.output / f'{site_index:02d}_{arm}'
                out.mkdir()
                model, initial = initialize(train, seed)
                initial_hashes.append(initial)
                if len(set(initial_hashes)) != 1:
                    raise ValueError('Unequal paired initialization')
                model.to('cuda')
                opts, groups = optimizers(model, arm)
                device_train = [torch.from_numpy(x).cuda() for x in train]
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                arm_started = time.perf_counter()
                history = []
                for step, indices in enumerate(batches):
                    if time.monotonic() - started > args.max_wall_seconds:
                        raise TimeoutError('Finite optimizer-pilot wall budget reached')
                    batch = torch.cat([x[indices] for x in device_train])
                    reconstructed, _ = model(batch)
                    pass_mse = (reconstructed - batch).square().reshape(2, 64, 3072).mean((1, 2))
                    loss = pass_mse.mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite reconstruction loss')
                    for opt in opts:
                        opt.zero_grad(set_to_none=True)
                    loss.backward()
                    for opt in opts:
                        opt.step()
                    model.normalize_decoder()
                    history.append(dict(step=step + 1, objective=float(loss.detach()), pass_mse=pass_mse.detach().tolist()))
                    if step == 0 or (step + 1) % 50 == 0:
                        state = dict(status='training', site=site, arm=arm, step=step + 1,
                            completed_conditions=len(summaries), elapsed_seconds=time.monotonic() - started)
                        write(args.output / 'run_status.json', state)
                        print(json.dumps(state), flush=True)
                torch.cuda.synchronize()
                training_seconds = time.perf_counter() - arm_started
                peak_bytes = torch.cuda.max_memory_allocated()
                model.cpu().eval().requires_grad_(False)
                del opt, opts, device_train, batch, reconstructed, pass_mse, loss
                torch.cuda.empty_cache()
                model.export(out / 'dictionary.npz', site=site)
                pooled_features = {'train': [[], []], 'eval': [[], []]}
                feature_records = []
                for index, r in enumerate(records):
                    encoded = [encode_rows(model, x)[0] for x in values[site][r['id']]]
                    z0, z1 = encoded
                    name = f'features_{index:02d}.npz'
                    np.savez_compressed(out / name, pass_0=z0, pass_1=z1, delta_z=z1-z0,
                        token_positions=np.asarray(r['positions']))
                    feature_records.append(dict(id=r['id'], split=r['split'], archive=name, sha256=digest(out / name)))
                    for i, z in enumerate(encoded):
                        pooled_features[r['split']][i].append(z)
                quality = {}
                for split, ids in [('train', training), ('eval', evaluation)]:
                    quality[split] = {}
                    for i in (0, 1):
                        x = np.concatenate([values[site][key][i] for key in ids])
                        _, q = encode_rows(model, x)
                        q.update(usage(np.concatenate(pooled_features[split][i])))
                        quality[split][f'pass_{i}'] = q
                    quality[split]['both_pass_usage'] = usage(np.concatenate([np.concatenate(z) for z in pooled_features[split]]))
                result = dict(status='sae_optimizer_condition_complete', site=site, arm=arm, seed=seed,
                    completed_steps=len(history), initial_parameters_sha256=initial,
                    batches_sha256=contract['batches_sha256'], parameter_groups=groups,
                    quality=quality, training_seconds=training_seconds, peak_cuda_bytes=peak_bytes,
                    timing_scope='Training only; uncontrolled laptop clocks and thermal state; no efficiency superiority claim', output=out.name)
                write(out / 'history.json', history)
                write(out / 'records.json', feature_records)
                write(out / 'manifest.json', dict(**result, files={f.name: digest(f) for f in out.iterdir() if f.is_file()}))
                summaries.append(result)
                write(args.output / 'summary.json', summaries)
                print(json.dumps(dict(completed=out.name, site=site, seconds=training_seconds,
                    eval_ev=[quality['eval'][f'pass_{i}']['explained_variance'] for i in (0, 1)])), flush=True)
                del model
        write(args.output / 'manifest.json', dict(status='sae_muon_control_complete', conditions=len(summaries),
            contract_sha256=digest(args.output / 'contract.json'),
            files={p.relative_to(args.output).as_posix(): digest(p) for p in args.output.rglob('*')
                   if p.is_file() and p.name != 'run_status.json'}))
        write(args.output / 'run_status.json', dict(status='sae_muon_control_complete', conditions=len(summaries)))
    except BaseException as exc:
        write(args.output / 'run_status.json', dict(status='needs_attention', error=f'{type(exc).__name__}: {exc}', completed_conditions=len(summaries)))
        raise


if __name__ == '__main__':
    main()
