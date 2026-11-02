"""Audit sign symmetry and BF16 resolution of saved component interventions.

A single coefficient magnitude does not establish a derivative, numerical
noise floor, or causal answer effect. Original arrays/proxies remain unchanged.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest, read


def energy(x):
    return float(np.square(x).sum(dtype=np.float64))


def quotient(a, b):
    return a/b if b else None


def stable_kl(reference, candidate):
    def log_probs(logits):
        shifted = logits.astype(np.float64)-np.max(logits)
        return shifted-np.log(np.exp(shifted).sum())
    lp,lq = log_probs(reference),log_probs(candidate)
    p,q = np.exp(lp),np.exp(lq)
    r = lq-lp
    direct = float(np.sum(p*(-r)))
    # Sum q-p is zero for normalized distributions. The equivalent generalized
    # KL form avoids subtracting nearly equal entropy terms near identical logits.
    small = np.abs(r)<50
    terms = np.empty_like(r)
    terms[small] = p[small]*(np.expm1(r[small])-r[small])
    terms[~small] = q[~small]-p[~small]-p[~small]*r[~small]
    stable = float(terms.sum())
    np.testing.assert_allclose(stable,direct,rtol=1e-8,atol=1e-12)
    if stable < 0 or not np.isfinite(stable):
        raise ValueError('Invalid recomputed divergence')
    return stable,direct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = read(args.fit/'manifest.json')
    verified = read(args.fit/'independent_verification.json')
    if (verified['status'] != 'multipass_saved_effects_independently_verified'
            or verified['manifest_sha256'] != digest(args.fit/'manifest.json')):
        raise ValueError('Verified saved effects required')
    if digest(args.fit/'effects.json') != manifest['artifacts']['effects.json']:
        raise ValueError('Changed effect metadata')
    effects = read(args.fit/'effects.json')
    if len(effects) != 20 or {r['alpha'] for r in effects} != {.05}:
        raise ValueError('Expected the fixed 20-record signed pilot')
    args.output.mkdir(parents=True, exist_ok=False)
    records, endpoint_rows = [], []
    groups = defaultdict(list)
    checks = 0
    for record in effects:
        source = (args.fit/record['archive']).resolve()
        if source.parent != args.fit.resolve() or digest(source) != record['sha256']:
            raise ValueError('Changed intervention archive')
        with np.load(source,allow_pickle=False) as arrays:
            reference = arrays['baseline__last_logits']
            baseline_path = args.fit/manifest['baseline_archives'][record['example_id']]
            if digest(baseline_path) != manifest['artifacts'][baseline_path.name]:
                raise ValueError('Changed baseline archive')
            with np.load(baseline_path,allow_pickle=False) as baseline:
                np.testing.assert_array_equal(reference,baseline['last_logits'])
            for condition, proxy in record['output_proxies'].items():
                logits = arrays[f'{condition}__last_logits']
                divergence,direct = stable_kl(reference,logits)
                candidate = int(reference[0].argmax())
                if (candidate != proxy['reference_top_token_id'] or
                        float(logits[0,candidate]-reference[0,candidate]) != proxy['reference_top_token_logit_delta']):
                    raise ValueError('Stored top-token proxy differs from raw logits')
                endpoint_rows.append(dict(example_id=record['example_id'],physical_module=record['physical_module'],
                    condition=condition,**proxy,recomputed_kl_float64=divergence,direct_kl_float64=direct,
                    candidate_argmax=int(logits[0].argmax()),argmax_changed=int(logits[0].argmax())!=candidate))
            for site in manifest['sites']:
                for invocation in (0,1):
                    key = f'{site}__pass_{invocation}__h'
                    raw = arrays[f'baseline__{key}']
                    if raw.dtype != np.float32 or np.any(raw.view(np.uint32)&65535):
                        raise ValueError('Baseline is not exactly representable in BF16')
                    base = raw.astype(np.float64)
                    # At a power-of-two boundary, use the larger adjacent step.
                    _, exponents = np.frexp(np.abs(base))
                    ulp = np.exp2(np.where(base==0,-133,np.maximum(exponents-8,-133)))
                    for scope in ('first','second','both'):
                        plus_raw,minus_raw = arrays[f'{scope}_plus__{key}'],arrays[f'{scope}_minus__{key}']
                        if any(x.dtype != np.float32 or np.any(x.view(np.uint32)&65535) for x in (plus_raw,minus_raw)):
                            raise ValueError('Perturbed state is not exactly representable in BF16')
                        plus,minus = plus_raw.astype(np.float64)-base,minus_raw.astype(np.float64)-base
                        odd,even = (plus-minus)/2,(plus+minus)/2
                        np.testing.assert_array_equal(odd+even,plus)
                        np.testing.assert_array_equal(even-odd,minus)
                        checks += 2
                        changed_plus,changed_minus = plus!=0,minus!=0
                        row = dict(example_id=record['example_id'],pair_id=record['pair_id'],
                            physical_module=record['physical_module'],component=record['component'],
                            alpha=record['alpha'],site=site,invocation=invocation,scope=scope,
                            base_energy=energy(base),plus_energy=energy(plus),minus_energy=energy(minus),
                            odd_energy=energy(odd),even_energy=energy(even),
                            plus_dot_negative_minus=float(np.sum(plus*(-minus))),
                            changed_plus=int(changed_plus.sum()),changed_minus=int(changed_minus.sum()),
                            changed_plus_within_one_larger_bf16_step=int(np.sum(changed_plus & (np.abs(plus)<=ulp))),
                            changed_minus_within_one_larger_bf16_step=int(np.sum(changed_minus & (np.abs(minus)<=ulp))),
                            total_coordinates=base.size)
                        row['even_to_odd_norm_ratio'] = quotient(np.sqrt(row['even_energy']),np.sqrt(row['odd_energy']))
                        row['positive_to_reversed_negative_cosine'] = quotient(row['plus_dot_negative_minus'],np.sqrt(row['plus_energy']*row['minus_energy']))
                        records.append(row)
                        groups[(record['physical_module'],site,invocation,scope)].append(row)
    pooled = []
    fields = ('base_energy','plus_energy','minus_energy','odd_energy','even_energy','plus_dot_negative_minus',
              'changed_plus','changed_minus','changed_plus_within_one_larger_bf16_step',
              'changed_minus_within_one_larger_bf16_step','total_coordinates')
    for (module,site,invocation,scope), selected in groups.items():
        totals = {f:sum(r[f] for r in selected) for f in fields}
        pooled.append(dict(physical_module=module,site=site,invocation=invocation,scope=scope,
            examples=len(selected),task_groups=len({r['pair_id'] for r in selected}),**totals,
            even_to_odd_norm_ratio=quotient(np.sqrt(totals['even_energy']),np.sqrt(totals['odd_energy'])),
            positive_to_reversed_negative_cosine=quotient(totals['plus_dot_negative_minus'],np.sqrt(totals['plus_energy']*totals['minus_energy'])),
            plus_relative_norm=quotient(np.sqrt(totals['plus_energy']),np.sqrt(totals['base_energy'])),
            fraction_changed_plus_within_one_larger_bf16_step=quotient(totals['changed_plus_within_one_larger_bf16_step'],totals['changed_plus'])))
    kl = [r['next_token_kl'] for r in endpoint_rows]
    endpoint_summary = dict(records=len(kl),negative_kl=sum(x<0 for x in kl),zero_kl=sum(x==0 for x in kl),
        minimum_kl=min(kl),maximum_kl=max(kl),
        all_kl_below_1e_minus_6=all(abs(x)<1e-6 for x in kl),
        recomputed_minimum_kl=min(r['recomputed_kl_float64'] for r in endpoint_rows),
        recomputed_maximum_kl=max(r['recomputed_kl_float64'] for r in endpoint_rows),
        changed_argmax_records=sum(r['argmax_changed'] for r in endpoint_rows),
        baseline_records=sum(r['condition']=='baseline' for r in endpoint_rows),
        maximum_absolute_kl_correction=max(abs(r['next_token_kl']-r['recomputed_kl_float64']) for r in endpoint_rows),
        numerical_note='Original float32 proxies retained unchanged. Full saved vocabulary logits were '
                       'recomputed in float64 with an equivalent generalized-KL form and checked against direct KL. '
                       'These are conditional next-token distributions, not scored answers.')
    output = dict(pooled=pooled,endpoint_summary=endpoint_summary,
        limitations=['Single magnitude at +/-0.05; not a dose sweep or validated derivative.',
                     'Even response can reflect finite-dose nonlinearity and numerical effects; their causes are not isolated.',
                     'BF16 spacing is a representational scale, not an empirically measured noise floor.',
                     'No repeated zero-dose or higher-precision model rerun was performed.',
                     'Token rows and multiple ARC inputs are not independent task replicates.',
                     'No sparse-feature semantics or scored final-answer effects are established.'])
    for name,value in [('records.json',records),('output_proxies.json',endpoint_rows),('summary.json',output)]:
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report = dict(status='signed_effect_resolution_audit_complete',records=len(records),
        exact_odd_even_reconstruction_checks=checks,fit_manifest_sha256=digest(args.fit/'manifest.json'),
        source_effects_sha256=digest(args.fit/'effects.json'),source_code_sha256=digest(Path(__file__)),
        files={p.name:digest(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=len(records),checks=checks,endpoint_summary=endpoint_summary,
        final_state=[{k:r[k] for k in ['physical_module','scope','even_to_odd_norm_ratio',
            'positive_to_reversed_negative_cosine','fraction_changed_plus_within_one_larger_bf16_step']}
            for r in pooled if r['site']=='model.norm' and r['invocation']==1]),indent=2))


if __name__=='__main__':
    main()
