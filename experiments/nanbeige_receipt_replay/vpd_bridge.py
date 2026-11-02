"""Use existing RecurrentTarget/measure_component for conditional ledger interventions.

The caller supplies the verified, loaded BF16 model. This module does not create
an alternative inference stack, train factors, or equate Q4 and BF16 behavior.
"""
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from experiments.nanbeige_looped.recurrent import RecurrentTarget
from experiments.nanbeige_looped.measure import measure_component
from .probes import predict
from .sae_analysis import describe_effect
from .io import digest


def attach_frozen_factors(model,factors_path,expected_sha256):
    if digest(factors_path)!=expected_sha256:raise ValueError('Frozen VPD checksum mismatch')
    with np.load(factors_path,allow_pickle=False) as z:
        paths=[k[:-3] for k in z.files if k.endswith('__V')]
        target=RecurrentTarget(model,{path:z[f'{path}__V'].shape[1] for path in paths})
        with torch.no_grad():
            for path in paths:
                for name in ['V','U']:
                    dest=getattr(target.physical[path].core,name)
                    value=torch.from_numpy(z[f'{path}__{name}']).to(device=dest.device,dtype=dest.dtype)
                    if value.shape!=dest.shape:raise ValueError('Component basis dimensions differ')
                    dest.copy_(value)
    return target


def measured_probe_effects(target,tokens,*,path,component,alpha,probe_site,probe_invocation,probe,dictionaries,numpy_dictionaries):
    """Existing signed parameter perturbations plus separate probe/SAE effects."""
    arrays,receipt=measure_component(target,tokens,list(dictionaries),dictionaries,path=path,component=component,alpha=alpha,rows=1)
    key=f'{probe_site}__pass_{probe_invocation-1}__h'
    baseline=arrays[f'baseline__{key}'];base_score=float(predict(probe,baseline)[-1]);rows=[];feature_arrays={}
    for scope in ['first','second','both']:
        for sign in ['plus','minus']:
            name=f'{scope}_{sign}';state=arrays[f'{name}__{key}'];score=float(predict(probe,state)[-1])
            effect,quality=describe_effect(numpy_dictionaries[probe_site],baseline,state)
            rows.append(dict(condition=name,baseline_probe_score=base_score,intervened_probe_score=score,
                probe_score_change=score-base_score,intervention_change_fidelity=quality))
            feature_arrays.update({f'{name}__{k}':v for k,v in effect.items()})
    return arrays,feature_arrays,dict(existing_measurement_receipt=receipt,probe_effects=rows,
        scope='BF16 fixed-prefix effects; probe trained on Q4 requires baseline transfer/behavior checks before a causal interpretation')


def continue_action(target,prefix,*,path,component,alpha,passes,eog_ids,max_new_tokens=32):
    """Bounded greedy continuation with the same existing pass-scoped wrapper.

RecurrentTarget resets invocation counters each forward. Full-prefix evaluation
avoids introducing another cache implementation. Caller grades the action with
the same ledger parser and environment; emitted tokens alone are not an outcome.
"""
    tokens=prefix.clone();generated=[];finished=False
    context=target.intervene(path,component,alpha,passes) if passes else nullcontext()
    with context,torch.inference_mode(),target.autocast():
        for _ in range(max_new_tokens):
            logits=target(tokens)[:,-1];next_id=int(logits[0].argmax())
            generated.append(next_id)
            if next_id in eog_ids:finished=True;break
            tokens=torch.cat((tokens,torch.tensor([[next_id]],device=tokens.device,dtype=tokens.dtype)),dim=1)
    return dict(output_ids=generated,finish_reason='stop' if finished else 'length',sampling='greedy',
        condition=dict(component=component,alpha=alpha,passes=passes),
        comparison_scope='Within BF16 deterministic continuations; distinct from the sampled Q4 pilot')
