"""Generate fresh held-out answers under learned, pass-specific VPD interventions.

Requires a completed, independently checked generated-prefix fit. Fitting tokens
are never used as an answer prefix: each generation starts at the public prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.nanbeige_looped.run_multipass import digest, write_json
from .pass_interventions import PassComponentIntervention
from .run_capture import cache_parity, require_cache_parity


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def conditions(alpha):
    result = [dict(name='baseline', path='model.layers.11.mlp.down_proj', alpha=0., passes=[0, 1])]
    for layer in (11, 21):
        for scope, selected in [('first', [0]), ('second', [1]), ('both', [0, 1])]:
            for sign, coefficient in [('plus', alpha), ('minus', -alpha)]:
                result.append(dict(name=f'layer_{layer+1}_{scope}_{sign}',
                    path=f'model.layers.{layer}.mlp.down_proj', alpha=coefficient, passes=selected))
    return result


def finalize(out, contract, records, expected):
    write_json(out / 'manifest.json', dict(status='answer_interventions_complete', contract=contract,
        records=len(records), expected_records=len(expected),
        limitations=['Single-component, single-coefficient pilot; dose and seed stability remain untested.',
                     'Fitted components came from BF16 analysis of censored Q4 trajectories.',
                     'Single-pass interventions deliberately break weight sharing.',
                     'Layer-token counts are a compute proxy and exclude component-update overhead.',
                     'No claim about Astra or adaptive neural routing.'],
        files={f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name not in ('run_status.json', 'manifest.json')}))
    write_json(out / 'run_status.json', dict(status='answer_interventions_complete', records=len(records)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--fit-input-contract', type=Path, required=True)
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--component', type=int, default=0)
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--max-new-tokens', type=int, default=8192)
    p.add_argument('--max-wall-seconds', type=int, default=21600)
    args = p.parse_args()
    if (args.component < 0 or not np.isfinite(args.alpha) or args.alpha <= 0
            or not 1 <= args.max_new_tokens <= 8192 or args.max_wall_seconds < 1):
        p.error('Invalid component, coefficient or generation bound')
    fit = read(args.fit / 'manifest.json')
    verified = read(args.fit / 'independent_verification.json')
    lineage = read(args.fit / 'cross_precision_verification.json')
    source_contract = read(args.fit_input_contract)
    if (fit['status'] != 'calibration_complete'
            or fit['model']['kind'] != 'local_pretrained_checkpoint'
            or fit['vpd_training_window']['kind'] != 'generated_prefixes'
            or verified['status'] != 'multipass_saved_effects_independently_verified'
            or verified['manifest_sha256'] != digest(args.fit / 'manifest.json')
            or lineage['status'] != 'cross_precision_reasoning_vpd_verified'
            or lineage['manifest_sha256'] != digest(args.fit / 'manifest.json')
            or lineage['input_contract_sha256'] != digest(args.fit_input_contract)
            or fit['model']['expected_revision'] != source_contract['revision']):
        raise ValueError('A verified pretrained generated-prefix VPD fit is required')
    for name in ('vpd_factors.npz', 'examples.json', 'vpd_training.json'):
        if digest(args.fit / name) != fit['artifacts'][name]:
            raise ValueError(f'Fitted artifact hash mismatch: {name}')
    for name, expected in source_contract['checkpoint_files'].items():
        if name != Path(name).name or digest(args.model_dir / name) != expected:
            raise ValueError(f'Checkpoint asset mismatch: {name}')
    for name, expected in fit['model']['checkpoint_files'].items():
        if name != Path(name).name or digest(args.model_dir / name) != expected:
            raise ValueError(f'Fit and generation checkpoint differ: {name}')
    questions = [json.loads(line) for line in args.questions.read_text(encoding='utf-8').splitlines()]
    source_examples = read(args.fit / 'examples.json')
    lookup = {r['id']: r for r in questions}
    if len(lookup) != len(questions) or set(lookup) != {r['id'] for r in source_examples}:
        raise ValueError('Question cohort differs from fit')
    for row in source_examples:
        question = lookup[row['id']]
        if (row['split'] != question['calibration_split'] or row['family'] != question['family']
                or row['token_ids'][:row['prompt_tokens']] != question['token_ids']):
            raise ValueError('Fit/public-prompt provenance mismatch')
    selected_questions = [q for q in questions if q['calibration_split'] == 'eval']
    arms = conditions(args.alpha)
    contract = dict(schema='nanbeige.answer_interventions.v1', question_sha256=digest(args.questions),
        fit_manifest_sha256=digest(args.fit / 'manifest.json'),
        fit_verification_sha256=digest(args.fit / 'independent_verification.json'),
        fit_lineage_sha256=digest(args.fit / 'cross_precision_verification.json'),
        fit_input_contract_sha256=digest(args.fit_input_contract),
        factors_sha256=digest(args.fit / 'vpd_factors.npz'), conditions=arms,
        evaluation_ids=[q['id'] for q in selected_questions], component=args.component,
        max_new_tokens=args.max_new_tokens, seed=42, decoding='greedy', dtype='bfloat16',
        intervention_phase='all_prefill_and_decode_calls',
        prompt_source='original public question only; no saved continuation',
        source_sha256={p.name: digest(p) for p in [Path(__file__),
            Path(__file__).with_name('pass_interventions.py'), Path(__file__).with_name('run_capture.py')]})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'contract.json').exists():
        if read(out / 'contract.json') != contract:
            raise ValueError('Resume contract changed')
    elif any(out.iterdir()):
        raise ValueError('Nonempty output directory lacks matching contract')
    else:
        write_json(out / 'contract.json', contract)
    records = ([json.loads(line) for line in (out / 'generations.jsonl').read_text(encoding='utf-8').splitlines()]
               if (out / 'generations.jsonl').exists() else [])
    expected = [(q['id'], arm['name']) for q in selected_questions for arm in arms]
    if [(r['id'], r['condition']) for r in records] != expected[:len(records)]:
        raise ValueError('Resume records are not the exact ordered prefix')
    for row in records:
        prompt = lookup[row['id']]['token_ids']
        if (row['token_ids'][:len(prompt)] != prompt or row['prompt_tokens'] != len(prompt)
                or len(row['token_ids']) != len(prompt) + row['generated_tokens']):
            raise ValueError('Resume token identity mismatch')
    if len(records) == len(expected):
        if (out / 'manifest.json').exists():
            completed = read(out / 'manifest.json')
            if (completed['status'] != 'answer_interventions_complete'
                    or completed['contract'] != contract
                    or completed['files']['generations.jsonl'] != digest(out / 'generations.jsonl')):
                raise ValueError('Existing completion manifest differs from saved generation')
        else:
            controls = sorted(out.glob('cache_parity_*.json'))
            if not controls:
                raise ValueError('Missing cached-generation controls')
            require_cache_parity(read(controls[-1]))
            finalize(out, contract, records, expected)
        print('Answer intervention generation already complete', flush=True)
        return
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for pretrained answer interventions')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(42)
    torch.set_num_threads(2)
    environment = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    if (out / 'environment.json').exists() and read(out / 'environment.json') != environment:
        raise ValueError('Resume execution environment changed')
    write_json(out / 'environment.json', environment)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda').eval()
    if (model.config.num_hidden_layers, model.config.num_loops, model.config.hidden_size) != (22, 2, 3072):
        raise ValueError('Unexpected model architecture')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
    interventions = {}
    with np.load(args.fit / 'vpd_factors.npz', allow_pickle=False) as factors:
        for path in {arm['path'] for arm in arms}:
            interventions[path] = PassComponentIntervention(model, path,
                torch.from_numpy(factors[path + '__V']), torch.from_numpy(factors[path + '__U']),
                component=args.component)
    start = time.monotonic()
    control_tokens = torch.tensor([selected_questions[0]['token_ids'][:32]], device='cuda')
    diagnostics, control_arrays = [], {'token_ids': control_tokens.cpu().numpy()}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for arm in arms:
            with interventions[arm['path']].apply(arm['alpha'], arm['passes']):
                diagnostic, full, cached = cache_parity(model, control_tokens, 2, return_logits=True)
            diagnostics.append(dict(condition=arm['name'], **diagnostic))
            control_arrays[arm['name'] + '__full'] = full
            control_arrays[arm['name'] + '__cached'] = cached
    # Save new controls under distinct names on each resume; never overwrite evidence.
    control_id = len(list(out.glob('cache_parity_*.json')))
    write_json(out / f'cache_parity_{control_id:03d}.json', diagnostics)
    np.savez_compressed(out / f'cache_parity_{control_id:03d}.npz', **control_arrays)
    require_cache_parity(diagnostics)
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    for index, (qid, condition) in enumerate(expected):
        if index < len(records):
            continue
        if time.monotonic() - start >= args.max_wall_seconds:
            write_json(out / 'run_status.json', dict(status='paused_at_time_bound', records=len(records), expected=len(expected)))
            raise SystemExit(75)
        question = lookup[qid]
        arm = next(a for a in arms if a['name'] == condition)
        tokens = torch.tensor([question['token_ids']], device='cuda')
        intervention = interventions[arm['path']]
        torch.cuda.synchronize()
        before = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), intervention.apply(arm['alpha'], arm['passes']):
            sequence = model.generate(tokens, attention_mask=torch.ones_like(tokens), do_sample=False,
                max_new_tokens=args.max_new_tokens, use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        torch.cuda.synchronize()
        all_ids = sequence[0].tolist()
        new_ids = all_ids[len(question['token_ids']):]
        if intervention.model_calls != len(new_ids):
            raise ValueError('Decode call count differs from generated-token count')
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        prefix = tokenizer.decode(question['token_ids'][-8:], skip_special_tokens=False)
        record = {k: question[k] for k in ('id', 'lane', 'family', 'split', 'calibration_split')}
        record.update(condition=condition, intervention=arm, component=args.component, loops=2,
            prompt_tokens=len(question['token_ids']), generated_tokens=len(new_ids), token_ids=all_ids,
            generated_text=text, scoring_text=('<think>' if '<think>' in prefix and '</think>' not in prefix else '') + text,
            elapsed_seconds=time.monotonic()-before, stopped_on_eos=bool(new_ids and new_ids[-1] in eos_ids),
            max_token_censored=bool(len(new_ids)==args.max_new_tokens and new_ids[-1] not in eos_ids),
            thinking_closed='</think>' in text, model_forward_calls=intervention.model_calls,
            physical_layer_token_executions=44*(len(question['token_ids']) + max(0,len(new_ids)-1)))
        with (out / 'generations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        records.append(record)
        print(f'{len(records)}/{len(expected)} {qid} {condition}: {len(new_ids)} tokens', flush=True)
    finalize(out, contract, records, expected)


if __name__ == '__main__':
    main()
