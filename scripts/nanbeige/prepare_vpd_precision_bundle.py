"""Freeze a precision/dose replay of the ten already measured VPD inputs."""
import json
from pathlib import Path
import tarfile

from experiments.nanbeige_reasoning.run_vpd_precision_control import condition_plan
from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest, read


def main():
    root=Path(__file__).resolve().parents[2]
    fit=root/'artifacts/nanbeige_reasoning/prime-20260906T204058Z-561/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration'
    manifest=read(fit/'manifest.json')
    for name,status in [('independent_verification.json','multipass_saved_effects_independently_verified'),
                        ('cross_precision_verification.json','cross_precision_reasoning_vpd_verified')]:
        check=read(fit/name)
        if check['status']!=status or check['manifest_sha256']!=digest(fit/'manifest.json'):
            raise ValueError('Verified completed fit required')
    for name in ('vpd_factors.npz','examples.json'):
        if digest(fit/name)!=manifest['artifacts'][name]:raise ValueError('Changed fitted input')
    source=read(root/'artifacts/nanbeige_reasoning/cross_precision_vpd_inputs_v2/contract.json')
    out=root/'artifacts/nanbeige_reasoning/vpd_precision_inputs_v1'
    out.mkdir(exist_ok=False)
    mapping={'vpd_factors.npz':'vpd_factors.npz','fit_manifest.json':'manifest.json',
             'fit_verification.json':'independent_verification.json','fit_lineage.json':'cross_precision_verification.json'}
    for dest,name in mapping.items():(out/dest).write_bytes((fit/name).read_bytes())
    examples=[r for r in read(fit/'examples.json') if r['split']=='eval']
    (out/'examples.json').write_text(json.dumps(examples,indent=2)+'\n',encoding='utf-8')
    contract=dict(schema='nanbeige.vpd_precision_inputs.v1',model_id=source['model_id'],revision=source['revision'],
        checkpoint_files=source['checkpoint_files'],source_fit_manifest_sha256=digest(fit/'manifest.json'),
        input_files={p.name:digest(p) for p in out.iterdir()},conditions=condition_plan(),
        examples=len(examples),precisions=['bf16','fp32_upcast'],expected_records=840,
        full_prior_context_retained=True,new_generation=False,component=0,tail_rows=8,
        max_remote_seconds=2700,max_launcher_seconds=7200,
        limitations=['Same ten held-out budget-censored trajectories; this is an exploratory VPD robustness control.',
                    'FP32 upcasts the same stored BF16 checkpoint values; it is not an independently trained FP32 model.',
                    'SDPA may select different kernels across precisions; execution and numeric effects are not fully isolated.',
                    'No final-answer accuracy, routing or model-scale conclusion.'])
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in (root/'experiments/nanbeige_looped').iterdir()
        if p.is_file() and (p.suffix in ('.py','.json','.txt') or p.name=='VPD_BACKEND_LICENSE'))
    files += [root/'experiments/nanbeige_reasoning'/name for name in ('__init__.py','run_vpd_precision_control.py')]
    files += [root/'scripts/nanbeige/run_vpd_precision_remote.sh',*sorted(out.iterdir())]
    dest=root/'artifacts/nanbeige_reasoning/vpd_precision_bundle.tar.gz'
    with tarfile.open(dest,'w:gz') as archive:
        for path in files:archive.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
    receipt=dict(sha256=digest(dest),files={p.relative_to(root).as_posix():digest(p) for p in files},
        input_contract_path=(out/'contract.json').relative_to(root).as_posix(),experiment_contract=contract)
    dest.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(files),bytes=dest.stat().st_size,sha256=receipt['sha256'],examples=len(examples),expected_records=840)))


if __name__=='__main__':main()
