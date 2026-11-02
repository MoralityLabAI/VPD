"""Measure frozen BF16 VPD coordinates on Q4 reasoning-layer inputs.

This is a basis-transfer diagnostic, not a Q4-fitted decomposition or causal
intervention. The original eight-component weight coverage was only about0.26%.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .fetch_sources import sha

ORIGINAL=Path('artifacts/nanbeige_looped/prime-20260906T034844Z-719/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration')


def analyze(captures):
    captures=Path(captures)
    manifest=json.loads((captures/'manifest.json').read_text())
    if manifest['status']!='q4_reasoning_capture_complete':raise ValueError('Incomplete Q4 capture')
    if sha(captures/'records.jsonl')!=manifest['files']['records.jsonl']:raise ValueError('Metadata hash mismatch')
    original=json.loads((ORIGINAL/'manifest.json').read_text())
    factors=ORIGINAL/'vpd_factors.npz'
    if sha(factors)!=original['artifacts']['vpd_factors.npz']:raise ValueError('Frozen VPD hash mismatch')
    records=list(map(json.loads,(captures/'records.jsonl').read_text(encoding='utf-8').splitlines()))
    out=captures/'frozen_vpd_transfer'
    out.mkdir(exist_ok=False)
    summaries=[]
    with np.load(factors,allow_pickle=False) as basis:
        for layer in (11,21):
            site=f'model.layers.{layer}.mlp.down_proj'
            v,u=basis[site+'__V'],basis[site+'__U']
            for index,record in enumerate(records):
                path=captures/record['archive']
                if sha(path)!=record['sha256']:raise ValueError('Activation hash mismatch')
                with np.load(path,allow_pickle=False) as a:
                    x=[a[f'{site}.input__pass_{i}'] for i in (0,1)]
                    c0,c1=[value@v for value in x]
                    y0,y1=c0@u,c1@u
                    delta=c1-c0
                    contribution_norms=np.linalg.norm(delta,axis=0)*np.linalg.norm(u,axis=1)
                    destination=out/f'layer_{layer+1}_{index:03d}.npz'
                    np.savez_compressed(destination,coordinates_0=c0,coordinates_1=c1,delta_coordinates=delta,
                        basis_output_0=y0,basis_output_1=y1,delta_basis_output=y1-y0,
                        token_positions=a['token_positions'])
                    summaries.append(dict(id=record['id'],site=site,lane=record['lane'],split=record['split'],
                        coordinate_difference_norm=float(np.linalg.norm(delta)),
                        per_component_output_difference_norm=contribution_norms.tolist(),
                        basis_output_difference_norm=float(np.linalg.norm(y1-y0)),
                        archive=destination.name,sha256=sha(destination)))
    (out/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(status='frozen_vpd_coordinate_transfer_complete',
        records=len(summaries),source_basis_weights='BF16',input_activation_weights='Q4_K_M',
        components_per_matrix=8,source_factors_sha256=sha(factors),new_vpd_fitted=False,
        causal_interventions=False,original_weight_energy_coverage_approx=.0026,
        interpretation='Actual x @ V coordinates and basis-output differences; sparse original basis, residual omitted from these diagnostic outputs. No task-effect or full-decomposition claim.',
        files={f.name:sha(f) for f in out.iterdir() if f.is_file()}),indent=2),encoding='utf-8')
    print('Saved',len(summaries),'frozen VPD coordinate-transfer records',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('captures',type=Path)
    analyze(p.parse_args().captures)
