"""Report verified frozen-dictionary VPD transfer without adding model claims."""
from pathlib import Path
import hashlib
import json


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path('artifacts/nanbeige_reasoning')
    run = root/'muon_vpd_transfer_v1'
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='frozen_sae_vpd_transfer_verified'
    assert manifest['feature_archives']==120 and manifest['independently_reencoded_conditions']==1680
    assert manifest['pooled_quality_checks']==24 and manifest['feature_arithmetic_checks']==2520
    for name, expected in manifest['files'].items():
        p=(run/name).resolve()
        assert p.is_relative_to(run.resolve()) and sha(p)==expected
    quality = json.loads((run/'quality.json').read_text())
    pairs={(r['site'],r['arm']):r['baseline_transfer_quality']['eval'] for r in quality}
    rows=[]
    for site,title in [('model.layers.11','Physical layer 12'),('model.layers.21','Physical layer 22'),('model.norm','Loop normalization')]:
        for i in (0,1):
            a=pairs[(site,'adamw')][f'pass_{i}']['explained_variance']
            m=pairs[(site,'muon_hybrid')][f'pass_{i}']['explained_variance']
            rows.append(f'| {title}, pass {i+1} | {a:.4f} | {m:.4f} |')
    section='''### 4.14 Frozen Muon dictionaries on saved VPD interventions

To connect the optimizer control to the parameter interventions, we freeze all six dictionaries from Section 4.13 and apply them to the verified BF16 baseline and intervention activations from Section 4.9. Those activations retain the final eight positions of the initial-budget trajectories, while the dictionaries were fitted on Q4 activations across prompt and generation regions of the longer run. Precision, execution runtime and position distribution therefore differ. We verify that all 18 BF16 source token sequences are exact prefixes of the corresponding longer Q4 sequences and that the original eight training and ten evaluation input assignments match. No dictionary is refitted, no new model invocation is run and no held-out input is used for fitting.

Before interpreting feature differences, we measure frozen-dictionary reconstruction on these BF16 baselines. Held-out explained variance pools 80 positions per pass across the ten evaluation input records:

| Site and pass | Frozen AdamW SAE EV | Frozen Muon-hybrid SAE EV |
|---|---:|---:|
'''+ '\n'.join(rows)+'''

At layer 22's second pass, the optimizer advantage persists in this particular transfer: EV is 0.3120 with Muon-hybrid versus -0.5350 with AdamW. It is lower than the Muon dictionary's 0.4533 on its original longer-run Q4 evaluation positions, and most variance remains unreconstructed. The weaker layer-12 and normalization reconstructions similarly constrain interpretation. This is a transfer-quality measurement; it does not isolate precision from position/context effects or validate feature semantics.

We then re-encode all 20 previously executed VPD intervention records: component 0 at each of the layer-12 and layer-22 MLP down projections, on ten held-out inputs. Each record contains its baseline and perturbations at coefficients +0.05 and -0.05 applied in the first pass, second pass or both. For every site and optimizer dictionary, archives retain both pass encodings under all seven conditions, their pass-to-pass differences, and each perturbed encoding minus its baseline encoding. These latter differences are obtained by re-encoding each full hidden state, not by encoding the hidden-state difference. Each comparison uses the same frozen dictionary; feature IDs are not aligned across the AdamW and Muon dictionaries.

The resulting 120 archives passed 1,680 independent NumPy float64 re-encoding checks, 24 pooled baseline reconstruction checks and 2,520 checks of saved feature arithmetic and causal ordering. In particular, a second-pass-only intervention leaves every captured first-pass feature vector unchanged. Saved token IDs/positions, source intervention hashes and dictionary provenance were checked. A corruption control verifies rejection of altered intervention-feature differences. These checks connect shared parameter components to observed changes in a fixed feature basis, while leaving the original model intervention results unchanged.

The analysis is post-hoc and reuses the existing signed VPD responses. It adds no dose sweep, independent training seed, feature-semantic validation or answer-level causal result. The original component coverage, finite-dose nonlinearity and BF16 numerical-resolution limits still apply. Artifacts and verified source links are in `muon_vpd_transfer_v1/{contract,quality,records,manifest}.json`, with the full feature arrays alongside them.

'''
    p=root/'PAPER.md';text=p.read_text(encoding='utf-8')
    if '### 4.14 ' in text:
        start=text.index('### 4.14 ')
        end=text.index('### 4.15 ',start) if '### 4.15 ' in text[start:] else text.index('## 5. Prospective tests and limits',start)
        text=text[:start]+section+text[end:]
    else:
        text=text.replace('## 5. Prospective tests and limits',section+'## 5. Prospective tests and limits',1)
    p.write_text(text,encoding='utf-8')
    receipt=dict(status='verified_transfer_added_to_paper',transfer_manifest_sha256=sha(run/'manifest.json'),paper_sha256=sha(p),table_rows=len(rows))
    (root/'muon_sae_verification/transfer_paper_update.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
