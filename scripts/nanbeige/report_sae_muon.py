"""Add the verified optimizer pilot to the paper without inventing missing results."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('artifacts/nanbeige_reasoning')
    run = root / 'muon_sae_control_v1'
    receipt_path = root / 'muon_sae_verification/complete.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'muon_control_independently_verified'
    assert receipt['completed_conditions'] == 6 and receipt['feature_archives_checked'] == 108
    assert receipt['contract_sha256'] == sha(run / 'contract.json')
    c = read(run / 'contract.json')
    pairs = {}
    for result in receipt['conditions']:
        index = c['sites'].index(result['site'])
        assert result['condition_manifest_sha256'] == sha(run / f"{index:02d}_{result['arm']}/manifest.json")
        pairs[(result['site'], result['arm'])] = result['quality']
    sites = [('model.layers.11', 'Physical layer 12'), ('model.layers.21', 'Physical layer 22'), ('model.norm', 'Loop normalization')]
    rows, inactive, changes, labels = [], [], [], []
    for site, title in sites:
        for invocation in (0, 1):
            a = pairs[(site, 'adamw')]['eval'][f'pass_{invocation}']['explained_variance']
            m = pairs[(site, 'muon_hybrid')]['eval'][f'pass_{invocation}']['explained_variance']
            changes.append(m-a); labels.append(f'{title}, pass {invocation+1}')
            rows.append(f'| {title}, pass {invocation+1} | {a:.4f} | {m:.4f} | {m-a:+.4f} |')
        a = pairs[(site, 'adamw')]['train']['both_pass_usage']['inactive_features_on_this_split']
        m = pairs[(site, 'muon_hybrid')]['train']['both_pass_usage']['inactive_features_on_this_split']
        inactive.append(f'| {title} | {a:,} | {m:,} | {m-a:+,} |')
    improved = sum(v > 0 for v in changes)
    l22 = [changes[2], changes[3]]
    fig, ax = plt.subplots(figsize=(8, 3.9), constrained_layout=True)
    y = np.arange(6)
    ax.axvline(0, color='#777777', linewidth=1)
    ax.hlines(y, 0, changes, color='#4877a1', linewidth=2)
    ax.scatter(changes, y, color=['#4877a1', '#c87938'] * 3, s=52, zorder=3)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlabel('Held-out explained variance: Muon-hybrid minus AdamW')
    ax.set_title('Paired SAE optimizer pilot: same initialization and batches\nOne seed, 1,000 updates per condition; larger values favor Muon-hybrid', fontsize=11)
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    directory = root / 'figures'; directory.mkdir(exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(directory / f'sae_muon_comparison.{extension}', dpi=180)
    plt.close(fig)
    section = '''### 4.13 Muon-hybrid control inspired by the constitutional-adapter SAE study

Dugan's earlier constitutional-adapter study reports improved SAE quality with Windows CUDA Muon-hybrid and recommends comparing activations in a fixed dictionary. It motivates an optimizer control here, particularly given the poor layer-22 second-pass reconstruction. Its PDF does not specify the original optimizer code or hyperparameters; this run is a documented adaptation using PyTorch Muon, not a replication of its numerical benchmark or causal steering results. The downloaded paper and local copy match SHA256 `036cc6b51fcb5d7307cf2772f4f967aea1adf0d6cd30ac4f2d8d03f9ee40da4b`. [Prior SAE paper](https://www.moralitylab.xyz/papers/Late_Layer_Feature_Reweighting_by_a_Constitutional_Adapter_An_Industrial_Sparse_Autoencoder_Analysis.pdf).

We train six SAEs: AdamW and Muon-hybrid at physical layers 12 and 22 and loop normalization, using the verified longer-run Q4 activations from Section 4.12. Each pair shares the exact initialization and stored batch indices, seed 42, eight training and ten held-out input records, 256 training positions per pass, 6,144 features, top-k 32, 1,000 updates, batch size 128 and raw pooled MSE. Decoder columns are normalized after each update. Held-out data are not used for fitting or learning-rate selection.

Muon updates the encoder and decoder matrices; AdamW updates the encoder bias and pre-bias. The nominal learning rate is 0.001 with zero weight decay. Muon uses momentum 0.95, Nesterov updates, five Newton-Schulz iterations with coefficients (3.4445, -4.775, 2.0315), epsilon 1e-7 and the `match_rms_adamw` learning-rate adjustment. Both arms run on Windows CUDA with PyTorch 2.10.0+cu126, float32 parameters and TF32 disabled; Muon's native orthogonalization calculation uses bfloat16. A fresh AdamW arm on that same runtime supplies the paired baseline; the earlier CPU results are not used as its substitute. The nominal LR match is not a claim of identical update dynamics or optimizer tuning. [PyTorch Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html).

| Site and pass | AdamW held-out EV | Muon-hybrid held-out EV | Muon minus AdamW |
|---|---:|---:|---:|
''' + '\n'.join(rows) + f'''

Muon-hybrid improves held-out EV in {improved}/6 site/pass comparisons in this single-seed pilot. At physical layer 22, the paired changes are {l22[0]:+.4f} on pass 1 and {l22[1]:+.4f} on pass 2. These are separate pass measurements, not six independent replications. The optimizer recipe was fixed before the full run; the two-update CUDA preflight is excluded from this table.

![Paired held-out reconstruction changes under Muon-hybrid; each point is one site and pass, with no uncertainty interval estimated from this single seed.](figures/sae_muon_comparison.png)

For feature usage, the following counts refer to features inactive across the union of both passes on the observed training split, out of 6,144 features:

| Shared SAE site | AdamW inactive | Muon-hybrid inactive | Muon minus AdamW |
|---|---:|---:|---:|
''' + '\n'.join(inactive) + '''

An inactive feature on this finite split is not necessarily permanently dead. More active features do not by themselves establish semantic clarity, transportability or better causal interventions. Each dictionary supports pass-to-pass feature differences within that fit; independently trained AdamW and Muon dictionaries do not supply aligned feature IDs.

All six dictionaries and 108 feature-difference archives passed independent NumPy float64 checks of re-encoding, reconstruction, pass differences, source hashes, split identity, paired initialization/batch receipts and decoder norms. All 24 pooled train/evaluation reconstruction measurements and feature-usage counts were reproduced. Three implementation tests cover baseline-recipe agreement, complete matrix/bias optimizer assignment and rejection of corrupted feature differences. Training itself was not independently rerun.

This is a post-hoc, single-seed control on an already inspected cohort. It does not establish a general optimizer ranking, feature semantics, model reasoning improvement or a capability threshold. Local training times and peak CUDA allocation are retained in the condition manifests; uncontrolled laptop clocks and thermal state prevent a hardware-efficiency claim. No language model was reloaded, no VPD was refitted and no answer intervention was performed. Artifacts are in `muon_sae_control_v1`, with the independent receipt in `muon_sae_verification/complete.json` and the reproducible protocol in `muon_sources/PROTOCOL.md`.

'''
    paper = root / 'PAPER.md'; text = paper.read_text(encoding='utf-8')
    if '### 4.13 ' in text:
        start = text.index('### 4.13 ')
        end = text.index('### 4.14 ', start) if '### 4.14 ' in text[start:] else text.index('## 5. Prospective tests and limits', start)
        text = text[:start] + section + text[end:]
    else:
        text = text.replace('## 5. Prospective tests and limits', section + '## 5. Prospective tests and limits', 1)
    paper.write_text(text, encoding='utf-8')
    report = dict(status='verified_muon_results_added_to_manuscript', independent_receipt_sha256=sha(receipt_path),
        source_sha256=sha(paper), comparisons=receipt['paired_changes'],
        figures={ext: sha(directory / f'sae_muon_comparison.{ext}') for ext in ['png', 'pdf', 'svg']}, visual_review_required=True)
    (root / 'muon_sae_verification/paper_update.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
