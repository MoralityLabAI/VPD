"""Plot verified generated-prefix calibration measurements, with their scope."""
import argparse
import json
import os
from pathlib import Path

from scripts.nanbeige.prepare_reasoning_vpd_bundle import digest, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit', type=Path)
    args = parser.parse_args()
    root = args.fit
    manifest_hash = digest(root / 'manifest.json')
    for name in ('independent_verification.json', 'cross_precision_verification.json'):
        if read(root / name)['manifest_sha256'] != manifest_hash:
            raise ValueError('Verification belongs to another fit')
    summary = read(root / 'effect_summary.json')
    coverage = read(root / 'factor_coverage.json')
    effects = read(root / 'effects.json')
    if len(effects) != 20 or {r['alpha'] for r in effects} != {.05}:
        raise ValueError('Figure labels require the specified 20-record, alpha .05 pilot')
    os.environ.setdefault('MPLCONFIGDIR', str(Path('artifacts/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8))
    sites = ['model.layers.11', 'model.layers.21', 'model.norm']
    labels = ['Layer 12', 'Layer 22', 'Loop norm']
    x = np.arange(3)
    for invocation, color in [(0, '#326a9e'), (1, '#d57a32')]:
        values = [summary['sae_quality'][s]['held_out_quality'][f'pass_{invocation}']['explained_variance']
                  for s in sites]
        bars = axes[0].bar(x + (invocation-.5)*.34, values, .34, color=color, label=f'Pass {invocation+1}')
        axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    axes[0].axhline(0, color='#777777', lw=.7)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel('Held-out explained variance')
    axes[0].set_title('A   Shared SAE reconstruction', loc='left', fontsize=11)
    axes[0].legend(frameon=False, loc='upper right')
    axes[0].margins(y=.25)
    entries = sorted(summary['final_norm_pass_2_pooled'], key=lambda r: r['physical_module'])
    if len(entries) != 2 or any(r['component'] != 0 or r['examples'] != 10 for r in entries):
        raise ValueError('Unexpected pooled intervention cohort')
    for i, (field, label, color) in enumerate([
            ('first_norm', 'First only', '#326a9e'), ('second_norm', 'Second only', '#d57a32'),
            ('both_norm', 'Both', '#428b75'), ('interaction_norm', 'Interaction', '#8863a6')]):
        values = [100*r[field]/r['baseline_norm'] for r in entries]
        axes[1].bar(np.arange(2)+(i-1.5)*.19, values, .19, color=color, label=label)
    axes[1].set_xticks([0, 1], labels[:2])
    axes[1].set_ylabel('Effect norm / baseline norm (%)')
    axes[1].set_title('B   Final-state change, component 0', loc='left', fontsize=11)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].margins(y=.3)
    values = [100*coverage[s+'.mlp.down_proj']['fraction_squared_weight_energy_reconstructed'] for s in sites[:2]]
    bars = axes[2].bar([0, 1], values, .5, color='#428b75')
    axes[2].bar_label(bars, fmt='%.2f%%', padding=4)
    axes[2].set_xticks([0, 1], labels[:2])
    axes[2].set_ylabel('Reconstructed squared weight energy (%)')
    axes[2].set_title('C   Coverage of 64 components', loc='left', fontsize=11)
    axes[2].margins(y=.2)
    fig.suptitle('Nanbeige: BF16 calibration on saved Q4 reasoning trajectories', x=.06, ha='left', fontsize=14)
    fig.text(.06, .055, '10 held-out inputs; final 8 token positions. Panel B: +0.05 dose, pooled norms; interaction includes BF16 rounding.\n'
             'Teacher-forced trajectories, all budget-censored. These are reconstruction and hidden-state effects, not answer accuracy.', fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.83, bottom=.23, wspace=.42)
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    paths = []
    for extension in ('png', 'pdf', 'svg'):
        path = out / f'reasoning_vpd.{extension}'
        fig.savefig(path, dpi=180, facecolor='white')
        paths.append(path)
    plt.close(fig)
    receipt = dict(source_manifest_sha256=manifest_hash,
        sources={name: digest(root/name) for name in ('effect_summary.json', 'factor_coverage.json', 'effects.json')},
        source_code_sha256=digest(Path(__file__)), figures={p.name: digest(p) for p in paths})
    (out/'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(str(paths[0]))


if __name__ == '__main__':
    main()
