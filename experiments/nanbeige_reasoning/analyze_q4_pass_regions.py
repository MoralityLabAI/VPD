"""Descriptive pass differences by physical layer and saved trajectory region.

This analyzes teacher-forced Q4 activations, without fitting or interpreting
semantic features. Inputs, rather than token positions, receive equal weight.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np
from .fetch_sources import sha


REGIONS = ('prompt', 'early', 'middle', 'late')


def measures(first, second):
    first, second = first.astype(np.float64), second.astype(np.float64)
    n0, n1 = np.linalg.norm(first), np.linalg.norm(second)
    return dict(rows=len(first), first_norm=float(n0), second_norm=float(n1),
        delta_norm=float(np.linalg.norm(second-first)),
        cosine=float(np.sum(first*second)/(n0*n1)) if n0*n1 else None,
        second_first_norm_ratio=float(n1/n0) if n0 else None,
        relative_delta=float(np.linalg.norm(second-first)/n0) if n0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.captures, args.output_dir
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['status'] != 'q4_reasoning_capture_complete':
        raise ValueError('Capture is incomplete')
    if sha(source/'records.jsonl') != manifest['files']['records.jsonl']:
        raise ValueError('Capture metadata hash differs')
    records = list(map(json.loads, (source/'records.jsonl').read_text().splitlines()))
    transfer = source/'frozen_vpd_transfer'
    transferred = json.loads((transfer/'summary.json').read_text())
    transfer_manifest = json.loads((transfer/'manifest.json').read_text())
    if sha(transfer/'summary.json') != transfer_manifest['files']['summary.json']:
        raise ValueError('VPD metadata hash differs')
    lookup = {(r['id'], r['site']): r for r in transferred}
    hidden, vpd = [], []
    sites = [f'model.layers.{i}' for i in range(22)] + ['model.norm']
    for record in records:
        path = source/record['archive']
        if sha(path) != record['sha256']:
            raise ValueError('Activation hash differs')
        with np.load(path, allow_pickle=False) as arrays:
            positions = arrays['token_positions']
            if not np.array_equal(positions, record['positions']):
                raise ValueError('Token positions differ')
            for site in sites:
                first, second = [arrays[f'{site}__pass_{i}'] for i in (0, 1)]
                if not np.isfinite(first).all() or not np.isfinite(second).all():
                    raise ValueError('Nonfinite activation')
                for region, (start, end) in record['regions'].items():
                    mask = (positions >= start) & (positions < end)
                    if mask.sum() != 8:
                        raise ValueError('Expected eight retained positions per region')
                    hidden.append(dict(id=record['id'], lane=record['lane'], split=record['split'],
                        site=site, region=region, **measures(first[mask], second[mask])))
        for layer in (11, 21):
            entry = lookup[(record['id'], f'model.layers.{layer}.mlp.down_proj')]
            path = transfer/entry['archive']
            if sha(path) != entry['sha256']:
                raise ValueError('VPD archive hash differs')
            with np.load(path, allow_pickle=False) as arrays:
                if not np.array_equal(arrays['token_positions'], positions):
                    raise ValueError('VPD token positions differ')
                for region, (start, end) in record['regions'].items():
                    mask = (positions >= start) & (positions < end)
                    delta = arrays['delta_coordinates'][mask].astype(np.float64)
                    vpd.append(dict(id=record['id'], lane=record['lane'], split=record['split'],
                        layer=layer+1, region=region,
                        coordinate_delta_rms_per_component=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                        basis_output=measures(arrays['basis_output_0'][mask], arrays['basis_output_1'][mask])))
    out.mkdir(parents=True, exist_ok=False)
    (out/'hidden_regions.jsonl').write_text('\n'.join(json.dumps(r) for r in hidden)+'\n')
    (out/'vpd_regions.jsonl').write_text('\n'.join(json.dumps(r) for r in vpd)+'\n')
    grouped = defaultdict(list)
    for row in hidden:
        grouped[(row['split'], row['site'], row['region'])].append(row)
    aggregate = [dict(split=split, site=site, region=region, input_records=len(rows),
        median_cosine=float(np.median([r['cosine'] for r in rows])),
        median_norm_ratio=float(np.median([r['second_first_norm_ratio'] for r in rows])),
        median_relative_delta=float(np.median([r['relative_delta'] for r in rows])))
        for (split, site, region), rows in grouped.items()]
    (out/'summary.json').write_text(json.dumps(aggregate, indent=2))
    os.environ.setdefault('MPLCONFIGDIR', str(Path('artifacts/nanbeige_reasoning/mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    colors = ('#62717b', '#187e99', '#ca7a24', '#a04274')
    for region, color in zip(REGIONS, colors):
        selected = [next(r for r in aggregate if r['split']=='eval' and r['region']==region
            and r['site']==f'model.layers.{i}') for i in range(22)]
        axes[0].plot(range(1, 23), [r['median_cosine'] for r in selected], label=region, color=color)
        axes[1].plot(range(1, 23), [r['median_norm_ratio'] for r in selected], label=region, color=color)
    axes[0].set(ylabel='Median pass cosine', ylim=(-.05, 1.05))
    axes[1].set(ylabel='Median norm ratio (pass 2 / pass 1)', yscale='log')
    axes[1].axhline(1, color='#777777', linestyle=':', linewidth=1)
    for ax in axes:
        ax.set(xlabel='Physical transformer layer', xticks=(1, 4, 8, 12, 16, 20, 22))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, ncols=2, fontsize=9)
    fig.suptitle('Q4 native trajectories: pass differences across layers\n'
        '10 evaluation inputs; eight retained positions per region; all continuations capped', fontsize=12)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'pass_regions.{suffix}', dpi=180)
    plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(dict(status='descriptive_region_analysis_complete',
        hidden_region_records=len(hidden), vpd_region_records=len(vpd),
        source_manifest_sha256=sha(source/'manifest.json'),
        transfer_manifest_sha256=sha(transfer/'manifest.json'), implementation_sha256=sha(Path(__file__)),
        aggregation='Median across input records, no statistical independence or causal effect claimed.',
        limitations=['Teacher-forced Q4 replay; four sampled greedy mismatches.',
            'All continuations censored; no answer-success contrast.',
            'Frozen BF16 VPD basis transfers only about 0.26% original weight energy; residual omitted.',
            'Norms and cosines do not establish semantic compression or routing.'],
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}), indent=2))
    print(f'Saved {len(hidden)} hidden-region and {len(vpd)} VPD-region records', flush=True)


if __name__ == '__main__':
    main()
