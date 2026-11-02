"""Seal the visually reviewed state-carry revision without changing prior versions."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import shutil
from pypdf import PdfReader
from .build_state_control_v007 import ROOT, VERSION, read, sha, save


def main():
    archive = ROOT / 'output/paper_versions'
    out = archive / VERSION
    index = read(archive / 'index.json')
    verification = read(out / 'content_verification.json')
    assert sha(archive / 'index.json') == verification['previous_index_sha256']
    assert not any(v['version'] == VERSION for v in index['versions'])
    for name, digest in verification['preserved_previous_files'].items():
        assert sha(ROOT / name) == digest, name
    receipt = read(out / 'paper.receipt.json')
    for name, key in [('PAPER.md', 'source_sha256'), ('paper.pdf', 'pdf_sha256'), ('render_paper.py', 'builder_sha256')]:
        assert sha(out / name) == receipt[key], name
    assert sha(out / 'build_state_control_v007.py') == sha(ROOT / 'experiments/nanbeige_reasoning/build_state_control_v007.py')
    pages = PdfReader(out / 'paper.pdf').pages
    assert len(pages) == 18
    assert all(len(p.extract_text().split()) >= 200 for p in pages)
    paper = (out / 'PAPER.md').read_text(encoding='utf-8')
    old = (archive / 'v006_2026-09-12_experiment_closeout/PAPER.md').read_text(encoding='utf-8')
    inherited = [line for line in old.splitlines() if line.startswith('|') and re.search(r'\d', line)]
    for row in inherited + verification['state_table'] + verification['crossed_table']:
        assert row in paper, row
    for phrase in ('450 saved conditions', '150 reset/one-pass pairs', '1,680 primary',
                   'They do not hold effective dependent depth fixed', 'No ledger episode has completed',
                   'not an externally preregistered study'):
        assert phrase in paper, phrase
    data = ROOT / 'artifacts/nanbeige_reasoning'
    close = read(data / 'recurrence_state_closeout_v1/verification.json')
    assert close['native_conditions'] == 450 and close['native_model_closed']
    assert close['tests_passed'] == 5 and not close['trained_architecture_causality_established']
    for directory, key in [('recurrence_state_validation_v1', 'state_validation_sha256'),
                           ('recurrence_state_analysis_validation_v1', 'metric_validation_sha256')]:
        assert sha(data / directory / 'verification.json') == close[key]
    assert sha(data / 'recurrence_state_control_v1/manifest.json') == verification['native_manifest_sha256']
    assert sha(data / 'recurrence_state_analysis_v1/manifest.json') == verification['analysis_manifest_sha256']
    metric = read(data / 'recurrence_state_analysis_validation_v1/verification.json')
    assert metric['primary_per_input_checks'] == 1680
    shutil.copy2(__file__, out / Path(__file__).name)
    visual = dict(status='all_final_pages_visually_reviewed', pages=18,
        pdf_sha256=sha(out / 'paper.pdf'), timestamp=datetime.now(timezone.utc).isoformat(),
        method='All pages reviewed in contact sheets; Figure 2 and the revised final appendix additionally inspected at full rendered size. Unchanged pages retained their layout after the final appendix was shortened.',
        findings='No clipping, overlap or missing visible glyphs. Tables, equations, figures and page transitions checked. Near-empty nineteenth page removed by shortening prose.',
        render_note='Poppler Symbol/ArialUnicode fallback warnings; visible equation and text rendering reviewed.',
        images={str(i): sha(ROOT / f'tmp/pdfs/v007_review/page-{i:02d}.png') for i in range(1, 19)})
    save(out / 'visual_review.json', visual)
    verification.update(status='revision_sealed_and_verified', pages=18,
        manuscript_sha256=sha(out / 'PAPER.md'), pdf_sha256=sha(out / 'paper.pdf'),
        inherited_numeric_rows_checked=len(inherited), tests_passed=5,
        primary_metric_checks=1680, visual_review_sha256=sha(out / 'visual_review.json'),
        native_model_closed=True, trained_architecture_causality_established=False,
        unfinished_experiments=['Full matched benchmark', 'Ledger episodes, semantic probes and behavioral mediation'])
    save(out / 'content_verification.json', verification)
    (out / 'README.md').write_text('''# Version 7: state carry and intervention fidelity

The 18-page manuscript adds Section 7 and Appendix E, with revised abstract and framing. Earlier numbered manuscripts are unchanged.

The new CPU experiment completed 450 conditions on ten complete held-out prompts from eight task groups: one pass, two passes carrying state, and two executed passes with every token reset at the boundary. All 150 reset/one-pass pairs agree exactly in saved states and logits. Five focused analysis tests passed; 1,680 primary metric values were independently checked.

Reconstruction failures persist without state carry. Freezing the SAE's baseline support removes a large selection-induced error spike. State carry nevertheless changes propagated responses: all six tested layer-12 writes have positive Muon layer-22 change fidelity under reset and negative fidelity under carry. This is a descriptive result for this cohort, dictionary and intervention direction, not a universal recurrence penalty or a separately trained architecture comparison.

The supplement contains analysis and both feature-encoding streams, independent checks, experiment protocol, records, native-archive manifest, figures and source snapshots. Native activation archives remain in artifacts/nanbeige_reasoning/recurrence_state_control_v1. Source hashes link the manuscript to those files. The existing inference and pass-specific intervention setup is reused; the new residual writes are distinguished from the earlier fixed-coefficient VPD weight edits.

No ledger episode completed, and no concealed chain-of-thought or semantic mediation result is claimed. The full benchmark remains unfinished. No GPU inference or paid cloud allocation was used for this control; its native model is closed.
''', encoding='utf-8')
    index['versions'].append(dict(version=VERSION, pages=18,
        note='State-carry and SAE-support controls: 450 native conditions, exact reset parity, matched residual writes, offline direction crossing; no training-architecture or hidden-CoT claim.',
        files={str(p.relative_to(out)).replace('\\', '/'):{'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}))
    index['current_manuscript'] = VERSION
    save(archive / 'index.json', index)
    overview = archive / 'README.md'
    text = overview.read_text(encoding='utf-8').replace('Current manuscript.', 'Previous manuscript.')
    text += '\n- [v007: 12 September, state-carry controls](v007_2026-09-12_state_carry_controls/paper.pdf), 18 pages. Current manuscript. Adds completed recurrence-state and sparse-support controls with exact parity checks, preserving the earlier results and explicit semantic limitations.\n'
    overview.write_text(text, encoding='utf-8')
    for entry in read(archive / 'index.json')['versions']:
        for name, value in entry['files'].items():
            assert sha(archive / entry['version'] / name) == value['sha256']
    print(json.dumps(dict(status='paper_revision_sealed', version=VERSION, pages=18,
        preserved_prior_files=len(verification['preserved_previous_files']),
        pdf_sha256=sha(out / 'paper.pdf'), full_benchmark_completed=False, ledger_episodes=0)))


if __name__ == '__main__':
    main()
