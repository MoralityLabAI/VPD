"""Seal the reviewed Future Work revision and preserve all earlier versions."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
from pypdf import PdfReader
from .build_monitorability_future_v008 import ROOT, VERSION, SECTION, read, sha, save


def main():
    archive = ROOT / 'output/paper_versions'
    out = archive / VERSION
    index = read(archive / 'index.json')
    v = read(out / 'content_verification.json')
    assert sha(archive / 'index.json') == v['previous_index_sha256']
    assert not any(entry['version'] == VERSION for entry in index['versions'])
    for name, digest in v['preserved_previous_files'].items():
        assert sha(ROOT / name) == digest, name
    receipt = read(out / 'paper.receipt.json')
    for name, key in [('PAPER.md', 'source_sha256'), ('paper.pdf', 'pdf_sha256'), ('render_paper.py', 'builder_sha256')]:
        assert sha(out / name) == receipt[key], name
    old = (archive / v['source_version'] / 'PAPER.md').read_text(encoding='utf-8')
    new = (out / 'PAPER.md').read_text(encoding='utf-8')
    assert new.count(SECTION) == 1
    assert new.replace(SECTION, '').replace('Version 8 | 12 September 2026', 'Version 7 | 12 September 2026') == old
    assert sha(out / 'build_monitorability_future_v008.py') == sha(ROOT / 'experiments/nanbeige_reasoning/build_monitorability_future_v008.py')
    pages = PdfReader(out / 'paper.pdf').pages
    assert len(pages) == 19 and all(len(p.extract_text().split()) >= 190 for p in pages)
    assert '9. Future work:' in pages[11].extract_text()
    assert 'computational budget' in pages[12].extract_text()
    visual = dict(status='all_final_pages_visually_reviewed', pages=19,
        pdf_sha256=sha(out / 'paper.pdf'), reviewed_utc=datetime.now(timezone.utc).isoformat(),
        method='Pages 1-18 inspected in contact sheets; new section on pages 12-13 and final page 19 additionally inspected at full rendered size.',
        findings='No clipping, overlap, missing visible glyphs or near-empty final page. Section and appendix transitions are clean. Inherited tables, equations and figures remain legible.',
        render_note='Poppler Symbol/ArialUnicode fallback warnings were checked against visible rendering.',
        images={str(i):sha(ROOT / f'tmp/pdfs/v008_review/page-{i:02d}.png') for i in range(1,20)})
    save(out / 'visual_review.json', visual)
    shutil.copy2(__file__, out / Path(__file__).name)
    v.update(status='revision_sealed_and_verified', pages=19,
        manuscript_sha256=sha(out / 'PAPER.md'), pdf_sha256=sha(out / 'paper.pdf'),
        visual_review_sha256=sha(out / 'visual_review.json'),
        content_check='Removing Section 9 and reverting version number exactly reproduces v007 manuscript source.',
        newly_run_model_experiments=0, full_benchmark_completed=False)
    save(out / 'content_verification.json', v)
    (out / 'README.md').write_text('''# Version 8: monitorability Future Work

Section 9, on pages 12-13 of the 19-page manuscript, frames internal monitoring as a prospective response to a measured loss of evidence in generated reasoning. It describes the matched behavioral conditions, distinct fact and decision labels, held-out linear probes, pass-specific VPD interventions, SAE reconstruction checks and a detection comparison at matched false-positive rate.

The section explicitly records the compute-budget boundary and zero completed ledger episodes. It makes no claim of hidden chain-of-thought detection or recovered monitorability. All v007 manuscript text remains identical apart from the version number; the completed experiment results and supplements are preserved. No new model experiment was run for this editorial revision.

The build refuses existing output directories. Content verification checks exact preservation of the inherited source, 16 rendered headings and 237 table cells. All 19 rendered pages were visually reviewed; earlier indexed versions remain unchanged.
''', encoding='utf-8')
    index['versions'].append(dict(version=VERSION, pages=19,
        note='Adds Section 9: future internal-monitorability study and success criteria, explicitly outside current compute budget; all existing evidence preserved.',
        files={str(p.relative_to(out)).replace('\\','/'):{'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}))
    index['current_manuscript'] = VERSION
    save(archive / 'index.json', index)
    overview = archive / 'README.md'
    text = overview.read_text(encoding='utf-8').replace('Current manuscript.', 'Previous manuscript.')
    text += '\n- [v008: 12 September, monitorability Future Work](v008_2026-09-12_monitorability_future_work/paper.pdf), 19 pages. Current manuscript. Adds the prospective behavioral, probing and causal study, its success criterion and compute-budget limitation. Existing results remain unchanged.\n'
    overview.write_text(text, encoding='utf-8')
    for entry in read(archive / 'index.json')['versions']:
        for name, value in entry['files'].items():
            assert sha(archive / entry['version'] / name) == value['sha256']
    print(json.dumps(dict(status='paper_revision_sealed', version=VERSION, pages=19,
        preserved_prior_files=len(v['preserved_previous_files']), new_experiments=0,
        pdf_sha256=sha(out / 'paper.pdf'))))


if __name__ == '__main__':
    main()
