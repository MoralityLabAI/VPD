"""Verify the reviewed paper and append it to the archive without changing old versions."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from pypdf import PdfReader

from build_paper_v005 import ROOT, VERSION, read, sha


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    archive = ROOT / 'output/paper_versions'
    out = archive / VERSION
    content = read(out / 'content_verification.json')
    index = read(archive / 'index.json')
    assert sha(archive / 'index.json') == content['previous_index_sha256']
    assert not any(v['version'] == VERSION for v in index['versions'])
    for name, digest in content['preserved_previous_files'].items():
        assert sha(ROOT / name) == digest, name
    receipt = read(out / 'paper.receipt.json')
    assert sha(out / 'PAPER.md') == receipt['source_sha256']
    assert sha(out / 'paper.pdf') == receipt['pdf_sha256']
    assert sha(out / 'render_paper.py') == receipt['builder_sha256']
    paper = (out / 'PAPER.md').read_text(encoding='utf-8')
    pdf = PdfReader(out / 'paper.pdf')
    assert len(pdf.pages) == 12
    assert all(len((page.extract_text() or '').split()) >= 200 for page in pdf.pages)
    old = (archive / 'v004_2026-09-09_intervention_fidelity/PAPER.md').read_text(encoding='utf-8')
    inherited_numeric_rows = [line for line in old.splitlines() if line.startswith('|')
                              and re.search(r'\| -?\d', line) and '`' not in line]
    for line in inherited_numeric_rows:
        assert line in paper, line
    families = {'arc_agi_1':'ARC-AGI-1', 'arc_agi_2':'ARC-AGI-2',
                'intellect3_logic':'INTELLECT-3 logic', 'intellect3_math':'INTELLECT-3 math'}
    for row in content['numeric_rows']:
        a, b, n = row['layer12'], row['layer22'], row['normalization']
        expected = f"| {families[row['family']]} | {b['inputs']} / 2 | {a['mean_cosine']:.3f} | {b['mean_cosine']:.3f} | {n['normalized_cosine']:.3f} |"
        assert expected in paper
    graded = read(out / 'supplement/comparator/summary.json')
    names = {'vanilla':'Vanilla generation', 'logic_skill':'Plain logic skill',
             'routed_candidate':'Routed candidate before repair', 'routed_final':'Routed final with repair',
             'plain_candidate_routed_repair':'Plain candidate with routed repair'}
    for arm, title in names.items():
        r = graded['totals'][arm]
        expected = f"| {title} | {int(r['official_correct'])} / {r['puzzles']} | {100*r['official_correct']/r['puzzles']:.1f}% |"
        assert expected in paper, expected
    protocol = read(out / 'supplement/comparator/protocol.json')
    assert protocol['thinking_enabled'] is False and protocol['max_new_tokens'] == 1024
    assert protocol['support_train_rows'] == 128 and protocol['repair']['exact_node_limit'] == 200000
    assert graded['totals']['vanilla']['capped_candidates'] == 109
    coverage = read(out / 'supplement/pass_dynamics/coverage.json')
    assert dict(Counter(r['lane'] for r in coverage)) == dict(arc_agi_1=5, arc_agi_2=5, intellect3_logic=4, intellect3_math=4)
    assert sum(r['max_token_censored'] for r in coverage) == 13
    for phrase in ('not a confidence interval', 'not a claim of uncontaminated benchmark testing',
                   'No ledger episode has completed', 'That benchmark remains incomplete',
                   'Matching task IDs therefore do not align token rows'):
        assert phrase in paper, phrase
    review = ROOT / 'tmp/pdfs/v005_review'
    images = {}
    for i in range(1,13):
        path = review / f'sealed-{i:02d}.png'
        assert path.exists()
        if i <= 9:
            assert sha(path) == sha(review / f'final-{i:02d}.png')
        images[str(i)] = sha(path)
    visual = dict(status='all_final_pages_visually_reviewed', pages=12, pdf_sha256=sha(out / 'paper.pdf'),
                  page_png_sha256=images, timestamp=datetime.now(timezone.utc).isoformat(),
                  method='Pages 1-9 inspected at 100 dpi and byte-matched to the final render; final pages 10-12 inspected after appendix layout edits.',
                  findings='Figure labels, equations, table columns, page numbers and links are legible. No clipped text, overlapping elements, missing glyphs or nearly empty ending page.',
                  render_note='Poppler emitted fallback-font warnings for Symbol/ArialUnicode; equations and all visible text were inspected successfully.')
    save(out / 'visual_review.json', visual)
    content.update(status='manuscript_numbers_and_render_verified', manuscript_sha256=sha(out / 'PAPER.md'),
                   pdf_sha256=sha(out / 'paper.pdf'), pages=12, tables=receipt['tables'],
                   inherited_numeric_rows_checked=len(inherited_numeric_rows), new_geometry_rows_checked=4,
                   comparator_rows_checked=5, visual_review_sha256=sha(out / 'visual_review.json'))
    save(out / 'content_verification.json', content)
    audit = read(out / 'supplement/pass_dynamics/goal_evidence_audit.json')
    audit.update(previous_goal_turn='progress: strengthened and tested ledger analysis input validation; 62 tests passed',
                 this_goal_turn='progress: incorporated verified four-family dynamics into a new 12-page mechanistic paper with normalization and comparator controls',
                 checked_at=datetime.now(timezone.utc).isoformat(), new_inference=False, goal_complete=False)
    for requirement in audit['requirements']:
        if requirement['requirement'] == 'Paper emphasizing mechanistic interpretation':
            requirement.update(status='new_v005_manuscript_rendered_and_visually_verified', path=str(out.relative_to(ROOT)),
                               new_analysis_incorporated_into_pdf=True, pdf_sha256=sha(out / 'paper.pdf'), pages=12)
    save(out / 'goal_evidence_audit.json', audit)
    shutil.copy2(__file__, out / 'seal_paper_v005.py')
    index['versions'].append(dict(version=VERSION, pages=12,
        note='Cross-family invocation geometry and loop-normalization control integrated with intervention-change fidelity; benchmark scope and incompatible execution variants explicit.',
        files={str(p.relative_to(out)).replace('\\','/'):{'sha256':sha(p)}
               for p in sorted(out.rglob('*')) if p.is_file()}))
    index['current_manuscript'] = VERSION
    save(archive / 'index.json', index)
    archive_readme = archive / 'README.md'
    overview = archive_readme.read_text(encoding='utf-8').replace('9 pages. Current manuscript:', '9 pages. Previous manuscript:')
    marker = '\nThe recovered v001'
    assert marker in overview
    overview = overview.replace(marker,
        '\n- [v005: 11 September, cross-family invocation dynamics](v005_2026-09-11_cross_family_dynamics/paper.pdf), 12 pages. Current manuscript. Adds ARC-AGI-1/2 and INTELLECT-3 logic/math profiles, normalization controls, preserved evidence and a qualified skills-comparison appendix. The matched cross-family benchmark remains incomplete.\n' + marker, 1)
    archive_readme.write_text(overview, encoding='utf-8')
    for entry in read(archive / 'index.json')['versions']:
        for name, value in entry['files'].items():
            assert sha(archive / entry['version'] / name) == value['sha256']
    print(json.dumps(dict(status='paper_revision_sealed', version=VERSION, pages=12,
                          goal_complete=False, preserved_previous_files=len(content['preserved_previous_files']),
                          pdf_sha256=sha(out / 'paper.pdf'))))


if __name__ == '__main__':
    main()
