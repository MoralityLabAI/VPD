"""Seal the visually reviewed revision and update the archive's current pointer."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil
from pypdf import PdfReader
from .build_closeout_v006 import ROOT,VERSION,read,sha,save


def main():
    archive=ROOT/'output/paper_versions';out=archive/VERSION
    index=read(archive/'index.json');v=read(out/'closeout_verification.json')
    assert sha(archive/'index.json')==v['previous_index_sha256']
    assert not any(entry['version']==VERSION for entry in index['versions'])
    for name,digest in v['preserved_previous_files'].items():assert sha(ROOT/name)==digest
    receipt=read(out/'paper.receipt.json')
    for name,key in [('PAPER.md','source_sha256'),('paper.pdf','pdf_sha256'),('render_paper.py','builder_sha256')]:
        assert sha(out/name)==receipt[key]
    pages=PdfReader(out/'paper.pdf').pages
    assert len(pages)==15 and all(len(p.extract_text().split())>=200 for p in pages)
    paper=(out/'PAPER.md').read_text(encoding='utf-8')
    old=(archive/'v005_2026-09-11_cross_family_dynamics/PAPER.md').read_text(encoding='utf-8')
    numeric=[line for line in old.splitlines() if line.startswith('|') and re.search(r'\| -?\d',line) and '`' not in line]
    for line in numeric+v['matched_table_rows']:assert line in paper,line
    for phrase in ('No ledger episode has completed','214 scheduled calls','not established to be random',
                   'none passes the stricter literal check','No probe accuracy is interpreted as semantic localization'):
        assert phrase in paper,phrase
    validation=ROOT/'artifacts/nanbeige_reasoning/partial_closeout_validation_v1'
    verified=read(validation/'verification.json')
    assert verified['rows']==96 and verified['official_and_literal_checks']==192
    shutil.copytree(validation,out/'supplement/independent_grading_validation')
    code=out/'supplement/closeout_code';code.mkdir()
    for name in ('grade_logic_skill_comparison.py','compare_skill_models.py','test_partial_skill_grading.py',
                 'verify_partial_closeout.py','verify_hermes_controller_control.py'):
        shutil.copy2(ROOT/'experiments/nanbeige_reasoning'/name,code/name)
    assert sha(code/'grade_logic_skill_comparison.py')==read(out/'supplement/partial_grading/verification.json')['grader_implementation_sha256']
    assert sha(code/'compare_skill_models.py')==read(out/'supplement/partial_comparison/verification.json')['code_sha256']
    shutil.copy2(__file__,out/Path(__file__).name)
    visual=dict(status='all_final_pages_visually_reviewed',pages=15,pdf_sha256=sha(out/'paper.pdf'),
        method='All 15 pages inspected in contact sheets; new comparison and gate tables additionally inspected at full rendered size.',
        findings='No clipping, overlap or missing visible glyphs; equations, charts, tables, page numbers and section transitions checked.',
        render_note='Poppler emitted Symbol/ArialUnicode fallback warnings; rendered mathematical figures and text inspected.',
        images={str(i):sha(ROOT/f'tmp/pdfs/v006_review/page-{i:02d}.png') for i in range(1,16)},
        timestamp=datetime.now(timezone.utc).isoformat())
    save(out/'visual_review.json',visual)
    save(out/'test_receipt.json',dict(status='passed',command='.venv/Scripts/python.exe -m unittest experiments.nanbeige_reasoning.test_partial_skill_grading experiments.nanbeige_reasoning.test_compare_skill_models experiments.nanbeige_reasoning.test_skill_resume experiments.nanbeige_reasoning.test_skill_finisher',
        tests=16,elapsed_seconds=10.320,observed_output='................\n----------------------------------------------------------------------\nRan 16 tests in 10.320s\n\nOK\n',
        historical_ledger_tests=dict(tests=80,log='artifacts/nanbeige_receipt_replay/tests_cooling_stability_v1.txt',sha256=sha(ROOT/'artifacts/nanbeige_receipt_replay/tests_cooling_stability_v1.txt'))))
    v.update(status='revision_closed_and_verified',pages=15,pdf_sha256=sha(out/'paper.pdf'),
             manuscript_sha256=sha(out/'PAPER.md'),inherited_numeric_rows_checked=len(numeric),
             independent_grade_checks=192,tests_passed=16,visual_review_sha256=sha(out/'visual_review.json'),
             completed_scope='Versioned paper revision and closeout of observed experiments',
             unfinished_experiments=['214 comparator calls without completed output','Ledger behavioral pilot and downstream probes/interventions','Matched ARC and math model/skill comparison'])
    save(out/'closeout_verification.json',v)
    (out/'README.md').write_text('''# Version 6: experimental closeout

The 15-page manuscript retains the mechanistic argument and completed VPD/SAE results. The new material is in Section 7 and Appendices A, C and D. Earlier numbered revisions remain unchanged.

## Evidence added

- All 45 completed Nanbeige comparator calls graded, including 11 capped outputs: 20 vanilla puzzle outcomes and 19 outcomes for each other reported procedure. The 214 missing calls remain unrun.
- Exact-puzzle comparison to archived Qwen results. Nanbeige routed repair accepts 6/19; Qwen accepts 3/19 on the same puzzles. This is a partial, exposed, answer-only cohort, not a general ranking.
- Published-verifier acceptance and prompt-literal validity remain distinct. All six accepted Nanbeige routed repairs fail the literal check; this known dataset contract discrepancy is explained in the paper.
- Ledger gate: one correct final response, two length-censored responses, nine assigned questions not run, zero ledger episodes. No ledger probe or monitoring result is claimed.
- All 96 observed procedure rows independently checked against grid/count/adjacency and bit-mask matching predicates (192 checks); 16 closeout/resumption/comparison tests pass. The earlier ledger suite records 80 passing tests.

## Reproducibility

The supplement contains immutable grading rows, exact paired IDs, protocol and terminal gate responses, recovery receipt, independent validation, and snapshots of the modified analysis scripts. Earlier supplements are retained as historical evidence, including their original progress records; the current status is in closeout_verification.json.

Run from the VPD workspace using its existing environment and source artifacts. Use fresh output directories when reproducing; the tools refuse to overwrite completed result directories.

```powershell
.venv/Scripts/python.exe -m experiments.nanbeige_reasoning.grade_logic_skill_comparison --run artifacts/nanbeige_reasoning/nanbeige_skill_comparison_run_v2 --preparation artifacts/nanbeige_reasoning/nanbeige_skill_comparison_inputs_v1 --output artifacts/nanbeige_reasoning/partial_grading_reproduction --allow-partial
```

The matching stage uses compare_skill_models with --allow-partial-nanbeige and the archived Qwen grading_v2; exact source paths and hashes are retained in the grading/comparison verification records. Analysis scripts require the repository's pinned verifier and helper dependencies. This revision directory is a manuscript and results bundle, not a replacement for the activation archive or model checkpoint.

No additional model inference was performed during closeout. GPU processes from these experiments are stopped. The original 85 C pause / 87 C hard stop policy is unchanged. The full benchmark and ledger semantic experiment are unfinished; their observed endpoints are now documented rather than treated as pending paper results.
''',encoding='utf-8')
    index['versions'].append(dict(version=VERSION,pages=15,
        note='Experimental closeout: graded 45-call Nanbeige prefix with matched Qwen denominators; terminal ledger gate; preserved mechanistic results and explicit unrun stages.',
        files={str(p.relative_to(out)).replace('\\','/'):{'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}))
    index['current_manuscript']=VERSION
    save(archive/'index.json',index)
    overview=archive/'README.md';text=overview.read_text(encoding='utf-8')
    text=text.replace('Current manuscript.','Previous manuscript.')
    text+='\n- [v006: 12 September, experimental closeout](v006_2026-09-12_experiment_closeout/paper.pdf), 15 pages. Current manuscript. Preserves the mechanistic essay, adds the verified partial comparison and terminal ledger gate, and records unfinished experiments explicitly.\n'
    overview.write_text(text,encoding='utf-8')
    for entry in read(archive/'index.json')['versions']:
        for name,value in entry['files'].items():assert sha(archive/entry['version']/name)==value['sha256']
    print(json.dumps(dict(status='paper_revision_sealed',version=VERSION,pages=15,
        preserved_prior_files=len(v['preserved_previous_files']),pdf_sha256=sha(out/'paper.pdf'),
        full_benchmark_completed=False,ledger_episodes=0)))


if __name__=='__main__':main()
