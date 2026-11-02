"""Integrate the completed, verified feature experiment into the methods draft."""
import json
from pathlib import Path
import shutil
from experiments.nanbeige_reasoning.check_llama_steering import read,sha,write
from scripts.nanbeige.report_q4_feature_answers import LABELS


def main():
    root=Path('artifacts/nanbeige_reasoning');run=root/'prime-20260908T200850Z-601'
    graded=read(run/'grading_v2/graded.json');previous=read(run/'grading/graded.json')
    manifest=read(run/'grading_v2/manifest.json')
    for name,digest in manifest['files'].items():assert sha(run/'grading_v2'/name)==digest
    assert len(graded)==70 and all(r['grade']['status']=='graded' for r in graded)
    pairs=list(zip(previous,graded))
    assert all((a['id'],a['arm'])==(b['id'],b['arm']) for a,b in pairs)
    differences=[dict(id=b['id'],arm=b['arm'],before=a['grade'],after=b['grade']) for a,b in pairs if a['grade']!=b['grade']]
    assert len(differences)==1 and differences[0]['before']['status']=='grader_error'
    summary=read(run/'paper_results_v2/summary.json')['arms']
    assert all(r['correct']==r['grader_errors']==0 and r['units']==8 for r in summary)
    successes=[r for r in graded if r['grade']['score']==1]
    assert {r['id'] for r in successes}=={'arc_agi_1/c074846d/1'}
    assert {r['arm'] for r in successes}=={'second_plus','second_minus','both_plus'}
    assert sum(r['max_token_censored'] for r in graded)==62
    assert sum(r['generated_tokens'] for r in graded)==561231
    assert all(r['grade']==r['native_render_grade'] for r in graded)
    report=read(run/'paper_results_v2/manifest.json')
    for name,digest in report['files'].items():assert sha(run/'paper_results_v2'/name)==digest
    for ext in ('png','svg','pdf'):
        shutil.copyfile(run/f'paper_results_v2/feature_answer_outcomes.{ext}',root/f'figures/feature_answer_outcomes.{ext}')
    paper=root/'PAPER.md';old=paper.read_text(encoding='utf-8')
    start=old.index('### 4.16 Answer-steering extension and decoding audit')
    end=old.index('### 4.17 Additional historical math and ARC-AGI receipts',start)
    section=['### 4.16 Completed answer steering and decoding audit','',
        'The extension completed 70 fresh Q4 continuations on ten held-out input records, using the selected layer-22 Muon decoder direction (feature 574). '
        'Seven conditions comprise a baseline and coefficients +0.1 and -0.1 on the first pass, second pass or both. All prefill and decode positions are eligible; '
        'the context allocation is 10,153 tokens and the output cap is 8,192. The A100 run independently repeats all eight serial cache controls at this allocation, '
        'obtaining zero full-vocabulary logit discrepancies over 64 comparisons. This short-prefix gate does not establish long-trajectory parity. All generated outputs '
        'were retrieved and checksummed before pod termination.','',
        'The continuations contain 561,231 generated tokens. Eight stop on an end token and 62 reach the output cap. After independent token decoding and grading, '
        'every condition scores zero on eight complete evaluation units: two ARC-AGI-1 tasks, two ARC-AGI-2 tasks, one local logic question, one historical Campsite question '
        'and two math questions. There are no missing records or remaining grader errors.','',
        '| Condition | Complete units correct | Individual inputs correct | Capped inputs | Generated tokens |',
        '|---|---:|---:|---:|---:|']
    for row in summary:
        correct=sum(r['grade']['score']==1 for r in graded if r['arm']==row['arm'])
        section.append(f"| {LABELS[row['arm']]} | 0/8 | {correct}/10 | {row['capped_inputs']}/10 | {row['generated_tokens']:,} |")
    section+=['',
        'The three individual-input successes concern the same second test input of ARC-AGI-1 task `c074846d`: pass-2 +0.1, pass-2 -0.1 and both-pass +0.1 finish correctly '
        'after 6,575, 7,453 and 8,078 generated tokens, respectively. The baseline is incorrect on this input. Each successful condition still fails the task\'s other input, '
        'so none produces a complete-task improvement. These are three condition outcomes on one input, not three independent solved tasks. Success with both signs of the '
        'pass-2 perturbation also does not support a monotonic feature interpretation.','',
        '![Complete-task outcomes under the seven feature-steering conditions. An asterisk denotes at least one token-capped input within that unit. Individual-input successes are reported separately in the table.](figures/feature_answer_outcomes.png)','',
        'Independent decoding of long checkpoints exposed a presentation difference missed by the eight-step gate. This GGUF enables a leading-space rule, and the pinned '
        'llama.cpp detokenizer performs three in-place punctuation/apostrophe cleanup passes. For example, raw token text `6 . 6` becomes `6. 6`, and `column0 != 0` becomes '
        '`column0!= 0`. A vocabulary-only control reproduces native rendering and verifies that independent HF text equals concatenated native token pieces. An exact '
        'reproduction of the pinned cleanup matches 18 checkpoint snapshots and ten synthetic boundary cases; the completed cohort subsequently passes the same rendering '
        'check on all 70 records. The original token IDs, native text, inference code and short-prefix verification remain unchanged. '
        '[Pinned detokenizer source](https://github.com/ggml-org/llama.cpp/blob/0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0/src/llama-vocab.cpp).','',
        'Primary scores use raw HF decoding, with native rendering graded separately: all 70 grading records agree between renderings. The first full grading attempt '
        'encountered a Windows command-line-length failure on one long math output. Passing the unchanged text through standard input into the unchanged published verifier '
        'resolves that transport error; regrading all 70 outputs leaves the other 69 grades unchanged. The affected response scores zero. Short-input parity controls and '
        'long correct, incorrect and unfinished-thinking controls verify the transport change. Original and revised grading receipts remain in `grading` and `grading_v2`.','',
        'This experiment demonstrates an auditable path from a frozen sparse feature to pass-specific intervention and scored output. It finds a narrow individual-input '
        'change but no complete-task gain at this feature, dose and cap. Heavy censoring, one selected feature, two signed doses and a previously inspected small cohort '
        'preclude a general reasoning-benefit claim. It does not establish feature semantics, an adaptive router, model-scale effects or reasoning compression. This '
        'two-pass zero-intervention baseline is distinct from the earlier one-pass ablation. All source receipts, raw outputs, decoding audits and completed tables are '
        'retained in `prime-20260908T200850Z-601`.','']
    new=old[:start]+'\n'.join(section)+'\n'+old[end:]
    anchor='Audited Hermes controls provide context rather than a matched scale comparison.'
    assert new.count(anchor)==1
    new=new.replace(anchor,'A completed 70-continuation sparse-feature steering test yields three correct condition outputs on one ARC test input, but zero complete-task gains; '
        '62 outputs reach the token cap. '+anchor)
    paper.write_text(new,encoding='utf-8')
    write(run/'grading_transport_revision.json',dict(status='stdin_transport_regrade_verified',records=70,
        unchanged_grades=69,differences=differences,original_manifest_sha256=sha(run/'grading/manifest.json'),
        revised_manifest_sha256=sha(run/'grading_v2/manifest.json'),published_verifier_sha256=manifest['published_math_verifier_sha256']))
    write(run/'paper_integration.json',dict(status='completed_answers_integrated_pdf_review_pending',
        source_before_sha256=__import__('hashlib').sha256(old.encode()).hexdigest(),source_after_sha256=sha(paper),
        grading_manifest_sha256=sha(run/'grading_v2/manifest.json'),report_manifest_sha256=sha(run/'paper_results_v2/manifest.json'),
        integrator_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status='completed_answers_integrated',records=70,individual_successes=3,complete_task_successes=0)))


if __name__=='__main__':main()
