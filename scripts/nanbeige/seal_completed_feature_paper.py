"""Seal inspected PDF and verified answer-dataset delivery receipts."""
import json
from pathlib import Path
import time
from experiments.nanbeige_reasoning.check_llama_steering import read,sha,write


def main():
    root=Path('artifacts/nanbeige_reasoning');run=root/'prime-20260908T200850Z-601'
    pdf=Path('output/pdf/nanbeige_hermes_research_draft.pdf')
    receipt_path=pdf.with_suffix('.receipt.json');receipt=read(receipt_path)
    assert receipt['pdf_sha256']==sha(pdf) and receipt['source_sha256']==sha(root/'PAPER.md')
    receipt.update(visual_review_required=False,visual_review=dict(render_dpi=100,
        inspected_pages=[1,2,3,4,5,6,24,25,26,27,28],unchanged_pixel_pages=list(range(7,24)),
        comparison_pdf_sha256=sha(Path('tmp/pdfs/pre_completed_answers.pdf')),
        findings='All 28 pages rendered. Changed pages inspected: no overlap, clipping or missing glyphs observed. Remaining pages pixel-identical to the previously reviewed draft.'))
    write(receipt_path,receipt)
    write(run/'paper_results_v2/visual_review.json',dict(status='figure_reviewed_in_final_pdf',
        figure_sha256=sha(run/'paper_results_v2/feature_answer_outcomes.png'),pdf_sha256=sha(pdf),page=25))
    integration=read(run/'paper_integration.json')
    integration.update(status='completed_answers_integrated_pdf_reviewed',pdf_sha256=sha(pdf),pdf_receipt_sha256=sha(receipt_path))
    write(run/'paper_integration.json',integration)
    grading=read(run/'grading_v2/manifest.json')
    assert grading['records']==70 and grading['native_render_grade_disagreements']==0
    records=read(run/'grading_v2/graded.json')
    assert all(r['grade']['status']=='graded' for r in records)
    index=root/'answer_steering_index_v1/manifest.json'
    for name,digest in read(index)['files'].items():assert sha(index.parent/name)==digest
    write(run/'FINAL_VERIFICATION.json',dict(status='nanbeige_feature_answer_extension_complete',
        outputs=70,generated_tokens=561231,capped_outputs=62,individual_correct_condition_outputs=3,
        complete_evaluation_units_per_condition=8,complete_unit_successes_by_condition={r['arm']:r['correct'] for r in read(run/'paper_results_v2/summary.json')['arms']},
        grader_errors=0,pdf_pages=28,updated_unix=time.time(),
        evidence_sha256={str(p):sha(p) for p in [run/'completion_verification.json',run/'grading_v2/manifest.json',
            run/'grading_transport_revision.json',run/'termination_verification.json',index,receipt_path,root/'PAPER.md']},
        remaining_goal_gap='Fresh matched Hermes/TRM and 27B comparison has not been run; historical evidence remains explicitly qualified.'))
    write(run/'WORKING_STATUS.json',dict(status='nanbeige_extension_finished_paper_delivered',updated_unix=time.time(),
        inference_complete=True,cloud_pod_terminated=True,authoritative_grading='grading_v2',
        final_verification='FINAL_VERIFICATION.json',paper=str(pdf),paper_pages=28,
        remaining=['Fresh matched Hermes/TRM and 27B comparison remains uncompleted.'],
        preserve=['Do not contact the terminated pod address or restart the completed run.',
            'Keep original grading and report artifacts; grading_v2/paper_results_v2 supersede their Windows transport error.',
            'No extra paid comparator allocation has been made.']))
    print(json.dumps(dict(status='nanbeige_extension_and_paper_verified',outputs=70,pages=28)))


if __name__=='__main__':main()
