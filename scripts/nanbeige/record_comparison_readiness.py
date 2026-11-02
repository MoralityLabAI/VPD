"""Record the remaining comparator identity gap from current local evidence."""
import json
from pathlib import Path
import subprocess
import time
from experiments.nanbeige_reasoning.check_llama_steering import read,sha,write


def main():
    root=Path('artifacts/nanbeige_reasoning')
    archive=Path('D:/research_engine/trmObservabilityLogs/trm_observability_harness_snapshot')
    run=archive/'data/qwen27b_intellect3_logic_hybrid_200'
    summary=read(run/'summary.json')
    assert summary['rows']==109
    source=Path(summary['source_path'])
    alternatives=[source,Path('D:/research_engine/Tesseract/Tesseract/data/normalized_trajectories/intellect_3_logic.jsonl')]
    checks=[dict(path=str(p),exists=p.exists()) for p in alternatives]
    assert not any(r['exists'] for r in checks)
    assert not any('sha' in k.lower() or 'hash' in k.lower() for k in summary)
    revision='07a177f2751c2adb1cc16dc819c1173c0cd57ad5'
    command=['git','-c','safe.directory=C:/projects/Hermes-Skills','-C','C:/projects/Hermes-Skills',
        'grep','-l','-i','-E','arc_agi|arc-agi',revision,'--','*.py','*.yaml','*/SKILL.md']
    search=subprocess.run(command,capture_output=True,text=True)
    assert search.returncode in (0,1)
    output=root/'comparison_readiness_v1';output.mkdir(exist_ok=False)
    write(output/'verification.json',dict(status='comparator_identity_gap_documented',updated_unix=time.time(),
        source_path_checks=checks,summary_has_input_checksum=False,
        historical_summary_sha256=sha(run/'summary.json'),historical_predictions_sha256=sha(run/'predictions.jsonl'),
        archived_run_files=sorted(p.name for p in run.iterdir() if p.is_file()),
        pinned_repository_revision=revision,arc_source_search=dict(command=command,exit_code=search.returncode,
            matching_files=search.stdout.splitlines(),scope='Only tracked Python, YAML and SKILL.md paths; no claim that generic runners cannot accept ARC.'),
        qwen_input_identity_receipt_sha256=sha(root/'comparator_sources/qwen35_27b/input_identity_verification.json'),
        completed_nanbeige_receipt_sha256=sha(root/'prime-20260908T200850Z-601/FINAL_VERIFICATION.json'),
        pending_user_input='Select the exact clearing run/configuration, or authorize using the pinned repository as a new comparison configuration.',
        new_paid_allocation=False,auditor_sha256=sha(Path(__file__))))
    (output/'STATUS.md').write_text('# Remaining comparison\n\n'
        'The Nanbeige activation/VPD/SAE dataset, Muon control, 70-answer extension and 28-page paper are complete and verified. '
        'The full thread goal remains open because a fresh matched Hermes/TRM and 27B comparison has not been executed.\n\n'
        'The historical 109-row logic run records a source path but no input checksum in its summary. The referenced source and the checked relocated path are absent. '
        'Its run directory contains only predictions and summary files. One Nanbeige pilot case is a candidate target match, but this does not establish the historical model prompt.\n\n'
        'The pinned repository search found no ARC-AGI references in tracked Python, YAML or SKILL.md paths. This bounded search is not proof that a generic runner lacks ARC support. '
        'The separately inspected ARC traces failed before receiving valid puzzles. A named clearing run/configuration is needed to reproduce that specific claim.\n\n'
        'The 27B baseline checkpoint and native model-facing inputs are pinned; weights and new inference are not present. '
        'No new cloud allocation has been made. Selecting the comparator is a research-identity question, not permission to extend the completed cloud run.\n',encoding='utf-8')
    print(json.dumps(dict(status='comparator_identity_gap_documented',path=str(output))))


if __name__=='__main__':main()
