"""Update the methods draft only after independently verifying all cache controls."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.nanbeige.verify_cached_steering import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path("artifacts/nanbeige_reasoning")
    run = root / "q4_cached_steering_burst_v1"
    receipt = verify(run)
    assert receipt["gate_passed"]
    contract = read(run / "contract.json")
    assert contract["chunk_size"] == 1
    assert read(run / "cleanup.json") == dict(model_closed=True, context_closed=True)
    for group in ("code_sha256", "continuation_code_sha256"):
        for name, digest in contract[group].items():
            assert sha(Path("experiments/nanbeige_reasoning") / name) == digest
    previous = root / "q4_cached_steering_guarded_v1"
    assert sha(previous / "contract.json") == contract["source_partial_contract_sha256"]
    assert sha(previous / "results.json") == contract["source_partial_results_sha256"]
    for row in read(previous / "results.json"):
        assert sha(previous / row["archive"]) == sha(run / row["archive"]) == row["sha256"]
    for name, digest in read(run / "checkpoints.json").items():
        assert sha(run / name) == digest
    restart = read(run / "guarded_restart_check.json")
    assert sha(run / "guarded_restart_check.npz") == restart["archive_sha256"]
    with np.load(run / "guarded_restart_check.npz") as replay:
        assert sorted(replay.files) == sorted(contract["carried_conditions"])
        for name in replay.files:
            with np.load(run / f"{name}.npz") as saved:
                np.testing.assert_array_equal(replay[name], saved["cached"])
    for row in read(run / "results.json"):
        with np.load(run / row["archive"]) as saved:
            assert len(saved["token_ids"]) == 383
            np.testing.assert_array_equal(saved["cached"], saved["full_prefix"])

    paper = root / "PAPER.md"
    old = paper.read_text(encoding="utf-8")
    paragraphs = old.split("\n\n")
    matches = [i for i, p in enumerate(paragraphs) if p.startswith("Using single-token processing throughout")]
    assert len(matches) == 1
    paragraphs[matches[0]] = (
        "Using single-token processing throughout prefill, cached decoding and fresh-prefix replay gives "
        "bitwise-identical logits in all eight conditions and agreement on all 64 greedy decisions. "
        "Maximum absolute discrepancy is zero; identity writes reproduce the baseline exactly. The "
        "original absolute and relative tolerances of 1e-4 are unchanged. Three controls were carried "
        "from earlier serial runs and five completed through guarded, checksummed checkpoint continuation. "
        "Re-executing the three carried cached trajectories under the added guard also reproduces their "
        "logits bitwise. Independent reloads verify all eight archives, mutation counts, checkpoint hashes "
        "and source lineage. This passes the serial gate for this calibration prompt and eight decoding "
        "steps; it does not explain the batched discrepancy, validate long-context generation or show "
        "answer-level benefit. The failed batched run remains in `q4_cached_steering_v1`; completed serial "
        "evidence and verification are in `q4_cached_steering_burst_v1`."
    )
    guide = root / "dataset_index_v1/README.md"
    old_guide = guide.read_text(encoding="utf-8")
    guide_paragraphs = old_guide.split("\n\n")
    matches = [i for i, p in enumerate(guide_paragraphs) if p.startswith("The separate cache controls are diagnostic artifacts")]
    assert len(matches) == 1
    guide_paragraphs[matches[0]] = (
        "The separate cache controls are diagnostic artifacts, not benchmark examples in this index. "
        "The batched numerical gate failed despite greedy-token agreement. All eight serial conditions "
        "now pass with bitwise-identical cached and fresh-prefix logits on one 383-token calibration "
        "prompt and eight decoding steps. Three controls were carried from earlier runs and five "
        "completed with guarded checkpoint continuation; their saved arrays and provenance were "
        "independently verified. See `../q4_cached_steering_burst_v1/verification.json` and "
        "`../cached_steering_completion_verification.json`. The answer-level intervention experiment "
        "and long-context cache validation remain unexecuted."
    )
    before = dict(paper=sha(paper), guide=sha(guide))
    paper.write_text("\n\n".join(paragraphs), encoding="utf-8")
    guide.write_text("\n\n".join(guide_paragraphs), encoding="utf-8")
    report = dict(
        status="complete_serial_cache_gate_independently_verified_and_reported",
        conditions=8, full_vocabulary_comparisons=64, maximum_absolute_logit_difference=0.0,
        carried_conditions=3, fresh_conditions=5,
        run_manifest_sha256=sha(run / "manifest.json"),
        numerical_verification_sha256=sha(run / "verification.json"),
        reporter_sha256=sha(Path(__file__)), before=before,
        paper_sha256=sha(paper), guide_sha256=sha(guide),
        scope="Calibration-only implementation control; no answer-level benefit or long-context validation.",
    )
    (root / "cached_steering_completion_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
