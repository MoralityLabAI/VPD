from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_plan_v2 import (
    build_report,
    verify_benchmark,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v2 import (
    _validate_split,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_canonical_sae_training_capture_plan_benchmark_v2/report.json"
)
SPLIT = ROOT / "artifacts/vpd_activation_category_capacity_split_v4/split_manifest.json"
DATASET = (
    ROOT
    / "artifacts/vpd_activation_category_capacity_split_v4/sae_training_episodes.jsonl"
)


def test_sae_training_capture_plan_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report()
    assert observed["summary"] == {"controls": 12, "passed": 12}
    assert verify_benchmark(REPORT)["status"] == "pass"


def test_recovered_sae_training_split_validates_under_v2() -> None:
    _, _, summary = _validate_split(
        split_manifest_path=SPLIT,
        split_role="sae_training",
        dataset_path=DATASET,
    )
    assert summary["split_role"] == "sae_training"
    assert summary["rows"] == 6292
    assert summary["pairs"] == 572
    assert summary["dataset_sha256"] == (
        "3673e72770c5bd1229a1455e4b5aa0e489e2f779f579fe47a42301458b7c71d4"
    )
    assert summary["episode_id_set_sha256"] == (
        "13494eb75e5e2aae3b398321112a1d7e4f79238b548ecb2ef0c884e9696cb24f"
    )


def test_v2_plan_does_not_claim_capture_or_training_authority() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    boundary = observed["scientific_boundary"]
    assert boundary["effect_capture_performed"] is False
    assert boundary["sae_training_authorized"] is False
    assert boundary["execution_authorized"] is False
