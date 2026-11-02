from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.activation_category_cluster_partition_identity_v4 import (
    canonical_partition,
)
from experiments.authz_mechinterp.benchmark_activation_category_cluster_partition_identity_v4 import (
    verify_benchmark as verify_partition_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_partition_identity_admission_v4 import (
    verify_benchmark as verify_admission_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
PARTITION_BENCHMARK = (
    ROOT
    / "artifacts/vpd_activation_category_cluster_partition_identity_benchmark_v4/report.json"
)
ADMISSION_BENCHMARK = (
    ROOT
    / "artifacts/vpd_cluster_candidate_partition_identity_admission_benchmark_v4/report.json"
)


def test_canonical_partition_ignores_numeric_label_names() -> None:
    component_ids = ["a", "b", "c", "d"]
    assert canonical_partition(
        component_ids, np.asarray([0, 0, 1, 1])
    ) == canonical_partition(component_ids, np.asarray([9, 9, 4, 4]))


def test_partition_identity_benchmark_rejects_same_count_membership_change() -> None:
    report = json.loads(PARTITION_BENCHMARK.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["negative_controls"]["stable_exact_partition_admitted"] is True
    assert report["negative_controls"]["same_count_membership_change_rejected"] is True
    assert report["negative_controls"]["cluster_label_rename_invariant"] is True
    assert report["negative_controls"]["single_replicate_membership_drift_rejected"] is True
    verification = verify_partition_benchmark(PARTITION_BENCHMARK)
    assert verification["negative_controls"] == 6


def test_production_shaped_partition_admission_preserves_stable_candidate() -> None:
    report = json.loads(ADMISSION_BENCHMARK.read_text(encoding="utf-8"))
    stable = report["scenarios"][0]
    assert stable["observed_v4_admitted"] is True
    assert stable["v4_partition_rejection_count"] == 0
    assert stable["v4_exact_categories"] == 8


def test_production_shaped_partition_admission_rejects_semantic_confound() -> None:
    report = json.loads(ADMISSION_BENCHMARK.read_text(encoding="utf-8"))
    confound = report["scenarios"][1]
    assert confound["observed_v4_admitted"] is False
    verification = verify_admission_benchmark(
        ADMISSION_BENCHMARK, project_root=ROOT
    )
    assert verification["status"] == "pass"
    assert verification["scenarios"] == 2
    assert verification["execution_authorized"] is False
