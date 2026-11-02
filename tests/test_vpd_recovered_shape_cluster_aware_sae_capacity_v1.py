from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_recovered_shape_cluster_aware_sae_capacity_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_recovered_shape_cluster_aware_sae_capacity_benchmark_v1/report.json"
)


def test_recovered_shape_capacity_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    assert observed["status"] == "pass"
    assert observed["summary"] == {"controls": 15, "passed": 15}
    assert observed["benchmark_id"] == (
        "e8021583211fde3fc1185dfb2423a7c56dd33089db66d3143ca5823692239390"
    )

    recovered = observed["recovered_incidence"]["role_summaries"]["sae_training"]
    assert recovered["rows"] == 6292
    assert recovered["pairs"] == 572

    scenarios = observed["results"]["scenarios"]
    assert scenarios["balanced"]["summary"]["specific_compartments"] == 16
    assert scenarios["balanced"]["summary"]["shared_compartments"] == 1
    assert scenarios["balanced"]["summary"]["routed_training_episodes"] == 6292
    assert scenarios["sparse_minority"]["summary"]["specific_compartments"] == 8
    assert scenarios["sparse_minority"]["under_supported_routes"] == 8
    assert scenarios["category_sparse"]["summary"]["specific_compartments"] == 15
    assert scenarios["category_sparse"]["under_supported_routes"] == 1
    assert observed["results"]["v1_balanced_specific_compartments"] == 0


def test_recovered_shape_capacity_verifier_preserves_boundaries() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["controls"] == 15
    assert verification["sae_training_rows"] == 6292
    assert verification["sae_training_pairs"] == 572
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False

