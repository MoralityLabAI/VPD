from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
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
PROTOCOL_V19 = ROOT / "configs/evals/vpd_activation_category_compartments_v19.json"
PROTOCOL_V20 = ROOT / "configs/evals/vpd_activation_category_compartments_v20.json"
RUNBOOK_V11 = ROOT / "artifacts/vpd_cheap_executor_runbook_v11/runbook.json"
RUNBOOK_V12 = ROOT / "artifacts/vpd_cheap_executor_runbook_v12/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v20_binds_partition_identity_evidence_and_pinned_v19() -> None:
    protocol = _load(PROTOCOL_V20)
    contract = protocol["component_compartments"][
        "cluster_candidate_partition_identity_admission_v4"
    ]
    assert protocol["predecessor_sha256"] == sha256_file(PROTOCOL_V19)
    assert contract["partition_identity_benchmark"]["sha256"] == sha256_file(
        PARTITION_BENCHMARK
    )
    assert contract["production_admission_benchmark"]["sha256"] == sha256_file(
        ADMISSION_BENCHMARK
    )
    assert contract["cluster_count_agreement_alone_is_insufficient"] is True
    assert contract["exact_partition_agreement_required_across_all_replicates"] is True
    assert contract["exact_candidate_consensus_membership_required"] is True


def test_protocol_v20_prohibits_count_only_empirical_admission() -> None:
    protocol = _load(PROTOCOL_V20)
    assert protocol["unified_lineage"]["count_only_candidate_admission_prohibited"] is True
    assert protocol["unified_lineage"][
        "exact_candidate_membership_required_for_sae_compartments"
    ] is True
    assert protocol["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v12_is_addressed_and_only_preflight_is_safe() -> None:
    runbook = _load(RUNBOOK_V12)
    body = dict(runbook)
    assert body.pop("runbook_id") == content_sha256(body)
    assert runbook["predecessor"]["runbook_id"] == _load(RUNBOOK_V11)["runbook_id"]
    assert runbook["source"]["research_protocol_v20"]["sha256"] == sha256_file(
        PROTOCOL_V20
    )
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]


def test_runbook_v12_requires_empirical_partition_identity_receipt() -> None:
    runbook = _load(RUNBOOK_V12)
    completion = runbook["completion_definition"]
    assert completion["cluster_partition_identity_runtime_ready"] is True
    assert completion["cluster_partition_identity_admission_runtime_ready"] is True
    assert completion["count_only_candidate_admission_forbidden"] is True
    assert completion["empirical_partition_identity_admission_ready"] is False
    reasons = runbook["stages"][1]["stop_if"]
    assert any("component co-assignment" in reason for reason in reasons)
    assert any("empirical partition-identity admission V4" in reason for reason in reasons)
    assert runbook["scientific_boundary"]["execution_authorized"] is False
