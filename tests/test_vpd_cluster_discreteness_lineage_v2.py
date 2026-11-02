from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v2 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT
    / "artifacts/vpd_activation_category_cluster_discreteness_benchmark_v2/report.json"
)
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v10.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v2/runbook.json"
RECEIPT = ROOT / "artifacts/vpd_unified_compartment_pipeline_v8/receipt.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v10_binds_cluster_discreteness_evidence() -> None:
    protocol = _load(PROTOCOL)
    benchmark = _load(BENCHMARK)
    contract = protocol["component_compartments"][
        "within_category_cluster_discreteness_v2"
    ]
    assert contract["benchmark"]["sha256"] == sha256_file(BENCHMARK)
    assert contract["benchmark"]["benchmark_id"] == benchmark["benchmark_id"]
    assert contract["benchmark"]["legacy_false_splits_reproduced"] == 2
    assert contract["required_for_next_empirical_category_candidate"] is True
    assert contract["direct_v1_cluster_count_selection_for_next_candidate_prohibited"]
    assert protocol["scientific_boundary"]["execution_authorized"] is False


def test_cheap_executor_runbook_v2_reproduces_and_stops_before_candidate() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["stages"] == 9
    assert verification["cluster_candidate_runtime_ready"] is False
    runbook = _load(RUNBOOK)
    safe = [stage["id"] for stage in runbook["stages"] if stage["safe_for_cheap_executor_now"]]
    assert safe == ["local_preflight"]
    gate = runbook["stages"][1]
    assert gate["id"] == "cluster_candidate_discreteness_v2"
    assert "production V2 candidate wrapper is absent" in gate["stop_if"]


def test_unified_receipt_v8_is_addressed_and_nonempirical() -> None:
    receipt = _load(RECEIPT)
    body = dict(receipt)
    identifier = body.pop("pipeline_receipt_id")
    assert identifier == content_sha256(body)
    assert receipt["source"]["research_protocol_v10"]["sha256"] == sha256_file(
        PROTOCOL
    )
    assert receipt["source"]["cheap_executor_runbook_v2"]["sha256"] == sha256_file(
        RUNBOOK
    )
    assert receipt["resources"]["legacy_false_split_scenarios"] == 2
    assert receipt["scientific_boundary"][
        "production_cluster_candidate_v2_runtime_present"
    ] is False
    assert receipt["scientific_boundary"]["execution_authorized"] is False
