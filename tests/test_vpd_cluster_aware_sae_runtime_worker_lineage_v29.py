from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v37 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v29 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v37.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v29/runbook.json"


def test_protocol_v37_reproduces_runtime_worker_core() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "48462a79bd32534c2472b64dfd8badd13c3d5246cd7bd7ff6742bc41fcde04e7"
    )
    assert verification["runtime_worker_benchmark_id"] == (
        "dbab569a095cd9c01449886d84f7a292b75edcf919ae65ba5cfc29af0a059188"
    )
    assert verification["runtime_worker_controls"] == 22
    assert verification["execution_jobs"] == 34
    assert verification["worker_core_implemented"] is True
    assert verification["production_cli_implemented"] is False
    assert verification["real_authority_verifiers_implemented"] is False
    assert verification["empirical_blueprint_present"] is False
    assert verification["sae_training_authorized"] is False


def test_protocol_v37_preserves_fail_closed_worker_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_runtime_worker_v1"
    ]
    assert component["job_and_planned_resources_reserved_before_runtime_attach"] is True
    assert component["failed_callbacks_remain_charged"] is True
    assert component["wrong_job_authority_rejected_before_reservation"] is True
    assert component["synthetic_blueprint_rejected_for_production"] is True
    assert component["worker_core_implemented"] is True
    assert component["production_cli_implemented"] is False
    assert component["real_scientific_release_verifier_implemented"] is False
    assert component["real_compute_authorization_verifier_implemented"] is False
    assert component["execution_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v29_reproduces_and_stops_before_production() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "d4d7e3d9099b5f454569e8d752fc16081c2f1d5d8b959c5a985f88dc3d3cb3f1"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["execution_jobs_in_fixture"] == 34
    assert verification["cluster_aware_runtime_worker_core_implemented"] is True
    assert verification["cluster_aware_production_cli_implemented"] is False
    assert verification["real_authority_verifiers_implemented"] is False
    assert verification["empirical_cluster_aware_blueprint_present"] is False
    assert verification["scientific_release_receipt_present"] is False
    assert verification["explicit_compute_authorization_present"] is False
    assert verification["identity_bindings_complete"] is False
    assert verification["sae_training_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_runtime_worker_benchmark_v1",
        "verify_research_protocol_v37",
    } <= command_ids
