from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v25 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v17 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v25.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v17/runbook.json"


def test_protocol_v25_reproduces_empirical_preflight_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "e1175601443bf4162df3400ad3e0740eb65c74b5e6eeeb30c92657998ddf7626"
    )
    assert verification["five_role_audit_id"] == (
        "c14b80e53ea6fe39e171fbd201ab0a87b86dc39737c091fd83616bb813d2cefc"
    )
    assert verification["empirical_preflight_id"] == (
        "edf41948607ac8471030545ea9782c1f6ea76dd2aec72f09aa7cffac5fb45f91"
    )
    assert verification["preflight_benchmark_id"] == (
        "03f8f0e82283b82b28997403a4f68bcd84d6d9630e1a23cc825ccbb805b28e13"
    )
    assert verification["missing_required_bindings"] == 13
    assert verification["empirical_capture_ready"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v25_preserves_four_role_staging_and_final_eval_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = observed["component_compartments"][
        "cluster_to_sae_empirical_preflight_v1"
    ]
    assert contract["cluster_selection_roles"] == [
        "discovery",
        "calibration",
        "cluster_validation",
    ]
    assert contract["sae_routing_extension_role"] == "sae_training"
    assert contract["final_evaluation_excluded"] is True
    assert contract["execution_authorized"] is False
    assert contract["sae_training_authorized"] is False


def test_runbook_v17_reproduces_and_stops_before_empirical_work() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "c8d9bf9a1ecdd327af6066a7d3f2084ddd7bdccb3782d2e321e904af8784dff4"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["missing_required_bindings"] == 13
    assert verification["empirical_capture_ready"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_five_role_recovered_incidence_audit_v1",
        "verify_cluster_to_sae_empirical_preflight_v1",
        "verify_cluster_to_sae_empirical_preflight_benchmark_v1",
        "verify_research_protocol_v25",
    } <= command_ids
    assert observed["scientific_boundary"]["model_framework_imported"] is False
    assert observed["scientific_boundary"]["model_or_adapter_weights_opened"] is False
    assert observed["scientific_boundary"]["sae_training_authorized"] is False

