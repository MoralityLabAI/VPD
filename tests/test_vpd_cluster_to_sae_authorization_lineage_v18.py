from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v26 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v18 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v26.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v18/runbook.json"


def test_protocol_v26_reproduces_cycle_free_authorization_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "3372cf627793588a9e5c2078f44f7955e028c8f43fb071d37d07735ced02eaba"
    )
    assert verification["preflight_v2_id"] == (
        "5465f17d583a0dd19f5c4ea4939883ac8f65b7a19e128f7ad130246eb2a816bf"
    )
    assert verification["authorization_benchmark_id"] == (
        "6d359a0c2608afe5f4d5bd8366ec4afe53a667b7be497cb521b814b22fee0f18"
    )
    assert verification["missing_execution_identity_bindings"] == 10
    assert verification["real_signed_authorization_receipt_present"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v26_preserves_v1_gap_and_forbids_scope_broadening() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    old = observed["component_compartments"][
        "cluster_to_sae_empirical_preflight_v1"
    ]
    contract = observed["component_compartments"][
        "cluster_to_sae_capture_authorization_v1"
    ]
    assert "superseded" in old["status"]
    assert contract["preflight_v2"][
        "authorization_receipt_is_deferred_one_way_handoff"
    ] is True
    assert contract["authorization_validator"][
        "signature_alone_does_not_authorize_runtime"
    ] is True
    assert contract["execution_authorized"] is False
    assert contract["sae_training_authorized"] is False
    assert contract["final_evaluation_authorized"] is False


def test_runbook_v18_reproduces_and_stops_before_empirical_work() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "c96df7676f62501531c33dbb259b8c5e846f8683b3d0e11b756d4d48ca8b140f"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["missing_execution_identity_bindings"] == 10
    assert verification["real_signed_authorization_receipt_present"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cycle_free_cluster_to_sae_preflight_v2",
        "verify_cluster_to_sae_capture_authorization_benchmark_v1",
        "verify_research_protocol_v26",
    } <= command_ids
    assert observed["scientific_boundary"]["real_authorization_receipt_present"] is False
    assert observed["scientific_boundary"]["execution_authorized"] is False
    assert observed["scientific_boundary"]["sae_training_authorized"] is False

