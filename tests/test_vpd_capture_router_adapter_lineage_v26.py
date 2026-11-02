from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v34 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v26 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v34.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v26/runbook.json"


def test_protocol_v34_reproduces_v3_capture_to_router_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "4a1d90e56b29f0eca513874aa62d5478731e6d5c529ec2a81f56f2fb2c35b370"
    )
    assert verification["capture_router_adapter_v2_benchmark_id"] == (
        "10486291b569bd2eeea91f8429702b6c668463d727013cad98baf2f676628688"
    )
    assert verification["capture_router_adapter_v2_controls"] == 18
    assert verification["v3_capture_identity_preserved_into_router"] is True
    assert verification["production_router_adapter_empirically_exercised"] is False
    assert verification["persistent_empirical_router_receipt_present"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v34_keeps_empirical_clustering_boundary_explicit() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    adapter = observed["component_compartments"][
        "guarded_capture_router_adapter_v2"
    ]
    assert adapter["v3_execution_identity_becomes_router_identity"] is True
    assert adapter["v2_payload_identity_preserved"] is True
    assert adapter["guarded_job_receipt_identity_preserved"] is True
    assert adapter["production_allocation_and_reservation_replay_required"] is True
    assert adapter["persistent_empirical_router_receipt_present"] is False
    assert observed["scientific_boundary"]["model_inference_performed"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v26_reproduces_and_stops_before_empirical_routing() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "746a4ed9b85be93e7dee9b95e949b6a685e7c98a2bda08d91ffd7e8822a81727"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["guarded_capture_router_adapter_v2_ready"] is True
    assert verification["v3_capture_identity_preserved_into_router"] is True
    assert verification["production_router_adapter_empirically_exercised"] is False
    assert verification["persistent_empirical_router_receipts_present"] is False
    assert verification["identity_bindings_complete"] is False
    assert verification["capture_started"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_guarded_capture_router_adapter_benchmark_v2",
        "verify_research_protocol_v34",
    } <= command_ids
