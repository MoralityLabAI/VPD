from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v32 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v24 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v32.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v24/runbook.json"


def test_protocol_v32_reproduces_exact_job_allocation_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "59749a0512ddbb0d4c7ea7fc3b91b9269da7cee8857e1e9a9f9bf11d20e13011"
    )
    assert verification["allocation_manifest_id"] == (
        "b046a591e380bc4882534f07329ba32c1ce0fdcb1735dcb99fc362196ea23a65"
    )
    assert verification["allocation_benchmark_id"] == (
        "bc929b7751eb09cc9b0a8d4ff258c546e59f27b5fa61e71decc4753e02e007a8"
    )
    assert verification["binding_benchmark_id"] == (
        "5eb11595d2d24b75fd8535e42625150c7257bcdeea4f44feb69f1e3f602225a9"
    )
    assert verification["jobs"] == 64
    assert verification["identity_bindings_complete"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v32_pins_regeneration_and_admission_boundaries() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    allocation = observed["component_compartments"][
        "guarded_capture_job_allocations_v1"
    ]
    binding = observed["component_compartments"]["guarded_capture_job_binding_v1"]
    assert allocation["regenerate_from_future_identity_complete_preflight"] is True
    assert allocation["identity_bindings_complete"] is False
    assert binding["exact_packet_membership_required"] is True
    assert binding["identity_complete_plan_still_requires_runtime_admission"] is True
    assert binding["production_runner_integration_present"] is False
    assert binding["execution_authorized"] is False


def test_runbook_v24_reproduces_and_stops_before_real_plans() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "7a40353f8c085840c587c0cb2fb239a27bf9eb26381a91690993f8c1557c6f6f"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["guarded_job_allocations_ready"] is True
    assert verification["guarded_job_binding_verifier_ready"] is True
    assert verification["identity_bindings_complete"] is False
    assert verification["production_runner_integration_present"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_guarded_capture_job_allocations_v1",
        "verify_guarded_capture_job_allocation_benchmark_v1",
        "verify_guarded_capture_job_binding_benchmark_v1",
        "verify_research_protocol_v32",
    } <= command_ids
    assert observed["scientific_boundary"]["sae_training_authorized"] is False
