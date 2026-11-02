from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v30 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v22 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v30.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v22/runbook.json"


def test_protocol_v30_reproduces_campaign_guard_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "6e231ace77b43b154e6622da1ee62017a02fbfebc44fbd2ad6b80720a9f47925"
    )
    assert verification["campaign_guard_benchmark_id"] == (
        "68555a4dd6e87341cadd4eb3a83bd49223314a44617add980a5337f02d1e6288"
    )
    assert verification["campaign_guard_controls"] == 22
    assert verification["roles"] == 4
    assert verification["real_guard_receipt_present"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v30_pins_counter_and_stage_boundaries() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = observed["component_compartments"]["cluster_capture_campaign_guard_v1"]
    assert contract["counter_increment_must_precede_compute"] is True
    assert contract["all_role_ceilings_sum_to_global_ceiling"] is True
    assert contract["wall_time_and_cost_must_be_monotonic_and_bounded"] is True
    assert contract["final_evaluation_role_forbidden"] is True
    assert contract["sae_training_authorized"] is False


def test_runbook_v22_reproduces_and_stops_before_worker_integration() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "8b39c4b66bfbe6edcd3688d0aff9842bdfd3fecc318bcf3ac2bd3c9a04e6ff1d"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["campaign_guard_ready"] is True
    assert verification["capture_worker_guard_integration_present"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_capture_campaign_guard_benchmark_v1",
        "verify_research_protocol_v30",
    } <= command_ids
    assert observed["scientific_boundary"]["sae_training_authorized"] is False
