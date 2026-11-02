from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v24 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v16 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v24.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v16/runbook.json"


def test_protocol_v24_reproduces_three_role_integration_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "c42dc6e43296c7e4a3d6054174e015620221cf141377bff44773d8cfc0add2b2"
    )
    assert verification["integration_benchmark_id"] == (
        "b27a7f901d191a81a3bc4e708244edf1a3325e968b979471d064986d2c773df6"
    )
    assert verification["planner_v2_ready"] is True
    assert verification["empirical_plan_ready"] is False
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = observed["component_compartments"][
        "cluster_aware_sae_training_plan_v2"
    ]
    assert contract["integration_benchmark"]["controls"] == 18
    assert contract["integration_benchmark"]["v1_specific_compartments"] == 0
    assert contract["integration_benchmark"]["v2_specific_compartments"] == 16
    assert contract["execution_authorized"] is False


def test_runbook_v16_reproduces_and_stops_before_empirical_work() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "321b0cab4fc95d0270395f39ba95d0139a3ea258e3cb9c05822b6c82f6ce5dd3"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_sae_planner_v2_ready"] is True
    assert verification["empirical_cluster_aware_sae_plan_ready"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert "verify_three_role_capture_to_sae_integration_v1" in command_ids
    assert observed["scientific_boundary"]["execution_authorized"] is False
    assert observed["scientific_boundary"]["sae_training_authorized"] is False
