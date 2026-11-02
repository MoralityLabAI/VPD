from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v22 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v14 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v22.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v14/runbook.json"


def test_protocol_v22_reproduces_capture_plan_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["capture_plan_benchmark_id"] == (
        "525fbf3433a410ad883b438ee301285f31d704815a86f955751a9cb29b0f09f2"
    )
    assert verification["recovered_sae_training_episodes"] == 6292
    assert verification["capture_worker_v2_ready"] is False
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = observed["component_compartments"][
        "sae_training_effect_capture_plan_v2"
    ]
    assert contract["recovered_sae_training_split"]["pairs"] == 572
    assert contract["sae_training_authorized"] is False


def test_runbook_v14_reproduces_and_stops_before_worker() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "bea30e4ef92a8237f44db800db816db436e2c29f9ec7233b37eff9ac8798690e"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["sae_training_role_capture_planner_ready"] is True
    assert verification["sae_training_role_capture_worker_ready"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert "verify_sae_training_capture_plan_benchmark_v2" in command_ids
    assert observed["scientific_boundary"]["execution_authorized"] is False
