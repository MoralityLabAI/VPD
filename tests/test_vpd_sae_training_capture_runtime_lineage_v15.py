from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v23 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v15 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v23.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v15/runbook.json"


def test_protocol_v23_reproduces_capture_runtime_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "8eda03fd5c982af058da674feff55c223b58598a8d38422fd8bb27afc34e5c49"
    )
    assert verification["runtime_benchmark_id"] == (
        "2023baf1646230680ed7d23d84fb1c6174f63d99abcc8cf629580f8f77086bf3"
    )
    assert verification["capture_worker_v2_ready"] is True
    assert verification["empirical_capture_ready"] is False
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = observed["component_compartments"][
        "sae_training_effect_capture_plan_v2"
    ]
    assert contract["runtime_benchmark"]["controls"] == 16
    assert contract["runtime_benchmark"]["forward_passes"] == 32
    assert contract["execution_authorized"] is False
    assert contract["sae_training_authorized"] is False


def test_runbook_v15_reproduces_and_stops_before_empirical_capture() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "60e3a3e8778a2efe435eeaa904d58c11621add35ed105345974fad26e5aa2699"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["sae_training_role_capture_worker_ready"] is True
    assert verification["empirical_sae_training_capture_ready"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert "verify_sae_training_capture_runtime_benchmark_v2" in command_ids
    assert observed["scientific_boundary"]["execution_authorized"] is False
    assert observed["scientific_boundary"]["sae_training_authorized"] is False
