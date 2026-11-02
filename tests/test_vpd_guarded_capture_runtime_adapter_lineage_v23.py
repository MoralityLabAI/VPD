from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v31 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v23 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v31.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v23/runbook.json"


def test_protocol_v31_reproduces_guarded_runtime_adapter_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "d36d6dc655b4879b391e604965f8a214ae4397cc9c569c1e70a0f66d35712104"
    )
    assert verification["adapter_benchmark_id"] == (
        "7dca595cfc6d37709b144a12089b9407b06ca3a19c5607c271bde7e854763ae2"
    )
    assert verification["adapter_controls"] == 20
    assert verification["production_runner_integration_present"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v31_pins_precall_accounting_boundaries() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = observed["component_compartments"][
        "guarded_capture_runtime_adapter_v1"
    ]
    assert contract["job_and_payload_reserved_before_first_callback"] is True
    assert contract["forward_and_actual_episode_counters_increment_before_callback"] is True
    assert contract["failed_callbacks_remain_charged"] is True
    assert contract["canonical_capture_worker_v2_modified"] is False
    assert contract["production_runner_integration_present"] is False
    assert contract["execution_authorized"] is False


def test_runbook_v23_reproduces_and_stops_before_production_runner() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "79f888f86513ad58c6640f6d00c4337716566c26ace55263207300c1f0e5bb2f"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["guarded_runtime_adapter_ready"] is True
    assert verification["production_runner_integration_present"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_guarded_capture_runtime_adapter_benchmark_v1",
        "verify_research_protocol_v31",
    } <= command_ids
    assert observed["scientific_boundary"]["sae_training_authorized"] is False
