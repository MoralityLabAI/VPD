from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v29 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v21 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v29.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v21/runbook.json"


def test_protocol_v29_reproduces_runtime_admission_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "8dfdfe72a6fd245d4d5a9b4d5612a1463bada0cdc3d2ae61afe32a342bef4a8b"
    )
    assert verification["runtime_admission_benchmark_id"] == (
        "32e17a9ba015320b44cf5ae4b9a88e56aa8eabde42e45e5e67635cbe2c366a66"
    )
    assert verification["runtime_admission_controls"] == 12
    assert verification["real_runtime_admission_present"] is False
    assert verification["persistent_output_root_created"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v29_pins_replay_and_capture_boundaries() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = observed["component_compartments"][
        "cluster_to_sae_runtime_admission_v1"
    ]
    assert len(contract["joined_proofs"]) == 4
    assert contract["all_proofs_must_join_same_preflight_id"] is True
    assert contract["atomic_output_creation_is_exact_receipt_replay_barrier"] is True
    assert contract["existing_output_root_is_terminal_fail_closed"] is True
    assert contract["reservation_never_starts_capture"] is True
    assert contract["sae_training_authorized"] is False


def test_runbook_v21_reproduces_and_keeps_admission_command_blocked() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "3dc3f49680e15add3a1e2346662576472ea72673bbee602b157b5714099c82a7"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["real_runtime_admission_present"] is False
    assert verification["persistent_output_root_created"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_to_sae_runtime_admission_benchmark_v1",
        "verify_research_protocol_v29",
    } <= command_ids
    pending = observed["stages"][1]["required_one_way_commands"]
    runtime = [row for row in pending if row["id"] == "build_readonly_real_runtime_admission"]
    assert len(runtime) == 1
    assert runtime[0]["authorized_now"] is False
    assert observed["scientific_boundary"]["sae_training_authorized"] is False
