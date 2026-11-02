from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v33 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v25 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v33.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v25/runbook.json"


def test_protocol_v33_reproduces_guarded_runner_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "cc25fac546c628e4618511e569485faf9a0a9984000affe85dfdbabaf0be2559"
    )
    assert verification["adapter_v2_benchmark_id"] == (
        "f6b9bef6f89f786e90e0ca1f8af4d24395c2dc54be50605d4d7cab339a5d2d36"
    )
    assert verification["reservation_benchmark_id"] == (
        "cc8938f2cb5fcf8d0b0ef7922ca07a22622712ae5e8bc8a27bb6a4947bd0c06a"
    )
    assert verification["runner_v3_benchmark_id"] == (
        "882f038f16bddca72b9db40b8b952406e860c21eb20e21de3163b399180f7a43"
    )
    assert verification["runner_v3_controls"] == 18
    assert verification["production_runner_code_path_present"] is True
    assert verification["production_runner_empirically_exercised"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v33_keeps_real_capture_boundaries_explicit() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    runner = observed["component_compartments"]["canonical_capture_runner_v3"]
    reservation = observed["component_compartments"][
        "output_reservation_transition_v1"
    ]
    assert runner["reservation_precedes_runtime_load"] is True
    assert runner["production_receipt_replays_allocation_and_reservation"] is True
    assert runner["actual_recovered_shape_capture_plans_present"] is False
    assert runner["real_reservation_receipt_present"] is False
    assert runner["capture_started"] is False
    assert reservation["persistent_output_root_created"] is False
    assert observed["scientific_boundary"]["model_inference_performed"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v25_reproduces_and_stops_at_receipt_verification() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "a6ba3497fc808e65305a096b113c0e141cdb03094296c42d0f58ba099da3f08a"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["guarded_runtime_adapter_v2_ready"] is True
    assert verification["output_reservation_transition_ready"] is True
    assert verification["canonical_capture_runner_v3_code_ready"] is True
    assert verification["production_runner_empirically_exercised"] is False
    assert verification["identity_bindings_complete"] is False
    assert verification["actual_capture_plans_present"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_guarded_capture_runtime_adapter_benchmark_v2",
        "verify_output_reservation_transition_benchmark_v1",
        "verify_canonical_capture_runner_benchmark_v3",
        "verify_research_protocol_v33",
    } <= command_ids
    assert observed["scientific_boundary"]["sae_training_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False
