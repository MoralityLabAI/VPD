from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v36 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v28 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v36.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v28/runbook.json"


def test_protocol_v36_reproduces_cluster_aware_execution_contracts() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "734ec49cd8c4b691333e4936e25fa9f691b3f5a4f6f84a6c9aa13b0da076ecc2"
    )
    assert verification["execution_blueprint_benchmark_id"] == (
        "7612bb0da7e388f029cbfe66b17a104ff93a32fef22763df3625a462db5daa75"
    )
    assert verification["execution_blueprint_controls"] == 17
    assert verification["activation_categories"] == 8
    assert verification["execution_jobs"] == 34
    assert verification["runtime_worker_implemented"] is False
    assert verification["empirical_blueprint_present"] is False
    assert verification["sae_training_authorized"] is False


def test_protocol_v36_keeps_worker_and_authority_absence_explicit() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_execution_blueprint_v1"
    ]
    assert component["exact_episode_and_pair_lists_required"] is True
    assert component["one_global_pair_split_required"] is True
    assert component["routing_and_stability_ids_required"] is True
    assert component["unique_relative_output_directories_required"] is True
    assert component["runtime_worker_implemented"] is False
    assert component["empirical_blueprint_present"] is False
    assert component["execution_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v28_reproduces_and_stops_before_runtime_worker() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "d19d71c3dbfc304bda207fad49f300e7709769b90e3f6ecb1964b42ea3e565b7"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_sae_execution_blueprint_ready"] is True
    assert verification["execution_jobs_in_fixture"] == 34
    assert verification["cluster_aware_runtime_worker_implemented"] is False
    assert verification["empirical_cluster_aware_blueprint_present"] is False
    assert verification["scientific_release_receipt_present"] is False
    assert verification["explicit_compute_authorization_present"] is False
    assert verification["identity_bindings_complete"] is False
    assert verification["sae_training_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_execution_blueprint_benchmark_v1",
        "verify_research_protocol_v36",
    } <= command_ids
