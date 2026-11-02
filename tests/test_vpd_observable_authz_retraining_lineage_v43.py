from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v51 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v43 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v51.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v43/runbook.json"
PROTOCOL_SHA256 = "1b49c83ed86571a9fb9c9f54b768b0ee470123d9a77d63d1bdf7ef637ad2ba2c"
RUNBOOK_ID = "870a7cbd0622e804cb6c91ad675160de3acc348df4f99256b576023b99017600"
SPEC_ID = "0c2f14cad8a05efeddda7a2e3428f2d323f49ee1a48f173dfc778a71f855dd86"
BENCHMARK_ID = "c0c88eeb6cdce5fc00564a38712d6e4ac1f771d59df216985d18fd8d497611a3"


def test_protocol_v51_reproduces_observable_retraining_design() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == PROTOCOL_SHA256
    assert verification["experiment_spec_id"] == SPEC_ID
    assert verification["benchmark_id"] == BENCHMARK_ID
    assert verification["controls"] == 25
    assert verification["activation_categories"] == 8
    assert verification["disjoint_partitions"] == 7
    assert verification["candidate_variants"] == 4
    assert verification["planned_pairs"] == 840
    assert verification["planned_rows"] == 1680
    assert verification["fresh_pack_present"] is False
    assert verification["training_performed"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["candidate_training_authorized"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v51_preserves_v50_and_design_only_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "observable_authz_category_retraining_experiment_v1"
    ]
    assert component["pair_template_and_condition_disjoint"] is True
    assert component["observable_invalid_maps_to_request"] is True
    assert component["observable_valid_maps_to_propose"] is True
    assert component["primary_candidate_id"] == "fresh_lora_margin_v1"
    assert component["fresh_pack_present"] is False
    assert component["training_performed"] is False
    predecessor = observed["protocol_derivation"]["predecessor_v50_verification"]
    assert predecessor["protocol_sha256"] == (
        "55dc12c6f948c6fc8ecaeb1d9d06d037fbc5a72ab2fd0a5990b453bd7271a951"
    )
    assert predecessor["deep_predecessor_replay_performed"] is False


def test_runbook_v43_reproduces_and_blocks_pack_generation() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == RUNBOOK_ID
    assert verification["safe_stage_now"] == 0
    assert verification["observable_retraining_experiment_verified"] is True
    assert verification["activation_categories"] == 8
    assert verification["disjoint_partitions"] == 7
    assert verification["candidate_variants"] == 4
    assert verification["planned_pairs"] == 840
    assert verification["planned_rows"] == 1680
    assert verification["fresh_pack_present"] is False
    assert verification["pack_generation_authorized"] is False
    assert verification["training_performed"] is False
    assert verification["persistent_dictionary_output_created"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["candidate_training_authorized"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_observable_authz_retraining_experiment_v1",
        "verify_observable_authz_retraining_benchmark_v1",
        "verify_research_protocol_v51",
    } <= command_ids
