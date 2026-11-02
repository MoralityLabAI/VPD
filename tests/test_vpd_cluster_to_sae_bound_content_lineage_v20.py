from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v28 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v20 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v28.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v20/runbook.json"


def test_protocol_v28_reproduces_bound_content_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "128d8ad5a7fc34b56b24e46b5221b919e035c74b4b5faa7d2d6abcb15dd316b5"
    )
    assert verification["bound_content_benchmark_id"] == (
        "dd035b55e6d56d941c369338123257f7bb5548326e4297f8c2d547315cb50f1a"
    )
    assert verification["bound_content_controls"] == 19
    assert verification["real_model_manifest_present"] is False
    assert verification["real_bound_content_verification_present"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v28_pins_one_way_preload_order() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = observed["component_compartments"]["cluster_to_sae_bound_content_v1"]
    assert contract["one_way_order"] == [
        "stream_model_files_and_write_manifest_outside_model_root",
        "bind_manifest_file_sha256_and_adapter_digests_in_preflight_v2",
        "verify_exact_model_tree_and_adapter_bytes_against_preflight",
        "verify_real_signed_scope_exact_authorization_receipt",
        "atomically_create_new_isolated_output_root",
        "begin_zero_action_capture",
    ]
    assert contract["model_manifest_preflight_back_reference_forbidden"] is True
    assert contract["model_or_adapter_loaded"] is False
    assert contract["sae_training_authorized"] is False


def test_runbook_v20_reproduces_and_keeps_real_commands_blocked() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "0586ec6de4f355743e2f3372c4642140de23ee6329deb6739cb156436d68def2"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["real_model_manifest_present"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_to_sae_bound_content_benchmark_v1",
        "verify_research_protocol_v28",
    } <= command_ids
    pending = observed["stages"][1]["required_one_way_commands"]
    assert len(pending) == 2
    assert all(row["authorized_now"] is False for row in pending)
    assert observed["scientific_boundary"]["real_model_or_adapter_content_opened"] is False
    assert observed["scientific_boundary"]["sae_training_authorized"] is False
