from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v27 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v19 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v27.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v19/runbook.json"


def test_protocol_v27_reproduces_binding_readiness_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "36c66bfb8591961342ea717bde6289f2511e32fbe2577959dad683ab51b94eca"
    )
    assert verification["readiness_id"] == (
        "77e2c766499f22d401ffa60166b31dbc8fe4d0ecaefa349ab3b636202fef1472"
    )
    assert verification["readiness_benchmark_id"] == (
        "987909d0fef72a41a3a50775c2330402632bb89458c5c6ba4b917634a9f0eda1"
    )
    assert verification["readiness_controls"] == 17
    assert verification["missing_execution_identity_bindings"] == 10
    assert verification["execution_authorized"] is False


def test_protocol_v27_keeps_output_and_training_boundaries_closed() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = observed["component_compartments"][
        "cluster_to_sae_binding_readiness_v1"
    ]
    isolation = contract["runtime_isolation_contract"]
    assert isolation["output_root_must_be_new_direct_child"] is True
    assert isolation["runtime_must_refuse_overwrite_or_resume"] is True
    assert isolation["writes_outside_output_root_forbidden"] is True
    assert contract["model_or_adapter_content_opened"] is False
    assert contract["execution_authorized"] is False
    assert contract["sae_training_authorized"] is False


def test_runbook_v19_reproduces_and_stops_before_empirical_work() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "74e48f8b1aee138f69135845f31d4ff2a8eda34eba189ff7506f2de0084fff44"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["missing_execution_identity_bindings"] == 10
    assert verification["output_isolation_verified_for_real_run"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_to_sae_binding_readiness_v1",
        "verify_cluster_to_sae_binding_readiness_benchmark_v1",
        "verify_research_protocol_v27",
    } <= command_ids
    assert observed["scientific_boundary"]["real_runtime_bindings_present"] is False
    assert observed["scientific_boundary"]["sae_training_authorized"] is False
