from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v7 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline_v5 import (
    canonical_inputs as receipt_inputs,
    verify_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v7.json"
RECEIPT = ROOT / "artifacts/vpd_unified_compartment_pipeline_v5/receipt.json"


def test_protocol_v7_binds_same_compartment_planner_executor_and_benchmark() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["same_compartment_control_required"] is True
    assert verification["execution_authorized"] is False
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    control = protocol["focused_evaluation"]["matched_parameter_control"]
    assert control["generic_cross_compartment_control_pool_forbidden"] is True
    assert control["numeric_control_selection_recomputed_before_execution"] is True
    assert control["empty_same_compartment_control_pool_rule"] == (
        "reject_focused_candidate_packet"
    )
    assert control["runner_v2"]["sha256"] == sha256_file(
        ROOT / control["runner_v2"]["path"]
    )
    assert control["executor_v2"]["sha256"] == sha256_file(
        ROOT / control["executor_v2"]["path"]
    )


def test_protocol_v7_rejects_readdressed_control_weakening(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["focused_evaluation"]["matched_parameter_control"][
        "generic_cross_compartment_control_pool_forbidden"
    ] = False
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="not reproducible"):
        verify_protocol(path, **protocol_inputs(ROOT))


def test_unified_lineage_v5_reproduces_category_control_chain() -> None:
    verification = verify_receipt(RECEIPT, **receipt_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["focused_control_scenarios"] == 8
    assert verification["guarded_sae_blueprint_jobs"] == 22
    assert verification["execution_authorized"] is False
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["lineage_amendment"][
        "numeric_control_selection_recomputed_before_execution"
    ] is True
    assert receipt["invariants"]["no_cross_compartment_parameter_control"] is True
    assert receipt["scientific_boundary"]["interventions_executed"] is False


def test_unified_lineage_rejects_readdressed_cross_compartment_bypass(
    tmp_path: Path,
) -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["invariants"]["no_cross_compartment_parameter_control"] = False
    body = copy.deepcopy(tampered)
    body.pop("pipeline_receipt_id")
    tampered["pipeline_receipt_id"] = content_sha256(body)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="not reproducible"):
        verify_receipt(path, **receipt_inputs(ROOT))
