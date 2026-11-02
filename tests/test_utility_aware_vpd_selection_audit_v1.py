from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_utility_aware_vpd_selection_audit_v1 import (
    verify_audit,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/vpd_utility_aware_selection_audit_v1/audit.json"


def test_utility_aware_selection_audit_reproduces_without_execution() -> None:
    result = verify_audit(AUDIT, project_root=ROOT)
    assert result["status"] == "pass"
    assert result["execution_authorized"] is False
    assert result["model_inference_performed"] is False


def test_selection_catalog_is_equal_budget_and_challenger_aware() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["authority"]["archive_only"] is True
    assert audit["evidence_boundary"]["separate_pass_dictionary_admissions"] == 0
    assert audit["evidence_boundary"]["equal_total_rank_checks"] == 36
    assert audit["evidence_boundary"]["third_domain_transfer_established"] is False
    catalog = audit["frozen_catalog"]
    assert catalog["candidate_ids"] == [
        "q_monolithic",
        "q_geometric",
        "q_interval_aware",
    ]
    assert catalog["shared_constraints"] == {
        "base_representation": "one pre-existing shared monolithic dictionary",
        "total_rank": 64,
        "maximum_edited_coordinates": 64,
        "additional_dictionary_fits": 0,
        "additional_model_calls_in_this_archive": 0,
        "same_frozen_case_blocks_and_candidate_pool": True,
        "selection_uses_discovery_and_calibration_blocks_only": True,
        "final_evaluation_blocks_are_never_used_to_choose_a_candidate": True,
    }
    envelope = audit["finite_challenger_envelope"]["regret_interval"]
    assert envelope["lower"] == "max(0, max_p(-U_qp))"
    assert envelope["upper"] == "max(0, max_p(-L_qp))"
    assert audit["cost_accounting"]["archive_values"]["model_calls"] == 0
    controls = {row["id"] for row in audit["controls"]}
    assert controls == {"held_out_family", "random_subspace", "pass_label_only"}
