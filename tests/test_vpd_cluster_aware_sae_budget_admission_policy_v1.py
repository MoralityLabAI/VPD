from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_budget_admission_policy_v1 import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_benchmark_v1/report.json"
)


def test_cluster_aware_sae_budget_admission_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "5a2bb38b94c43856bcc9d7e7932db98e3cb89192af60e3418706d33713b6db64"
    )
    assert verification["budget_policy_id"] == (
        "5e9f4fc0841d0e71a2eef6f9ac91f5e27d0702c0e0aadfa5a6f511dad50719f2"
    )
    assert verification["controls"] == 32
    assert verification["activation_categories"] == 8
    assert verification["compartments"] == 17
    assert verification["jobs_ceiling"] == 34
    assert verification["planned_steps_ceiling"] == 68000
    assert verification["planned_activation_rows_ceiling"] == 20480
    assert verification["shared_fallback_charged_to_categories"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_budget_policy_ceiling_isolation_and_authority_controls() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    for control in (
        "category_ceiling_broadening_rejected",
        "specific_compartment_ceiling_broadening_rejected",
        "shared_fallback_ceiling_broadening_rejected",
        "shared_fallback_category_charge_rejected",
        "shared_fallback_global_double_count_rejected",
        "specific_dictionary_width_substitution_rejected",
        "shared_dictionary_width_substitution_rejected",
        "output_directory_reuse_rejected",
        "authority_broadening_rejected",
        "policy_content_tamper_rejected",
        "readdressed_policy_tamper_rejected",
        "budget_admission_transfers_no_authority",
    ):
        assert controls[control] is True
    boundary = observed["scientific_boundary"]
    assert boundary["synthetic_reference_proposal_only"] is True
    assert boundary["empirical_blueprint_present"] is False
    assert boundary["empirical_evidence_created"] is False
    assert boundary["model_loaded"] is False
    assert boundary["training_performed"] is False
    assert boundary["persistent_dictionary_output_created"] is False
    assert boundary["execution_authorized"] is False
