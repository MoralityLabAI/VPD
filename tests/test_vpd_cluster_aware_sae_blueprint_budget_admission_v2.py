from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_blueprint_budget_admission_v2 import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_blueprint_budget_admission_benchmark_v2/report.json"
)
BENCHMARK_ID = "1c239f6416c57374a810732496c20f36a7245d882b41c2282c72b8465678da58"
PROPOSAL_ID = "50e09fa4b4247483bf616c9f31226bdd021d7ca609b20a6b911cd361175bb400"
ADMISSION_ID = "dadb42b793f491486bcd9677a840f39ad7644f73460254f3b793c5c7c3a60f72"


def test_blueprint_bound_budget_admission_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == BENCHMARK_ID
    assert verification["proposal_id"] == PROPOSAL_ID
    assert verification["admission_id"] == ADMISSION_ID
    assert verification["controls"] == 27
    assert verification["execution_jobs"] == 34
    assert verification["planned_steps"] == 68000
    assert verification["planned_activation_rows"] == 20480
    assert verification["synthetic_blueprint_persisted"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_blueprint_bound_budget_shape_and_attacks_remain_explicit() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed["benchmark_id"] == BENCHMARK_ID
    assert observed["summary"] == {
        "activation_categories": 8,
        "controls": 27,
        "execution_jobs": 34,
        "passed": 27,
        "planned_activation_rows": 20480,
        "planned_steps": 68000,
        "shared_fallback_compartments": 1,
        "specific_compartments": 16,
        "synthetic_blueprint_persisted": False,
    }
    for control in (
        "proposal_binds_exact_blueprint_id",
        "proposal_binds_exact_blueprint_file_hash",
        "admission_repeats_exact_blueprint_binding",
        "proposal_resource_substitution_rejected",
        "readdressed_blueprint_resource_widening_rejected",
        "readdressed_blueprint_category_rebinding_rejected",
        "readdressed_blueprint_output_reuse_rejected",
        "proposal_and_admission_transfer_no_authority",
        "benchmark_imports_no_model_frameworks",
    ):
        assert observed["controls"][control] is True
    assert observed["scientific_boundary"] == {
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "model_framework_imported": False,
        "model_inference_performed": False,
        "model_loaded": False,
        "paid_compute_authorized": False,
        "persistent_dictionary_output_created": False,
        "sae_training_authorized": False,
        "synthetic_blueprint_fixture_only": True,
        "training_performed": False,
        "weights_loaded_or_mutated": False,
    }
