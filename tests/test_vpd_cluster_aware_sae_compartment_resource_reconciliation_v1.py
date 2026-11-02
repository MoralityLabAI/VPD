from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_compartment_resource_reconciliation_v1 import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_benchmark_v1/report.json"
)


def test_compartment_resource_reconciliation_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "0c9800d968a49df90b7b1ab942ed5f59017da4f0ee491eb8440b6257aad0f642"
    )
    assert verification["controls"] == 24
    assert verification["resource_reconciliation_id"] == (
        "c9759d5e1f89cd3f2e6d3a9fa264e27f0ae7e1eee933c3413de5ac3de54bf0ff"
    )
    assert verification["activation_categories"] == 8
    assert verification["compartments"] == 17
    assert verification["jobs"] == 34
    assert verification["planned_steps"] == 68000
    assert verification["planned_activation_rows"] == 20480
    assert verification["raw_campaign_receipts_recovered"] is False
    assert verification["retry_authorized"] is False
    assert verification["training_performed"] is False
    assert verification["execution_authorized"] is False


def test_reconstructed_resource_and_terminal_failure_boundaries() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["provenance_explicitly_reconstructed_not_raw_receipt"] is True
    assert controls["exact_8_category_accounts"] is True
    assert controls["exact_17_compartment_accounts"] is True
    assert controls["each_category_has_two_specific_compartments"] is True
    assert controls["each_compartment_preserves_two_seed_jobs"] is True
    assert controls["shared_fallback_is_separate_two_job_account"] is True
    assert controls["specific_and_shared_jobs_sum_to_global"] is True
    assert controls["specific_and_shared_steps_sum_to_global"] is True
    assert controls["specific_and_shared_rows_sum_to_global"] is True
    assert controls["dictionary_widths_remain_compartment_specific"] is True
    assert controls["success_reconciliation_charges_34_without_training"] is True
    assert controls["failure_reconciliation_is_7_success_1_failure_26_uninvoked"] is True
    assert controls["retry_authority_tamper_rejected"] is True
    assert controls["failure_and_uninvoked_jobs_require_fresh_authority"] is True
    assert controls["resource_ledger_remains_nonexecuting"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["synthetic_reconstruction_only"] is True
    assert boundary["empirical_evidence_created"] is False
    assert boundary["model_loaded"] is False
    assert boundary["training_performed"] is False
    assert boundary["persistent_dictionary_output_created"] is False
    assert boundary["execution_authorized"] is False
