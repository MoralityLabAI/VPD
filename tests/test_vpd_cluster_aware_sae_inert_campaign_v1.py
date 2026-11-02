from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_campaign_v1 import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "artifacts/vpd_cluster_aware_sae_inert_campaign_benchmark_v1/report.json"
)


def test_cluster_aware_sae_inert_campaign_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "34204198cc2cd605e74fff0f24da7b5d09337613f16da74872f354c53f8b32f5"
    )
    assert verification["controls"] == 28
    assert verification["categories"] == 8
    assert verification["compartments"] == 17
    assert verification["jobs"] == 34
    assert verification["unique_output_dirs"] == 34
    assert verification["success_callbacks_charged"] == 34
    assert verification["failure_campaign_callbacks_charged"] == 8
    assert verification["production_campaign_executed"] is False
    assert verification["model_loaded"] is False
    assert verification["training_performed"] is False
    assert verification["persistent_output_written"] is False
    assert verification["execution_authorized"] is False


def test_inert_campaign_exact_coverage_failure_and_isolation_controls() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["exact_34_job_coverage_required_before_campaign_claim"] is True
    assert controls["duplicate_job_rejected_before_campaign_claim"] is True
    assert controls[
        "campaign_and_per_job_claims_precede_all_success_callbacks"
    ] is True
    assert controls["all_34_success_callbacks_invoked_exactly_once"] is True
    assert controls["success_campaign_covers_8_categories"] is True
    assert controls["success_campaign_covers_17_compartments"] is True
    assert controls["success_campaign_charges_exactly_34_callbacks"] is True
    assert controls["success_campaign_preserves_34_unique_output_dirs"] is True
    assert controls["failure_campaign_stops_at_exact_job"] is True
    assert controls["failure_campaign_charges_successes_plus_failure"] is True
    assert controls["failure_campaign_leaves_remaining_uninvoked"] is True
    assert controls["campaigns_create_no_dictionary_output"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["synthetic_campaign_callbacks_invoked"] is True
    assert boundary["production_trainer_callback_invoked"] is False
    assert boundary["model_loaded"] is False
    assert boundary["training_performed"] is False
    assert boundary["execution_authorized"] is False
