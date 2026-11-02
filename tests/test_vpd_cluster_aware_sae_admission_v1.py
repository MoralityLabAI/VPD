from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_admission_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_cluster_aware_sae_admission_benchmark_v1/report.json"


def test_cluster_aware_sae_admission_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "98c5613d4fe7615ca217d51e80460811e1901d05d177d7839a4aaf8d6f8d79b2"
    )
    assert verification["controls"] == 24
    assert verification["dry_run_cli_implemented"] is True
    assert verification["durable_nonce_consumption_implemented"] is False
    assert verification["output_reservation_implemented"] is False
    assert verification["trainer_callback_invoked"] is False
    assert verification["sae_training_authorized"] is False


def test_signed_admission_is_exact_and_nonexecuting() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["signed_synthetic_release_verifies_without_authority"] is True
    assert controls["signed_synthetic_compute_verifies_without_authority"] is True
    assert controls[
        "release_binds_exact_blueprint_plan_materialization_and_global_split"
    ] is True
    assert controls[
        "compute_binds_one_exact_job_device_samples_resources_and_output"
    ] is True
    assert controls["real_release_rejects_synthetic_blueprint"] is True
    assert controls["consumed_release_nonce_rejected"] is True
    assert controls["consumed_compute_nonce_rejected"] is True
    assert controls["wrong_compute_job_rejected"] is True
    assert controls["wrong_compute_device_rejected"] is True
    assert controls["base_model_mutation_broadening_rejected"] is True
    assert controls["dry_run_never_reserves_or_attaches_or_invokes"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["private_key_persisted"] is False
    assert boundary["model_loaded"] is False
    assert boundary["persistent_output_created"] is False
    assert boundary["trainer_callback_invoked"] is False
    assert boundary["execution_authorized"] is False
