from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_execution_blueprint_benchmark_v1/report.json"
)


def test_cluster_aware_sae_execution_blueprint_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "7612bb0da7e388f029cbfe66b17a104ff93a32fef22763df3625a462db5daa75"
    )
    assert verification["controls"] == 17
    assert verification["runtime_worker_implemented"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False


def test_blueprint_binds_exact_samples_routes_and_global_pairs() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["exact_episode_and_pair_lists_bound"] is True
    assert controls["one_global_pair_split_for_all_jobs"] is True
    assert controls["training_and_validation_pair_lists_are_disjoint"] is True
    assert controls["routing_and_stability_ids_bound"] is True
    assert controls["specific_jobs_bind_component_membership"] is True
    assert controls["shared_jobs_have_no_fake_cluster_identity"] is True
    assert controls["output_directories_are_unique_and_relative"] is True
    assert controls["authorization_placeholders_are_explicit"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["runtime_worker_implemented"] is False
    assert boundary["empirical_blueprint_produced"] is False
    assert boundary["model_loaded"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["sae_training_authorized"] is False
