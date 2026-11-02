from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1 import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_benchmark_v1/report.json"
)


def test_empirical_budget_binding_readiness_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "b95ab40c46c621534b0cf2aed26c342848180a9c5bc7fd48b9674498292159a3"
    )
    assert verification["binding_readiness_id"] == (
        "08bdaaf4733afeae33d847e25bba0ea5657a2723ed49b58137f64c9904084c7a"
    )
    assert verification["controls"] == 29
    assert verification["required_bindings"] == 50
    assert verification["missing_bindings"] == 50
    assert verification["phases"] == 6
    assert verification["budget_binding_verified"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_empirical_budget_binding_phase_and_authority_controls() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    for control in (
        "phase_partition_is_exact_and_exhaustive",
        "all_default_phases_are_incomplete_and_nonauthorizing",
        "placeholder_text_rejected",
        "uniform_placeholder_digest_rejected",
        "uppercase_digest_rejected",
        "output_outside_dedicated_parent_rejected",
        "synthetic_complete_fixture_reaches_syntax_only",
        "synthetic_completion_does_not_verify_cross_artifact_graph",
        "synthetic_completion_does_not_verify_budget_binding",
        "synthetic_completion_does_not_verify_signed_authority",
        "synthetic_completion_does_not_verify_runtime_transition",
        "worksheet_remains_nonexecuting",
    ):
        assert controls[control] is True
    boundary = observed["scientific_boundary"]
    assert boundary["synthetic_complete_fixture_used_in_memory_only"] is True
    assert boundary["empirical_artifact_graph_present"] is False
    assert boundary["model_loaded"] is False
    assert boundary["training_performed"] is False
    assert boundary["persistent_output_created"] is False
    assert boundary["budget_binding_verified"] is False
    assert boundary["execution_authorized"] is False
