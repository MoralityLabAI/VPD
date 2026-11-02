from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_empirical_preflight_v1 import (
    build_report,
    verify_benchmark,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v1 import (
    build_preflight,
    verify_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "artifacts/vpd_cluster_to_sae_empirical_preflight_v1/preflight.json"
BENCHMARK = (
    ROOT / "artifacts/vpd_cluster_to_sae_empirical_preflight_benchmark_v1/report.json"
)


def test_cluster_to_sae_empirical_preflight_reproduces_blocked_template() -> None:
    observed = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert observed == build_preflight(project_root=ROOT)
    assert observed["preflight_id"] == (
        "edf41948607ac8471030545ea9782c1f6ea76dd2aec72f09aa7cffac5fb45f91"
    )
    assert observed["status"] == (
        "blocked_missing_immutable_identities_and_explicit_authorization"
    )
    assert len(observed["missing_required_bindings"]) == 13
    assert observed["scientific_boundary"]["execution_authorized"] is False
    assert observed["scientific_boundary"]["model_or_adapter_weights_opened"] is False


def test_cluster_to_sae_preflight_resource_ceiling_is_exact() -> None:
    observed = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    ceiling = observed["resource_ceiling"]
    assert ceiling["maximum_compute_jobs"] == 64
    assert ceiling["maximum_batch_forwards_at_batch_size_4"] == 41_352_192
    assert ceiling["maximum_episode_forward_equivalents"] == 164_900_736
    assert ceiling["estimated_float32_effect_payload_bytes"] == 4_046_192_640
    assert set(observed["capture_roles"]) == {
        "discovery",
        "calibration",
        "cluster_validation",
        "sae_training",
    }
    assert "final_evaluation" not in observed["capture_roles"]


def test_cluster_to_sae_preflight_verifier_preserves_boundaries() -> None:
    verification = verify_preflight(PREFLIGHT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["missing_required_bindings"] == 13
    assert verification["maximum_batch_forwards"] == 41_352_192
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_cluster_to_sae_empirical_preflight_benchmark_reproduces() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    assert observed["benchmark_id"] == (
        "03f8f0e82283b82b28997403a4f68bcd84d6d9630e1a23cc825ccbb805b28e13"
    )
    assert observed["summary"] == {"controls": 15, "passed": 15}
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["controls"] == 15
    assert verification["execution_authorized"] is False


def test_cluster_to_sae_preflight_rejects_unknown_binding() -> None:
    with pytest.raises(ValueError, match="unknown bindings"):
        build_preflight(project_root=ROOT, bindings={"run_model_now": True})

