from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_capture_authorization_v1 import (
    build_report,
    verify_benchmark,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 import (
    build_preflight,
    verify_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "artifacts/vpd_cluster_to_sae_empirical_preflight_v2/preflight.json"
BENCHMARK = (
    ROOT
    / "artifacts/vpd_cluster_to_sae_capture_authorization_benchmark_v1/report.json"
)


def test_cycle_free_preflight_v2_reproduces() -> None:
    observed = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert observed == build_preflight(project_root=ROOT)
    assert observed["preflight_id"] == (
        "5465f17d583a0dd19f5c4ea4939883ac8f65b7a19e128f7ad130246eb2a816bf"
    )
    assert len(observed["missing_required_bindings"]) == 10
    assert "authorization_receipt_id" not in observed["immutable_bindings"]
    assert "authorization_receipt_sha256" not in observed["immutable_bindings"]
    assert observed["authorization_handoff"][
        "receipt_id_is_not_an_input_to_this_preflight"
    ] is True
    assert observed["scientific_boundary"]["execution_authorized"] is False


def test_cycle_free_preflight_v2_verifier_preserves_boundaries() -> None:
    verification = verify_preflight(PREFLIGHT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["missing_required_bindings"] == 10
    assert verification["authorization_receipt_present"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_capture_authorization_benchmark_reproduces() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    assert observed["benchmark_id"] == (
        "6d359a0c2608afe5f4d5bd8366ec4afe53a667b7be497cb521b814b22fee0f18"
    )
    assert observed["summary"] == {"controls": 16, "passed": 16}
    clean = observed["results"]["clean_verification"]
    assert clean["signature_valid"] is True
    assert clean["scope_exact"] is True
    assert clean["execution_authorized"] is False
    assert clean["sae_training_authorized"] is False


def test_capture_authorization_benchmark_verifier_is_non_authorizing() -> None:
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["controls"] == 16
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_preflight_v2_rejects_deferred_receipt_as_input() -> None:
    with pytest.raises(ValueError, match="unknown bindings"):
        build_preflight(
            project_root=ROOT,
            bindings={"authorization_receipt_id": "0" * 64},
        )

