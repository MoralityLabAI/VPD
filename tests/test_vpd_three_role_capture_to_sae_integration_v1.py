from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_three_role_capture_to_sae_integration_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "artifacts/vpd_three_role_capture_to_sae_integration_benchmark_v1/report.json"
)


def test_three_role_capture_to_sae_integration_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    assert observed["summary"] == {"controls": 18, "passed": 18}
    assert observed["results"]["v1_specific_compartments"] == 0
    assert observed["results"]["v2_summary"]["specific_compartments"] == 16
    assert observed["results"]["v2_summary"]["shared_compartments"] == 1
    assert observed["results"]["v2_summary"]["training_jobs"] == 34
    assert observed["results"]["v2_summary"]["routed_training_episodes"] == 80
    assert observed["results"]["v2_summary"]["routed_training_pairs"] == 10


def test_three_role_integration_verifier_preserves_boundaries() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["specific_compartments"] == 16
    assert verification["shared_compartments"] == 1
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False
