from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_runtime_v2 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_canonical_sae_training_capture_runtime_benchmark_v2/report.json"
)


def test_sae_training_capture_runtime_v2_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report()
    assert observed["summary"] == {"controls": 16, "passed": 16}
    assert observed["scenario"]["forward_passes"] == 32
    assert observed["scenario"]["split_role"] == "sae_training"


def test_sae_training_capture_runtime_v2_verifier_preserves_boundaries() -> None:
    verification = verify_benchmark(REPORT)
    assert verification["status"] == "pass"
    assert verification["split_role"] == "sae_training"
    assert verification["actual_model_loaded"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False
