from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_capture_campaign_guard_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts/vpd_cluster_capture_campaign_guard_benchmark_v1/report.json"


def test_capture_campaign_guard_benchmark_reproduces() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["controls"] == 22
    assert verification["roles"] == 4
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_capture_campaign_guard_covers_stage_budget_and_action_boundaries() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["calibration_stage_skip_rejected"] is True
    assert controls["sae_training_stage_skip_rejected"] is True
    assert controls["batch_forward_overage_rejected"] is True
    assert controls["wall_time_overage_rejected"] is True
    assert controls["generated_token_attempt_rejected"] is True
    assert controls["real_action_attempt_rejected"] is True
    assert controls["persistent_mutation_attempt_rejected"] is True
    assert controls["guard_receipt_remains_non_authorizing"] is True
