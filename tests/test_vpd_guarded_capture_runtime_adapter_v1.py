from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v1/report.json"
)


def test_guarded_capture_runtime_adapter_benchmark_reproduces() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["controls"] == 20
    assert verification["real_model_inference_performed"] is False
    assert verification["capture_authorized"] is False
    assert verification["execution_authorized"] is False


def test_guarded_adapter_charges_before_call_and_fails_closed() -> None:
    controls = json.loads(BENCHMARK.read_text(encoding="utf-8"))["controls"]
    assert controls["job_and_payload_reserved_before_first_callback"] is True
    assert controls["first_callback_observes_preincremented_counters"] is True
    assert controls["second_partial_batch_charged_by_actual_length"] is True
    assert controls["failed_callback_remains_charged"] is True
    assert controls["per_job_forward_overage_stops_before_callback"] is True
    assert controls["per_job_episode_overage_stops_before_callback"] is True
    assert controls["role_forward_allocation_overage_rejected_before_session"] is True
    assert controls["exact_job_receipt_is_content_addressed"] is True
    assert controls["receipt_remains_non_authorizing"] is True
