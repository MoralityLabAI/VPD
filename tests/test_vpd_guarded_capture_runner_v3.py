from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v3 import (
    build_report as build_runner_report,
    verify_benchmark as verify_runner_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_output_reservation_v1 import (
    build_report as build_reservation_report,
    verify_benchmark as verify_reservation_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v2 import (
    build_report as build_adapter_report,
    verify_benchmark as verify_adapter_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v2/report.json"
RESERVATION = (
    ROOT / "artifacts/vpd_cluster_to_sae_output_reservation_benchmark_v1/report.json"
)
RUNNER = ROOT / "artifacts/vpd_canonical_capture_job_runtime_benchmark_v3/report.json"


def test_adapter_v2_reserves_before_runtime_load() -> None:
    observed = json.loads(ADAPTER.read_text(encoding="utf-8"))
    assert observed == build_adapter_report(ROOT)
    verification = verify_adapter_benchmark(ADAPTER, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "f6b9bef6f89f786e90e0ca1f8af4d24395c2dc54be50605d4d7cab339a5d2d36"
    )
    assert verification["controls"] == 14
    assert verification["reservation_precedes_runtime_load"] is True
    assert verification["execution_authorized"] is False


def test_output_reservation_transition_is_fail_closed() -> None:
    observed = json.loads(RESERVATION.read_text(encoding="utf-8"))
    assert observed == build_reservation_report(ROOT)
    verification = verify_reservation_benchmark(RESERVATION, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "cc8938f2cb5fcf8d0b0ef7922ca07a22622712ae5e8bc8a27bb6a4947bd0c06a"
    )
    assert verification["controls"] == 13
    assert verification["persistent_output_created"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False


def test_guarded_capture_runner_v3_reproduces_synthetic_integration() -> None:
    observed = json.loads(RUNNER.read_text(encoding="utf-8"))
    assert observed == build_runner_report(ROOT)
    verification = verify_runner_benchmark(RUNNER, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "882f038f16bddca72b9db40b8b952406e860c21eb20e21de3163b399180f7a43"
    )
    assert verification["controls"] == 18
    assert verification["production_runner_empirically_exercised"] is False
    assert verification["capture_authorized"] is False
    assert verification["execution_authorized"] is False
    controls = observed["controls"]
    assert controls["reservation_precedes_loader"] is True
    assert controls["callback_failure_remains_charged"] is True
    assert controls["bad_allocation_rejected_before_loader"] is True
    assert controls["v3_and_v2_receipts_verify"] is True
