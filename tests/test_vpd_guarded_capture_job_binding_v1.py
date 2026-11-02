from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_job_binding_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "artifacts/vpd_guarded_capture_job_binding_benchmark_v1/report.json"
)


def test_guarded_capture_job_binding_benchmark_reproduces() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "5eb11595d2d24b75fd8535e42625150c7257bcdeea4f44feb69f1e3f602225a9"
    )
    assert verification["controls"] == 15
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False


def test_binding_rejects_all_shape_and_identity_substitutions() -> None:
    controls = json.loads(BENCHMARK.read_text(encoding="utf-8"))["controls"]
    for name in (
        "component_substitution_rejected",
        "role_substitution_rejected",
        "dataset_substitution_rejected",
        "forward_allocation_tamper_rejected",
        "episode_allocation_tamper_rejected",
        "batch_size_tamper_rejected",
        "category_count_tamper_rejected",
        "sketch_shape_tamper_rejected",
        "complete_identity_tamper_rejected",
        "complete_output_escape_rejected",
    ):
        assert controls[name] is True
    assert controls["binding_never_authorizes_execution"] is True
