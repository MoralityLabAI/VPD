from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_job_allocations_v1 import (
    build_report,
    verify_benchmark,
)
from experiments.authz_mechinterp.plan_vpd_guarded_capture_job_allocations_v1 import (
    verify_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/vpd_guarded_capture_job_allocations_v1/manifest.json"
BENCHMARK = (
    ROOT / "artifacts/vpd_guarded_capture_job_allocations_benchmark_v1/report.json"
)


def test_guarded_capture_allocations_reproduce_exact_global_ceiling() -> None:
    verification = verify_manifest(MANIFEST, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["allocation_manifest_id"] == (
        "b046a591e380bc4882534f07329ba32c1ce0fdcb1735dcb99fc362196ea23a65"
    )
    assert verification["roles"] == 4
    assert verification["jobs"] == 64
    assert verification["identity_bindings_complete"] is False
    observed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert observed["global_totals"] == {
        "batch_forwards": 41_352_192,
        "compute_jobs": 64,
        "episode_forward_equivalents": 164_900_736,
        "float32_effect_payload_bytes": 4_046_192_640,
    }
    assert observed["scientific_boundary"]["execution_authorized"] is False


def test_guarded_capture_allocation_benchmark_reproduces() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "bc929b7751eb09cc9b0a8d4ff258c546e59f27b5fa61e71decc4753e02e007a8"
    )
    assert verification["controls"] == 18
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False

