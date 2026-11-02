from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_runtime_admission_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts/vpd_cluster_to_sae_runtime_admission_benchmark_v1/report.json"


def test_runtime_admission_benchmark_reproduces() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["controls"] == 12
    assert verification["persistent_output_root_created"] is False
    assert verification["capture_started"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_runtime_admission_controls_cover_join_and_atomic_replay_barrier() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["four_proofs_join_same_preflight"] is True
    assert controls["synthetic_authorization_blocks_reservation"] is True
    assert controls["content_mutation_rejected_at_join"] is True
    assert controls["consumed_nonce_rejected_at_join"] is True
    assert controls["double_reservation_rejected_fail_closed"] is True
    assert controls["admission_join_does_not_start_capture"] is True
