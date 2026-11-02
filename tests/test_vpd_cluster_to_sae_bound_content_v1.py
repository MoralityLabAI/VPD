from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_bound_content_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts/vpd_cluster_to_sae_bound_content_benchmark_v1/report.json"


def test_bound_content_benchmark_reproduces() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["controls"] == 19
    assert verification["real_model_or_adapter_content_opened"] is False
    assert verification["model_loaded"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_bound_content_controls_cover_cycle_and_mutation_boundaries() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["manifest_is_one_way_and_cycle_free"] is True
    assert controls["model_file_mutation_rejected"] is True
    assert controls["unlisted_model_file_rejected"] is True
    assert controls["adapter_weight_mutation_rejected"] is True
    assert controls["adapter_config_mutation_rejected"] is True
    assert controls["synthetic_reparse_point_rejected"] is True
    assert controls["all_verification_paths_remain_non_authorizing"] is True
