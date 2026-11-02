from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_materialization_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "artifacts/vpd_cluster_aware_sae_materialization_benchmark_v1/report.json"
)


def test_guarded_cluster_aware_sae_materialization_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "a8dfb72bd021847f2511258c1d66f08d8bf7ce9c8e1a822f41d4b1af437d3601"
    )
    assert verification["controls"] == 18
    assert verification["categories"] == 8
    assert verification["roles"] == 3
    assert verification["cluster_bound_training_jobs"] == 34
    assert verification["empirical_plan_produced"] is False
    assert verification["sae_training_authorized"] is False


def test_materialization_is_exact_shared_pair_safe_and_non_authorizing() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["exact_eight_categories_bound"] is True
    assert controls["twenty_four_adapter_receipts_bound"] is True
    assert controls["twenty_four_router_receipts_bound"] is True
    assert controls["three_guarded_v3_role_captures_bound"] is True
    assert controls["sixteen_specific_clusters_and_shared_fallback"] is True
    assert controls["shared_pair_safe_global_split"] is True
    assert controls["v3_identity_is_router_source_for_every_role_category"] is True
    assert controls["synthetic_evidence_as_empirical_rejected"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["persistent_empirical_bundle_produced"] is False
    assert boundary["empirical_plan_produced"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["sae_training_authorized"] is False
