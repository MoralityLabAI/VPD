from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_cluster_aware_sae_training_plan_v1 import (
    _plan_kwargs,
    build_report,
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.plan_cluster_aware_sae_training_v1 import (
    build_cluster_aware_sae_plan_v1,
    verify_cluster_aware_sae_plan_v1,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_cluster_aware_sae_training_plan_benchmark_v1/report.json"


def test_cluster_aware_plan_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    assert observed["summary"] == {"controls": 17, "passed": 17}
    assert verify_benchmark(REPORT, project_root=ROOT)["status"] == "pass"


def test_plan_binds_specific_membership_and_resolves_job_capacity() -> None:
    plan = build_cluster_aware_sae_plan_v1(**_plan_kwargs(ROOT))
    assert verify_cluster_aware_sae_plan_v1(plan)["status"] == "pass"
    contracts = {
        row["compartment_id"]: row for row in plan["compartments"]
    }
    assert sum(row["support"]["episodes"] for row in contracts.values()) == 22
    assert sum(row["support"]["sae_activation_rows"] for row in contracts.values()) == 2816
    assert all(
        row["support"]["routing_effect_rows"] == row["support"]["episodes"] * 2
        for row in contracts.values()
    )
    for job in plan["training_jobs"]:
        contract = contracts[job["compartment_id"]]
        assert job["dictionary_features"] == contract["dictionary_features"]
        assert job["profile"]["dictionary_features"] == contract["dictionary_features"]
        if contract["kind"] == "specific_cluster":
            assert job["required_sample_contract"][
                "component_membership_sha256"
            ] == contract["component_membership_sha256"]


def test_plan_rejects_readdressed_internal_routing_support() -> None:
    kwargs = _plan_kwargs(ROOT)
    routing = kwargs["category_evidence"]["valid_authority"]["sae_training"]
    route = sorted(routing["routing"]["support_by_route"])[0]
    routing["routing"]["support_by_route"][route]["episodes"] += 1
    body = dict(routing)
    body.pop("cluster_episode_routing_id")
    routing["cluster_episode_routing_id"] = content_sha256(body)
    with pytest.raises(ValueError, match="derived support differs"):
        build_cluster_aware_sae_plan_v1(**kwargs)


def test_plan_receipt_tamper_is_rejected() -> None:
    plan = build_cluster_aware_sae_plan_v1(**_plan_kwargs(ROOT))
    tampered = copy.deepcopy(plan)
    tampered["training_jobs"][0]["dictionary_features"] += 1
    with pytest.raises(ValueError, match="address differs"):
        verify_cluster_aware_sae_plan_v1(tampered)
