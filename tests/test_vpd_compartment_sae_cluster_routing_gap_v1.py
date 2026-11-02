from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.authz_mechinterp.audit_compartment_sae_cluster_routing_gap_v1 import (
    audit_training_plan,
    verify_audit,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
REPORT = (
    ROOT
    / "artifacts/vpd_compartment_sae_cluster_routing_gap_audit_v1/report.json"
)


def test_current_plan_is_category_only_and_audit_reproduces() -> None:
    observed = audit_training_plan(PLAN)
    assert observed["finding"] == "category_only_no_within_category_cluster_routing"
    assert observed["observed"]["training_jobs"] == 18
    assert observed["observed"]["unique_compartments"] == 9
    assert observed["observed"]["cluster_bound_training_jobs"] == 0
    assert verify_audit(REPORT, plan_path=PLAN)["status"] == "pass"


def test_audit_detects_a_cluster_bound_successor_job(tmp_path: Path) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    changed = copy.deepcopy(plan)
    changed["training_jobs"][0]["required_sample_contract"]["cluster_id"] = (
        "cluster-0001"
    )
    changed_path = tmp_path / "plan.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    observed = audit_training_plan(changed_path)
    assert observed["finding"] == "cluster_bound_training_jobs_present"
    assert observed["observed"]["cluster_bound_training_jobs"] == 1
