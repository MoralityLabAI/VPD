from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.admit_compartment_sae_training_release import (
    RECEIPT_STATUS,
    evaluate_release_contract,
)
from experiments.authz_mechinterp.benchmark_compartment_sae_training_release_gate import (
    _fixtures,
    verify_benchmark,
)
from experiments.authz_mechinterp.plan_compartment_sae_training_release_gate import (
    verify_release_gate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/vpd_compartment_sae_training_release_gate_v1/plan.json"
TRAINING = ROOT / "artifacts/vpd_compartment_sae_training_plan_v3/plan.json"
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v6.json"
AUDIT_GATE = ROOT / "artifacts/vpd_partition_audit_gate_v2/plan.json"
AUDIT_PACK = ROOT / "artifacts/vpd_partition_audit_pack_v1"
BENCHMARK = (
    ROOT
    / "artifacts/vpd_compartment_sae_training_release_gate_benchmark_v1/report.json"
)


def _plan_inputs() -> dict[str, Path]:
    return {
        "training_plan_path": TRAINING,
        "protocol_config_path": PROTOCOL,
        "partition_audit_gate_plan_path": AUDIT_GATE,
        "partition_audit_pack_dir": AUDIT_PACK,
    }


def test_release_gate_plan_is_reproducible_and_guards_every_blueprint_job() -> None:
    verification = verify_release_gate_plan(PLAN, **_plan_inputs())
    assert verification["status"] == "pass"
    assert verification["guarded_jobs"] == 22
    assert verification["training_jobs"] == 18
    assert verification["execution_authorized"] is False
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    release_job_id = plan["release_job"]["job_id"]
    assert all(
        row["depends_on_training_release_job_id"] == release_job_id
        for row in plan["guarded_execution"]["guarded_jobs"]
    )
    assert plan["guarded_execution"][
        "direct_execution_of_predecessor_blueprint_prohibited"
    ] is True


def test_clean_synthetic_evidence_emits_non_authorizing_release_receipt() -> None:
    receipt = evaluate_release_contract(**_fixtures(ROOT))
    assert receipt["status"] == RECEIPT_STATUS
    assert receipt["released_partition"]["component_count"] == 4032
    assert receipt["released_partition"]["specific_failure_count"] == 0
    assert len(receipt["released_partition"]["compartments"]) == 9
    assert receipt["scientific_boundary"]["compute_authorized"] is False


def test_training_blueprint_substitution_is_rejected() -> None:
    inputs = _fixtures(ROOT)
    inputs["training_plan_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different training blueprint"):
        evaluate_release_contract(**inputs)


def test_partition_audit_verification_cannot_be_omitted() -> None:
    inputs = _fixtures(ROOT)
    inputs["partition_audit_verification"] = {}
    with pytest.raises(ValueError, match="verification did not release"):
        evaluate_release_contract(**inputs)


def test_release_gate_plan_tamper_breaks_content_address(tmp_path: Path) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(plan)
    tampered["scientific_boundary"]["execution_authorized"] = True
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content address mismatch"):
        verify_release_gate_plan(path, **_plan_inputs())


def test_release_gate_synthetic_benchmark_is_reproducible() -> None:
    verification = verify_benchmark(BENCHMARK, ROOT)
    assert verification == {
        "schema": "sch.vpd-compartment-sae-training-release-gate-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": "f706337b2141574a537a013c8d04ea2ec28f5c4e8fac5f4a580b75f9c00b90a3",
        "scenarios": 9,
        "released": 1,
        "rejected_before_release": 8,
    }
