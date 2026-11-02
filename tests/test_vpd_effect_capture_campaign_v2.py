from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file
from experiments.authz_mechinterp.plan_vpd_effect_capture_campaign_v2 import (
    WRAPPER_MODULE,
    canonical_inputs,
    verify_campaign_plan_v2,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts" / "vpd_effect_capture_campaign_v4" / "plan.json"


def test_seed_corrected_campaign_v2_reproduces_and_preserves_predecessor() -> None:
    verification = verify_campaign_plan_v2(PLAN, **canonical_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["compute_jobs"] == 48
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert {
        tuple(row["arguments"][:2]) for row in plan["compute_jobs"]
    } == {("-m", WRAPPER_MODULE)}
    correction = plan["execution_correction"]
    assert correction["predecessor_worker_preserved_byte_for_byte"] is True
    assert correction["execution_authorized"] is False
    predecessor = plan["source"]["predecessor_campaign"]
    assert predecessor["sha256"] == sha256_file(
        ROOT / "artifacts" / "vpd_effect_capture_campaign_v3" / "plan.json"
    )


def test_seed_corrected_campaign_rejects_readdressed_worker_substitution(
    tmp_path: Path,
) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(plan)
    tampered["compute_jobs"][0]["arguments"][1] = (
        "experiments.authz_mechinterp.compute_vpd_effect_shard"
    )
    job = tampered["compute_jobs"][0]
    job_body = copy.deepcopy(job)
    job_body.pop("job_id")
    job["job_id"] = content_sha256(job_body)
    body = copy.deepcopy(tampered)
    body.pop("campaign_plan_id")
    tampered["campaign_plan_id"] = content_sha256(body)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="not reproducible"):
        verify_campaign_plan_v2(path, **canonical_inputs(ROOT))
