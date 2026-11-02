from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v1 import (
    canonical_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_PATH = ROOT / "artifacts/vpd_cheap_executor_runbook_v1/runbook.json"


def _load() -> dict:
    return json.loads(RUNBOOK_PATH.read_text(encoding="utf-8"))


def test_runbook_reproduces() -> None:
    verification = verify_runbook(RUNBOOK_PATH, **canonical_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["stages"] == 8
    assert verification["guarded_jobs"] == 22
    assert verification["execution_authorized"] is False


def test_only_local_preflight_is_safe_now() -> None:
    runbook = _load()
    safe = [stage["id"] for stage in runbook["stages"] if stage["safe_for_cheap_executor_now"]]
    assert safe == ["local_preflight"]
    assert runbook["executor_policy"]["never_generate_external_compute_authorization"]
    assert runbook["scientific_boundary"]["execution_authorized"] is False


def test_paid_schedule_covers_exact_plan_with_canary_first() -> None:
    runbook = _load()
    stages = {stage["id"]: stage for stage in runbook["stages"]}
    canary = stages["canary_dictionary"]["job"]
    remaining = stages["remaining_dictionaries"]["jobs"]
    assert canary["compartment"] == "other_invalid"
    assert canary["initialization_seed"] == 17
    job_ids = [canary["job_id"], *(job["job_id"] for job in remaining)]
    assert len(job_ids) == 18
    assert len(set(job_ids)) == 18
    assert stages["dictionary_bundle_admission"]["job"][
        "depends_on_training_job_ids"
    ] == sorted(job_ids)
