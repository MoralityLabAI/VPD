from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.admit_compartment_sae_training_release_v2 import (
    COMPUTE_AUTHORIZATION_SCHEMA,
    verify_explicit_compute_authorization,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_compartment_sae_training_plan_v4 import (
    canonical_inputs as plan_v4_inputs,
    verify_training_plan_v4,
)
from experiments.authz_mechinterp.derive_compartment_sae_training_release_gate_v2 import (
    canonical_inputs as gate_v2_inputs,
    verify_release_gate_v2,
)
from experiments.authz_mechinterp.run_compartment_sae_capture_job_v2 import (
    upgrade_capture_manifest,
    validate_capture_execution,
)
from experiments.authz_mechinterp.run_compartment_sae_training_job_v2 import (
    upgrade_dictionary_manifest_v2,
    validate_dictionary_manifest_v2,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
GATE_PATH = (
    ROOT / "artifacts/vpd_compartment_sae_training_release_gate_v2/plan.json"
)
BLUEPRINT_PATH = (
    ROOT
    / "artifacts/vpd_compartment_sae_training_execution_blueprint_v2/blueprint.json"
)


def _inputs() -> tuple[dict, dict, dict]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    blueprint = json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))
    job = plan["training_jobs"][0]
    return plan, blueprint, job


def _release(job_id: str) -> tuple[dict, dict]:
    release = {
        "schema": "sch.vpd-compartment-sae-training-release-verification.v2",
        "status": "pass",
        "training_release_receipt_id": "a" * 64,
        "training_release_receipt_sha256": "b" * 64,
        "release_gate_plan_id": "c" * 64,
        "training_plan_id": "d" * 64,
        "compute_authorized": False,
    }
    compute = {
        "schema": "sch.vpd-explicit-sae-compute-authorization-verification.v1",
        "status": "pass",
        "compute_authorization_id": "e" * 64,
        "compute_authorization_sha256": "f" * 64,
        "required_job_id": job_id,
        "required_action": "compartment_sae_training",
    }
    return release, compute


def _predecessor_manifest(plan: dict, job: dict) -> dict:
    sample = job["required_sample_contract"]
    partition = sample["pair_atomic_partition"]
    body = {
        "schema": "sch.vpd-fixed-sae-dictionary-manifest.v1",
        "source": {
            "runner_sha256": plan["source"]["implementation"][
                "predecessor_sae_trainer"
            ]["sha256"]
        },
        "model": {"model_id": "synthetic/base", "adapter_sha256": None},
        "dictionary": {
            "activation_key": plan["profile"]["activation_key"],
            "input_dim": plan["profile"]["hidden_size"],
            "dict_dim": plan["profile"]["dictionary_features"],
            "top_k": plan["profile"]["top_k"],
            "expansion": plan["profile"]["expansion"],
            "steps": job["required_manifest_hyperparameters"]["steps"],
            "seed": job["initialization_seed"],
            "data_selection_seed": job["data_selection_seed"],
            "activation_compartment": {
                "activation_category": job["compartment"],
                "capacity_authorization": {
                    "selected_activation_rows": sample["selected_activation_rows"],
                    "selected_episodes": sample["selected_episodes"],
                    "selected_unique_pairs": sample["selected_unique_pairs"],
                },
            },
            "train_validation_split": {
                key: partition[key]
                for key in (
                    "training_pairs",
                    "validation_pairs",
                    "training_rows",
                    "validation_rows",
                    "training_pair_id_set_sha256",
                    "validation_pair_id_set_sha256",
                )
            },
        },
        "artifacts": {
            "dictionary": {"path": "sae_dictionary.npz", "sha256": "1" * 64}
        },
    }
    return {**body, "dictionary_id": content_sha256(body)}


def _upgrade() -> tuple[dict, dict, dict, dict]:
    plan, blueprint, job = _inputs()
    release, compute = _release(str(job["job_id"]))
    predecessor = _predecessor_manifest(plan, job)
    upgraded = upgrade_dictionary_manifest_v2(
        predecessor_manifest=predecessor,
        predecessor_manifest_sha256="2" * 64,
        plan=plan,
        training_plan_path=PLAN_PATH,
        job=job,
        blueprint=blueprint,
        execution_blueprint_path=BLUEPRINT_PATH,
        release_verification=release,
        compute_authorization_verification=compute,
    )
    return plan, blueprint, job, upgraded


def _readdress(value: dict, id_field: str) -> dict:
    body = {key: item for key, item in value.items() if key != id_field}
    return {**body, id_field: content_sha256(body)}


def test_manifest_v2_upgrade_and_independent_validation() -> None:
    plan, blueprint, job, upgraded = _upgrade()
    verification = validate_dictionary_manifest_v2(
        manifest=upgraded,
        plan=plan,
        training_plan_path=PLAN_PATH,
        job=job,
        blueprint=blueprint,
        execution_blueprint_path=BLUEPRINT_PATH,
    )
    assert verification["status"] == "pass"
    assert upgraded["dictionary"]["batch_size"] == 128
    assert upgraded["dictionary"]["learning_rate"] == 0.0003
    assert upgraded["dictionary"]["exact_sample_contract"] == job[
        "required_sample_contract"
    ]


def test_readdressed_manifest_cannot_change_bound_batch_size() -> None:
    plan, blueprint, job, upgraded = _upgrade()
    changed = copy.deepcopy(upgraded)
    changed["dictionary"]["batch_size"] = 64
    changed = _readdress(changed, "dictionary_id")
    with pytest.raises(ValueError, match="execution bindings"):
        validate_dictionary_manifest_v2(
            manifest=changed,
            plan=plan,
            training_plan_path=PLAN_PATH,
            job=job,
            blueprint=blueprint,
            execution_blueprint_path=BLUEPRINT_PATH,
        )


def test_upgrade_requires_job_scoped_compute_authorization() -> None:
    plan, blueprint, job = _inputs()
    release, compute = _release("0" * 64)
    with pytest.raises(ValueError, match="does not cover"):
        upgrade_dictionary_manifest_v2(
            predecessor_manifest=_predecessor_manifest(plan, job),
            predecessor_manifest_sha256="2" * 64,
            plan=plan,
            training_plan_path=PLAN_PATH,
            job=job,
            blueprint=blueprint,
            execution_blueprint_path=BLUEPRINT_PATH,
            release_verification=release,
            compute_authorization_verification=compute,
        )


def test_capture_execution_receipt_binds_both_authorizations() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    job = plan["capture_jobs"][0]
    release, compute = _release(str(job["job_id"]))
    compute["required_action"] = "base_activation_capture"
    capture_body = {
        "schema": "sch.vpd-activation-capture-manifest.v1",
        "source": {
            "runner_sha256": plan["source"]["implementation"][
                "predecessor_capture_worker"
            ]["sha256"]
        },
        "capture": {"rows": 8, "generated_tokens": 0},
        "artifacts": {},
    }
    predecessor = {**capture_body, "capture_id": content_sha256(capture_body)}
    upgraded = upgrade_capture_manifest(
        predecessor_manifest=predecessor,
        plan=plan,
        training_plan_path=PLAN_PATH,
        job=job,
        release_verification=release,
        compute_verification=compute,
    )
    verification = validate_capture_execution(
        manifest=upgraded,
        plan=plan,
        training_plan_path=PLAN_PATH,
        job=job,
    )
    assert verification["status"] == "pass"
    assert verification["training_release_receipt_id"] == "a" * 64
    assert verification["compute_authorization_id"] == "e" * 64


def test_external_compute_authorization_is_exactly_job_scoped(tmp_path: Path) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    job = plan["training_jobs"][0]
    release, _ = _release(str(job["job_id"]))
    body = {
        "schema": COMPUTE_AUTHORIZATION_SCHEMA,
        "status": "authorized_by_user",
        "source": {
            "training_plan_id": plan["training_plan_id"],
            "training_plan_sha256": sha256_file(PLAN_PATH),
            "release_gate_plan_id": gate["release_gate_plan_id"],
            "release_gate_plan_sha256": sha256_file(GATE_PATH),
            "training_release_receipt_id": release[
                "training_release_receipt_id"
            ],
            "training_release_receipt_sha256": release[
                "training_release_receipt_sha256"
            ],
        },
        "scope": {
            "authorized_actions": ["compartment_sae_training"],
            "authorized_job_ids": [job["job_id"]],
            "real_actions_authorized": False,
            "base_model_weights_mutable": False,
            "sae_dictionary_weight_training_authorized": True,
        },
    }
    authorization = {**body, "compute_authorization_id": content_sha256(body)}
    path = tmp_path / "authorization.json"
    write_json(path, authorization)
    verification = verify_explicit_compute_authorization(
        path,
        release_verification=release,
        release_gate_plan_path=GATE_PATH,
        training_plan_path=PLAN_PATH,
        required_job_id=str(job["job_id"]),
        required_action="compartment_sae_training",
    )
    assert verification["status"] == "pass"
    with pytest.raises(ValueError, match="scope is insufficient"):
        verify_explicit_compute_authorization(
            path,
            release_verification=release,
            release_gate_plan_path=GATE_PATH,
            training_plan_path=PLAN_PATH,
            required_job_id=str(plan["training_jobs"][1]["job_id"]),
            required_action="compartment_sae_training",
        )


def test_plan_v4_and_release_gate_v2_reproduce() -> None:
    plan_verification = verify_training_plan_v4(
        PLAN_PATH, **plan_v4_inputs(ROOT)
    )
    gate_verification = verify_release_gate_v2(
        GATE_PATH, **gate_v2_inputs(ROOT)
    )
    assert plan_verification["guarded_jobs"] == 22
    assert gate_verification["guarded_jobs"] == 22
    assert gate_verification["compute_authorized"] is False
