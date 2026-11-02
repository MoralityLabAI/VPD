from __future__ import annotations

import argparse
import base64
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1 import (
    COMPUTE_ACTION,
    COMPUTE_RECEIPT_SCHEMA,
    REGISTRY_SCHEMA,
    RELEASE_ACTION,
    RELEASE_RECEIPT_SCHEMA,
    _compute_scope,
    _release_scope,
    build_dry_run_admission,
    sign_receipt,
    verify_compute_authorization_receipt,
    verify_scientific_release_receipt,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1 import (
    _synthetic_materialization,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1 import (
    build_execution_blueprint,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import crypto_types
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    SIMULATION_STATUS,
    validate_execution_blueprint_content,
)


AS_OF = "2030-01-01T00:30:00Z"
DEVICE = "synthetic-device-not-used"


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def _registry(
    private_key: Any,
    *,
    authority_id: str,
    action: str,
    synthetic: bool = True,
    enabled: bool = True,
) -> dict[str, Any]:
    _, _, serialization, _ = crypto_types()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    body = {
        "schema": REGISTRY_SCHEMA,
        "environment": "synthetic_benchmark_only" if synthetic else "production",
        "authorities": [
            {
                "authority_id": authority_id,
                "algorithm": "ed25519",
                "public_key_base64": base64.b64encode(public_bytes).decode("ascii"),
                "enabled": enabled,
                "synthetic_fixture_only": synthetic,
                "allowed_actions": [action],
                "maximum_scope": {
                    "planned_steps": 10_000,
                    "planned_activation_rows": 10_000_000,
                    "dictionary_features": 65_536,
                    "allowed_devices": [DEVICE],
                },
            }
        ],
        "revoked_receipt_ids": [],
        "consumed_nonces": [],
    }
    return {**body, "registry_id": content_sha256(body)}


def _readdress_registry(registry: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in registry.items() if key != "registry_id"}
    return {**body, "registry_id": content_sha256(body)}


def _release_body(
    *,
    blueprint: dict[str, Any],
    blueprint_sha256: str,
    synthetic: bool = True,
) -> dict[str, Any]:
    check = validate_execution_blueprint_content(blueprint)
    routing_ids = sorted(
        {
            str(identifier)
            for job in blueprint["execution_jobs"]
            for identifier in job["routing_receipt_ids"]
        }
    )
    stability_ids = sorted(
        {
            str(identifier)
            for job in blueprint["execution_jobs"]
            for identifier in job["routing_stability_ids"]
        }
    )
    return {
        "schema": RELEASE_RECEIPT_SCHEMA,
        "status": "synthetic_fixture_only" if synthetic else "authorized_by_user",
        "synthetic_fixture_only": synthetic,
        "authority_id": "synthetic-release-authority" if synthetic else "real-release-authority",
        "issued_at": "2030-01-01T00:00:00Z",
        "not_before": "2030-01-01T00:00:00Z",
        "expires_at": "2030-01-01T01:00:00Z",
        "nonce": "1" * 64,
        "explicit_release_statement": (
            "synthetic benchmark only; no scientific release or execution authority"
            if synthetic
            else "explicit scientific release for the exactly bound blueprint"
        ),
        "source": {
            "execution_blueprint_id": check["execution_blueprint_id"],
            "execution_blueprint_sha256": blueprint_sha256,
        },
        "scope": _release_scope(blueprint, check),
        "release_evidence": {
            "routing_receipt_ids": routing_ids,
            "routing_stability_ids": stability_ids,
            "exact_job_contracts_verified": True,
            "one_global_pair_split_verified": True,
            "independent_routing_stability_verified": True,
        },
    }


def _compute_body(
    *,
    blueprint: dict[str, Any],
    job: dict[str, Any],
    release_verification: dict[str, Any],
    synthetic: bool = True,
) -> dict[str, Any]:
    check = validate_execution_blueprint_content(blueprint)
    return {
        "schema": COMPUTE_RECEIPT_SCHEMA,
        "status": "synthetic_fixture_only" if synthetic else "authorized_by_user",
        "synthetic_fixture_only": synthetic,
        "authority_id": "synthetic-compute-authority" if synthetic else "real-compute-authority",
        "issued_at": "2030-01-01T00:00:00Z",
        "not_before": "2030-01-01T00:00:00Z",
        "expires_at": "2030-01-01T01:00:00Z",
        "nonce": "2" * 64,
        "explicit_compute_statement": (
            "synthetic benchmark only; no compute or SAE training authority"
            if synthetic
            else "explicit compute authorization for exactly one SAE job"
        ),
        "source": {
            "scientific_release_receipt_id": release_verification[
                "scientific_release_receipt_id"
            ],
            "scientific_release_receipt_sha256": release_verification[
                "scientific_release_receipt_sha256"
            ],
        },
        "scope": _compute_scope(blueprint_check=check, job=job, device=DEVICE),
    }


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    materialization = _synthetic_materialization(project_root)
    blueprint = build_execution_blueprint(materialization=materialization)
    blueprint_sha256 = content_sha256(blueprint)
    job = blueprint["execution_jobs"][0]
    Ed25519PrivateKey, _, _, _ = crypto_types()
    release_key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
    compute_key = Ed25519PrivateKey.from_private_bytes(bytes(range(33, 65)))
    release_registry = _registry(
        release_key,
        authority_id="synthetic-release-authority",
        action=RELEASE_ACTION,
    )
    compute_registry = _registry(
        compute_key,
        authority_id="synthetic-compute-authority",
        action=COMPUTE_ACTION,
    )
    release_body = _release_body(
        blueprint=blueprint, blueprint_sha256=blueprint_sha256
    )
    release_receipt = sign_receipt(release_body, release_key)
    release_verification = verify_scientific_release_receipt(
        receipt=release_receipt,
        registry=release_registry,
        blueprint=blueprint,
        blueprint_sha256=blueprint_sha256,
        as_of=AS_OF,
    )
    compute_body = _compute_body(
        blueprint=blueprint,
        job=job,
        release_verification=release_verification,
    )
    compute_receipt = sign_receipt(compute_body, compute_key)
    compute_verification = verify_compute_authorization_receipt(
        receipt=compute_receipt,
        registry=compute_registry,
        blueprint=blueprint,
        execution_job_id=job["execution_job_id"],
        device=DEVICE,
        release_verification=release_verification,
        as_of=AS_OF,
    )
    admission = build_dry_run_admission(
        blueprint=blueprint,
        blueprint_sha256=blueprint_sha256,
        execution_job_id=job["execution_job_id"],
        device=DEVICE,
        release_receipt=release_receipt,
        release_registry=release_registry,
        compute_receipt=compute_receipt,
        compute_registry=compute_registry,
        as_of=AS_OF,
    )

    bad_release_signature = copy.deepcopy(release_receipt)
    bad_release_signature["signature"]["value_base64"] = base64.b64encode(
        b"x" * 64
    ).decode("ascii")

    def signed_release_mutation(
        mutator: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        changed = copy.deepcopy(release_body)
        mutator(changed)
        return sign_receipt(changed, release_key)

    def signed_compute_mutation(
        mutator: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        changed = copy.deepcopy(compute_body)
        mutator(changed)
        return sign_receipt(changed, compute_key)

    wrong_blueprint_release = signed_release_mutation(
        lambda body: body["source"].__setitem__("execution_blueprint_id", "f" * 64)
    )
    wrong_evidence_release = signed_release_mutation(
        lambda body: body["release_evidence"].__setitem__(
            "routing_stability_ids", []
        )
    )
    expired_release = signed_release_mutation(
        lambda body: body.__setitem__("expires_at", "2030-01-01T00:15:00Z")
    )
    disabled_release_registry = _registry(
        release_key,
        authority_id="synthetic-release-authority",
        action=RELEASE_ACTION,
        enabled=False,
    )
    revoked_release_registry = copy.deepcopy(release_registry)
    revoked_release_registry["revoked_receipt_ids"] = [release_receipt["receipt_id"]]
    revoked_release_registry = _readdress_registry(revoked_release_registry)
    consumed_release_registry = copy.deepcopy(release_registry)
    consumed_release_registry["consumed_nonces"] = [release_receipt["nonce"]]
    consumed_release_registry = _readdress_registry(consumed_release_registry)

    wrong_job_compute = signed_compute_mutation(
        lambda body: body["scope"].__setitem__(
            "execution_job_id", blueprint["execution_jobs"][1]["execution_job_id"]
        )
    )
    wrong_release_compute = signed_compute_mutation(
        lambda body: body["source"].__setitem__(
            "scientific_release_receipt_id", "e" * 64
        )
    )
    broad_base_mutation_compute = signed_compute_mutation(
        lambda body: body["scope"].__setitem__("base_model_weights_mutable", True)
    )
    wrong_output_compute = signed_compute_mutation(
        lambda body: body["scope"].__setitem__("output_dir", "different/output")
    )
    wrong_steps_compute = signed_compute_mutation(
        lambda body: body["scope"].__setitem__(
            "planned_steps", int(body["scope"]["planned_steps"]) + 1
        )
    )
    bad_compute_signature = copy.deepcopy(compute_receipt)
    bad_compute_signature["signature"]["value_base64"] = base64.b64encode(
        b"y" * 64
    ).decode("ascii")
    consumed_compute_registry = copy.deepcopy(compute_registry)
    consumed_compute_registry["consumed_nonces"] = [compute_receipt["nonce"]]
    consumed_compute_registry = _readdress_registry(consumed_compute_registry)

    real_release_registry = _registry(
        release_key,
        authority_id="real-release-authority",
        action=RELEASE_ACTION,
        synthetic=False,
    )
    real_release_body = _release_body(
        blueprint=blueprint,
        blueprint_sha256=blueprint_sha256,
        synthetic=False,
    )
    real_release_receipt = sign_receipt(real_release_body, release_key)

    def verify_release(candidate: dict[str, Any], registry: dict[str, Any]) -> Any:
        return verify_scientific_release_receipt(
            receipt=candidate,
            registry=registry,
            blueprint=blueprint,
            blueprint_sha256=blueprint_sha256,
            as_of=AS_OF,
        )

    def verify_compute(candidate: dict[str, Any], registry: dict[str, Any]) -> Any:
        return verify_compute_authorization_receipt(
            receipt=candidate,
            registry=registry,
            blueprint=blueprint,
            execution_job_id=job["execution_job_id"],
            device=DEVICE,
            release_verification=release_verification,
            as_of=AS_OF,
        )

    controls = {
        "signed_synthetic_release_verifies_without_authority": release_verification[
            "status"
        ]
        == SIMULATION_STATUS
        and release_verification["signature_valid"] is True
        and release_verification["execution_authorized"] is False,
        "release_binds_exact_blueprint_plan_materialization_and_global_split": all(
            release_verification[field_name]
            == {
                "execution_blueprint_id": blueprint["execution_blueprint_id"],
                "materialization_id": blueprint["source"]["materialization_id"],
                "cluster_aware_sae_plan_id": blueprint["source"][
                    "cluster_aware_sae_plan_id"
                ],
                "global_pair_split_id": blueprint["source"]["global_pair_split_id"],
            }[field_name]
            for field_name in (
                "execution_blueprint_id",
                "materialization_id",
                "cluster_aware_sae_plan_id",
                "global_pair_split_id",
            )
        ),
        "signed_synthetic_compute_verifies_without_authority": compute_verification[
            "status"
        ]
        == SIMULATION_STATUS
        and compute_verification["signature_valid"] is True
        and compute_verification["execution_authorized"] is False,
        "compute_binds_one_exact_job_device_samples_resources_and_output": compute_receipt[
            "scope"
        ]
        == _compute_scope(
            blueprint_check=validate_execution_blueprint_content(blueprint),
            job=job,
            device=DEVICE,
        ),
        "dry_run_admission_is_content_addressed": admission["admission_id"]
        == content_sha256(
            {key: value for key, value in admission.items() if key != "admission_id"}
        ),
        "dry_run_never_reserves_or_attaches_or_invokes": admission[
            "runtime_transition"
        ]
        == {
            "durable_nonce_consumption_implemented": False,
            "output_reservation_implemented": False,
            "worker_ledger_reservation_performed": False,
            "runtime_attached": False,
            "trainer_callback_invoked": False,
        },
        "invalid_release_signature_rejected": _rejects(
            lambda: verify_release(bad_release_signature, release_registry)
        ),
        "wrong_release_blueprint_rejected": _rejects(
            lambda: verify_release(wrong_blueprint_release, release_registry)
        ),
        "wrong_release_evidence_rejected": _rejects(
            lambda: verify_release(wrong_evidence_release, release_registry)
        ),
        "expired_release_rejected": _rejects(
            lambda: verify_release(expired_release, release_registry)
        ),
        "disabled_release_authority_rejected": _rejects(
            lambda: verify_release(release_receipt, disabled_release_registry)
        ),
        "revoked_release_rejected": _rejects(
            lambda: verify_release(release_receipt, revoked_release_registry)
        ),
        "consumed_release_nonce_rejected": _rejects(
            lambda: verify_release(release_receipt, consumed_release_registry)
        ),
        "real_release_rejects_synthetic_blueprint": _rejects(
            lambda: verify_release(real_release_receipt, real_release_registry)
        ),
        "wrong_compute_job_rejected": _rejects(
            lambda: verify_compute(wrong_job_compute, compute_registry)
        ),
        "wrong_compute_release_rejected": _rejects(
            lambda: verify_compute(wrong_release_compute, compute_registry)
        ),
        "base_model_mutation_broadening_rejected": _rejects(
            lambda: verify_compute(broad_base_mutation_compute, compute_registry)
        ),
        "wrong_compute_output_rejected": _rejects(
            lambda: verify_compute(wrong_output_compute, compute_registry)
        ),
        "wrong_compute_steps_rejected": _rejects(
            lambda: verify_compute(wrong_steps_compute, compute_registry)
        ),
        "wrong_compute_device_rejected": _rejects(
            lambda: verify_compute_authorization_receipt(
                receipt=compute_receipt,
                registry=compute_registry,
                blueprint=blueprint,
                execution_job_id=job["execution_job_id"],
                device="different-device",
                release_verification=release_verification,
                as_of=AS_OF,
            )
        ),
        "invalid_compute_signature_rejected": _rejects(
            lambda: verify_compute(bad_compute_signature, compute_registry)
        ),
        "consumed_compute_nonce_rejected": _rejects(
            lambda: verify_compute(compute_receipt, consumed_compute_registry)
        ),
        "admission_imports_no_model_frameworks": before_frameworks
        == (MODEL_FRAMEWORKS & set(sys.modules)),
        "all_admission_paths_remain_nonexecuting": all(
            admission["scientific_boundary"][field_name] in (False, 0)
            for field_name in (
                "model_loaded",
                "model_inference_performed",
                "persistent_output_created",
                "weights_loaded_or_mutated",
                "trainer_callback_invoked",
                "execution_authorized",
                "sae_training_authorized",
                "real_actions_authorized",
            )
        ),
    }
    if not all(controls.values()):
        raise ValueError(
            "cluster-aware SAE admission benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-admission-benchmark.v1",
        "status": "pass",
        "source": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "admission_verifier_sha256": sha256_file(
                Path(__file__).with_name("admit_cluster_aware_sae_training_v1.py")
            ),
            "runtime_worker_sha256": sha256_file(
                Path(__file__).with_name("run_cluster_aware_sae_training_job_v1.py")
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "ed25519_release_and_compute_receipts": True,
            "release_binds_complete_blueprint": True,
            "compute_binds_exactly_one_job": True,
            "dry_run_cli_implemented": True,
            "durable_nonce_consumption_implemented": False,
            "output_reservation_implemented": False,
            "production_trainer_invocation_implemented": False,
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "execution_job_id": job["execution_job_id"],
            "scientific_release_receipt_id": release_receipt["receipt_id"],
            "compute_authorization_receipt_id": compute_receipt["receipt_id"],
            "dry_run_admission_id": admission["admission_id"],
        },
        "scientific_boundary": {
            "synthetic_keys_and_receipts_only": True,
            "private_key_persisted": False,
            "empirical_blueprint_present": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "persistent_output_created": False,
            "weights_loaded_or_mutated": False,
            "trainer_callback_invoked": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster-aware SAE admission benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("cluster-aware SAE admission benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster-aware SAE admission benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-aware-sae-admission-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": int(observed["summary"]["controls"]),
        "dry_run_cli_implemented": True,
        "durable_nonce_consumption_implemented": False,
        "output_reservation_implemented": False,
        "trainer_callback_invoked": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cluster-aware SAE signed admission V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
