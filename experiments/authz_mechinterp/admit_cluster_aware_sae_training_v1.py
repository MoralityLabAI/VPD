from __future__ import annotations

import argparse
import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    canonical_bytes,
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import crypto_types
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    REQUIRED_ACTION,
    SIMULATION_COMPUTE_SCHEMA,
    SIMULATION_RELEASE_SCHEMA,
    SIMULATION_STATUS,
    select_execution_job,
    validate_execution_blueprint_content,
)


REGISTRY_SCHEMA = "sch.vpd-cluster-aware-sae-authority-registry.v1"
RELEASE_RECEIPT_SCHEMA = "sch.vpd-cluster-aware-sae-scientific-release-receipt.v1"
COMPUTE_RECEIPT_SCHEMA = "sch.vpd-cluster-aware-sae-compute-authorization-receipt.v1"
ADMISSION_SCHEMA = "sch.vpd-cluster-aware-sae-runtime-admission.v1"
RELEASE_ACTION = "release_cluster_aware_sae_dictionary_training"
COMPUTE_ACTION = REQUIRED_ACTION
DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _parse_utc(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field_name} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field_name} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{field_name} is not UTC")
    return parsed


def _content_id(value: Any, *, field_name: str) -> str:
    observed = str(value)
    if DIGEST_RE.fullmatch(observed) is None:
        raise ValueError(f"{field_name} is not a lowercase SHA-256 content ID")
    return observed


def receipt_body(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_id", "signature"}
    }


def sign_receipt(body: dict[str, Any], private_key: Any) -> dict[str, Any]:
    if body.get("schema") not in {RELEASE_RECEIPT_SCHEMA, COMPUTE_RECEIPT_SCHEMA}:
        raise ValueError("cluster-aware SAE receipt body schema differs")
    if "receipt_id" in body or "signature" in body:
        raise ValueError("cluster-aware SAE receipt body is already addressed")
    identifier = content_sha256(body)
    signature = private_key.sign(canonical_bytes(body))
    return {
        **body,
        "receipt_id": identifier,
        "signature": {
            "algorithm": "ed25519",
            "value_base64": base64.b64encode(signature).decode("ascii"),
        },
    }


def _verify_registry(registry: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("cluster-aware SAE authority registry schema differs")
    body = dict(registry)
    identifier = _content_id(body.pop("registry_id", ""), field_name="registry ID")
    if identifier != content_sha256(body):
        raise ValueError("cluster-aware SAE authority registry address differs")
    authorities = registry.get("authorities")
    if not isinstance(authorities, list) or not authorities:
        raise ValueError("cluster-aware SAE authority registry is empty")
    by_id = {str(row.get("authority_id")): row for row in authorities}
    if len(by_id) != len(authorities):
        raise ValueError("cluster-aware SAE authority registry repeats an authority")
    return identifier, by_id


def _verify_signed_receipt(
    *,
    receipt: dict[str, Any],
    registry: dict[str, Any],
    expected_schema: str,
    expected_action: str,
    as_of: str,
) -> dict[str, Any]:
    if receipt.get("schema") != expected_schema:
        raise ValueError("cluster-aware SAE receipt schema differs")
    receipt_id = _content_id(receipt.get("receipt_id", ""), field_name="receipt ID")
    if receipt_id != content_sha256(receipt_body(receipt)):
        raise ValueError("cluster-aware SAE receipt address differs")
    registry_id, authorities = _verify_registry(registry)
    synthetic = bool(receipt.get("synthetic_fixture_only"))
    expected_status = "synthetic_fixture_only" if synthetic else "authorized_by_user"
    if receipt.get("status") != expected_status:
        raise ValueError("cluster-aware SAE receipt status differs")
    issued = _parse_utc(receipt.get("issued_at"), field_name="issued_at")
    not_before = _parse_utc(receipt.get("not_before"), field_name="not_before")
    expires = _parse_utc(receipt.get("expires_at"), field_name="expires_at")
    observed_time = _parse_utc(as_of, field_name="as_of")
    if not issued <= not_before < expires or not not_before <= observed_time < expires:
        raise ValueError("cluster-aware SAE receipt is inactive or expired")
    nonce = _content_id(receipt.get("nonce", ""), field_name="receipt nonce")
    if receipt_id in registry.get("revoked_receipt_ids", []):
        raise ValueError("cluster-aware SAE receipt is revoked")
    if nonce in registry.get("consumed_nonces", []):
        raise ValueError("cluster-aware SAE receipt nonce was already consumed")
    authority_id = str(receipt.get("authority_id", ""))
    authority = authorities.get(authority_id)
    if authority is None or authority.get("enabled") is not True:
        raise ValueError("cluster-aware SAE receipt authority is absent or disabled")
    if (
        authority.get("algorithm") != "ed25519"
        or bool(authority.get("synthetic_fixture_only")) != synthetic
        or expected_action not in set(authority.get("allowed_actions", []))
        or receipt.get("scope", {}).get("authorized_action") != expected_action
    ):
        raise ValueError("cluster-aware SAE receipt authority scope differs")
    signature = receipt.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        raise ValueError("cluster-aware SAE receipt signature is absent")
    try:
        signature_bytes = base64.b64decode(
            str(signature["value_base64"]), validate=True
        )
        public_bytes = base64.b64decode(
            str(authority["public_key_base64"]), validate=True
        )
        if len(public_bytes) != 32:
            raise ValueError("cluster-aware SAE public key length differs")
        _, Ed25519PublicKey, _, _ = crypto_types()
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(
            signature_bytes, canonical_bytes(receipt_body(receipt))
        )
    except Exception as exc:
        raise ValueError("cluster-aware SAE receipt signature verification failed") from exc
    return {
        "receipt_id": receipt_id,
        "registry_id": registry_id,
        "authority_id": authority_id,
        "nonce": nonce,
        "synthetic_fixture_only": synthetic,
        "signature_valid": True,
        "active_at_as_of": True,
        "nonce_unconsumed": True,
    }


def _release_scope(
    blueprint: dict[str, Any], blueprint_check: dict[str, Any]
) -> dict[str, Any]:
    return {
        "authorized_action": RELEASE_ACTION,
        "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
        "materialization_id": blueprint_check["materialization_id"],
        "cluster_aware_sae_plan_id": blueprint_check["cluster_aware_sae_plan_id"],
        "global_pair_split_id": blueprint_check["global_pair_split_id"],
        "evidence_kind": blueprint_check["evidence_kind"],
        "execution_job_ids": sorted(
            str(job["execution_job_id"]) for job in blueprint["execution_jobs"]
        ),
        "exact_cluster_routed_samples_required": True,
        "one_global_pair_split_required": True,
        "routing_and_stability_receipts_required": True,
        "compute_authorized": False,
        "base_model_weight_mutation_authorized": False,
        "real_actions_authorized": False,
    }


def verify_scientific_release_receipt(
    *,
    receipt: dict[str, Any],
    registry: dict[str, Any],
    blueprint: dict[str, Any],
    blueprint_sha256: str,
    as_of: str,
) -> dict[str, Any]:
    blueprint_check = validate_execution_blueprint_content(blueprint)
    signed = _verify_signed_receipt(
        receipt=receipt,
        registry=registry,
        expected_schema=RELEASE_RECEIPT_SCHEMA,
        expected_action=RELEASE_ACTION,
        as_of=as_of,
    )
    synthetic = signed["synthetic_fixture_only"]
    if synthetic != (blueprint_check["evidence_kind"] == "synthetic_benchmark"):
        raise ValueError("cluster-aware SAE release evidence environment differs")
    if not synthetic and blueprint_check["evidence_kind"] != "empirical_guarded_v3_capture":
        raise ValueError("real cluster-aware SAE release requires empirical evidence")
    source = receipt.get("source", {})
    if (
        source.get("execution_blueprint_id")
        != blueprint_check["execution_blueprint_id"]
        or source.get("execution_blueprint_sha256") != blueprint_sha256
        or receipt.get("scope") != _release_scope(blueprint, blueprint_check)
    ):
        raise ValueError("cluster-aware SAE release blueprint scope differs")
    evidence = receipt.get("release_evidence", {})
    expected_routing_ids = sorted(
        {
            str(identifier)
            for job in blueprint["execution_jobs"]
            for identifier in job["routing_receipt_ids"]
        }
    )
    expected_stability_ids = sorted(
        {
            str(identifier)
            for job in blueprint["execution_jobs"]
            for identifier in job["routing_stability_ids"]
        }
    )
    if (
        evidence.get("routing_receipt_ids") != expected_routing_ids
        or evidence.get("routing_stability_ids") != expected_stability_ids
        or evidence.get("exact_job_contracts_verified") is not True
        or evidence.get("one_global_pair_split_verified") is not True
        or evidence.get("independent_routing_stability_verified") is not True
    ):
        raise ValueError("cluster-aware SAE scientific release evidence differs")
    return {
        "schema": SIMULATION_RELEASE_SCHEMA,
        "status": SIMULATION_STATUS if synthetic else "pass",
        "scientific_release_receipt_id": signed["receipt_id"],
        "scientific_release_receipt_sha256": _content_id(
            content_sha256(receipt), field_name="scientific release receipt SHA-256"
        ),
        "authority_registry_id": signed["registry_id"],
        "authority_id": signed["authority_id"],
        "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
        "materialization_id": blueprint_check["materialization_id"],
        "cluster_aware_sae_plan_id": blueprint_check["cluster_aware_sae_plan_id"],
        "global_pair_split_id": blueprint_check["global_pair_split_id"],
        "signature_valid": True,
        "active_at_as_of": True,
        "nonce_unconsumed": True,
        "synthetic_fixture_only": synthetic,
        "compute_authorized": False,
        "execution_authorized": False,
    }


def _compute_scope(
    *, blueprint_check: dict[str, Any], job: dict[str, Any], device: str
) -> dict[str, Any]:
    return {
        "authorized_action": COMPUTE_ACTION,
        "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
        "execution_job_id": job["execution_job_id"],
        "source_training_job_id": job["source_training_job_id"],
        "compartment_id": job["compartment_id"],
        "compartment_contract_id": job["compartment_contract_id"],
        "exact_sample_contract_sha256": content_sha256(
            job["exact_sample_contract"]
        ),
        "global_pair_split_id": job["global_pair_split_id"],
        "planned_steps": int(job["profile"]["steps"]),
        "planned_activation_rows": int(
            job["exact_sample_contract"]["sae_activation_rows"]
        ),
        "dictionary_features": int(job["dictionary_features"]),
        "initialization_seed": int(job["initialization_seed"]),
        "data_selection_seed": int(job["data_selection_seed"]),
        "output_dir": str(job["output_dir"]),
        "device": device,
        "base_model_weights_mutable": False,
        "real_actions_authorized": False,
        "sae_dictionary_weight_training_authorized": True,
    }


def verify_compute_authorization_receipt(
    *,
    receipt: dict[str, Any],
    registry: dict[str, Any],
    blueprint: dict[str, Any],
    execution_job_id: str,
    device: str,
    release_verification: dict[str, Any],
    as_of: str,
) -> dict[str, Any]:
    blueprint_check = validate_execution_blueprint_content(blueprint)
    job = select_execution_job(blueprint, execution_job_id=execution_job_id)
    signed = _verify_signed_receipt(
        receipt=receipt,
        registry=registry,
        expected_schema=COMPUTE_RECEIPT_SCHEMA,
        expected_action=COMPUTE_ACTION,
        as_of=as_of,
    )
    synthetic = signed["synthetic_fixture_only"]
    if synthetic != bool(release_verification.get("synthetic_fixture_only")):
        raise ValueError("cluster-aware SAE compute and release environments differ")
    expected_release_status = SIMULATION_STATUS if synthetic else "pass"
    if (
        release_verification.get("schema") != SIMULATION_RELEASE_SCHEMA
        or release_verification.get("status") != expected_release_status
        or release_verification.get("execution_blueprint_id")
        != blueprint_check["execution_blueprint_id"]
    ):
        raise ValueError("cluster-aware SAE compute release verification differs")
    source = receipt.get("source", {})
    if (
        source.get("scientific_release_receipt_id")
        != release_verification["scientific_release_receipt_id"]
        or source.get("scientific_release_receipt_sha256")
        != release_verification["scientific_release_receipt_sha256"]
        or receipt.get("scope")
        != _compute_scope(blueprint_check=blueprint_check, job=job, device=device)
    ):
        raise ValueError("cluster-aware SAE compute authorization scope differs")
    maximum_scope = {
        str(row.get("authority_id")): row.get("maximum_scope", {})
        for row in registry.get("authorities", [])
    }.get(signed["authority_id"], {})
    scope = receipt["scope"]
    for field_name in (
        "planned_steps",
        "planned_activation_rows",
        "dictionary_features",
    ):
        if type(maximum_scope.get(field_name)) is not int or scope[field_name] > int(
            maximum_scope[field_name]
        ):
            raise ValueError(
                f"cluster-aware SAE compute exceeds authority ceiling: {field_name}"
            )
    allowed_devices = maximum_scope.get("allowed_devices")
    if not isinstance(allowed_devices, list) or device not in allowed_devices:
        raise ValueError("cluster-aware SAE compute device is outside authority scope")
    return {
        "schema": SIMULATION_COMPUTE_SCHEMA,
        "status": SIMULATION_STATUS if synthetic else "pass",
        "compute_authorization_id": signed["receipt_id"],
        "compute_authorization_sha256": _content_id(
            content_sha256(receipt), field_name="compute authorization SHA-256"
        ),
        "authority_registry_id": signed["registry_id"],
        "authority_id": signed["authority_id"],
        "scientific_release_receipt_id": release_verification[
            "scientific_release_receipt_id"
        ],
        "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
        "required_job_id": job["execution_job_id"],
        "required_action": COMPUTE_ACTION,
        "device": device,
        "signature_valid": True,
        "active_at_as_of": True,
        "nonce_unconsumed": True,
        "synthetic_fixture_only": synthetic,
        "base_model_weights_mutable": False,
        "real_actions_authorized": False,
        "sae_dictionary_weight_training_authorized": not synthetic,
        "execution_authorized": not synthetic,
    }


def build_dry_run_admission(
    *,
    blueprint: dict[str, Any],
    blueprint_sha256: str,
    execution_job_id: str,
    device: str,
    release_receipt: dict[str, Any],
    release_registry: dict[str, Any],
    compute_receipt: dict[str, Any],
    compute_registry: dict[str, Any],
    as_of: str,
) -> dict[str, Any]:
    blueprint_check = validate_execution_blueprint_content(blueprint)
    job = select_execution_job(blueprint, execution_job_id=execution_job_id)
    release = verify_scientific_release_receipt(
        receipt=release_receipt,
        registry=release_registry,
        blueprint=blueprint,
        blueprint_sha256=blueprint_sha256,
        as_of=as_of,
    )
    compute = verify_compute_authorization_receipt(
        receipt=compute_receipt,
        registry=compute_registry,
        blueprint=blueprint,
        execution_job_id=execution_job_id,
        device=device,
        release_verification=release,
        as_of=as_of,
    )
    synthetic = bool(release["synthetic_fixture_only"])
    body = {
        "schema": ADMISSION_SCHEMA,
        "status": (
            "blocked_authenticated_synthetic_fixture"
            if synthetic
            else "readonly_checks_pass_runtime_not_reserved_or_attached"
        ),
        "as_of": as_of,
        "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
        "execution_job_id": job["execution_job_id"],
        "scientific_release_verification": release,
        "compute_authorization_verification": compute,
        "exact_job_binding": {
            "compartment_id": job["compartment_id"],
            "exact_sample_contract_sha256": content_sha256(
                job["exact_sample_contract"]
            ),
            "global_pair_split_id": job["global_pair_split_id"],
            "output_dir": job["output_dir"],
            "device": device,
        },
        "runtime_transition": {
            "durable_nonce_consumption_implemented": False,
            "output_reservation_implemented": False,
            "worker_ledger_reservation_performed": False,
            "runtime_attached": False,
            "trainer_callback_invoked": False,
        },
        "stop_reason": (
            "synthetic_receipts_cannot_authorize_execution"
            if synthetic
            else "dry_run_only_durable_nonce_and_output_reservation_not_performed"
        ),
        "scientific_boundary": {
            "read_only_dry_run": True,
            "synthetic_fixture_only": synthetic,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "persistent_output_created": False,
            "weights_loaded_or_mutated": False,
            "trainer_callback_invoked": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "real_actions_authorized": False,
        },
    }
    return {**body, "admission_id": content_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only cluster-aware SAE release/compute admission; never "
            "reserve output, attach a runtime, or invoke training."
        )
    )
    parser.add_argument("--execution-blueprint", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--scientific-release-receipt", type=Path, required=True)
    parser.add_argument("--scientific-release-registry", type=Path, required=True)
    parser.add_argument("--compute-authorization-receipt", type=Path, required=True)
    parser.add_argument("--compute-authority-registry", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blueprint = json.loads(args.execution_blueprint.read_text(encoding="utf-8"))
    result = build_dry_run_admission(
        blueprint=blueprint,
        blueprint_sha256=sha256_file(args.execution_blueprint),
        execution_job_id=args.job_id,
        device=args.device,
        release_receipt=json.loads(
            args.scientific_release_receipt.read_text(encoding="utf-8")
        ),
        release_registry=json.loads(
            args.scientific_release_registry.read_text(encoding="utf-8")
        ),
        compute_receipt=json.loads(
            args.compute_authorization_receipt.read_text(encoding="utf-8")
        ),
        compute_registry=json.loads(
            args.compute_authority_registry.read_text(encoding="utf-8")
        ),
        as_of=args.as_of,
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
