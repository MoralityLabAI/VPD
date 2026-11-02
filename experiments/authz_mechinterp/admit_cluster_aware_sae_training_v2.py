from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp import admit_cluster_aware_sae_training_v1 as v1
from experiments.authz_mechinterp.build_cluster_aware_sae_blueprint_budget_admission_v2 import (
    build_blueprint_bound_proposal,
    evaluate_blueprint_bound_budget_proposal,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1 import (
    validate_budget_policy_content,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    canonical_bytes,
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    SIMULATION_COMPUTE_SCHEMA,
    SIMULATION_RELEASE_SCHEMA,
    SIMULATION_STATUS,
    select_execution_job,
    validate_execution_blueprint_content,
)


RELEASE_RECEIPT_SCHEMA = (
    "sch.vpd-cluster-aware-sae-budget-bound-scientific-release-receipt.v2"
)
RELEASE_VERIFICATION_SCHEMA = (
    "sch.vpd-cluster-aware-sae-budget-bound-scientific-release-verification.v2"
)
RUNTIME_ADMISSION_SCHEMA = (
    "sch.vpd-cluster-aware-sae-budget-bound-runtime-admission.v2"
)
RUNTIME_VERIFICATION_SCHEMA = (
    "sch.vpd-cluster-aware-sae-budget-bound-runtime-admission-verification.v2"
)
DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"budget-bound release {field} is not a SHA-256 digest")
    return value


def release_receipt_body(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_id", "signature"}
    }


def sign_scientific_release_receipt_v2(
    body: dict[str, Any], private_key: Any
) -> dict[str, Any]:
    if body.get("schema") != RELEASE_RECEIPT_SCHEMA:
        raise ValueError("budget-bound scientific release body schema differs")
    if "receipt_id" in body or "signature" in body:
        raise ValueError("budget-bound scientific release body is already addressed")
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


def _reproduce_budget_chain(
    *,
    policy: dict[str, Any],
    proposal: dict[str, Any],
    admission: dict[str, Any],
    blueprint: dict[str, Any],
    blueprint_sha256: str,
) -> dict[str, Any]:
    validate_budget_policy_content(policy)
    blueprint_file_sha256 = _digest(
        blueprint_sha256, field="execution blueprint file SHA-256"
    )
    expected_proposal = build_blueprint_bound_proposal(
        policy=policy,
        blueprint=blueprint,
        blueprint_sha256=blueprint_file_sha256,
    )
    if proposal != expected_proposal:
        raise ValueError("budget-bound scientific release proposal is not reproducible")
    expected_admission = evaluate_blueprint_bound_budget_proposal(
        policy=policy,
        blueprint=blueprint,
        blueprint_sha256=blueprint_file_sha256,
        proposal=proposal,
    )
    if admission != expected_admission:
        raise ValueError("budget-bound scientific release admission is not reproducible")
    if (
        admission.get("budget_admissible") is not True
        or admission.get("blueprint_binding_verified") is not True
        or admission.get("scientific_boundary", {}).get(
            "budget_admission_is_execution_authority"
        )
        is not False
    ):
        raise ValueError("budget-bound scientific release admission boundary differs")
    return {
        "budget_policy_id": _digest(
            policy.get("budget_policy_id"), field="budget policy ID"
        ),
        "budget_proposal_id": _digest(
            proposal.get("proposal_id"), field="budget proposal ID"
        ),
        "budget_admission_id": _digest(
            admission.get("admission_id"), field="budget admission ID"
        ),
        "execution_blueprint_id": _digest(
            admission.get("source", {}).get("execution_blueprint_id"),
            field="execution blueprint ID",
        ),
        "execution_blueprint_sha256": blueprint_file_sha256,
        "execution_jobs": int(admission["blueprint_shape"]["execution_jobs"]),
        "admitted_shape_sha256": content_sha256(admission["admitted_shape"]),
    }


def expected_release_scope_v2(
    *,
    blueprint: dict[str, Any],
    blueprint_check: dict[str, Any],
    budget_check: dict[str, Any],
    policy_sha256: str,
    proposal_sha256: str,
    admission_sha256: str,
) -> dict[str, Any]:
    return {
        **v1._release_scope(blueprint, blueprint_check),
        "budget_policy_id": budget_check["budget_policy_id"],
        "budget_policy_sha256": _digest(
            policy_sha256, field="budget policy file SHA-256"
        ),
        "budget_proposal_id": budget_check["budget_proposal_id"],
        "budget_proposal_sha256": _digest(
            proposal_sha256, field="budget proposal file SHA-256"
        ),
        "budget_admission_id": budget_check["budget_admission_id"],
        "budget_admission_sha256": _digest(
            admission_sha256, field="budget admission file SHA-256"
        ),
        "budget_admitted_shape_sha256": budget_check["admitted_shape_sha256"],
        "budget_admissible": True,
        "budget_admission_is_execution_authority": False,
    }


def _expected_release_evidence(blueprint: dict[str, Any]) -> dict[str, Any]:
    return {
        "routing_receipt_ids": sorted(
            {
                str(identifier)
                for job in blueprint["execution_jobs"]
                for identifier in job["routing_receipt_ids"]
            }
        ),
        "routing_stability_ids": sorted(
            {
                str(identifier)
                for job in blueprint["execution_jobs"]
                for identifier in job["routing_stability_ids"]
            }
        ),
        "exact_job_contracts_verified": True,
        "one_global_pair_split_verified": True,
        "independent_routing_stability_verified": True,
        "blueprint_bound_budget_proposal_reproduced": True,
        "blueprint_bound_budget_admission_reproduced": True,
    }


def verify_scientific_release_receipt_content_v2(
    *,
    receipt: dict[str, Any],
    registry: dict[str, Any],
    blueprint: dict[str, Any],
    blueprint_sha256: str,
    policy: dict[str, Any],
    policy_sha256: str,
    proposal: dict[str, Any],
    proposal_sha256: str,
    admission: dict[str, Any],
    admission_sha256: str,
    as_of: str,
) -> dict[str, Any]:
    blueprint_check = validate_execution_blueprint_content(blueprint)
    budget_check = _reproduce_budget_chain(
        policy=policy,
        proposal=proposal,
        admission=admission,
        blueprint=blueprint,
        blueprint_sha256=blueprint_sha256,
    )
    signed = v1._verify_signed_receipt(
        receipt=receipt,
        registry=registry,
        expected_schema=RELEASE_RECEIPT_SCHEMA,
        expected_action=v1.RELEASE_ACTION,
        as_of=as_of,
    )
    synthetic = bool(signed["synthetic_fixture_only"])
    evidence_kind = str(blueprint_check["evidence_kind"])
    if synthetic != (evidence_kind == "synthetic_benchmark"):
        raise ValueError("budget-bound release evidence environment differs")
    if not synthetic and evidence_kind != "empirical_guarded_v3_capture":
        raise ValueError("real budget-bound release requires empirical evidence")
    expected_scope = expected_release_scope_v2(
        blueprint=blueprint,
        blueprint_check=blueprint_check,
        budget_check=budget_check,
        policy_sha256=policy_sha256,
        proposal_sha256=proposal_sha256,
        admission_sha256=admission_sha256,
    )
    expected_source = {
        "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
        "execution_blueprint_sha256": _digest(
            blueprint_sha256, field="execution blueprint file SHA-256"
        ),
        "budget_policy_id": budget_check["budget_policy_id"],
        "budget_policy_sha256": _digest(
            policy_sha256, field="budget policy file SHA-256"
        ),
        "budget_proposal_id": budget_check["budget_proposal_id"],
        "budget_proposal_sha256": _digest(
            proposal_sha256, field="budget proposal file SHA-256"
        ),
        "budget_admission_id": budget_check["budget_admission_id"],
        "budget_admission_sha256": _digest(
            admission_sha256, field="budget admission file SHA-256"
        ),
    }
    if receipt.get("source") != expected_source:
        raise ValueError("budget-bound scientific release source differs")
    if receipt.get("scope") != expected_scope:
        raise ValueError("budget-bound scientific release scope differs")
    if receipt.get("release_evidence") != _expected_release_evidence(blueprint):
        raise ValueError("budget-bound scientific release evidence differs")
    return {
        "schema": RELEASE_VERIFICATION_SCHEMA,
        "status": SIMULATION_STATUS if synthetic else "pass",
        "scientific_release_receipt_id": signed["receipt_id"],
        "scientific_release_receipt_sha256": content_sha256(receipt),
        "authority_registry_id": signed["registry_id"],
        "authority_id": signed["authority_id"],
        "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
        "execution_blueprint_sha256": expected_source[
            "execution_blueprint_sha256"
        ],
        "materialization_id": blueprint_check["materialization_id"],
        "cluster_aware_sae_plan_id": blueprint_check[
            "cluster_aware_sae_plan_id"
        ],
        "global_pair_split_id": blueprint_check["global_pair_split_id"],
        "budget_policy_id": budget_check["budget_policy_id"],
        "budget_policy_sha256": expected_source["budget_policy_sha256"],
        "budget_proposal_id": budget_check["budget_proposal_id"],
        "budget_proposal_sha256": expected_source["budget_proposal_sha256"],
        "budget_admission_id": budget_check["budget_admission_id"],
        "budget_admission_sha256": expected_source["budget_admission_sha256"],
        "admitted_shape_sha256": budget_check["admitted_shape_sha256"],
        "execution_jobs": budget_check["execution_jobs"],
        "signature_valid": True,
        "authority_enabled": True,
        "active_at_as_of": True,
        "not_revoked": True,
        "nonce_unconsumed": True,
        "budget_proposal_reproduced": True,
        "budget_admission_reproduced": True,
        "synthetic_fixture_only": synthetic,
        "compute_authorized": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def verify_scientific_release_receipt_v2(
    path: Path,
    *,
    registry_path: Path,
    blueprint_path: Path,
    policy_path: Path,
    proposal_path: Path,
    admission_path: Path,
    as_of: str,
) -> dict[str, Any]:
    return verify_scientific_release_receipt_content_v2(
        receipt=_load(path),
        registry=_load(registry_path),
        blueprint=_load(blueprint_path),
        blueprint_sha256=sha256_file(blueprint_path),
        policy=_load(policy_path),
        policy_sha256=sha256_file(policy_path),
        proposal=_load(proposal_path),
        proposal_sha256=sha256_file(proposal_path),
        admission=_load(admission_path),
        admission_sha256=sha256_file(admission_path),
        as_of=as_of,
    )


def _legacy_release_verification(
    release: dict[str, Any]
) -> dict[str, Any]:
    return {
        **release,
        "schema": SIMULATION_RELEASE_SCHEMA,
    }


def build_budget_bound_runtime_admission_v2(
    *,
    release_receipt: dict[str, Any],
    release_registry: dict[str, Any],
    blueprint: dict[str, Any],
    blueprint_sha256: str,
    policy: dict[str, Any],
    policy_sha256: str,
    proposal: dict[str, Any],
    proposal_sha256: str,
    budget_admission: dict[str, Any],
    budget_admission_sha256: str,
    compute_receipt: dict[str, Any],
    compute_registry: dict[str, Any],
    execution_job_id: str,
    device: str,
    as_of: str,
) -> dict[str, Any]:
    blueprint_check = validate_execution_blueprint_content(blueprint)
    job = select_execution_job(blueprint, execution_job_id=execution_job_id)
    release = verify_scientific_release_receipt_content_v2(
        receipt=release_receipt,
        registry=release_registry,
        blueprint=blueprint,
        blueprint_sha256=blueprint_sha256,
        policy=policy,
        policy_sha256=policy_sha256,
        proposal=proposal,
        proposal_sha256=proposal_sha256,
        admission=budget_admission,
        admission_sha256=budget_admission_sha256,
        as_of=as_of,
    )
    compute = v1.verify_compute_authorization_receipt(
        receipt=compute_receipt,
        registry=compute_registry,
        blueprint=blueprint,
        execution_job_id=execution_job_id,
        device=device,
        release_verification=_legacy_release_verification(release),
        as_of=as_of,
    )
    synthetic = bool(release["synthetic_fixture_only"])
    body = {
        "schema": RUNTIME_ADMISSION_SCHEMA,
        "status": (
            "blocked_authenticated_budget_bound_synthetic_fixture"
            if synthetic
            else "readonly_budget_bound_checks_pass_transition_not_performed"
        ),
        "as_of": as_of,
        "source": {
            "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
            "execution_blueprint_sha256": release["execution_blueprint_sha256"],
            "budget_policy_id": release["budget_policy_id"],
            "budget_policy_sha256": release["budget_policy_sha256"],
            "budget_proposal_id": release["budget_proposal_id"],
            "budget_proposal_sha256": release["budget_proposal_sha256"],
            "budget_admission_id": release["budget_admission_id"],
            "budget_admission_sha256": release["budget_admission_sha256"],
            "scientific_release_receipt_id": release[
                "scientific_release_receipt_id"
            ],
            "scientific_release_receipt_sha256": release[
                "scientific_release_receipt_sha256"
            ],
            "compute_authorization_id": compute["compute_authorization_id"],
            "compute_authorization_sha256": compute[
                "compute_authorization_sha256"
            ],
        },
        "execution_job_id": job["execution_job_id"],
        "scientific_release_verification": release,
        "compute_authorization_verification": compute,
        "exact_job_binding": {
            "compartment_id": job["compartment_id"],
            "exact_sample_contract_sha256": content_sha256(
                job["exact_sample_contract"]
            ),
            "global_pair_split_id": job["global_pair_split_id"],
            "planned_steps": int(job["profile"]["steps"]),
            "planned_activation_rows": int(
                job["exact_sample_contract"]["sae_activation_rows"]
            ),
            "dictionary_features": int(job["dictionary_features"]),
            "output_dir": job["output_dir"],
            "device": device,
        },
        "verified_joins": {
            "budget_admission_to_signed_release": True,
            "signed_release_to_compute_authorization": True,
            "budget_admission_to_runtime_admission": True,
            "full_budget_proposal_reproduced": True,
            "full_budget_admission_reproduced": True,
        },
        "runtime_transition": {
            "nonce_consumed": False,
            "output_reserved": False,
            "worker_ledger_reserved": False,
            "durable_transition_created": False,
            "production_runtime_attached": False,
            "trainer_callback_invoked": False,
        },
        "stop_reason": (
            "synthetic_receipts_cannot_authorize_execution"
            if synthetic
            else "readonly_only_durable_transition_and_runtime_attachment_absent"
        ),
        "scientific_boundary": {
            "read_only_admission": True,
            "synthetic_fixture_only": synthetic,
            "budget_admission_is_execution_authority": False,
            "synthetic_signatures_are_execution_authority": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "persistent_output_created": False,
            "weights_loaded_or_mutated": False,
            "trainer_callback_invoked": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "real_actions_authorized": False,
        },
    }
    return {**body, "runtime_admission_id": content_sha256(body)}


def build_budget_bound_runtime_admission_from_paths_v2(
    *,
    release_receipt_path: Path,
    release_registry_path: Path,
    blueprint_path: Path,
    policy_path: Path,
    proposal_path: Path,
    budget_admission_path: Path,
    compute_receipt_path: Path,
    compute_registry_path: Path,
    execution_job_id: str,
    device: str,
    as_of: str,
) -> dict[str, Any]:
    return build_budget_bound_runtime_admission_v2(
        release_receipt=_load(release_receipt_path),
        release_registry=_load(release_registry_path),
        blueprint=_load(blueprint_path),
        blueprint_sha256=sha256_file(blueprint_path),
        policy=_load(policy_path),
        policy_sha256=sha256_file(policy_path),
        proposal=_load(proposal_path),
        proposal_sha256=sha256_file(proposal_path),
        budget_admission=_load(budget_admission_path),
        budget_admission_sha256=sha256_file(budget_admission_path),
        compute_receipt=_load(compute_receipt_path),
        compute_registry=_load(compute_registry_path),
        execution_job_id=execution_job_id,
        device=device,
        as_of=as_of,
    )


def verify_runtime_admission_v2(
    path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    observed = _load(path)
    expected = build_budget_bound_runtime_admission_from_paths_v2(**kwargs)
    if observed != expected:
        raise ValueError("budget-bound runtime admission V2 is not reproducible")
    body = {
        key: value for key, value in observed.items() if key != "runtime_admission_id"
    }
    if observed.get("runtime_admission_id") != content_sha256(body):
        raise ValueError("budget-bound runtime admission V2 address differs")
    return {
        "schema": RUNTIME_VERIFICATION_SCHEMA,
        "status": "pass",
        "runtime_admission_id": observed["runtime_admission_id"],
        "execution_blueprint_id": observed["source"]["execution_blueprint_id"],
        "budget_proposal_id": observed["source"]["budget_proposal_id"],
        "budget_admission_id": observed["source"]["budget_admission_id"],
        "scientific_release_receipt_id": observed["source"]
        ["scientific_release_receipt_id"],
        "compute_authorization_id": observed["source"][
            "compute_authorization_id"
        ],
        "execution_job_id": observed["execution_job_id"],
        "budget_release_join_verified": True,
        "full_reproduction_compared": True,
        "durable_transition_created": False,
        "production_runtime_attached": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a budget-bound scientific release or build/verify a read-only "
            "budget-bound runtime admission V2."
        )
    )
    parser.add_argument("--scientific-release-receipt", type=Path, required=True)
    parser.add_argument("--scientific-release-registry", type=Path, required=True)
    parser.add_argument("--execution-blueprint", type=Path, required=True)
    parser.add_argument("--budget-policy", type=Path, required=True)
    parser.add_argument("--budget-proposal", type=Path, required=True)
    parser.add_argument("--budget-admission", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-release", action="store_true")
    mode.add_argument("--output-runtime-admission", type=Path)
    mode.add_argument("--verify-runtime-admission", type=Path)
    parser.add_argument("--compute-authorization-receipt", type=Path)
    parser.add_argument("--compute-authority-registry", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--device")
    args = parser.parse_args()
    release_kwargs = {
        "registry_path": args.scientific_release_registry,
        "blueprint_path": args.execution_blueprint,
        "policy_path": args.budget_policy,
        "proposal_path": args.budget_proposal,
        "admission_path": args.budget_admission,
        "as_of": args.as_of,
    }
    if args.verify_release:
        result = verify_scientific_release_receipt_v2(
            args.scientific_release_receipt, **release_kwargs
        )
    else:
        if any(
            value is None
            for value in (
                args.compute_authorization_receipt,
                args.compute_authority_registry,
                args.job_id,
                args.device,
            )
        ):
            raise ValueError(
                "runtime admission requires compute receipt, registry, job ID, and device"
            )
        runtime_kwargs = {
            "release_receipt_path": args.scientific_release_receipt,
            "release_registry_path": args.scientific_release_registry,
            "blueprint_path": args.execution_blueprint,
            "policy_path": args.budget_policy,
            "proposal_path": args.budget_proposal,
            "budget_admission_path": args.budget_admission,
            "compute_receipt_path": args.compute_authorization_receipt,
            "compute_registry_path": args.compute_authority_registry,
            "execution_job_id": args.job_id,
            "device": args.device,
            "as_of": args.as_of,
        }
        if args.output_runtime_admission is not None:
            result = build_budget_bound_runtime_admission_from_paths_v2(
                **runtime_kwargs
            )
            write_json(args.output_runtime_admission, result)
        else:
            result = verify_runtime_admission_v2(
                args.verify_runtime_admission, **runtime_kwargs
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
