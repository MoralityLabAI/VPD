from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1 import (
    canonical_inputs as budget_policy_inputs,
    verify_budget_policy,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    BINDING_CONTRACTS,
    REQUIRED_EXECUTION_BINDINGS,
    build_readiness as build_capture_binding_readiness,
    verify_readiness as verify_capture_binding_readiness,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-cluster-aware-sae-empirical-budget-binding-readiness.v1"
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER_TEXT = re.compile(r"(?:todo|tbd|placeholder|unknown|missing|unset)", re.I)

ARTIFACT_BINDING_PHASES: dict[str, tuple[str, ...]] = {
    "empirical_capture": (
        "empirical_preflight_v2_id",
        "empirical_preflight_v2_sha256",
        "capture_authorization_receipt_id",
        "capture_authorization_receipt_sha256",
        "capture_authority_registry_id",
        "capture_authority_registry_sha256",
        "empirical_capture_manifest_id",
        "empirical_capture_manifest_sha256",
    ),
    "routing_and_partition": (
        "empirical_routing_bundle_id",
        "empirical_routing_bundle_sha256",
        "routing_stability_manifest_id",
        "routing_stability_manifest_sha256",
        "global_pair_split_id",
        "global_pair_split_sha256",
        "cluster_aware_sae_plan_id",
        "cluster_aware_sae_plan_sha256",
        "materialization_id",
        "materialization_sha256",
    ),
    "execution_blueprint_and_budget": (
        "execution_blueprint_id",
        "execution_blueprint_sha256",
        "budget_proposal_id",
        "budget_proposal_sha256",
        "budget_admission_id",
        "budget_admission_sha256",
    ),
    "signed_release_and_compute_authority": (
        "scientific_release_receipt_id",
        "scientific_release_receipt_sha256",
        "release_authority_registry_id",
        "release_authority_registry_sha256",
        "compute_authorization_manifest_id",
        "compute_authorization_manifest_sha256",
        "compute_authority_registry_id",
        "compute_authority_registry_sha256",
    ),
    "durable_runtime_transition": (
        "nonce_consumption_manifest_id",
        "nonce_consumption_manifest_sha256",
        "output_reservation_manifest_id",
        "output_reservation_manifest_sha256",
        "worker_ledger_claim_id",
        "worker_ledger_claim_sha256",
        "runtime_attachment_claim_id",
        "runtime_attachment_claim_sha256",
    ),
}
ARTIFACT_BINDINGS = tuple(
    field_name
    for phase_fields in ARTIFACT_BINDING_PHASES.values()
    for field_name in phase_fields
)
ALL_BINDINGS = (*REQUIRED_EXECUTION_BINDINGS, *ARTIFACT_BINDINGS)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _validate_artifact_bindings(bindings: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_name in ARTIFACT_BINDINGS:
        value = bindings.get(field_name)
        if value is None or value == "":
            missing.append(field_name)
            continue
        if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
            raise ValueError(f"empirical budget binding {field_name} is not a digest")
        if len(set(value)) == 1:
            raise ValueError(f"empirical budget binding {field_name} is a placeholder digest")
    return missing


def _phase_for(field_name: str) -> str:
    for phase, fields in ARTIFACT_BINDING_PHASES.items():
        if field_name in fields:
            return phase
    return "capture_execution_identity"


def _artifact_contract(field_name: str) -> dict[str, Any]:
    object_name = field_name.removesuffix("_sha256").removesuffix("_id")
    return {
        "kind": "sha256",
        "completion_evidence": (
            f"authenticated content ID or file digest for {object_name}; placeholder digests forbidden"
        ),
        "content_opened_by_this_check": False,
        "cross_artifact_binding_verified_by_this_check": False,
    }


def build_binding_readiness(
    *,
    project_root: Path,
    budget_policy_path: Path,
    capture_binding_readiness_path: Path,
    bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    supplied = dict(bindings or {})
    unknown = sorted(set(supplied) - set(ALL_BINDINGS))
    if unknown:
        raise ValueError(
            "empirical budget binding worksheet has unknown bindings: "
            + ", ".join(unknown)
        )
    for field_name, value in supplied.items():
        if isinstance(value, str) and PLACEHOLDER_TEXT.search(value):
            raise ValueError(f"empirical budget binding {field_name} contains placeholder text")

    policy_verification = verify_budget_policy(
        budget_policy_path, **budget_policy_inputs(project_root)
    )
    capture_template_verification = verify_capture_binding_readiness(
        capture_binding_readiness_path, project_root=project_root
    )
    capture_bindings = {
        field_name: supplied[field_name]
        for field_name in REQUIRED_EXECUTION_BINDINGS
        if field_name in supplied
    }
    capture_readiness = build_capture_binding_readiness(
        project_root=project_root, bindings=capture_bindings
    )
    capture_missing = list(capture_readiness["summary"]["missing_bindings"])
    artifact_missing = _validate_artifact_bindings(supplied)
    missing = [*capture_missing, *artifact_missing]
    syntax_complete = not missing

    rows = []
    for field_name in ALL_BINDINGS:
        contract = (
            {
                **BINDING_CONTRACTS[field_name],
                "cross_artifact_binding_verified_by_this_check": False,
            }
            if field_name in REQUIRED_EXECUTION_BINDINGS
            else _artifact_contract(field_name)
        )
        rows.append(
            {
                "phase": _phase_for(field_name),
                "field": field_name,
                "value": supplied.get(field_name),
                "status": "missing" if field_name in missing else "supplied_unopened",
                **contract,
            }
        )

    phase_summaries = []
    phase_order = ("capture_execution_identity", *ARTIFACT_BINDING_PHASES)
    for phase in phase_order:
        phase_fields = (
            REQUIRED_EXECUTION_BINDINGS
            if phase == "capture_execution_identity"
            else ARTIFACT_BINDING_PHASES[phase]
        )
        phase_missing = [field_name for field_name in phase_fields if field_name in missing]
        phase_summaries.append(
            {
                "phase": phase,
                "required_bindings": len(phase_fields),
                "supplied_bindings": len(phase_fields) - len(phase_missing),
                "missing_bindings": phase_missing,
                "syntax_complete": not phase_missing,
                "content_verified": False,
                "authority_granted": False,
            }
        )

    body = {
        "schema": SCHEMA,
        "status": (
            "syntax_complete_unopened_awaiting_real_artifact_graph_and_signed_authority"
            if syntax_complete
            else "blocked_missing_empirical_blueprint_budget_and_authority_bindings"
        ),
        "source": {
            "budget_policy": {
                "path": budget_policy_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(budget_policy_path),
                "budget_policy_id": policy_verification["budget_policy_id"],
                "verification": policy_verification,
            },
            "capture_binding_readiness_template": {
                "path": capture_binding_readiness_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(capture_binding_readiness_path),
                "readiness_id": capture_template_verification["readiness_id"],
                "verification": capture_template_verification,
            },
            "builder_sha256": sha256_file(Path(__file__)),
        },
        "binding_rows": rows,
        "phase_summaries": phase_summaries,
        "summary": {
            "required_bindings": len(ALL_BINDINGS),
            "capture_execution_identity_bindings": len(REQUIRED_EXECUTION_BINDINGS),
            "artifact_graph_bindings": len(ARTIFACT_BINDINGS),
            "supplied_bindings": len(ALL_BINDINGS) - len(missing),
            "missing_bindings": missing,
            "syntax_complete": syntax_complete,
            "capture_output_isolation_verified": capture_readiness["summary"][
                "output_isolation_verified"
            ],
            "cross_artifact_content_verified": False,
            "budget_binding_verified": False,
            "signed_authority_verified": False,
            "durable_runtime_transition_verified": False,
        },
        "completed_syntax_audit": (
            {
                "schema": "sch.vpd-cluster-aware-sae-empirical-budget-binding-syntax-audit.v1",
                "status": "pass_syntax_and_output_isolation_only",
                "bindings_complete": True,
                "binding_fields": list(ALL_BINDINGS),
                "capture_binding_audit": capture_readiness[
                    "completed_binding_audit"
                ],
                "artifact_digests_nonplaceholder": True,
                "model_or_adapter_content_opened": False,
                "artifact_graph_content_opened": False,
                "cross_artifact_binding_verified": False,
                "execution_authorized": False,
                "sae_training_authorized": False,
            }
            if syntax_complete
            else None
        ),
        "binding_contract": {
            "all_fifty_bindings_required_for_syntax_completion": True,
            "null_empty_placeholder_or_uniform_digest_rejected": True,
            "content_ids_and_file_digests_must_be_authenticated_separately": True,
            "execution_blueprint_must_bind_empirical_routing_and_global_split": True,
            "budget_proposal_and_admission_must_bind_the_pinned_policy": True,
            "release_and_compute_receipts_must_bind_the_exact_blueprint": True,
            "nonce_output_worker_and_runtime_claims_must_be_fresh_and_single_use": True,
            "syntax_completion_transfers_execution_authority": False,
        },
        "stop_conditions": [
            "any_of_fifty_bindings_is_missing_or_placeholder",
            "any_content_id_or_file_digest_is_not_independently_authenticated",
            "empirical_capture_routing_split_plan_materialization_or_blueprint_binding_differs",
            "budget_proposal_or_admission_does_not_bind_the_pinned_policy_and_exact_blueprint",
            "signed_release_or_compute_authority_is_absent_invalid_inactive_revoked_consumed_or_scope_mismatched",
            "nonce_output_worker_or_runtime_claim_is_absent_reused_colliding_or_not_durable",
            "worksheet_syntax_completion_is_treated_as_empirical_evidence_or_execution_authority",
        ],
        "scientific_boundary": {
            "worksheet_only": True,
            "synthetic_complete_fixture_persisted": False,
            "model_or_adapter_content_opened": False,
            "artifact_graph_content_opened": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_capture_present": False,
            "empirical_routing_bundle_present": False,
            "empirical_execution_blueprint_present": False,
            "real_signed_receipts_present": False,
            "durable_runtime_transition_present": False,
            "budget_binding_verified": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "paid_compute_authorized": False,
            "real_actions_authorized": False,
        },
    }
    return {**body, "binding_readiness_id": content_sha256(body)}


def verify_binding_readiness(
    path: Path,
    *,
    project_root: Path,
    budget_policy_path: Path,
    capture_binding_readiness_path: Path,
) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("binding_readiness_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("empirical budget binding readiness address differs")
    bindings = {
        row["field"]: row["value"]
        for row in observed["binding_rows"]
        if row["value"] is not None
    }
    if observed != build_binding_readiness(
        project_root=project_root,
        budget_policy_path=budget_policy_path,
        capture_binding_readiness_path=capture_binding_readiness_path,
        bindings=bindings,
    ):
        raise ValueError("empirical budget binding readiness is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-empirical-budget-binding-readiness-verification.v1",
        "status": "pass",
        "binding_readiness_id": identifier,
        "required_bindings": observed["summary"]["required_bindings"],
        "missing_bindings": len(observed["summary"]["missing_bindings"]),
        "syntax_complete": observed["summary"]["syntax_complete"],
        "budget_binding_verified": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "project_root": project_root,
        "budget_policy_path": project_root
        / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json",
        "capture_binding_readiness_path": project_root
        / "artifacts/vpd_cluster_to_sae_binding_readiness_v1/readiness.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify nonexecuting empirical-blueprint budget binding readiness V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--bindings", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    inputs = canonical_inputs(root)
    if args.output is not None:
        bindings = _load(args.bindings) if args.bindings is not None else {}
        result = build_binding_readiness(**inputs, bindings=bindings)
        write_json(args.output, result)
    else:
        if args.bindings is not None:
            raise ValueError("--bindings is only valid with --output")
        result = verify_binding_readiness(args.verify, **inputs)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "binding_readiness_id": result["binding_readiness_id"],
                "required_bindings": result.get(
                    "required_bindings", result.get("summary", {}).get("required_bindings")
                ),
                "missing_bindings": result.get(
                    "missing_bindings",
                    len(result.get("summary", {}).get("missing_bindings", [])),
                ),
                "execution_authorized": result.get(
                    "execution_authorized",
                    result.get("scientific_boundary", {}).get("execution_authorized"),
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
