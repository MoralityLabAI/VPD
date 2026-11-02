from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.admit_cluster_aware_sae_training_v2 import (
    verify_runtime_admission_v2,
    verify_scientific_release_receipt_v2,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.dispatch_cluster_aware_sae_semantic_slice_v1 import (
    _load,
    _safe_file,
    _safe_root,
    dispatch_semantic_slice,
)


SCHEMA = "sch.vpd-cluster-aware-sae-semantic-dispatch-slice.v2"


def dispatch_semantic_slice_v2(
    *,
    project_root: Path,
    artifact_root: str,
    materialization: str,
    routing_bundle: str,
    predecessor_training_plan: str,
    execution_blueprint: str,
    budget_policy: str,
    budget_proposal: str,
    budget_admission: str,
    scientific_release_receipt: str,
    scientific_release_registry: str,
    compute_authorization_receipt: str,
    compute_authority_registry: str,
    runtime_admission: str,
    execution_job_id: str,
    device: str,
    as_of: str,
    expected_graph_kind: str,
) -> dict[str, Any]:
    if expected_graph_kind != "synthetic_benchmark_fixture":
        raise ValueError(
            "semantic dispatch V2 is restricted to non-authorizing synthetic fixtures"
        )
    project_root = project_root.resolve()
    root = _safe_root(project_root, artifact_root)
    paths = {
        "materialization": _safe_file(root, materialization, label="materialization"),
        "routing_bundle": _safe_file(root, routing_bundle, label="routing bundle"),
        "predecessor_training_plan": _safe_file(
            root, predecessor_training_plan, label="predecessor training plan"
        ),
        "execution_blueprint": _safe_file(
            root, execution_blueprint, label="execution blueprint"
        ),
        "budget_policy": _safe_file(root, budget_policy, label="budget policy"),
        "budget_proposal": _safe_file(
            root, budget_proposal, label="budget proposal"
        ),
        "budget_admission": _safe_file(
            root, budget_admission, label="budget admission"
        ),
        "scientific_release_receipt": _safe_file(
            root,
            scientific_release_receipt,
            label="scientific release receipt",
        ),
        "scientific_release_registry": _safe_file(
            root,
            scientific_release_registry,
            label="scientific release registry",
        ),
        "compute_authorization_receipt": _safe_file(
            root,
            compute_authorization_receipt,
            label="compute authorization receipt",
        ),
        "compute_authority_registry": _safe_file(
            root,
            compute_authority_registry,
            label="compute authority registry",
        ),
        "runtime_admission": _safe_file(
            root, runtime_admission, label="runtime admission"
        ),
    }
    base = dispatch_semantic_slice(
        project_root=project_root,
        artifact_root=artifact_root,
        materialization=materialization,
        routing_bundle=routing_bundle,
        predecessor_training_plan=predecessor_training_plan,
        execution_blueprint=execution_blueprint,
        budget_policy=budget_policy,
        budget_proposal=budget_proposal,
        budget_admission=budget_admission,
        expected_graph_kind=expected_graph_kind,
    )
    release = verify_scientific_release_receipt_v2(
        paths["scientific_release_receipt"],
        registry_path=paths["scientific_release_registry"],
        blueprint_path=paths["execution_blueprint"],
        policy_path=paths["budget_policy"],
        proposal_path=paths["budget_proposal"],
        admission_path=paths["budget_admission"],
        as_of=as_of,
    )
    runtime = verify_runtime_admission_v2(
        paths["runtime_admission"],
        release_receipt_path=paths["scientific_release_receipt"],
        release_registry_path=paths["scientific_release_registry"],
        blueprint_path=paths["execution_blueprint"],
        policy_path=paths["budget_policy"],
        proposal_path=paths["budget_proposal"],
        budget_admission_path=paths["budget_admission"],
        compute_receipt_path=paths["compute_authorization_receipt"],
        compute_registry_path=paths["compute_authority_registry"],
        execution_job_id=execution_job_id,
        device=device,
        as_of=as_of,
    )
    budget_admission_document = _load(paths["budget_admission"])
    runtime_document = _load(paths["runtime_admission"])
    compute_receipt_document = _load(paths["compute_authorization_receipt"])
    if release["synthetic_fixture_only"] is not True:
        raise ValueError("semantic dispatch V2 received real signed authority")
    if (
        release["budget_admission_id"]
        != budget_admission_document["admission_id"]
        or release["budget_admission_sha256"]
        != sha256_file(paths["budget_admission"])
        or runtime_document["source"]["budget_admission_id"]
        != release["budget_admission_id"]
        or runtime_document["source"]["budget_admission_sha256"]
        != release["budget_admission_sha256"]
        or runtime_document["source"]["scientific_release_receipt_id"]
        != release["scientific_release_receipt_id"]
        or runtime_document["source"]["compute_authorization_id"]
        != compute_receipt_document["receipt_id"]
        or runtime["runtime_admission_id"]
        != runtime_document["runtime_admission_id"]
    ):
        raise ValueError("semantic dispatch V2 release/runtime identity join differs")
    boundary = runtime_document.get("scientific_boundary", {})
    if any(
        boundary.get(field) is True
        for field in (
            "execution_authorized",
            "sae_training_authorized",
            "real_actions_authorized",
            "model_framework_imported",
            "model_loaded",
            "model_inference_performed",
            "training_performed",
            "persistent_output_created",
            "weights_loaded_or_mutated",
            "trainer_callback_invoked",
        )
    ):
        raise ValueError("semantic dispatch V2 runtime boundary broadens authority")

    dispatched = list(base["dispatched_nodes"])
    dispatched.extend(
        [
            {
                "node": "budget_bound_scientific_release_receipt_v2",
                "id": release["scientific_release_receipt_id"],
                "verifier_status": release["status"],
                "verification_kind": (
                    "ed25519_signature_plus_full_budget_chain_reproduction"
                ),
                "full_reproduction_compared": False,
            },
            {
                "node": "compute_authorization_receipt_v1",
                "id": runtime["compute_authorization_id"],
                "verifier_status": "authenticated_synthetic_fixture",
                "verification_kind": "ed25519_signature_plus_exact_job_scope",
                "full_reproduction_compared": False,
            },
            {
                "node": "budget_bound_runtime_admission_v2",
                "id": runtime["runtime_admission_id"],
                "verifier_status": runtime["status"],
                "verification_kind": "full_reproduction",
                "full_reproduction_compared": True,
            },
        ]
    )
    body = {
        "schema": SCHEMA,
        "status": "synthetic_budget_bound_runtime_slice_verified_non_authorizing",
        "source": {
            "predecessor_semantic_dispatch_slice_id": base[
                "semantic_dispatch_slice_id"
            ],
            "semantic_graph_contract_id": base["source"][
                "semantic_graph_contract_id"
            ],
            "artifact_root_policy": base["source"]["artifact_root_policy"],
            "artifact_file_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
        },
        "dispatched_nodes": dispatched,
        "verified_joins": {
            **base["verified_joins"],
            "budget_admission_to_signed_release": release[
                "budget_admission_id"
            ],
            "budget_admission_file_sha256_to_signed_release": release[
                "budget_admission_sha256"
            ],
            "signed_release_to_compute_authorization": runtime[
                "scientific_release_receipt_id"
            ],
            "budget_admission_to_runtime_admission": runtime[
                "budget_admission_id"
            ],
            "runtime_admission_to_exact_job": runtime["execution_job_id"],
        },
        "slice_summary": {
            "dispatched_nodes": len(dispatched),
            "full_reproduction_comparisons": sum(
                row["full_reproduction_compared"] is True for row in dispatched
            ),
            "cryptographic_receipt_verifications": 2,
            "execution_jobs": release["execution_jobs"],
            "graph_kind": expected_graph_kind,
            "terminal_gap_count": 2,
        },
        "closed_terminal_joins": [
            "budget_admission_to_signed_release",
            "budget_admission_to_runtime_admission",
        ],
        "terminal_gaps": [
            "runtime_admission_to_durable_transition",
            "production_runtime_attachment",
        ],
        "scientific_boundary": {
            "semantic_slice_only": True,
            "end_to_end_semantic_dispatch_complete": False,
            "synthetic_signed_receipts_verified": True,
            "real_signed_authority_dispatched": False,
            "nonce_consumed": False,
            "output_reserved": False,
            "durable_transition_dispatched": False,
            "production_runtime_attachment_dispatched": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "weights_loaded_or_mutated": False,
            "persistent_dictionary_output_created": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    return {**body, "semantic_dispatch_slice_id": content_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch the synthetic, model-free semantic slice through signed "
            "release and read-only runtime admission."
        )
    )
    parser.add_argument("--artifact-root", required=True)
    for name in (
        "materialization",
        "routing-bundle",
        "predecessor-training-plan",
        "execution-blueprint",
        "budget-policy",
        "budget-proposal",
        "budget-admission",
        "scientific-release-receipt",
        "scientific-release-registry",
        "compute-authorization-receipt",
        "compute-authority-registry",
        "runtime-admission",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--expected-graph-kind",
        choices=("synthetic_benchmark_fixture", "empirical_candidate"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = dispatch_semantic_slice_v2(
        project_root=Path(__file__).resolve().parents[2],
        artifact_root=args.artifact_root,
        materialization=args.materialization,
        routing_bundle=args.routing_bundle,
        predecessor_training_plan=args.predecessor_training_plan,
        execution_blueprint=args.execution_blueprint,
        budget_policy=args.budget_policy,
        budget_proposal=args.budget_proposal,
        budget_admission=args.budget_admission,
        scientific_release_receipt=args.scientific_release_receipt,
        scientific_release_registry=args.scientific_release_registry,
        compute_authorization_receipt=args.compute_authorization_receipt,
        compute_authority_registry=args.compute_authority_registry,
        runtime_admission=args.runtime_admission,
        execution_job_id=args.job_id,
        device=args.device,
        as_of=args.as_of,
        expected_graph_kind=args.expected_graph_kind,
    )
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
