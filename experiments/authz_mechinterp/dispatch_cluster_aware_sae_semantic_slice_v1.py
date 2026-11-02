from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_cluster_aware_sae_blueprint_budget_admission_v2 import (
    verify_admission,
    verify_proposal,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1 import (
    verify_execution_blueprint,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_semantic_graph_contract_v2 import (
    canonical_inputs as contract_inputs,
    verify_contract,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1 import (
    verify_materialization,
)


SCHEMA = "sch.vpd-cluster-aware-sae-semantic-dispatch-slice.v1"
ALLOWED_INPUT_PARENT = "artifacts/vpd_cluster_aware_sae_semantic_dispatch_inputs"
CONTRACT_ID = "6db000c29e0619d6bec26fb85f8d1739bd5fae16e46cccc03516a2171d79adb0"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _is_reparse_component(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _has_reparse_component(base: Path, relative: Path) -> bool:
    current = base
    for part in relative.parts:
        current = current / part
        if _is_reparse_component(current):
            return True
    return False


def _safe_root(project_root: Path, value: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("semantic dispatch artifact root must be a safe relative path")
    if _has_reparse_component(project_root, relative):
        raise ValueError("semantic dispatch artifact root contains a reparse point")
    root = (project_root / relative).resolve()
    allowed = (project_root / ALLOWED_INPUT_PARENT).resolve()
    if root == allowed or not root.is_relative_to(allowed) or not root.is_dir():
        raise ValueError("semantic dispatch artifact root escapes its dedicated parent")
    return root


def _safe_file(root: Path, value: str, *, label: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"semantic dispatch {label} must be a safe relative path")
    if _has_reparse_component(root, relative):
        raise ValueError(f"semantic dispatch {label} contains a reparse point")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"semantic dispatch {label} escapes its root or is absent")
    return path


def dispatch_semantic_slice(
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
    expected_graph_kind: str,
) -> dict[str, Any]:
    if expected_graph_kind not in {"synthetic_benchmark_fixture", "empirical_candidate"}:
        raise ValueError("semantic dispatch expected graph kind differs")
    root = _safe_root(project_root.resolve(), artifact_root)
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
    }
    contract_path = (
        project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_v2/contract.json"
    )
    contract_check = verify_contract(
        contract_path, **contract_inputs(project_root)
    )
    if (
        contract_check["semantic_graph_contract_id"] != CONTRACT_ID
        or contract_check["end_to_end_semantic_dispatch_allowed"] is not False
        or contract_check["terminal_gap_joins"] != 3
    ):
        raise ValueError("semantic dispatch contract boundary differs")

    materialization_check = verify_materialization(
        paths["materialization"],
        bundle_path=paths["routing_bundle"],
        predecessor_training_plan_path=paths["predecessor_training_plan"],
        project_root=root,
    )
    materialization_document = _load(paths["materialization"])
    blueprint_document = _load(paths["execution_blueprint"])
    blueprint_check = verify_execution_blueprint(
        blueprint_document, materialization=materialization_document
    )
    proposal_check = verify_proposal(
        paths["budget_proposal"],
        policy_path=paths["budget_policy"],
        blueprint_path=paths["execution_blueprint"],
    )
    admission_check = verify_admission(
        paths["budget_admission"],
        proposal_path=paths["budget_proposal"],
        policy_path=paths["budget_policy"],
        blueprint_path=paths["execution_blueprint"],
    )
    policy_document = _load(paths["budget_policy"])
    proposal_document = _load(paths["budget_proposal"])
    admission_document = _load(paths["budget_admission"])

    empirical = bool(materialization_document["boundary"]["empirical_plan"])
    observed_kind = "empirical_candidate" if empirical else "synthetic_benchmark_fixture"
    if observed_kind != expected_graph_kind:
        raise ValueError("semantic dispatch evidence kind differs")
    if (
        blueprint_document["source"]["materialization_id"]
        != materialization_check["materialization_id"]
        or proposal_document["source"]["execution_blueprint_id"]
        != blueprint_check["execution_blueprint_id"]
        or admission_document["source"]["execution_blueprint_id"]
        != blueprint_check["execution_blueprint_id"]
        or proposal_document["source"]["execution_blueprint_sha256"]
        != sha256_file(paths["execution_blueprint"])
        or admission_document["source"]["execution_blueprint_sha256"]
        != sha256_file(paths["execution_blueprint"])
        or proposal_document["source"]["budget_policy_id"]
        != policy_document["budget_policy_id"]
        or admission_document["source"]["budget_policy_id"]
        != policy_document["budget_policy_id"]
        or admission_document["source"]["proposal_id"]
        != proposal_document["proposal_id"]
    ):
        raise ValueError("semantic dispatch cross-artifact identity join differs")
    if any(
        document.get("scientific_boundary", {}).get(field) is True
        for document in (proposal_document, admission_document)
        for field in (
            "execution_authorized",
            "paid_compute_authorized",
            "weight_mutation_authorized",
        )
    ):
        raise ValueError("semantic dispatch budget artifact broadens authority")

    body = {
        "schema": SCHEMA,
        "status": (
            "synthetic_semantic_slice_verified_non_authorizing"
            if not empirical
            else "empirical_candidate_semantic_slice_verified_non_authorizing"
        ),
        "source": {
            "semantic_graph_contract_id": contract_check[
                "semantic_graph_contract_id"
            ],
            "artifact_root_policy": "dedicated_safe_self_contained_root_verified",
            "artifact_file_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
        },
        "dispatched_nodes": [
            {
                "node": "materialization",
                "id": materialization_check["materialization_id"],
                "verifier_status": materialization_check["status"],
                "full_reproduction_compared": True,
            },
            {
                "node": "execution_blueprint",
                "id": blueprint_check["execution_blueprint_id"],
                "verifier_status": blueprint_check["status"],
                "full_reproduction_compared": True,
            },
            {
                "node": "blueprint_bound_budget_proposal_v2",
                "id": proposal_check["proposal_id"],
                "verifier_status": proposal_check["status"],
                "full_reproduction_compared": True,
            },
            {
                "node": "blueprint_bound_budget_admission_v2",
                "id": admission_check["admission_id"],
                "verifier_status": admission_check["status"],
                "full_reproduction_compared": True,
            },
        ],
        "verified_joins": {
            "materialization_to_routing_bundle": materialization_check[
                "routing_evidence_bundle_id"
            ],
            "blueprint_to_materialization": blueprint_check["materialization_id"],
            "proposal_to_blueprint": proposal_check["execution_blueprint_id"],
            "admission_to_blueprint": admission_check["execution_blueprint_id"],
            "admission_to_proposal": admission_check["proposal_id"],
            "exact_blueprint_file_sha256": admission_check[
                "execution_blueprint_sha256"
            ],
        },
        "slice_summary": {
            "dispatched_nodes": 4,
            "full_reproduction_comparisons": 4,
            "execution_jobs": admission_check["execution_jobs"],
            "graph_kind": observed_kind,
            "terminal_gap_count": 3,
        },
        "terminal_gaps": [
            "budget_admission_to_signed_release",
            "budget_admission_to_runtime_admission",
            "production_runtime_attachment",
        ],
        "scientific_boundary": {
            "semantic_slice_only": True,
            "end_to_end_semantic_dispatch_complete": False,
            "empirical_graph_present": empirical,
            "signed_authority_dispatched": False,
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
        description="Dispatch the model-free materialization-to-budget semantic slice."
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--materialization", required=True)
    parser.add_argument("--routing-bundle", required=True)
    parser.add_argument("--predecessor-training-plan", required=True)
    parser.add_argument("--execution-blueprint", required=True)
    parser.add_argument("--budget-policy", required=True)
    parser.add_argument("--budget-proposal", required=True)
    parser.add_argument("--budget-admission", required=True)
    parser.add_argument(
        "--expected-graph-kind",
        choices=("synthetic_benchmark_fixture", "empirical_candidate"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = dispatch_semantic_slice(
        project_root=Path(__file__).resolve().parents[2],
        artifact_root=args.artifact_root,
        materialization=args.materialization,
        routing_bundle=args.routing_bundle,
        predecessor_training_plan=args.predecessor_training_plan,
        execution_blueprint=args.execution_blueprint,
        budget_policy=args.budget_policy,
        budget_proposal=args.budget_proposal,
        budget_admission=args.budget_admission,
        expected_graph_kind=args.expected_graph_kind,
    )
    if args.output is not None:
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "semantic_dispatch_slice_id": result["semantic_dispatch_slice_id"],
                "dispatched_nodes": result["slice_summary"]["dispatched_nodes"],
                "terminal_gap_count": result["slice_summary"]["terminal_gap_count"],
                "execution_authorized": result["scientific_boundary"][
                    "execution_authorized"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
