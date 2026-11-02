from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.verify_cluster_aware_sae_cross_artifact_graph_v1 import (
    EDGE_SPECS,
    NODE_SPECS,
)


SCHEMA = "sch.vpd-cluster-aware-sae-semantic-dispatch-coverage-audit.v1"


IMPLEMENTATION_ROWS: tuple[dict[str, Any], ...] = (
    {
        "node": "empirical_preflight_v2",
        "implemented_schema": "sch.vpd-cluster-to-sae-empirical-preflight.v2",
        "implemented_id_field": "preflight_id",
        "module": "experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2",
        "callable": "verify_preflight",
        "verification_mode": "standalone_path",
        "dispatch_ready_under_v1_graph": True,
        "gap": None,
    },
    {
        "node": "capture_authority_registry",
        "implemented_schema": "sch.vpd-cluster-to-sae-capture-authority-registry.v1",
        "implemented_id_field": "registry_id",
        "module": "experiments.authz_mechinterp.verify_vpd_cluster_to_sae_capture_authorization_v1",
        "callable": "verify_authorization_receipt",
        "verification_mode": "joint_registry_receipt_preflight_and_as_of",
        "dispatch_ready_under_v1_graph": False,
        "gap": "capture receipt does not embed source.registry_id and its ID field differs",
    },
    {
        "node": "capture_authorization_receipt",
        "implemented_schema": "sch.vpd-cluster-to-sae-capture-authorization-receipt.v1",
        "implemented_id_field": "authorization_receipt_id",
        "module": "experiments.authz_mechinterp.verify_vpd_cluster_to_sae_capture_authorization_v1",
        "callable": "verify_authorization_receipt",
        "verification_mode": "joint_registry_receipt_preflight_and_as_of",
        "dispatch_ready_under_v1_graph": False,
        "gap": "V1 graph declares receipt_id instead of authorization_receipt_id",
    },
    {
        "node": "empirical_capture_manifest",
        "implemented_schema": "sch.vpd-activation-capture-manifest.v1",
        "implemented_id_field": "capture_id",
        "module": "experiments.authz_mechinterp.verify_activation_capture_semantics",
        "callable": "verify_prompt_window_capture_semantics",
        "verification_mode": "capture_directory_dataset_split_and_role_sidecars",
        "dispatch_ready_under_v1_graph": False,
        "gap": "V1 graph declares capture_manifest_id and omits required capture directory, dataset, split, and role sidecars",
    },
    {
        "node": "empirical_routing_bundle",
        "implemented_schema": "sch.vpd-guarded-cluster-routing-evidence-bundle.v1",
        "implemented_id_field": "routing_evidence_bundle_id",
        "module": "experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1",
        "callable": "load_and_verify_routing_evidence_bundle",
        "verification_mode": "standalone_path_loading_adapter_routing_and_stability_receipts",
        "dispatch_ready_under_v1_graph": False,
        "gap": "V1 graph declares routing_bundle_id and invents a direct capture-manifest edge absent from the bundle",
    },
    {
        "node": "routing_stability_manifest",
        "implemented_schema": "sch.vpd-cluster-routing-stability-manifest.v1",
        "implemented_id_field": "routing_stability_id",
        "module": "experiments.authz_mechinterp.route_vpd_cluster_episodes_v1",
        "callable": "verify_routing_stability_v1",
        "verification_mode": "standalone_document_and_joined_inside_routing_bundle",
        "dispatch_ready_under_v1_graph": True,
        "gap": None,
    },
    {
        "node": "global_pair_split",
        "implemented_schema": "sch.vpd-cluster-aware-sae-global-pair-split.v2",
        "implemented_id_field": "global_pair_split_id",
        "module": "experiments.authz_mechinterp.plan_cluster_aware_sae_training_v2",
        "callable": "verify_cluster_aware_sae_plan_v2",
        "verification_mode": "embedded_inside_cluster_aware_plan",
        "dispatch_ready_under_v1_graph": False,
        "gap": "implemented global split is embedded in the plan rather than persisted as an independent node",
    },
    {
        "node": "cluster_aware_sae_plan",
        "implemented_schema": "sch.vpd-cluster-aware-sae-training-plan.v2",
        "implemented_id_field": "cluster_aware_sae_plan_id",
        "module": "experiments.authz_mechinterp.plan_cluster_aware_sae_training_v2",
        "callable": "verify_cluster_aware_sae_plan_v2",
        "verification_mode": "standalone_document_with_embedded_global_split",
        "dispatch_ready_under_v1_graph": True,
        "gap": None,
    },
    {
        "node": "materialization",
        "implemented_schema": "sch.vpd-cluster-aware-sae-plan-materialization.v1",
        "implemented_id_field": "materialization_id",
        "module": "experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1",
        "callable": "verify_materialization",
        "verification_mode": "path_plus_routing_bundle_and_predecessor_plan_sidecars",
        "dispatch_ready_under_v1_graph": False,
        "gap": "V1 graph omits the predecessor training-plan sidecar and treats embedded plan/split values as independent sources",
    },
    {
        "node": "execution_blueprint",
        "implemented_schema": "sch.vpd-cluster-aware-sae-execution-blueprint.v1",
        "implemented_id_field": "execution_blueprint_id",
        "module": "experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1",
        "callable": "verify_execution_blueprint",
        "verification_mode": "document_plus_materialization_document",
        "dispatch_ready_under_v1_graph": True,
        "gap": None,
    },
    {
        "node": "budget_proposal",
        "implemented_schema": "sch.vpd-cluster-aware-sae-budget-proposal.v1",
        "implemented_id_field": "proposal_id",
        "module": "experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1",
        "callable": "evaluate_budget_proposal",
        "verification_mode": "joint_policy_proposal_reproducing_admission",
        "dispatch_ready_under_v1_graph": False,
        "gap": "implemented V1 proposal binds only the policy; V1 graph requires blueprint ID and file hash fields that do not exist",
    },
    {
        "node": "budget_admission",
        "implemented_schema": "sch.vpd-cluster-aware-sae-budget-admission.v1",
        "implemented_id_field": "admission_id",
        "module": "experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1",
        "callable": "evaluate_budget_proposal",
        "verification_mode": "reproduced_output_of_policy_and_proposal",
        "dispatch_ready_under_v1_graph": False,
        "gap": "implemented V1 admission has no execution-blueprint binding required by the graph",
    },
    {
        "node": "release_authority_registry",
        "implemented_schema": "sch.vpd-cluster-aware-sae-authority-registry.v1",
        "implemented_id_field": "registry_id",
        "module": "experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1",
        "callable": "verify_scientific_release_receipt",
        "verification_mode": "joint_registry_receipt_blueprint_and_as_of",
        "dispatch_ready_under_v1_graph": False,
        "gap": "receipt-to-registry is cryptographically joined but no source.registry_id field exists",
    },
    {
        "node": "scientific_release_receipt",
        "implemented_schema": "sch.vpd-cluster-aware-sae-scientific-release-receipt.v1",
        "implemented_id_field": "receipt_id",
        "module": "experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1",
        "callable": "verify_scientific_release_receipt",
        "verification_mode": "joint_registry_receipt_blueprint_and_as_of",
        "dispatch_ready_under_v1_graph": False,
        "gap": "implemented release receipt does not bind the separate V1 budget-admission artifact",
    },
    {
        "node": "compute_authority_registry",
        "implemented_schema": "sch.vpd-cluster-aware-sae-authority-registry.v1",
        "implemented_id_field": "registry_id",
        "module": "experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1",
        "callable": "verify_compute_authorization_receipt",
        "verification_mode": "joint_registry_receipt_blueprint_release_job_device_and_as_of",
        "dispatch_ready_under_v1_graph": False,
        "gap": "compute receipt representation and graph edges differ",
    },
    {
        "node": "compute_authorization_manifest",
        "implemented_schema": "sch.vpd-cluster-aware-sae-compute-authorization-receipt.v1",
        "implemented_id_field": "receipt_id",
        "module": "experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1",
        "callable": "verify_compute_authorization_receipt",
        "verification_mode": "joint_registry_receipt_blueprint_release_job_device_and_as_of",
        "dispatch_ready_under_v1_graph": False,
        "gap": "V1 graph invents a compute-authorization manifest and ID; implementation uses a signed receipt and receipt_id",
    },
    {
        "node": "nonce_consumption_manifest",
        "implemented_schema": "sch.vpd-cluster-aware-sae-job-transition-reservation.v1",
        "implemented_id_field": "transition_reservation_id",
        "module": "experiments.authz_mechinterp.reserve_cluster_aware_sae_training_transition_v1",
        "callable": "verify_training_transition",
        "verification_mode": "embedded_combined_transition_receipt",
        "dispatch_ready_under_v1_graph": False,
        "gap": "nonce consumption is embedded in one durable transition receipt; no standalone manifest exists",
    },
    {
        "node": "output_reservation_manifest",
        "implemented_schema": "sch.vpd-cluster-aware-sae-job-transition-reservation.v1",
        "implemented_id_field": "transition_reservation_id",
        "module": "experiments.authz_mechinterp.reserve_cluster_aware_sae_training_transition_v1",
        "callable": "verify_training_transition",
        "verification_mode": "embedded_combined_transition_receipt",
        "dispatch_ready_under_v1_graph": False,
        "gap": "output reservation is embedded in one durable transition receipt; no standalone manifest exists",
    },
    {
        "node": "worker_ledger_claim",
        "implemented_schema": "sch.vpd-cluster-aware-sae-job-transition-reservation.v1",
        "implemented_id_field": "transition_reservation_id",
        "module": "experiments.authz_mechinterp.reserve_cluster_aware_sae_training_transition_v1",
        "callable": "verify_training_transition",
        "verification_mode": "embedded_combined_transition_receipt",
        "dispatch_ready_under_v1_graph": False,
        "gap": "worker-ledger reservation is embedded in one durable transition receipt; no standalone claim exists",
    },
    {
        "node": "runtime_attachment_claim",
        "implemented_schema": "sch.vpd-cluster-aware-sae-inert-runtime-attachment-claim.v1",
        "implemented_id_field": "attachment_claim_id",
        "module": "experiments.authz_mechinterp.attach_cluster_aware_sae_inert_runtime_v1",
        "callable": "verify_inert_runtime_attachment_claim",
        "verification_mode": "synthetic_inert_only_path",
        "dispatch_ready_under_v1_graph": False,
        "gap": "V1 graph schema and ID differ; only a synthetic inert attachment exists and no production attachment verifier exists",
    },
)


EXACT_IMPLEMENTED_EDGE_NAMES = {
    "capture_receipt_to_preflight_id",
    "blueprint_to_materialization_id",
    "blueprint_to_plan_id",
    "blueprint_to_split_id",
    "proposal_to_policy_id",
    "admission_to_policy_id",
    "admission_to_proposal_id",
    "release_to_blueprint_id",
    "compute_to_release_id",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _callable_evidence(row: dict[str, Any], project_root: Path) -> dict[str, Any]:
    module = importlib.import_module(row["module"])
    callback = getattr(module, row["callable"], None)
    module_path = Path(str(module.__file__)).resolve()
    if not module_path.is_relative_to(project_root.resolve()) or not callable(callback):
        raise ValueError(f"semantic verifier evidence differs for {row['node']}")
    return {
        "module": row["module"],
        "callable": row["callable"],
        "module_path": module_path.relative_to(project_root.resolve()).as_posix(),
        "module_sha256": sha256_file(module_path),
        "callable_exists": True,
    }


def build_audit(*, project_root: Path, graph_contract_path: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    contract = _load(graph_contract_path)
    contract_body = dict(contract)
    contract_id = str(contract_body.pop("graph_contract_id", ""))
    if contract_id != content_sha256(contract_body):
        raise ValueError("cross-artifact graph contract address differs")
    graph_specs = {row["node"]: row for row in contract.get("node_specs", [])}
    implementation = {row["node"]: row for row in IMPLEMENTATION_ROWS}
    if set(graph_specs) != set(implementation) or len(EDGE_SPECS) != 37:
        raise ValueError("semantic coverage audit inventory differs from graph contract")

    node_rows = []
    for name in (row["node"] for row in NODE_SPECS):
        graph = graph_specs[name]
        actual = implementation[name]
        schema_matches = graph["schema"] == actual["implemented_schema"]
        id_field_matches = graph["id_field"] == actual["implemented_id_field"]
        node_rows.append(
            {
                "node": name,
                "graph_schema": graph["schema"],
                "graph_id_field": graph["id_field"],
                "implemented_schema": actual["implemented_schema"],
                "implemented_id_field": actual["implemented_id_field"],
                "schema_matches": schema_matches,
                "id_field_matches": id_field_matches,
                "schema_and_id_match": schema_matches and id_field_matches,
                "verification_mode": actual["verification_mode"],
                "verifier": _callable_evidence(actual, project_root),
                "dispatch_ready_under_v1_graph": actual[
                    "dispatch_ready_under_v1_graph"
                ],
                "gap": actual["gap"],
            }
        )

    edge_rows = [
        {
            **edge,
            "implemented_field_path_and_target_match": edge["edge"]
            in EXACT_IMPLEMENTED_EDGE_NAMES,
            "status": (
                "exact_implemented_field_edge"
                if edge["edge"] in EXACT_IMPLEMENTED_EDGE_NAMES
                else "representation_or_lineage_mismatch"
            ),
        }
        for edge in EDGE_SPECS
    ]
    if before_frameworks != (MODEL_FRAMEWORKS & set(sys.modules)):
        raise ValueError("semantic coverage audit imported a model framework")
    matching_nodes = sum(row["schema_and_id_match"] for row in node_rows)
    dispatch_ready = sum(row["dispatch_ready_under_v1_graph"] for row in node_rows)
    matching_edges = sum(
        row["implemented_field_path_and_target_match"] for row in edge_rows
    )
    if (matching_nodes, dispatch_ready, matching_edges) != (12, 4, 9):
        raise ValueError("semantic coverage audit expected mismatch counts differ")
    body = {
        "schema": SCHEMA,
        "status": "fail_closed_v1_graph_not_semantically_dispatchable",
        "source": {
            "graph_contract_path": graph_contract_path.relative_to(
                project_root
            ).as_posix(),
            "graph_contract_sha256": sha256_file(graph_contract_path),
            "graph_contract_id": contract_id,
            "audit_builder_sha256": sha256_file(Path(__file__)),
        },
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "summary": {
            "graph_nodes": len(node_rows),
            "schema_and_id_matching_nodes": matching_nodes,
            "dispatch_ready_nodes_under_v1_graph": dispatch_ready,
            "graph_edges": len(edge_rows),
            "exact_implemented_field_edges": matching_edges,
            "mismatched_edges": len(edge_rows) - matching_edges,
            "production_runtime_attachment_verifier_present": False,
            "semantic_dispatch_allowed": False,
        },
        "required_successor_corrections": [
            "preserve V1 graph as structural-gap evidence and do not mutate its sealed artifact",
            "replace raw capture-manifest lineage with guarded V3 capture execution receipts and their semantic sidecars",
            "use routing_evidence_bundle_id and its nested adapter routing and stability receipts",
            "represent global pair split and cluster-aware plan as embedded materialization content unless separately persisted by a new builder",
            "add a budget proposal/admission successor that binds the exact execution blueprint ID and file hash",
            "use signed compute authorization receipt schema and receipt_id",
            "represent nonce output and worker reservation with the implemented combined job-transition receipt",
            "keep inert runtime attachment synthetic-only and leave production attachment absent until separately implemented and authorized",
            "define successor edges from implemented field paths and benchmark every semantic join before dispatch",
        ],
        "scientific_boundary": {
            "audit_only": True,
            "v1_structural_graph_verified_previously": True,
            "v1_semantic_graph_verified": False,
            "empirical_graph_present": False,
            "semantic_verifiers_invoked_on_empirical_inputs": False,
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
    return {**body, "semantic_dispatch_coverage_audit_id": content_sha256(body)}


def verify_audit(
    path: Path, *, project_root: Path, graph_contract_path: Path
) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("semantic_dispatch_coverage_audit_id", ""))
    if identifier != content_sha256(body) or observed != build_audit(
        project_root=project_root, graph_contract_path=graph_contract_path
    ):
        raise ValueError("semantic dispatch coverage audit is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-semantic-dispatch-coverage-audit-verification.v1",
        "status": "pass",
        "semantic_dispatch_coverage_audit_id": identifier,
        **observed["summary"],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Path]:
    return {
        "project_root": project_root,
        "graph_contract_path": project_root
        / "artifacts/vpd_cluster_aware_sae_cross_artifact_graph_contract_v1/contract.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify semantic-dispatch coverage audit V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_audit(**inputs)
        write_json(args.output, result)
    else:
        result = verify_audit(args.verify, **inputs)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result.get("status", "audited"),
                "semantic_dispatch_coverage_audit_id": result.get(
                    "semantic_dispatch_coverage_audit_id"
                ),
                "summary": result.get("summary", result),
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
