from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1 import (
    canonical_inputs as budget_policy_inputs,
    verify_budget_policy,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_empirical_budget_binding_readiness_v1 import (
    canonical_inputs as readiness_inputs,
    verify_binding_readiness,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    REQUIRED_EXECUTION_BINDINGS,
    build_readiness as build_capture_binding_readiness,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


CONTRACT_SCHEMA = "sch.vpd-cluster-aware-sae-cross-artifact-graph-contract.v1"
MANIFEST_SCHEMA = "sch.vpd-cluster-aware-sae-cross-artifact-graph-manifest.v1"
ALLOWED_GRAPH_PARENT = "artifacts/vpd_cluster_aware_sae_empirical_graph_inputs"


NODE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "node": "empirical_preflight_v2",
        "phase": "empirical_capture",
        "schema": "sch.vpd-cluster-to-sae-empirical-preflight.v2",
        "id_field": "preflight_id",
        "address_excluded_fields": ["preflight_id"],
    },
    {
        "node": "capture_authority_registry",
        "phase": "empirical_capture",
        "schema": "sch.vpd-cluster-to-sae-capture-authority-registry.v1",
        "id_field": "registry_id",
        "address_excluded_fields": ["registry_id"],
    },
    {
        "node": "capture_authorization_receipt",
        "phase": "empirical_capture",
        "schema": "sch.vpd-cluster-to-sae-capture-authorization-receipt.v1",
        "id_field": "receipt_id",
        "address_excluded_fields": ["receipt_id", "signature"],
    },
    {
        "node": "empirical_capture_manifest",
        "phase": "empirical_capture",
        "schema": "sch.vpd-activation-capture-manifest.v1",
        "id_field": "capture_manifest_id",
        "address_excluded_fields": ["capture_manifest_id"],
    },
    {
        "node": "empirical_routing_bundle",
        "phase": "routing_and_partition",
        "schema": "sch.vpd-guarded-cluster-routing-evidence-bundle.v1",
        "id_field": "routing_bundle_id",
        "address_excluded_fields": ["routing_bundle_id"],
    },
    {
        "node": "routing_stability_manifest",
        "phase": "routing_and_partition",
        "schema": "sch.vpd-cluster-routing-stability-manifest.v1",
        "id_field": "routing_stability_id",
        "address_excluded_fields": ["routing_stability_id"],
    },
    {
        "node": "global_pair_split",
        "phase": "routing_and_partition",
        "schema": "sch.vpd-cluster-aware-sae-global-pair-split.v2",
        "id_field": "global_pair_split_id",
        "address_excluded_fields": ["global_pair_split_id"],
    },
    {
        "node": "cluster_aware_sae_plan",
        "phase": "routing_and_partition",
        "schema": "sch.vpd-cluster-aware-sae-training-plan.v2",
        "id_field": "cluster_aware_sae_plan_id",
        "address_excluded_fields": ["cluster_aware_sae_plan_id"],
    },
    {
        "node": "materialization",
        "phase": "routing_and_partition",
        "schema": "sch.vpd-cluster-aware-sae-plan-materialization.v1",
        "id_field": "materialization_id",
        "address_excluded_fields": ["materialization_id"],
    },
    {
        "node": "execution_blueprint",
        "phase": "execution_blueprint_and_budget",
        "schema": "sch.vpd-cluster-aware-sae-execution-blueprint.v1",
        "id_field": "execution_blueprint_id",
        "address_excluded_fields": ["execution_blueprint_id"],
    },
    {
        "node": "budget_proposal",
        "phase": "execution_blueprint_and_budget",
        "schema": "sch.vpd-cluster-aware-sae-budget-proposal.v1",
        "id_field": "proposal_id",
        "address_excluded_fields": ["proposal_id"],
    },
    {
        "node": "budget_admission",
        "phase": "execution_blueprint_and_budget",
        "schema": "sch.vpd-cluster-aware-sae-budget-admission.v1",
        "id_field": "admission_id",
        "address_excluded_fields": ["admission_id"],
    },
    {
        "node": "release_authority_registry",
        "phase": "signed_release_and_compute_authority",
        "schema": "sch.vpd-cluster-aware-sae-authority-registry.v1",
        "id_field": "registry_id",
        "address_excluded_fields": ["registry_id"],
    },
    {
        "node": "scientific_release_receipt",
        "phase": "signed_release_and_compute_authority",
        "schema": "sch.vpd-cluster-aware-sae-scientific-release-receipt.v1",
        "id_field": "receipt_id",
        "address_excluded_fields": ["receipt_id", "signature"],
    },
    {
        "node": "compute_authority_registry",
        "phase": "signed_release_and_compute_authority",
        "schema": "sch.vpd-cluster-aware-sae-authority-registry.v1",
        "id_field": "registry_id",
        "address_excluded_fields": ["registry_id"],
    },
    {
        "node": "compute_authorization_manifest",
        "phase": "signed_release_and_compute_authority",
        "schema": "sch.vpd-cluster-aware-sae-compute-authorization-manifest.v1",
        "id_field": "compute_authorization_manifest_id",
        "address_excluded_fields": ["compute_authorization_manifest_id"],
    },
    {
        "node": "nonce_consumption_manifest",
        "phase": "durable_runtime_transition",
        "schema": "sch.vpd-cluster-aware-sae-nonce-consumption-manifest.v1",
        "id_field": "nonce_consumption_manifest_id",
        "address_excluded_fields": ["nonce_consumption_manifest_id"],
    },
    {
        "node": "output_reservation_manifest",
        "phase": "durable_runtime_transition",
        "schema": "sch.vpd-cluster-aware-sae-output-reservation-manifest.v1",
        "id_field": "output_reservation_manifest_id",
        "address_excluded_fields": ["output_reservation_manifest_id"],
    },
    {
        "node": "worker_ledger_claim",
        "phase": "durable_runtime_transition",
        "schema": "sch.vpd-cluster-aware-sae-worker-ledger-claim.v1",
        "id_field": "worker_ledger_claim_id",
        "address_excluded_fields": ["worker_ledger_claim_id"],
    },
    {
        "node": "runtime_attachment_claim",
        "phase": "durable_runtime_transition",
        "schema": "sch.vpd-cluster-aware-sae-runtime-attachment-claim.v1",
        "id_field": "runtime_attachment_claim_id",
        "address_excluded_fields": ["runtime_attachment_claim_id"],
    },
)


def _edge(
    name: str,
    source_node: str,
    source_field_path: str,
    target_node: str,
    target_value: str,
) -> dict[str, str]:
    return {
        "edge": name,
        "source_node": source_node,
        "source_field_path": source_field_path,
        "target_node": target_node,
        "target_value": target_value,
    }


EDGE_SPECS: tuple[dict[str, str], ...] = (
    _edge("capture_receipt_to_preflight_id", "capture_authorization_receipt", "source.preflight_id", "empirical_preflight_v2", "content_id"),
    _edge("capture_receipt_to_preflight_hash", "capture_authorization_receipt", "source.preflight_sha256", "empirical_preflight_v2", "file_sha256"),
    _edge("capture_receipt_to_registry_id", "capture_authorization_receipt", "source.registry_id", "capture_authority_registry", "content_id"),
    _edge("capture_manifest_to_preflight_id", "empirical_capture_manifest", "source.preflight_id", "empirical_preflight_v2", "content_id"),
    _edge("capture_manifest_to_receipt_id", "empirical_capture_manifest", "source.capture_authorization_receipt_id", "capture_authorization_receipt", "content_id"),
    _edge("routing_bundle_to_capture_id", "empirical_routing_bundle", "source.capture_manifest_id", "empirical_capture_manifest", "content_id"),
    _edge("routing_bundle_to_capture_hash", "empirical_routing_bundle", "source.capture_manifest_sha256", "empirical_capture_manifest", "file_sha256"),
    _edge("stability_to_routing_id", "routing_stability_manifest", "source.routing_bundle_id", "empirical_routing_bundle", "content_id"),
    _edge("split_to_routing_id", "global_pair_split", "source.routing_bundle_id", "empirical_routing_bundle", "content_id"),
    _edge("split_to_stability_id", "global_pair_split", "source.routing_stability_id", "routing_stability_manifest", "content_id"),
    _edge("plan_to_routing_id", "cluster_aware_sae_plan", "source.routing_bundle_id", "empirical_routing_bundle", "content_id"),
    _edge("plan_to_stability_id", "cluster_aware_sae_plan", "source.routing_stability_id", "routing_stability_manifest", "content_id"),
    _edge("plan_to_split_id", "cluster_aware_sae_plan", "source.global_pair_split_id", "global_pair_split", "content_id"),
    _edge("materialization_to_plan_id", "materialization", "source.cluster_aware_sae_plan_id", "cluster_aware_sae_plan", "content_id"),
    _edge("materialization_to_split_id", "materialization", "source.global_pair_split_id", "global_pair_split", "content_id"),
    _edge("blueprint_to_materialization_id", "execution_blueprint", "source.materialization_id", "materialization", "content_id"),
    _edge("blueprint_to_plan_id", "execution_blueprint", "source.cluster_aware_sae_plan_id", "cluster_aware_sae_plan", "content_id"),
    _edge("blueprint_to_split_id", "execution_blueprint", "source.global_pair_split_id", "global_pair_split", "content_id"),
    _edge("proposal_to_policy_id", "budget_proposal", "source.budget_policy_id", "__budget_policy__", "content_id"),
    _edge("proposal_to_blueprint_id", "budget_proposal", "source.execution_blueprint_id", "execution_blueprint", "content_id"),
    _edge("proposal_to_blueprint_hash", "budget_proposal", "source.execution_blueprint_sha256", "execution_blueprint", "file_sha256"),
    _edge("admission_to_policy_id", "budget_admission", "source.budget_policy_id", "__budget_policy__", "content_id"),
    _edge("admission_to_proposal_id", "budget_admission", "source.proposal_id", "budget_proposal", "content_id"),
    _edge("admission_to_blueprint_id", "budget_admission", "source.execution_blueprint_id", "execution_blueprint", "content_id"),
    _edge("release_to_registry_id", "scientific_release_receipt", "source.registry_id", "release_authority_registry", "content_id"),
    _edge("release_to_blueprint_id", "scientific_release_receipt", "source.execution_blueprint_id", "execution_blueprint", "content_id"),
    _edge("release_to_admission_id", "scientific_release_receipt", "source.budget_admission_id", "budget_admission", "content_id"),
    _edge("compute_to_registry_id", "compute_authorization_manifest", "source.registry_id", "compute_authority_registry", "content_id"),
    _edge("compute_to_release_id", "compute_authorization_manifest", "source.scientific_release_receipt_id", "scientific_release_receipt", "content_id"),
    _edge("compute_to_blueprint_id", "compute_authorization_manifest", "source.execution_blueprint_id", "execution_blueprint", "content_id"),
    _edge("nonce_to_compute_id", "nonce_consumption_manifest", "source.compute_authorization_manifest_id", "compute_authorization_manifest", "content_id"),
    _edge("reservation_to_compute_id", "output_reservation_manifest", "source.compute_authorization_manifest_id", "compute_authorization_manifest", "content_id"),
    _edge("reservation_to_nonce_id", "output_reservation_manifest", "source.nonce_consumption_manifest_id", "nonce_consumption_manifest", "content_id"),
    _edge("worker_to_reservation_id", "worker_ledger_claim", "source.output_reservation_manifest_id", "output_reservation_manifest", "content_id"),
    _edge("worker_to_compute_id", "worker_ledger_claim", "source.compute_authorization_manifest_id", "compute_authorization_manifest", "content_id"),
    _edge("runtime_to_worker_id", "runtime_attachment_claim", "source.worker_ledger_claim_id", "worker_ledger_claim", "content_id"),
    _edge("runtime_to_reservation_id", "runtime_attachment_claim", "source.output_reservation_manifest_id", "output_reservation_manifest", "content_id"),
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _resolve_field(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"cross-artifact edge field {dotted_path} is absent")
        current = current[part]
    return current


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


def _safe_graph_root(project_root: Path, graph_root_value: Any) -> Path:
    if not isinstance(graph_root_value, str) or not graph_root_value:
        raise ValueError("cross-artifact graph root is absent")
    relative = Path(graph_root_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("cross-artifact graph root must be a safe relative path")
    if _has_reparse_component(project_root, relative):
        raise ValueError("cross-artifact graph root must not contain a reparse point")
    resolved = (project_root / relative).resolve()
    allowed = (project_root / ALLOWED_GRAPH_PARENT).resolve()
    if not resolved.is_relative_to(allowed) or resolved == allowed:
        raise ValueError("cross-artifact graph root escapes its dedicated parent")
    if not resolved.is_dir():
        raise ValueError("cross-artifact graph root is absent or linked")
    return resolved


def build_graph_contract(
    *,
    project_root: Path,
    binding_readiness_path: Path,
    budget_policy_path: Path,
) -> dict[str, Any]:
    readiness = verify_binding_readiness(
        binding_readiness_path, **readiness_inputs(project_root)
    )
    policy = verify_budget_policy(
        budget_policy_path, **budget_policy_inputs(project_root)
    )
    body = {
        "schema": CONTRACT_SCHEMA,
        "status": "sealed_nonexecuting_exact_20_node_37_edge_graph_contract",
        "source": {
            "binding_readiness_path": binding_readiness_path.relative_to(project_root).as_posix(),
            "binding_readiness_sha256": sha256_file(binding_readiness_path),
            "binding_readiness_id": readiness["binding_readiness_id"],
            "budget_policy_path": budget_policy_path.relative_to(project_root).as_posix(),
            "budget_policy_sha256": sha256_file(budget_policy_path),
            "budget_policy_id": policy["budget_policy_id"],
            "builder_sha256": sha256_file(Path(__file__)),
        },
        "required_execution_identity_bindings": list(REQUIRED_EXECUTION_BINDINGS),
        "node_specs": list(NODE_SPECS),
        "edge_specs": list(EDGE_SPECS),
        "graph_contract": {
            "required_nodes": 20,
            "required_edges": len(EDGE_SPECS),
            "all_node_files_must_be_json_objects": True,
            "all_node_paths_must_remain_under_dedicated_graph_root": True,
            "all_node_file_hashes_and_content_ids_must_match": True,
            "all_content_addresses_use_declared_exclusions": True,
            "all_37_edges_must_resolve_exactly": len(EDGE_SPECS) == 37,
            "all_ten_execution_identity_bindings_must_match_preflight": True,
            "synthetic_graph_can_only_prove_structural_integrity": True,
            "structural_integrity_transfers_execution_authority": False,
        },
        "scientific_boundary": {
            "contract_only": True,
            "empirical_graph_supplied": False,
            "model_or_adapter_content_opened": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "weights_loaded_or_mutated": False,
            "persistent_dictionary_output_created": False,
            "signed_authority_verified": False,
            "durable_runtime_transition_verified": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    return {**body, "graph_contract_id": content_sha256(body)}


def verify_graph_contract(
    path: Path,
    *,
    project_root: Path,
    binding_readiness_path: Path,
    budget_policy_path: Path,
) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("graph_contract_id", ""))
    if identifier != content_sha256(body) or observed != build_graph_contract(
        project_root=project_root,
        binding_readiness_path=binding_readiness_path,
        budget_policy_path=budget_policy_path,
    ):
        raise ValueError("cross-artifact graph contract is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-cross-artifact-graph-contract-verification.v1",
        "status": "pass",
        "graph_contract_id": identifier,
        "required_nodes": len(observed["node_specs"]),
        "required_edges": len(observed["edge_specs"]),
        "execution_identity_bindings": len(
            observed["required_execution_identity_bindings"]
        ),
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def validate_graph_manifest(
    *,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    contract_body = dict(contract)
    contract_id = str(contract_body.pop("graph_contract_id", ""))
    if contract.get("schema") != CONTRACT_SCHEMA or contract_id != content_sha256(
        contract_body
    ):
        raise ValueError("cross-artifact graph contract content address differs")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("cross-artifact graph manifest schema differs")
    manifest_body = dict(manifest)
    manifest_id = str(manifest_body.pop("graph_manifest_id", ""))
    if manifest_id != content_sha256(manifest_body):
        raise ValueError("cross-artifact graph manifest content address differs")
    if manifest.get("graph_contract_id") != contract_id:
        raise ValueError("cross-artifact graph manifest binds a different contract")
    if manifest.get("graph_kind") not in {
        "synthetic_benchmark_fixture",
        "empirical_candidate",
    }:
        raise ValueError("cross-artifact graph kind differs")

    execution_bindings = manifest.get("execution_identity_bindings")
    if not isinstance(execution_bindings, dict):
        raise ValueError("cross-artifact execution identity bindings are absent")
    graph_root = _safe_graph_root(project_root, manifest.get("graph_root"))
    node_specs = {row["node"]: row for row in contract["node_specs"]}
    node_rows = manifest.get("nodes")
    if not isinstance(node_rows, list):
        raise ValueError("cross-artifact graph nodes are absent")
    by_name: dict[str, dict[str, Any]] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for row in node_rows:
        if not isinstance(row, dict):
            raise ValueError("cross-artifact graph node descriptor differs")
        name = str(row.get("node", ""))
        if name in by_name or name not in node_specs:
            raise ValueError("cross-artifact graph node is duplicated or unknown")
        if set(row) != {"node", "path", "file_sha256", "content_id"}:
            raise ValueError("cross-artifact graph node descriptor fields differ")
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("cross-artifact node path is unsafe")
        if _has_reparse_component(project_root, relative):
            raise ValueError("cross-artifact node path contains a reparse point")
        path = (project_root / relative).resolve()
        if not path.is_relative_to(graph_root) or not path.is_file():
            raise ValueError("cross-artifact node file escapes graph root or is absent")
        if sha256_file(path) != row["file_sha256"]:
            raise ValueError("cross-artifact node file SHA-256 differs")
        document = _load(path)
        spec = node_specs[name]
        if document.get("schema") != spec["schema"]:
            raise ValueError("cross-artifact node schema differs")
        if not isinstance(document.get("synthetic_fixture_only"), bool):
            raise ValueError("cross-artifact node synthetic boundary is absent")
        content_id = str(document.get(spec["id_field"], ""))
        if content_id != row["content_id"]:
            raise ValueError("cross-artifact node content ID differs")
        address_body = {
            key: value
            for key, value in document.items()
            if key not in set(spec["address_excluded_fields"])
        }
        if content_id != content_sha256(address_body):
            raise ValueError("cross-artifact node content address differs")
        by_name[name] = row
        loaded[name] = document
    if set(by_name) != set(node_specs):
        raise ValueError("cross-artifact graph must contain all twenty nodes")
    preflight_bindings = loaded["empirical_preflight_v2"].get("immutable_bindings")
    if preflight_bindings != execution_bindings:
        raise ValueError("cross-artifact preflight execution identity binding differs")

    policy_target = {
        "content_id": contract["source"]["budget_policy_id"],
        "file_sha256": contract["source"]["budget_policy_sha256"],
    }
    for edge in contract["edge_specs"]:
        actual = _resolve_field(loaded[edge["source_node"]], edge["source_field_path"])
        target = (
            policy_target
            if edge["target_node"] == "__budget_policy__"
            else by_name[edge["target_node"]]
        )
        if actual != target[edge["target_value"]]:
            raise ValueError(f"cross-artifact edge {edge['edge']} differs")

    synthetic_values = {
        document["synthetic_fixture_only"] for document in loaded.values()
    }
    if len(synthetic_values) != 1:
        raise ValueError("cross-artifact graph mixes synthetic and empirical nodes")
    synthetic = synthetic_values == {True}
    if synthetic != (manifest["graph_kind"] == "synthetic_benchmark_fixture"):
        raise ValueError("cross-artifact graph kind and node boundary differ")
    capture_readiness = build_capture_binding_readiness(
        project_root=project_root, bindings=execution_bindings
    )
    if capture_readiness["summary"]["missing_bindings"]:
        raise ValueError("cross-artifact graph requires all execution identity bindings")
    body = {
        "schema": "sch.vpd-cluster-aware-sae-cross-artifact-graph-verification.v1",
        "status": (
            "synthetic_graph_structurally_verified_non_authorizing"
            if synthetic
            else "empirical_candidate_graph_structurally_verified_non_authorizing"
        ),
        "source": {
            "graph_contract_id": contract_id,
            "graph_manifest_id": manifest_id,
        },
        "verified": {
            "nodes": len(by_name),
            "edges": len(contract["edge_specs"]),
            "execution_identity_bindings": len(execution_bindings),
            "file_hashes": len(by_name),
            "content_addresses": len(by_name),
            "graph_integrity_verified": True,
            "synthetic_fixture_only": synthetic,
        },
        "scientific_boundary": {
            "structural_integrity_only": True,
            "node_specific_semantic_verifiers_invoked": False,
            "receipt_signatures_verified": False,
            "authority_status_verified": False,
            "nonce_freshness_verified": False,
            "output_reservation_exclusivity_verified": False,
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
    return {**body, "graph_verification_id": content_sha256(body)}


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "project_root": project_root,
        "binding_readiness_path": project_root
        / "artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1/readiness.json",
        "budget_policy_path": project_root
        / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/verify the cross-artifact graph contract or validate a supplied manifest."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-contract", type=Path)
    mode.add_argument("--verify-contract", type=Path)
    mode.add_argument("--validate-manifest", type=Path)
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    inputs = canonical_inputs(root)
    if args.output_contract is not None:
        result = build_graph_contract(**inputs)
        write_json(args.output_contract, result)
    elif args.verify_contract is not None:
        if args.contract is not None:
            raise ValueError("--contract is only valid with --validate-manifest")
        result = verify_graph_contract(args.verify_contract, **inputs)
    else:
        if args.contract is None:
            raise ValueError("--validate-manifest requires --contract")
        result = validate_graph_manifest(
            contract=_load(args.contract),
            manifest=_load(args.validate_manifest),
            project_root=root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
