from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1 import (
    canonical_inputs as budget_policy_inputs,
    verify_budget_policy,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


CONTRACT_SCHEMA = "sch.vpd-cluster-aware-sae-semantic-graph-contract.v2"
AUDIT_SCHEMA = "sch.vpd-cluster-aware-sae-semantic-dispatch-coverage-audit.v1"
BUDGET_BENCHMARK_SCHEMA = (
    "sch.vpd-cluster-aware-sae-blueprint-bound-budget-admission-benchmark.v2"
)
AUDIT_ID = "bd3cc043f7b658029fd20c99492c43edbace23550a7e237d36d692e46297eaa7"
BUDGET_BENCHMARK_ID = (
    "1c239f6416c57374a810732496c20f36a7245d882b41c2282c72b8465678da58"
)


def _node(
    name: str,
    phase: str,
    schema: str | None,
    id_field: str | None,
    module: str | None,
    callback: str | None,
    verification_mode: str,
    required_companions: tuple[str, ...] = (),
    address_excluded_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "node": name,
        "phase": phase,
        "schema": schema,
        "id_field": id_field,
        "module": module,
        "callable": callback,
        "verification_mode": verification_mode,
        "required_companions": list(required_companions),
        "address_excluded_fields": list(address_excluded_fields),
        "implemented": module is not None and callback is not None,
    }


NODE_SPECS: tuple[dict[str, Any], ...] = (
    _node(
        "empirical_preflight_v2",
        "empirical_capture",
        "sch.vpd-cluster-to-sae-empirical-preflight.v2",
        "preflight_id",
        "experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2",
        "verify_preflight",
        "standalone_path_reproduction",
        address_excluded_fields=("preflight_id",),
    ),
    _node(
        "capture_authority_registry",
        "empirical_capture",
        "sch.vpd-cluster-to-sae-capture-authority-registry.v1",
        "registry_id",
        "experiments.authz_mechinterp.verify_vpd_cluster_to_sae_capture_authorization_v1",
        "verify_authorization_receipt",
        "joint_cryptographic_registry_receipt_preflight_as_of",
        required_companions=("capture_authorization_receipt", "empirical_preflight_v2", "as_of"),
        address_excluded_fields=("registry_id",),
    ),
    _node(
        "capture_authorization_receipt",
        "empirical_capture",
        "sch.vpd-cluster-to-sae-capture-authorization-receipt.v1",
        "authorization_receipt_id",
        "experiments.authz_mechinterp.verify_vpd_cluster_to_sae_capture_authorization_v1",
        "verify_authorization_receipt",
        "joint_cryptographic_registry_receipt_preflight_as_of",
        required_companions=("capture_authority_registry", "empirical_preflight_v2", "as_of"),
        address_excluded_fields=("authorization_receipt_id", "signature"),
    ),
    _node(
        "guarded_v3_capture_receipt",
        "empirical_capture",
        "sch.vpd-canonical-category-capture-job-execution.v3",
        "capture_execution_v3_id",
        "experiments.authz_mechinterp.run_canonical_vpd_capture_job_v3",
        "verify_capture_execution_receipt",
        "path_plus_plan_authorization_v2_and_guard_receipt_companions",
        required_companions=(
            "capture_plan",
            "capture_authorization",
            "predecessor_v2_execution",
            "guarded_job_receipt",
            "allocation_manifest_if_empirical",
            "reservation_receipt_if_empirical",
        ),
        address_excluded_fields=("capture_execution_v3_id",),
    ),
    _node(
        "capture_execution_v2_receipt",
        "empirical_capture",
        "sch.vpd-canonical-category-capture-job-execution.v2",
        "capture_execution_id",
        "experiments.authz_mechinterp.run_canonical_vpd_capture_job_v2",
        "verify_capture_execution_receipt",
        "path_plus_plan_authorization_and_capture_artifact_companions",
        required_companions=(
            "capture_plan",
            "capture_authorization",
            "capture_manifest",
            "effect_manifest",
            "semantic_verification_receipt",
        ),
        address_excluded_fields=("capture_execution_id",),
    ),
    _node(
        "routing_evidence_bundle",
        "routing_and_partition",
        "sch.vpd-guarded-cluster-routing-evidence-bundle.v1",
        "routing_evidence_bundle_id",
        "experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1",
        "load_and_verify_routing_evidence_bundle",
        "path_loading_adapter_routing_and_stability_receipts",
        required_companions=(
            "24_capture_router_adapter_receipts",
            "24_routing_receipts",
            "8_routing_stability_receipts",
        ),
        address_excluded_fields=("routing_evidence_bundle_id",),
    ),
    _node(
        "routing_stability_manifest",
        "routing_and_partition",
        "sch.vpd-cluster-routing-stability-manifest.v1",
        "routing_stability_id",
        "experiments.authz_mechinterp.route_vpd_cluster_episodes_v1",
        "verify_routing_stability_v1",
        "standalone_document_and_joined_inside_routing_bundle",
        address_excluded_fields=("routing_stability_id",),
    ),
    _node(
        "predecessor_training_plan",
        "routing_and_partition",
        "sch.vpd-compartment-sae-training-plan.v2",
        "training_plan_id",
        "experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1",
        "build_materialization",
        "content_address_and_full_reproduction_companion",
        address_excluded_fields=("training_plan_id",),
    ),
    _node(
        "cluster_aware_sae_plan_with_embedded_split",
        "routing_and_partition",
        "sch.vpd-cluster-aware-sae-training-plan.v2",
        "cluster_aware_sae_plan_id",
        "experiments.authz_mechinterp.plan_cluster_aware_sae_training_v2",
        "verify_cluster_aware_sae_plan_v2",
        "standalone_document_with_embedded_global_pair_split",
        required_companions=("embedded_global_pair_split",),
        address_excluded_fields=("cluster_aware_sae_plan_id",),
    ),
    _node(
        "materialization",
        "routing_and_partition",
        "sch.vpd-cluster-aware-sae-plan-materialization.v1",
        "materialization_id",
        "experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1",
        "verify_materialization",
        "path_plus_routing_bundle_and_predecessor_plan_full_reproduction",
        required_companions=("routing_evidence_bundle", "predecessor_training_plan"),
        address_excluded_fields=("materialization_id",),
    ),
    _node(
        "execution_blueprint",
        "execution_blueprint_and_budget",
        "sch.vpd-cluster-aware-sae-execution-blueprint.v1",
        "execution_blueprint_id",
        "experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1",
        "verify_execution_blueprint",
        "document_plus_materialization_full_reproduction",
        required_companions=("materialization",),
        address_excluded_fields=("execution_blueprint_id",),
    ),
    _node(
        "budget_policy",
        "execution_blueprint_and_budget",
        "sch.vpd-cluster-aware-sae-budget-admission-policy.v1",
        "budget_policy_id",
        "experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1",
        "verify_budget_policy",
        "canonical_pinned_policy_reproduction",
        address_excluded_fields=("budget_policy_id",),
    ),
    _node(
        "blueprint_bound_budget_proposal_v2",
        "execution_blueprint_and_budget",
        "sch.vpd-cluster-aware-sae-blueprint-bound-budget-proposal.v2",
        "proposal_id",
        "experiments.authz_mechinterp.build_cluster_aware_sae_blueprint_budget_admission_v2",
        "verify_proposal",
        "path_plus_policy_and_exact_blueprint_full_reproduction",
        required_companions=("budget_policy", "execution_blueprint"),
        address_excluded_fields=("proposal_id",),
    ),
    _node(
        "blueprint_bound_budget_admission_v2",
        "execution_blueprint_and_budget",
        "sch.vpd-cluster-aware-sae-blueprint-bound-budget-admission.v2",
        "admission_id",
        "experiments.authz_mechinterp.build_cluster_aware_sae_blueprint_budget_admission_v2",
        "verify_admission",
        "path_plus_proposal_policy_and_exact_blueprint_full_reproduction",
        required_companions=(
            "blueprint_bound_budget_proposal_v2",
            "budget_policy",
            "execution_blueprint",
        ),
        address_excluded_fields=("admission_id",),
    ),
    _node(
        "release_authority_registry",
        "signed_release_and_compute_authority",
        "sch.vpd-cluster-aware-sae-authority-registry.v1",
        "registry_id",
        "experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1",
        "verify_scientific_release_receipt",
        "joint_cryptographic_registry_receipt_blueprint_as_of",
        required_companions=("scientific_release_receipt", "execution_blueprint", "as_of"),
        address_excluded_fields=("registry_id",),
    ),
    _node(
        "scientific_release_receipt",
        "signed_release_and_compute_authority",
        "sch.vpd-cluster-aware-sae-scientific-release-receipt.v1",
        "receipt_id",
        "experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1",
        "verify_scientific_release_receipt",
        "joint_cryptographic_registry_receipt_blueprint_as_of",
        required_companions=("release_authority_registry", "execution_blueprint", "as_of"),
        address_excluded_fields=("receipt_id", "signature"),
    ),
    _node(
        "compute_authority_registry",
        "signed_release_and_compute_authority",
        "sch.vpd-cluster-aware-sae-authority-registry.v1",
        "registry_id",
        "experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1",
        "verify_compute_authorization_receipt",
        "joint_cryptographic_registry_receipt_release_blueprint_job_device_as_of",
        required_companions=(
            "compute_authorization_receipt",
            "scientific_release_verification",
            "execution_blueprint",
            "execution_job_id",
            "device",
            "as_of",
        ),
        address_excluded_fields=("registry_id",),
    ),
    _node(
        "compute_authorization_receipt",
        "signed_release_and_compute_authority",
        "sch.vpd-cluster-aware-sae-compute-authorization-receipt.v1",
        "receipt_id",
        "experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1",
        "verify_compute_authorization_receipt",
        "joint_cryptographic_registry_receipt_release_blueprint_job_device_as_of",
        required_companions=(
            "compute_authority_registry",
            "scientific_release_verification",
            "execution_blueprint",
            "execution_job_id",
            "device",
            "as_of",
        ),
        address_excluded_fields=("receipt_id", "signature"),
    ),
    _node(
        "runtime_admission",
        "durable_runtime_transition",
        "sch.vpd-cluster-aware-sae-runtime-admission.v1",
        "admission_id",
        "experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1",
        "build_dry_run_admission",
        "full_reproduction_from_blueprint_release_compute_job_device_and_as_of",
        required_companions=(
            "execution_blueprint",
            "scientific_release_receipt",
            "release_authority_registry",
            "compute_authorization_receipt",
            "compute_authority_registry",
            "execution_job_id",
            "device",
            "as_of",
        ),
        address_excluded_fields=("admission_id",),
    ),
    _node(
        "durable_job_transition",
        "durable_runtime_transition",
        "sch.vpd-cluster-aware-sae-job-transition-reservation.v1",
        "transition_reservation_id",
        "experiments.authz_mechinterp.reserve_cluster_aware_sae_training_transition_v1",
        "verify_training_transition",
        "single_combined_nonce_output_worker_ledger_transition_receipt",
        required_companions=("reserved_job_root", "campaign_receipt", "worker_ledger"),
        address_excluded_fields=("transition_reservation_id",),
    ),
    _node(
        "production_runtime_attachment",
        "durable_runtime_transition",
        None,
        None,
        None,
        None,
        "terminal_prerequisite_gap_no_production_verifier",
    ),
)


def _join(
    name: str,
    source: str,
    target: str,
    kind: str,
    evidence: str,
    source_path: str | None = None,
) -> dict[str, Any]:
    return {
        "join": name,
        "source_node": source,
        "target_node": target,
        "join_kind": kind,
        "source_field_path": source_path,
        "semantic_evidence": evidence,
    }


JOIN_SPECS: tuple[dict[str, Any], ...] = (
    _join("capture_receipt_to_preflight_id", "capture_authorization_receipt", "empirical_preflight_v2", "exact_field_to_content_id", "verify_authorization_receipt", "source.preflight_id"),
    _join("capture_receipt_to_preflight_canonical_digest", "capture_authorization_receipt", "empirical_preflight_v2", "exact_field_to_canonical_object_digest_not_file_hash", "verify_authorization_receipt", "source.preflight_sha256"),
    _join("capture_receipt_to_registry", "capture_authorization_receipt", "capture_authority_registry", "cryptographic_authority_membership_join", "verify_authorization_receipt", "authority_id"),
    _join("v3_capture_to_predecessor_v2", "guarded_v3_capture_receipt", "capture_execution_v2_receipt", "compound_receipt_internal_predecessor", "verify_capture_execution_receipt", "predecessor_v2_execution"),
    _join("routing_bundle_to_guarded_v3_captures", "routing_evidence_bundle", "guarded_v3_capture_receipt", "nested_adapter_receipt_identity_join", "load_and_verify_routing_evidence_bundle", "categories.*.roles.*.adapter"),
    _join("routing_bundle_to_stability", "routing_evidence_bundle", "routing_stability_manifest", "nested_path_hash_id_and_semantic_join", "load_and_verify_routing_evidence_bundle", "categories.*.stability"),
    _join("materialization_to_routing_bundle_id", "materialization", "routing_evidence_bundle", "exact_field_to_content_id", "verify_materialization", "source.routing_evidence_bundle_id"),
    _join("materialization_to_routing_bundle_hash", "materialization", "routing_evidence_bundle", "exact_field_to_file_sha256", "verify_materialization", "source.routing_evidence_bundle_sha256"),
    _join("materialization_to_predecessor_plan_id", "materialization", "predecessor_training_plan", "exact_field_to_content_id", "verify_materialization", "source.predecessor_training_plan_id"),
    _join("materialization_to_predecessor_plan_hash", "materialization", "predecessor_training_plan", "exact_field_to_file_sha256", "verify_materialization", "source.predecessor_training_plan_sha256"),
    _join("materialization_to_plan", "materialization", "cluster_aware_sae_plan_with_embedded_split", "embedded_full_object_and_verification", "verify_materialization", "cluster_aware_sae_plan"),
    _join("plan_to_global_split", "cluster_aware_sae_plan_with_embedded_split", "cluster_aware_sae_plan_with_embedded_split", "embedded_content_addressed_subdocument", "verify_cluster_aware_sae_plan_v2", "global_pair_split"),
    _join("blueprint_to_materialization_id", "execution_blueprint", "materialization", "exact_field_to_content_id_plus_full_reproduction", "verify_execution_blueprint", "source.materialization_id"),
    _join("blueprint_to_plan_id", "execution_blueprint", "cluster_aware_sae_plan_with_embedded_split", "exact_field_to_embedded_content_id", "verify_execution_blueprint", "source.cluster_aware_sae_plan_id"),
    _join("blueprint_to_routing_bundle_id", "execution_blueprint", "routing_evidence_bundle", "exact_field_to_content_id", "verify_execution_blueprint", "source.routing_evidence_bundle_id"),
    _join("blueprint_to_split_id", "execution_blueprint", "cluster_aware_sae_plan_with_embedded_split", "exact_field_to_embedded_subdocument_id", "verify_execution_blueprint", "source.global_pair_split_id"),
    _join("proposal_to_policy_id", "blueprint_bound_budget_proposal_v2", "budget_policy", "exact_field_to_content_id", "verify_proposal", "source.budget_policy_id"),
    _join("proposal_to_blueprint_id", "blueprint_bound_budget_proposal_v2", "execution_blueprint", "exact_field_to_content_id_plus_full_resource_derivation", "verify_proposal", "source.execution_blueprint_id"),
    _join("proposal_to_blueprint_hash", "blueprint_bound_budget_proposal_v2", "execution_blueprint", "exact_field_to_file_sha256", "verify_proposal", "source.execution_blueprint_sha256"),
    _join("budget_admission_to_policy_id", "blueprint_bound_budget_admission_v2", "budget_policy", "exact_field_to_content_id", "verify_admission", "source.budget_policy_id"),
    _join("budget_admission_to_proposal_id", "blueprint_bound_budget_admission_v2", "blueprint_bound_budget_proposal_v2", "exact_field_to_content_id_plus_full_reproduction", "verify_admission", "source.proposal_id"),
    _join("budget_admission_to_blueprint_id", "blueprint_bound_budget_admission_v2", "execution_blueprint", "exact_field_to_content_id", "verify_admission", "source.execution_blueprint_id"),
    _join("budget_admission_to_blueprint_hash", "blueprint_bound_budget_admission_v2", "execution_blueprint", "exact_field_to_file_sha256", "verify_admission", "source.execution_blueprint_sha256"),
    _join("release_receipt_to_registry", "scientific_release_receipt", "release_authority_registry", "cryptographic_authority_membership_join", "verify_scientific_release_receipt", "authority_id"),
    _join("release_receipt_to_blueprint_id", "scientific_release_receipt", "execution_blueprint", "exact_signed_field_and_scope_join", "verify_scientific_release_receipt", "source.execution_blueprint_id"),
    _join("release_receipt_to_blueprint_hash", "scientific_release_receipt", "execution_blueprint", "exact_signed_field_to_file_sha256", "verify_scientific_release_receipt", "source.execution_blueprint_sha256"),
    _join("budget_admission_to_release_receipt", "scientific_release_receipt", "blueprint_bound_budget_admission_v2", "terminal_missing_signed_join", "no_implemented_verifier", None),
    _join("compute_receipt_to_registry", "compute_authorization_receipt", "compute_authority_registry", "cryptographic_authority_membership_join", "verify_compute_authorization_receipt", "authority_id"),
    _join("compute_receipt_to_release_id", "compute_authorization_receipt", "scientific_release_receipt", "exact_signed_field_to_verified_receipt_id", "verify_compute_authorization_receipt", "source.scientific_release_receipt_id"),
    _join("compute_receipt_to_release_hash", "compute_authorization_receipt", "scientific_release_receipt", "exact_signed_field_to_canonical_object_digest", "verify_compute_authorization_receipt", "source.scientific_release_receipt_sha256"),
    _join("compute_receipt_to_blueprint_job", "compute_authorization_receipt", "execution_blueprint", "exact_signed_scope_to_selected_job", "verify_compute_authorization_receipt", "scope"),
    _join("runtime_admission_to_blueprint", "runtime_admission", "execution_blueprint", "full_reproduction_and_exact_job_join", "build_dry_run_admission", "execution_blueprint_id"),
    _join("runtime_admission_to_release", "runtime_admission", "scientific_release_receipt", "embedded_verification_result", "build_dry_run_admission", "scientific_release_verification"),
    _join("runtime_admission_to_compute", "runtime_admission", "compute_authorization_receipt", "embedded_verification_result", "build_dry_run_admission", "compute_authorization_verification"),
    _join("budget_admission_to_runtime_admission", "runtime_admission", "blueprint_bound_budget_admission_v2", "terminal_missing_runtime_join", "no_implemented_verifier", None),
    _join("transition_to_runtime_admission", "durable_job_transition", "runtime_admission", "exact_content_id_and_reconstruction_join", "verify_training_transition", "source.admission_id"),
    _join("transition_combines_nonce_output_and_worker_ledger", "durable_job_transition", "durable_job_transition", "single_atomic_combined_receipt", "verify_training_transition", "transition"),
    _join("transition_to_production_attachment", "durable_job_transition", "production_runtime_attachment", "terminal_prerequisite_gap", "no_production_runtime_attachment_verifier", None),
)


V1_MISMATCH_CORRECTIONS: dict[str, str] = {
    "capture_receipt_to_preflight_hash": "canonical_object_digest_not_file_sha256",
    "capture_receipt_to_registry_id": "cryptographic_authority_membership_join_not_source_registry_id",
    "capture_manifest_to_preflight_id": "removed_raw_manifest_edge_use_guarded_v3_compound_receipt",
    "capture_manifest_to_receipt_id": "removed_raw_manifest_edge_use_guarded_v3_compound_receipt",
    "routing_bundle_to_capture_id": "nested_adapter_receipts_join_guarded_v3_capture_ids",
    "routing_bundle_to_capture_hash": "nested_adapter_receipts_and_guarded_companions_verify_bytes",
    "stability_to_routing_id": "stability_joins_two_role_routes_inside_bundle_verifier",
    "split_to_routing_id": "split_is_embedded_in_plan_not_independent_node",
    "split_to_stability_id": "split_is_embedded_in_plan_not_independent_node",
    "plan_to_routing_id": "plan_is_reproduced_from_verified_bundle_category_evidence",
    "plan_to_stability_id": "plan_is_reproduced_from_verified_bundle_category_evidence",
    "plan_to_split_id": "global_split_is_embedded_content_addressed_subdocument",
    "materialization_to_plan_id": "materialization_embeds_and_verifies_full_plan_object",
    "materialization_to_split_id": "split_identity_is_verified_inside_embedded_plan",
    "proposal_to_blueprint_id": "closed_by_blueprint_bound_budget_proposal_v2",
    "proposal_to_blueprint_hash": "closed_by_blueprint_bound_budget_proposal_v2",
    "admission_to_blueprint_id": "closed_by_blueprint_bound_budget_admission_v2",
    "release_to_registry_id": "cryptographic_authority_membership_join_not_source_registry_id",
    "release_to_admission_id": "terminal_gap_release_receipt_does_not_bind_budget_admission_v2",
    "compute_to_registry_id": "cryptographic_authority_membership_join_not_source_registry_id",
    "compute_to_blueprint_id": "blueprint_job_binding_is_in_exact_signed_scope",
    "nonce_to_compute_id": "folded_into_single_durable_job_transition",
    "reservation_to_compute_id": "folded_into_single_durable_job_transition",
    "reservation_to_nonce_id": "folded_into_single_durable_job_transition",
    "worker_to_reservation_id": "folded_into_single_durable_job_transition",
    "worker_to_compute_id": "folded_into_single_durable_job_transition",
    "runtime_to_worker_id": "production_runtime_attachment_is_explicit_terminal_gap",
    "runtime_to_reservation_id": "production_runtime_attachment_is_explicit_terminal_gap",
}


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


def _module_evidence(spec: dict[str, Any], project_root: Path) -> dict[str, Any] | None:
    if not spec["implemented"]:
        return None
    module = importlib.import_module(str(spec["module"]))
    callback = getattr(module, str(spec["callable"]), None)
    module_path = Path(str(module.__file__)).resolve()
    root = project_root.resolve()
    if (
        not module_path.is_relative_to(root)
        or _is_reparse_component(module_path)
        or not callable(callback)
    ):
        raise ValueError(f"semantic graph verifier evidence differs for {spec['node']}")
    return {
        "module_path": module_path.relative_to(root).as_posix(),
        "module_sha256": sha256_file(module_path),
        "callable_exists": True,
    }


def build_contract(
    *,
    project_root: Path,
    audit_path: Path,
    budget_policy_path: Path,
    budget_benchmark_path: Path,
) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    audit = _load(audit_path)
    audit_body = dict(audit)
    audit_id = str(audit_body.pop("semantic_dispatch_coverage_audit_id", ""))
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit_id != AUDIT_ID
        or audit_id != content_sha256(audit_body)
        or audit["summary"]["mismatched_edges"] != 28
    ):
        raise ValueError("semantic graph V1 audit evidence differs")
    mismatches = {
        str(row["edge"])
        for row in audit["edge_rows"]
        if not row["implemented_field_path_and_target_match"]
    }
    if mismatches != set(V1_MISMATCH_CORRECTIONS):
        raise ValueError("semantic graph V1 mismatch correction inventory differs")

    policy = verify_budget_policy(
        budget_policy_path, **budget_policy_inputs(project_root)
    )
    budget_benchmark = _load(budget_benchmark_path)
    budget_body = dict(budget_benchmark)
    budget_id = str(budget_body.pop("benchmark_id", ""))
    if (
        budget_benchmark.get("schema") != BUDGET_BENCHMARK_SCHEMA
        or budget_id != BUDGET_BENCHMARK_ID
        or budget_id != content_sha256(budget_body)
        or budget_benchmark["summary"]["passed"]
        != budget_benchmark["summary"]["controls"]
    ):
        raise ValueError("blueprint-bound budget benchmark evidence differs")

    node_rows = [
        {**spec, "verifier_evidence": _module_evidence(spec, project_root)}
        for spec in NODE_SPECS
    ]
    if before_frameworks != (MODEL_FRAMEWORKS & set(sys.modules)):
        raise ValueError("semantic graph contract imported a model framework")
    implemented = sum(bool(row["implemented"]) for row in node_rows)
    terminal = sum(not bool(row["implemented"]) for row in node_rows)
    terminal_joins = sum(
        str(row["join_kind"]).startswith("terminal_") for row in JOIN_SPECS
    )
    body = {
        "schema": CONTRACT_SCHEMA,
        "status": "corrected_interface_contract_nonexecuting_with_explicit_terminal_gaps",
        "source": {
            "v1_semantic_audit_path": audit_path.relative_to(project_root).as_posix(),
            "v1_semantic_audit_sha256": sha256_file(audit_path),
            "v1_semantic_audit_id": audit_id,
            "budget_policy_path": budget_policy_path.relative_to(project_root).as_posix(),
            "budget_policy_sha256": sha256_file(budget_policy_path),
            "budget_policy_id": policy["budget_policy_id"],
            "blueprint_budget_benchmark_path": budget_benchmark_path.relative_to(project_root).as_posix(),
            "blueprint_budget_benchmark_sha256": sha256_file(budget_benchmark_path),
            "blueprint_budget_benchmark_id": budget_id,
            "builder_sha256": sha256_file(Path(__file__)),
        },
        "node_specs": node_rows,
        "join_specs": list(JOIN_SPECS),
        "v1_mismatch_corrections": [
            {"v1_edge": name, "v2_representation": V1_MISMATCH_CORRECTIONS[name]}
            for name in sorted(V1_MISMATCH_CORRECTIONS)
        ],
        "path_policy": {
            "candidate_inputs_must_use_safe_relative_paths": True,
            "candidate_inputs_must_remain_under_dedicated_graph_root": True,
            "path_traversal_forbidden": True,
            "symlink_and_windows_junction_components_forbidden": True,
            "file_sha256_required_before_semantic_dispatch": True,
        },
        "summary": {
            "logical_nodes": len(node_rows),
            "implemented_node_representations": implemented,
            "terminal_gap_nodes": terminal,
            "semantic_joins": len(JOIN_SPECS),
            "terminal_gap_joins": terminal_joins,
            "v1_mismatches_accounted": len(V1_MISMATCH_CORRECTIONS),
            "blueprint_budget_v2_controls_passed": budget_benchmark["summary"]["passed"],
            "production_runtime_attachment_verifier_present": False,
            "end_to_end_semantic_dispatch_allowed": False,
        },
        "terminal_gaps": [
            {
                "gap": "budget_admission_to_signed_release",
                "reason": "scientific release receipt V1 does not bind blueprint-bound budget admission V2",
                "required_successor": "new signed release receipt and verifier version binding admission ID and exact file digest",
            },
            {
                "gap": "budget_admission_to_runtime_admission",
                "reason": "runtime admission V1 reproduces signed release and compute scope but does not consume budget admission V2",
                "required_successor": "new runtime admission version reproducing blueprint-bound budget admission V2",
            },
            {
                "gap": "production_runtime_attachment",
                "reason": "only the synthetic inert attachment verifier exists",
                "required_successor": "separately authorized production attachment implementation and semantic verifier",
            },
        ],
        "scientific_boundary": {
            "contract_only": True,
            "v1_contract_modified": False,
            "empirical_graph_present": False,
            "semantic_dispatch_performed": False,
            "signed_authority_verified": False,
            "durable_transition_performed": False,
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
    return {**body, "semantic_graph_contract_id": content_sha256(body)}


def verify_contract(
    path: Path,
    *,
    project_root: Path,
    audit_path: Path,
    budget_policy_path: Path,
    budget_benchmark_path: Path,
) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("semantic_graph_contract_id", ""))
    if identifier != content_sha256(body) or observed != build_contract(
        project_root=project_root,
        audit_path=audit_path,
        budget_policy_path=budget_policy_path,
        budget_benchmark_path=budget_benchmark_path,
    ):
        raise ValueError("semantic graph V2 contract is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-semantic-graph-contract-verification.v2",
        "status": "pass",
        "semantic_graph_contract_id": identifier,
        **observed["summary"],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "project_root": project_root,
        "audit_path": project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_audit_v1/audit.json",
        "budget_policy_path": project_root
        / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json",
        "budget_benchmark_path": project_root
        / "artifacts/vpd_cluster_aware_sae_blueprint_budget_admission_benchmark_v2/report.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify corrected semantic graph interface contract V2."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_contract(**inputs)
        write_json(args.output, result)
    else:
        result = verify_contract(args.verify, **inputs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
