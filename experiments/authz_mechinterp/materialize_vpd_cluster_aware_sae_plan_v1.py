from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import CATEGORY_PROTOCOLS
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.load_vpd_capture_router_inputs_v2 import (
    verify_capture_router_adapter_receipt_v2,
)
from experiments.authz_mechinterp.plan_cluster_aware_sae_training_v2 import (
    build_cluster_aware_sae_plan_v2,
    verify_cluster_aware_sae_plan_v2,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    verify_routing_receipt_v1,
    verify_routing_stability_v1,
)


BUNDLE_SCHEMA = "sch.vpd-guarded-cluster-routing-evidence-bundle.v1"
MATERIALIZATION_SCHEMA = "sch.vpd-cluster-aware-sae-plan-materialization.v1"
ROLES = ("discovery", "cluster_validation", "sae_training")
EVIDENCE_KINDS = {"synthetic_benchmark", "empirical_guarded_v3_capture"}
POLICY_FIELDS = {
    "initialization_seeds",
    "split_seed",
    "validation_fraction",
    "minimum_specific_episodes",
    "minimum_specific_pairs",
    "minimum_shared_episodes",
    "minimum_shared_pairs",
    "specific_dictionary_features",
    "shared_dictionary_features",
    "sae_activation_rows_per_episode",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cluster-routing evidence JSON must be an object")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    project = root.resolve()
    if not resolved.is_relative_to(project):
        raise ValueError("cluster-routing evidence path escapes project")
    return resolved.relative_to(project).as_posix()


def _resolve(relative: Any, root: Path) -> Path:
    raw = Path(str(relative))
    if raw.is_absolute():
        raise ValueError("cluster-routing evidence path must be project-relative")
    resolved = (root.resolve() / raw).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("cluster-routing evidence path escapes project")
    if not resolved.is_file():
        raise ValueError("cluster-routing evidence file is absent")
    return resolved


def _file_record(path: Path, *, project_root: Path, id_field: str) -> dict[str, Any]:
    value = _load(path)
    identifier = str(value.get(id_field, ""))
    if len(identifier) != 64:
        raise ValueError(f"cluster-routing evidence omits {id_field}")
    return {
        "path": _relative(path, project_root),
        "sha256": sha256_file(path),
        id_field: identifier,
    }


def build_routing_evidence_bundle(
    *,
    category_files: dict[str, dict[str, Any]],
    evidence_kind: str,
    planner_policy: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    if evidence_kind not in EVIDENCE_KINDS:
        raise ValueError("cluster-routing evidence kind differs")
    expected_categories = sorted(CATEGORY_PROTOCOLS)
    if sorted(category_files) != expected_categories:
        raise ValueError("cluster-routing evidence must cover exactly eight categories")
    if set(planner_policy) != POLICY_FIELDS:
        raise ValueError("cluster-routing planner policy fields differ")
    categories: dict[str, Any] = {}
    for category in expected_categories:
        category_record = category_files[category]
        if set(category_record) != {"roles", "stability_path"}:
            raise ValueError("cluster-routing category file inventory differs")
        roles = category_record["roles"]
        if set(roles) != set(ROLES):
            raise ValueError("cluster-routing evidence role inventory differs")
        role_records: dict[str, Any] = {}
        for role in ROLES:
            paths = roles[role]
            if set(paths) != {"adapter_path", "routing_path"}:
                raise ValueError("cluster-routing role file inventory differs")
            role_records[role] = {
                "adapter": _file_record(
                    Path(paths["adapter_path"]),
                    project_root=project_root,
                    id_field="capture_router_adapter_v2_id",
                ),
                "routing": _file_record(
                    Path(paths["routing_path"]),
                    project_root=project_root,
                    id_field="cluster_episode_routing_id",
                ),
            }
        categories[category] = {
            "roles": role_records,
            "stability": _file_record(
                Path(category_record["stability_path"]),
                project_root=project_root,
                id_field="routing_stability_id",
            ),
        }
    body = {
        "schema": BUNDLE_SCHEMA,
        "status": "complete_guarded_cluster_routing_evidence_inventory",
        "evidence_kind": evidence_kind,
        "activation_categories": expected_categories,
        "required_roles": list(ROLES),
        "categories": categories,
        "planner_policy": dict(planner_policy),
        "boundary": {
            "exact_eight_category_coverage_required": True,
            "discovery_and_independent_cluster_validation_required": True,
            "guarded_v3_capture_identity_required": True,
            "persistent_empirical_evidence": evidence_kind
            == "empirical_guarded_v3_capture",
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "routing_evidence_bundle_id": content_sha256(body)}


def load_and_verify_routing_evidence_bundle(
    bundle_path: Path, *, project_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    bundle = _load(bundle_path)
    body = dict(bundle)
    identifier = str(body.pop("routing_evidence_bundle_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster-routing evidence bundle address differs")
    evidence_kind = str(bundle.get("evidence_kind", ""))
    expected_categories = sorted(CATEGORY_PROTOCOLS)
    if (
        bundle.get("schema") != BUNDLE_SCHEMA
        or bundle.get("status")
        != "complete_guarded_cluster_routing_evidence_inventory"
        or evidence_kind not in EVIDENCE_KINDS
        or bundle.get("activation_categories") != expected_categories
        or bundle.get("required_roles") != list(ROLES)
        or sorted(bundle.get("categories", {})) != expected_categories
        or set(bundle.get("planner_policy", {})) != POLICY_FIELDS
    ):
        raise ValueError("cluster-routing evidence bundle contract differs")
    category_evidence: dict[str, dict[str, Any]] = {}
    v3_ids_by_role: dict[str, set[str]] = {role: set() for role in ROLES}
    adapter_ids: set[str] = set()
    routing_ids: set[str] = set()
    for category in expected_categories:
        record = bundle["categories"][category]
        if set(record) != {"roles", "stability"} or set(record["roles"]) != set(
            ROLES
        ):
            raise ValueError("cluster-routing category evidence inventory differs")
        routes: dict[str, dict[str, Any]] = {}
        for role in ROLES:
            role_record = record["roles"][role]
            if set(role_record) != {"adapter", "routing"}:
                raise ValueError("cluster-routing role evidence inventory differs")
            adapter_path = _resolve(role_record["adapter"]["path"], project_root)
            routing_path = _resolve(role_record["routing"]["path"], project_root)
            if (
                sha256_file(adapter_path) != role_record["adapter"]["sha256"]
                or sha256_file(routing_path) != role_record["routing"]["sha256"]
            ):
                raise ValueError("cluster-routing evidence bytes differ")
            adapter = _load(adapter_path)
            routing = _load(routing_path)
            adapter_check = verify_capture_router_adapter_receipt_v2(adapter)
            routing_check = verify_routing_receipt_v1(routing)
            adapter_id = adapter_check["capture_router_adapter_v2_id"]
            routing_id = routing_check["cluster_episode_routing_id"]
            if (
                role_record["adapter"]["capture_router_adapter_v2_id"]
                != adapter_id
                or role_record["routing"]["cluster_episode_routing_id"]
                != routing_id
                or adapter_check["activation_category"] != category
                or routing["source"]["activation_category"] != category
                or adapter_check["split_role"] != role
                or routing["source"]["split_role"] != role
                or routing["source"]["capture_execution_id"]
                != adapter_check["capture_execution_v3_id"]
            ):
                raise ValueError("cluster-routing adapter and route lineage differs")
            synthetic = bool(
                adapter["scientific_boundary"]["synthetic_benchmark_only"]
            )
            if (evidence_kind == "synthetic_benchmark") != synthetic:
                raise ValueError("cluster-routing evidence kind and adapter differ")
            if not synthetic and (
                adapter["source"].get("job_allocation_id") is None
                or adapter["source"].get("allocation_manifest_id") is None
                or adapter["source"].get("reservation_transition_id") is None
            ):
                raise ValueError("empirical cluster-routing adapter lineage is incomplete")
            if adapter_id in adapter_ids or routing_id in routing_ids:
                raise ValueError("cluster-routing category-role evidence repeats")
            adapter_ids.add(adapter_id)
            routing_ids.add(routing_id)
            v3_ids_by_role[role].add(adapter_check["capture_execution_v3_id"])
            routes[role] = routing
        stability_path = _resolve(record["stability"]["path"], project_root)
        if sha256_file(stability_path) != record["stability"]["sha256"]:
            raise ValueError("cluster-routing stability bytes differ")
        stability = _load(stability_path)
        stability_check = verify_routing_stability_v1(stability)
        if (
            record["stability"]["routing_stability_id"]
            != stability_check["routing_stability_id"]
            or stability_check["eligible_for_cluster_aware_sae_planning"] is not True
            or stability["source"]["activation_category"] != category
            or stability["source"]["discovery_routing_id"]
            != routes["discovery"]["cluster_episode_routing_id"]
            or stability["source"]["cluster_validation_routing_id"]
            != routes["cluster_validation"]["cluster_episode_routing_id"]
        ):
            raise ValueError("cluster-routing stability lineage differs")
        category_evidence[category] = {
            "discovery": routes["discovery"],
            "cluster_validation": routes["cluster_validation"],
            "stability": stability,
            "sae_training": routes["sae_training"],
        }
    if any(len(values) != 1 for values in v3_ids_by_role.values()):
        raise ValueError("cluster-routing role categories do not share one V3 capture")
    boundary = bundle.get("boundary", {})
    if (
        boundary.get("exact_eight_category_coverage_required") is not True
        or boundary.get("discovery_and_independent_cluster_validation_required")
        is not True
        or boundary.get("guarded_v3_capture_identity_required") is not True
        or boundary.get("persistent_empirical_evidence")
        != (evidence_kind == "empirical_guarded_v3_capture")
        or any(
            boundary.get(field) is not False
            for field in (
                "model_loaded",
                "model_inference_performed",
                "weights_loaded_or_mutated",
                "sae_training_authorized",
                "execution_authorized",
            )
        )
    ):
        raise ValueError("cluster-routing evidence boundary differs")
    verification = {
        "schema": "sch.vpd-guarded-cluster-routing-evidence-bundle-verification.v1",
        "status": "pass",
        "routing_evidence_bundle_id": identifier,
        "evidence_kind": evidence_kind,
        "categories": len(expected_categories),
        "roles_per_category": len(ROLES),
        "adapter_receipts": len(adapter_ids),
        "routing_receipts": len(routing_ids),
        "v3_capture_ids_by_role": {
            role: next(iter(values)) for role, values in v3_ids_by_role.items()
        },
        "sae_training_authorized": False,
        "execution_authorized": False,
    }
    return category_evidence, verification


def build_materialization(
    *,
    bundle_path: Path,
    predecessor_training_plan_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    category_evidence, bundle_verification = load_and_verify_routing_evidence_bundle(
        bundle_path, project_root=project_root
    )
    bundle = _load(bundle_path)
    predecessor = _load(predecessor_training_plan_path)
    predecessor_body = dict(predecessor)
    predecessor_id = str(predecessor_body.pop("training_plan_id", ""))
    if (
        predecessor_id != content_sha256(predecessor_body)
        or predecessor.get("schema") != "sch.vpd-compartment-sae-training-plan.v2"
        or not isinstance(predecessor.get("profile"), dict)
    ):
        raise ValueError("predecessor SAE training plan identity differs")
    policy = bundle["planner_policy"]
    plan = build_cluster_aware_sae_plan_v2(
        category_evidence=category_evidence,
        predecessor_training_plan_id=predecessor_id,
        predecessor_training_plan_sha256=sha256_file(predecessor_training_plan_path),
        profile=predecessor["profile"],
        **policy,
    )
    plan_verification = verify_cluster_aware_sae_plan_v2(plan)
    body = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "complete_manifest_only_guarded_cluster_aware_sae_plan",
        "source": {
            "routing_evidence_bundle_path": _relative(bundle_path, project_root),
            "routing_evidence_bundle_sha256": sha256_file(bundle_path),
            "routing_evidence_bundle_id": bundle_verification[
                "routing_evidence_bundle_id"
            ],
            "predecessor_training_plan_path": _relative(
                predecessor_training_plan_path, project_root
            ),
            "predecessor_training_plan_sha256": sha256_file(
                predecessor_training_plan_path
            ),
            "predecessor_training_plan_id": predecessor_id,
            "planner_v2_sha256": sha256_file(
                Path(__file__).with_name("plan_cluster_aware_sae_training_v2.py")
            ),
        },
        "bundle_verification": bundle_verification,
        "cluster_aware_sae_plan": plan,
        "cluster_aware_sae_plan_verification": plan_verification,
        "boundary": {
            "manifest_only": True,
            "evidence_kind": bundle_verification["evidence_kind"],
            "empirical_plan": bundle_verification["evidence_kind"]
            == "empirical_guarded_v3_capture",
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "materialization_id": content_sha256(body)}


def verify_materialization(
    path: Path,
    *,
    bundle_path: Path,
    predecessor_training_plan_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("materialization_id", ""))
    if identifier != content_sha256(body) or observed != build_materialization(
        bundle_path=bundle_path,
        predecessor_training_plan_path=predecessor_training_plan_path,
        project_root=project_root,
    ):
        raise ValueError("cluster-aware SAE plan materialization is not reproducible")
    plan_check = verify_cluster_aware_sae_plan_v2(
        observed["cluster_aware_sae_plan"]
    )
    return {
        "schema": "sch.vpd-cluster-aware-sae-plan-materialization-verification.v1",
        "status": "pass",
        "materialization_id": identifier,
        "routing_evidence_bundle_id": observed["source"][
            "routing_evidence_bundle_id"
        ],
        "cluster_aware_sae_plan_id": plan_check["cluster_aware_sae_plan_id"],
        "categories": observed["bundle_verification"]["categories"],
        "compartments": plan_check["compartments"],
        "training_jobs": plan_check["training_jobs"],
        "shared_pair_safe": plan_check["shared_pair_safe"],
        "empirical_plan": observed["boundary"]["empirical_plan"],
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a guarded cluster-aware SAE plan from routing evidence."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--predecessor-plan",
        type=Path,
        default=Path("artifacts/vpd_compartment_sae_training_plan_v4/plan.json"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_materialization(
            bundle_path=args.bundle,
            predecessor_training_plan_path=args.predecessor_plan,
            project_root=project_root,
        )
        write_json(args.output, result)
    else:
        result = verify_materialization(
            args.verify,
            bundle_path=args.bundle,
            predecessor_training_plan_path=args.predecessor_plan,
            project_root=project_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
