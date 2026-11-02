from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_category_split_contract import (
    verify_activation_category_split_bundle,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v4 import (
    verify_protocol,
)
from experiments.authz_mechinterp.plan_compartment_sae_training import (
    verify_training_plan,
)
from experiments.authz_mechinterp.plan_vpd_effect_capture_campaign import (
    verify_campaign_plan,
)


SCHEMA = "sch.vpd-unified-compartment-pipeline-receipt.v1"
ROLES = (
    "sae_training",
    "discovery",
    "calibration",
    "cluster_validation",
    "final_evaluation",
)


def _relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("unified pipeline input escapes the project")
    return resolved.relative_to(root).as_posix()


def build_receipt(
    *,
    protocol_path: Path,
    predecessor_protocol_path: Path,
    effect_plan_path: Path,
    training_plan_path: Path,
    bundle_dir: Path,
    split_dir: Path,
    pack_dir: Path,
    hierarchy_plan_path: Path,
    capacity_plan_path: Path,
    capture_resource_plan_path: Path,
    quality_benchmark_path: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    protocol_verification = verify_protocol(
        protocol_path,
        predecessor_path=predecessor_protocol_path,
    )
    split_verification = verify_activation_category_split_bundle(
        split_dir,
        source_path=pack_dir / "gated_episodes.jsonl",
    )
    effect_verification = verify_campaign_plan(
        effect_plan_path,
        bundle_dir=bundle_dir,
        split_dir=split_dir,
        protocol_config_path=protocol_path,
        hierarchy_plan_path=hierarchy_plan_path,
    )
    training_verification = verify_training_plan(
        training_plan_path,
        capacity_plan_path=capacity_plan_path,
        capture_resource_plan_path=capture_resource_plan_path,
        pack_dir=pack_dir,
        split_dir=split_dir,
        protocol_config_path=protocol_path,
        quality_benchmark_path=quality_benchmark_path,
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    split = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    effect = json.loads(effect_plan_path.read_text(encoding="utf-8"))
    training = json.loads(training_plan_path.read_text(encoding="utf-8"))
    protocol_sha = sha256_file(protocol_path)
    split_sha = sha256_file(split_dir / "split_manifest.json")
    split_id = str(split["split_id"])
    if protocol["data_pack"]["split_id"] != split_id:
        raise ValueError("unified protocol and split IDs differ")
    if protocol["unified_lineage"]["split_manifest_sha256"] != split_sha:
        raise ValueError("unified protocol and split manifests differ")
    if (
        effect["source"]["protocol"]["sha256"] != protocol_sha
        or training["source"]["protocol_config_sha256"] != protocol_sha
    ):
        raise ValueError("unified plans use different protocol bytes")
    if (
        effect["source"]["split"]["split_id"] != split_id
        or effect["source"]["split"]["sha256"] != split_sha
        or training["source"]["split_id"] != split_id
        or training["source"]["split_manifest_sha256"] != split_sha
    ):
        raise ValueError("unified plans use different split provenance")
    configured_effect_path = str(protocol["effect_capture"]["campaign_plan"]["path"])
    configured_training_path = str(
        protocol["sae_training"]["replica_training_plan"]["path"]
    )
    if _relative(effect_plan_path, project_root) != configured_effect_path or _relative(
        training_plan_path, project_root
    ) != configured_training_path:
        raise ValueError("unified plan paths differ from the protocol")

    role_receipts: dict[str, dict[str, Any]] = {}
    effect_roles = effect["source"]["split"]["roles"]
    training_capture_by_role = {
        str(row["role"]): row for row in training["capture_jobs"]
    }
    for role in ROLES:
        record = split["splits"][role]
        template_hash = str(record["template_group_set_sha256"])
        evidence = "split_manifest_only"
        if role in effect_roles:
            observed = effect_roles[role]
            if (
                observed["sha256"] != record["artifact"]["sha256"]
                or observed["template_group_set_sha256"] != template_hash
            ):
                raise ValueError(f"effect plan role differs from split: {role}")
            evidence = "effect_campaign_and_split_manifest"
        if role in training_capture_by_role:
            observed = training_capture_by_role[role]["input"]
            if (
                observed["sha256"] != record["artifact"]["sha256"]
                or observed["template_group_set_sha256"] != template_hash
            ):
                raise ValueError(f"SAE plan role differs from split: {role}")
            evidence = (
                "effect_campaign_sae_plan_and_split_manifest"
                if role in effect_roles
                else "sae_plan_and_split_manifest"
            )
        role_receipts[role] = {
            "rows": int(record["rows"]),
            "pairs": int(record["pairs"]),
            "episode_id_set_sha256": str(record["episode_id_set_sha256"]),
            "pair_id_set_sha256": str(record["pair_id_set_sha256"]),
            "template_group_set_sha256": template_hash,
            "template_groups": sorted(record["template_group_counts"]),
            "evidence": evidence,
        }
    template_hashes = [
        role_receipts[role]["template_group_set_sha256"] for role in ROLES
    ]
    if len(set(template_hashes)) != len(ROLES):
        raise ValueError("unified pipeline semantic role sets overlap")

    body = {
        "schema": SCHEMA,
        "status": "pass_locally_verified_requires_runtime_identities_and_compute_authorization",
        "source": {
            "protocol": {
                "path": _relative(protocol_path, project_root),
                "sha256": protocol_sha,
                "verification": protocol_verification,
            },
            "split": {
                "path": _relative(split_dir / "split_manifest.json", project_root),
                "sha256": split_sha,
                "split_id": split_id,
                "verification": split_verification,
            },
            "effect_campaign": {
                "path": _relative(effect_plan_path, project_root),
                "sha256": sha256_file(effect_plan_path),
                "verification": effect_verification,
            },
            "sae_training": {
                "path": _relative(training_plan_path, project_root),
                "sha256": sha256_file(training_plan_path),
                "verification": training_verification,
            },
            "verifier_sha256": sha256_file(Path(__file__)),
        },
        "roles": role_receipts,
        "resources": {
            "effect_episode_forward_equivalents": effect["resources"]["total"][
                "episode_forward_equivalents"
            ],
            "effect_batch_forwards_at_planned_batch_size": effect["resources"][
                "total"
            ]["batch_forwards_at_planned_batch_size"],
            "sae_capture_rows_by_role": dict(
                training["resource_summary"]["capture_rows_by_role"]
            ),
            "sae_capture_rows_total": training["resource_summary"][
                "capture_rows_total"
            ],
            "sae_training_jobs": training["resource_summary"]["training_jobs"],
        },
        "invariants": {
            "one_exact_split_for_effects_sae_and_final_evaluation": True,
            "five_semantic_role_sets_pairwise_disjoint": True,
            "effect_discovery_calibration_and_cluster_validation_bound": True,
            "sae_training_and_quality_calibration_bound": True,
            "final_evaluation_excluded_from_all_planned_selection_jobs": True,
            "cross_pack_role_join_forbidden": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_bonsai_categories_or_clusters_claimed": False,
            "execution_authorized": False,
            "v7_negative_release_decision_unchanged": True,
        },
    }
    return {**body, "pipeline_receipt_id": content_sha256(body)}


def verify_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(receipt)
    receipt_id = str(body.pop("pipeline_receipt_id", ""))
    if content_sha256(body) != receipt_id:
        raise ValueError("unified pipeline receipt content address mismatch")
    if build_receipt(**kwargs) != receipt:
        raise ValueError("unified pipeline receipt is not reproducible")
    return {
        "schema": "sch.vpd-unified-compartment-pipeline-verification.v1",
        "status": "pass",
        "pipeline_receipt_id": receipt_id,
        "split_id": receipt["source"]["split"]["split_id"],
        "roles": len(receipt["roles"]),
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the unified effect-cluster-to-SAE lineage receipt."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predecessor-protocol", type=Path, required=True)
    parser.add_argument("--effect-plan", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-plan", type=Path, required=True)
    parser.add_argument("--capacity-plan", type=Path, required=True)
    parser.add_argument("--capture-resource-plan", type=Path, required=True)
    parser.add_argument("--quality-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    common = {
        "protocol_path": args.protocol,
        "predecessor_protocol_path": args.predecessor_protocol,
        "effect_plan_path": args.effect_plan,
        "training_plan_path": args.training_plan,
        "bundle_dir": args.bundle_dir,
        "split_dir": args.split_dir,
        "pack_dir": args.pack_dir,
        "hierarchy_plan_path": args.hierarchy_plan,
        "capacity_plan_path": args.capacity_plan,
        "capture_resource_plan_path": args.capture_resource_plan,
        "quality_benchmark_path": args.quality_benchmark,
    }
    if args.verify is not None:
        result = verify_receipt(args.verify, **common)
    else:
        if args.output is None:
            parser.error("receipt generation requires --output")
        result = build_receipt(**common)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
