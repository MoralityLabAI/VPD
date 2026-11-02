from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import (
    COMPONENT_COMPARTMENT_PROTOCOLS,
    category_contract,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    verify_bundle,
    write_json,
)
from experiments.authz_mechinterp.evaluate_vpd_partition_audit import (
    verify_report as verify_production_partition_audit_report,
)
from experiments.authz_mechinterp.plan_compartment_sae_training import (
    validate_training_plan_structure,
)


PLAN_SCHEMA = "sch.vpd-compartment-sae-training-release-gate-plan.v1"
RECEIPT_SCHEMA = "sch.vpd-compartment-sae-training-release-receipt.v1"
RECEIPT_STATUS = "pass_scientific_release_prerequisite_compute_not_authorized"


def _validate_content_address(document: dict[str, Any], id_field: str, label: str) -> str:
    body = copy.deepcopy(document)
    identifier = str(body.pop(id_field, ""))
    if len(identifier) != 64 or content_sha256(body) != identifier:
        raise ValueError(f"{label} content address mismatch")
    return identifier


def _require_digest(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return digest


def evaluate_release_contract(
    *,
    release_gate_plan: dict[str, Any],
    release_gate_plan_sha256: str,
    training_plan: dict[str, Any],
    training_plan_sha256: str,
    protocol: dict[str, Any],
    protocol_sha256: str,
    bundle_manifest: dict[str, Any],
    bundle_manifest_sha256: str,
    bundle_verification: dict[str, Any],
    partition_audit_report: dict[str, Any],
    partition_audit_report_sha256: str,
    partition_audit_verification: dict[str, Any],
    partition_audit_gate_plan: dict[str, Any],
    partition_audit_gate_plan_sha256: str,
    partition_audit_pack_manifest: dict[str, Any],
    partition_audit_pack_manifest_sha256: str,
) -> dict[str, Any]:
    """Join already-verified evidence into a fail-closed SAE training release receipt.

    The production entry point below obtains the verification objects by replaying
    the bundle and partition-audit verifiers.  Keeping the joins pure makes every
    rejection branch cheap to exercise without model inference.
    """

    if release_gate_plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported compartment SAE training release gate plan schema")
    release_gate_plan_id = _validate_content_address(
        release_gate_plan, "release_gate_plan_id", "training release gate plan"
    )
    release_gate_plan_sha256 = _require_digest(
        release_gate_plan_sha256, "training release gate plan file digest"
    )
    if release_gate_plan.get("scientific_boundary", {}).get("execution_authorized") is not False:
        raise ValueError("training release gate plan improperly authorizes execution")

    training_plan_id = _validate_content_address(
        training_plan, "training_plan_id", "compartment SAE training plan"
    )
    validate_training_plan_structure(training_plan)
    training_plan_sha256 = _require_digest(
        training_plan_sha256, "compartment SAE training plan file digest"
    )
    protocol_sha256 = _require_digest(protocol_sha256, "protocol file digest")
    bundle_manifest_sha256 = _require_digest(
        bundle_manifest_sha256, "category-aware bundle manifest digest"
    )
    partition_audit_report_sha256 = _require_digest(
        partition_audit_report_sha256, "partition audit report digest"
    )
    partition_audit_gate_plan_sha256 = _require_digest(
        partition_audit_gate_plan_sha256, "partition audit gate plan digest"
    )
    partition_audit_pack_manifest_sha256 = _require_digest(
        partition_audit_pack_manifest_sha256, "partition audit pack manifest digest"
    )

    source = release_gate_plan.get("source", {})
    expected_training = source.get("training_blueprint", {})
    if (
        expected_training.get("training_plan_id") != training_plan_id
        or expected_training.get("sha256") != training_plan_sha256
    ):
        raise ValueError("training release gate references a different training blueprint")
    expected_protocol = source.get("protocol", {})
    if expected_protocol.get("sha256") != protocol_sha256:
        raise ValueError("training release gate references a different protocol")

    partition_gate_id = _validate_content_address(
        partition_audit_gate_plan,
        "gate_plan_id",
        "partition audit gate plan",
    )
    pack_id = _validate_content_address(
        partition_audit_pack_manifest,
        "pack_id",
        "partition audit pack manifest",
    )
    expected_gate = source.get("partition_audit_gate", {})
    expected_pack = source.get("fresh_partition_audit_pack", {})
    if (
        expected_gate.get("gate_plan_id") != partition_gate_id
        or expected_gate.get("sha256") != partition_audit_gate_plan_sha256
    ):
        raise ValueError("training release gate references a stale partition audit gate")
    if (
        expected_pack.get("pack_id") != pack_id
        or expected_pack.get("manifest_sha256")
        != partition_audit_pack_manifest_sha256
    ):
        raise ValueError("training release gate references a stale partition audit pack")

    protocol_gate = protocol.get("sae_training", {}).get("partition_audit_gate", {})
    if (
        protocol_gate.get("gate_plan_id") != partition_gate_id
        or protocol_gate.get("category_specific_training_requires_pass") is not True
        or protocol_gate.get("same_pack_failure_cannot_be_used_for_tuning") is not True
        or protocol_gate.get("shared_only_training_not_promoted_by_audit") is not True
        or protocol_gate.get("execution_authorized") is not False
    ):
        raise ValueError("protocol partition audit release contract differs")
    replica_contract = protocol.get("sae_training", {}).get(
        "replica_training_plan", {}
    )
    if (
        replica_contract.get("path") != expected_training.get("path")
        or replica_contract.get("schema") != training_plan.get("schema")
        or int(replica_contract.get("compartments", -1)) != 9
        or replica_contract.get("status") != training_plan.get("status")
    ):
        raise ValueError("protocol references a different training blueprint")

    if bundle_manifest.get("schema") != "sch.vpd-sae-cluster-manifest.v1":
        raise ValueError("training release requires a VPD category-aware bundle")
    compartments = bundle_manifest.get("algorithm", {}).get(
        "activation_category_compartments", {}
    )
    invariants = bundle_manifest.get("invariants", {})
    expected_categories = sorted(
        name for name in COMPONENT_COMPARTMENT_PROTOCOLS if name != "shared_cross_category"
    )
    observed_categories = list(compartments.get("categories", []))
    component_count = int(invariants.get("component_count", 0))
    component_id_set_sha256 = _require_digest(
        invariants.get("component_id_set_sha256", ""),
        "category-aware bundle component set",
    )
    if (
        bundle_manifest.get("status") != "complete"
        or compartments.get("enabled") is not True
        or compartments.get("contract_sha256") != content_sha256(category_contract())
        or observed_categories != expected_categories
        or invariants.get("exact_partition") is not True
        or component_count < 1
        or int(invariants.get("assigned_component_count", -1)) != component_count
        or int(invariants.get("unique_assigned_component_count", -1))
        != component_count
    ):
        raise ValueError("category-aware bundle partition contract differs")
    if (
        bundle_verification.get("status") != "pass"
        or int(bundle_verification.get("components", -1)) != component_count
        or bundle_verification.get("exact_partition") is not True
    ):
        raise ValueError("category-aware bundle verification differs")

    audit_id = _validate_content_address(
        partition_audit_report,
        "partition_audit_id",
        "partition audit report",
    )
    audit_summary = partition_audit_report.get("summary", {})
    audit_source = partition_audit_report.get("source", {})
    if (
        partition_audit_report.get("schema")
        != "sch.vpd-frozen-category-partition-audit.v1"
        or partition_audit_report.get("status") != "pass_partition_audit"
        or audit_summary.get("eligible_for_category_specific_sae_training") is not True
        or int(audit_summary.get("specific_failure_count", -1)) != 0
        or int(audit_summary.get("frozen_specific_component_count", 0)) < 1
        or int(audit_summary.get("component_count", -1)) != component_count
        or partition_audit_report.get("component_id_set_sha256")
        != component_id_set_sha256
        or list(partition_audit_report.get("category_names", []))
        != observed_categories
    ):
        raise ValueError("partition audit did not release category-specific SAE training")
    if (
        partition_audit_verification.get("status") != "pass"
        or partition_audit_verification.get("partition_audit_id") != audit_id
        or partition_audit_verification.get("audit_decision")
        != "pass_partition_audit"
        or partition_audit_verification.get(
            "eligible_for_category_specific_sae_training"
        )
        is not True
        or int(partition_audit_verification.get("specific_failure_count", -1)) != 0
    ):
        raise ValueError("partition audit verification did not release training")
    if (
        audit_source.get("bundle_manifest_sha256") != bundle_manifest_sha256
        or audit_source.get("partition_audit_gate_plan_id") != partition_gate_id
        or audit_source.get("partition_audit_pack_id") != pack_id
    ):
        raise ValueError("partition audit lineage differs from the release inputs")

    measurement = audit_source.get("cross_archive_measurement_identity", {})
    if measurement.get("status") != "pass":
        raise ValueError("partition audit lacks a passed cross-archive identity")
    shared_identity = measurement.get("shared_identity", {})
    if (
        shared_identity.get("effect_space_id") != training_plan["profile"]["activation_key"]
        or int(shared_identity.get("generated_tokens", -1)) != 0
        or shared_identity.get("phase") != "pre_gate"
    ):
        raise ValueError("partition audit activation identity differs from SAE training")
    model_id = str(shared_identity.get("model_id", ""))
    revision = str(shared_identity.get("revision", ""))
    adapter_sha256 = _require_digest(
        shared_identity.get("adapter_sha256", ""),
        "partition audit adapter identity",
    )
    if not model_id or not revision:
        raise ValueError("partition audit model identity is incomplete")

    planned_compartments = sorted(
        {str(job["compartment"]) for job in training_plan["training_jobs"]}
    )
    if planned_compartments != sorted([*observed_categories, "shared_cross_category"]):
        raise ValueError("training blueprint compartment grid differs from the bundle")

    body = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "source": {
            "release_gate_plan_id": release_gate_plan_id,
            "release_gate_plan_sha256": release_gate_plan_sha256,
            "training_plan_id": training_plan_id,
            "training_plan_sha256": training_plan_sha256,
            "protocol_sha256": protocol_sha256,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "partition_audit_id": audit_id,
            "partition_audit_report_sha256": partition_audit_report_sha256,
            "partition_audit_gate_plan_id": partition_gate_id,
            "partition_audit_gate_plan_sha256": partition_audit_gate_plan_sha256,
            "partition_audit_pack_id": pack_id,
            "partition_audit_pack_manifest_sha256": partition_audit_pack_manifest_sha256,
        },
        "released_partition": {
            "activation_categories": observed_categories,
            "compartments": planned_compartments,
            "component_count": component_count,
            "component_id_set_sha256": component_id_set_sha256,
            "specific_component_count": int(
                audit_summary["frozen_specific_component_count"]
            ),
            "shared_component_count": int(
                audit_summary["frozen_shared_component_count"]
            ),
            "specific_failure_count": 0,
        },
        "released_runtime_identity": {
            "model_id": model_id,
            "revision": revision,
            "partition_measurement_adapter_sha256": adapter_sha256,
            "sae_capture_model_variant": "base",
            "sae_capture_adapter_sha256": None,
            "activation_key": training_plan["profile"]["activation_key"],
            "phase": "pre_gate",
            "generated_tokens": 0,
        },
        "invariants": {
            "frozen_bundle_verified": True,
            "fresh_pack_partition_audit_reproduced": True,
            "zero_specific_failures": True,
            "bundle_audit_component_census_equal": True,
            "bundle_audit_training_categories_equal": True,
            "audit_measurement_identity_bound": True,
            "same_pack_failure_cannot_tune_candidate": True,
            "receipt_required_before_base_capture_or_sae_training": True,
        },
        "scientific_boundary": {
            "model_inference_performed_by_release_gate": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "dictionary_quality_claimed": False,
            "compute_authorized": False,
            "v7_negative_release_decision_unchanged": True,
        },
    }
    return {**body, "training_release_receipt_id": content_sha256(body)}


def build_release_receipt(
    *,
    release_gate_plan_path: Path,
    training_plan_path: Path,
    protocol_config_path: Path,
    bundle_dir: Path,
    hierarchy_plan_path: Path,
    discovery_effects_path: Path,
    discovery_merge_manifest_path: Path,
    audit_merge_dir: Path,
    partition_audit_pack_dir: Path,
    partition_audit_gate_plan_path: Path,
    partition_audit_report_path: Path,
) -> dict[str, Any]:
    release_gate_plan = json.loads(release_gate_plan_path.read_text(encoding="utf-8"))
    training_plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_config_path.read_text(encoding="utf-8"))
    bundle_manifest_path = bundle_dir / "manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    partition_audit_report = json.loads(
        partition_audit_report_path.read_text(encoding="utf-8")
    )
    partition_audit_gate_plan = json.loads(
        partition_audit_gate_plan_path.read_text(encoding="utf-8")
    )
    partition_audit_pack_manifest_path = (
        partition_audit_pack_dir / "artifact_manifest.json"
    )
    partition_audit_pack_manifest = json.loads(
        partition_audit_pack_manifest_path.read_text(encoding="utf-8")
    )
    bundle_verification = verify_bundle(bundle_dir)
    partition_audit_verification = verify_production_partition_audit_report(
        partition_audit_report_path,
        bundle_dir=bundle_dir,
        hierarchy_plan_path=hierarchy_plan_path,
        discovery_effects_path=discovery_effects_path,
        discovery_merge_manifest_path=discovery_merge_manifest_path,
        audit_merge_dir=audit_merge_dir,
        pack_dir=partition_audit_pack_dir,
        gate_plan_path=partition_audit_gate_plan_path,
    )
    return evaluate_release_contract(
        release_gate_plan=release_gate_plan,
        release_gate_plan_sha256=sha256_file(release_gate_plan_path),
        training_plan=training_plan,
        training_plan_sha256=sha256_file(training_plan_path),
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_config_path),
        bundle_manifest=bundle_manifest,
        bundle_manifest_sha256=sha256_file(bundle_manifest_path),
        bundle_verification=bundle_verification,
        partition_audit_report=partition_audit_report,
        partition_audit_report_sha256=sha256_file(partition_audit_report_path),
        partition_audit_verification=partition_audit_verification,
        partition_audit_gate_plan=partition_audit_gate_plan,
        partition_audit_gate_plan_sha256=sha256_file(
            partition_audit_gate_plan_path
        ),
        partition_audit_pack_manifest=partition_audit_pack_manifest,
        partition_audit_pack_manifest_sha256=sha256_file(
            partition_audit_pack_manifest_path
        ),
    )


def verify_release_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    receipt_id = _validate_content_address(
        observed,
        "training_release_receipt_id",
        "compartment SAE training release receipt",
    )
    expected = build_release_receipt(**kwargs)
    if observed != expected:
        raise ValueError("compartment SAE training release receipt is not reproducible")
    return {
        "schema": "sch.vpd-compartment-sae-training-release-verification.v1",
        "status": "pass",
        "training_release_receipt_id": receipt_id,
        "component_count": observed["released_partition"]["component_count"],
        "compartments": len(observed["released_partition"]["compartments"]),
        "compute_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Admit a passed frozen partition audit before compartment SAE work."
    )
    parser.add_argument("--release-gate-plan", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-plan", type=Path, required=True)
    parser.add_argument("--discovery-effects", type=Path, required=True)
    parser.add_argument("--discovery-merge-manifest", type=Path, required=True)
    parser.add_argument("--audit-merge-dir", type=Path, required=True)
    parser.add_argument("--partition-audit-pack", type=Path, required=True)
    parser.add_argument("--partition-audit-gate", type=Path, required=True)
    parser.add_argument("--partition-audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    kwargs = {
        "release_gate_plan_path": args.release_gate_plan,
        "training_plan_path": args.training_plan,
        "protocol_config_path": args.protocol,
        "bundle_dir": args.bundle_dir,
        "hierarchy_plan_path": args.hierarchy_plan,
        "discovery_effects_path": args.discovery_effects,
        "discovery_merge_manifest_path": args.discovery_merge_manifest,
        "audit_merge_dir": args.audit_merge_dir,
        "partition_audit_pack_dir": args.partition_audit_pack,
        "partition_audit_gate_plan_path": args.partition_audit_gate,
        "partition_audit_report_path": args.partition_audit_report,
    }
    if args.verify:
        result = verify_release_receipt(args.output, **kwargs)
    else:
        result = build_release_receipt(**kwargs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
