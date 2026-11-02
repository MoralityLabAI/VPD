from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp import admit_compartment_sae_training_release as v1
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_compartment_sae_training_job_v2 import (
    load_training_plan_v2,
)


PLAN_SCHEMA = "sch.vpd-compartment-sae-training-release-gate-plan.v2"
RECEIPT_SCHEMA = "sch.vpd-compartment-sae-training-release-receipt.v2"
RECEIPT_STATUS = (
    "pass_scientific_release_prerequisite_compute_not_authorized_v2"
)
COMPUTE_AUTHORIZATION_SCHEMA = "sch.vpd-explicit-sae-compute-authorization.v1"


def _load_addressed(
    path: Path, *, schema: str, id_field: str
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"unsupported content-addressed schema: {path}")
    body = dict(value)
    identifier = str(body.pop(id_field, ""))
    if content_sha256(body) != identifier:
        raise ValueError(f"content address mismatch: {path}")
    return value


def load_release_gate_plan_v2(path: Path) -> dict[str, Any]:
    plan = _load_addressed(
        path, schema=PLAN_SCHEMA, id_field="release_gate_plan_id"
    )
    guarded = plan.get("guarded_execution", {}).get("guarded_jobs")
    if not isinstance(guarded, list) or len(guarded) != 22:
        raise ValueError("release gate V2 must guard all 22 blueprint jobs")
    if len({str(row.get("source_job_id")) for row in guarded}) != 22:
        raise ValueError("release gate V2 repeats guarded job IDs")
    if plan.get("scientific_boundary", {}).get("execution_authorized") is not False:
        raise ValueError("release gate V2 weakens its execution boundary")
    return plan


def build_release_receipt_v2(
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
    gate = load_release_gate_plan_v2(release_gate_plan_path)
    plan = load_training_plan_v2(training_plan_path)
    protocol = json.loads(protocol_config_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "sch.vpd-activation-category-protocol.v9":
        raise ValueError("release receipt V2 requires research protocol V9")
    if (
        str(gate["source"]["training_blueprint"]["training_plan_id"])
        != str(plan["training_plan_id"])
        or str(gate["source"]["training_blueprint"]["sha256"])
        != sha256_file(training_plan_path)
    ):
        raise ValueError("release gate V2 source lineage differs")
    runtime = protocol.get("sae_training", {}).get("execution_runtime_v2", {})
    if (
        str(protocol.get("predecessor_sha256"))
        != str(gate["source"]["research_protocol_v8"]["sha256"])
        or str(runtime.get("training_plan", {}).get("training_plan_id"))
        != str(plan["training_plan_id"])
        or str(runtime.get("training_plan", {}).get("sha256"))
        != sha256_file(training_plan_path)
        or str(runtime.get("release_gate", {}).get("release_gate_plan_id"))
        != str(gate["release_gate_plan_id"])
        or str(runtime.get("release_gate", {}).get("sha256"))
        != sha256_file(release_gate_plan_path)
    ):
        raise ValueError("research protocol V9 runtime lineage differs")
    root = Path(__file__).resolve().parents[2]
    predecessor_gate_path = root / str(
        gate["source"]["predecessor_release_gate_v1"]["path"]
    )
    predecessor_plan_path = root / str(
        plan["source"]["predecessor_training_plan"]["path"]
    )
    predecessor_protocol_path = root / str(
        gate["source"]["predecessor_protocol_v6"]["path"]
    )
    predecessor = v1.build_release_receipt(
        release_gate_plan_path=predecessor_gate_path,
        training_plan_path=predecessor_plan_path,
        protocol_config_path=predecessor_protocol_path,
        bundle_dir=bundle_dir,
        hierarchy_plan_path=hierarchy_plan_path,
        discovery_effects_path=discovery_effects_path,
        discovery_merge_manifest_path=discovery_merge_manifest_path,
        audit_merge_dir=audit_merge_dir,
        partition_audit_pack_dir=partition_audit_pack_dir,
        partition_audit_gate_plan_path=partition_audit_gate_plan_path,
        partition_audit_report_path=partition_audit_report_path,
    )
    predecessor_body = {
        key: value
        for key, value in predecessor.items()
        if key != "training_release_receipt_id"
    }
    if content_sha256(predecessor_body) != predecessor[
        "training_release_receipt_id"
    ]:
        raise ValueError("predecessor release receipt address mismatch")
    result = json.loads(json.dumps(predecessor))
    result["schema"] = RECEIPT_SCHEMA
    result["status"] = RECEIPT_STATUS
    result["source"]["predecessor_v1_release_receipt_id"] = str(
        predecessor["training_release_receipt_id"]
    )
    result["source"]["predecessor_v1_release_receipt_content_sha256"] = (
        content_sha256(predecessor)
    )
    result["source"]["predecessor_v1_source"] = json.loads(
        json.dumps(predecessor["source"])
    )
    result["source"]["release_gate_plan_id"] = str(gate["release_gate_plan_id"])
    result["source"]["release_gate_plan_sha256"] = sha256_file(
        release_gate_plan_path
    )
    result["source"]["training_plan_id"] = str(plan["training_plan_id"])
    result["source"]["training_plan_sha256"] = sha256_file(training_plan_path)
    result["source"]["protocol_sha256"] = sha256_file(protocol_config_path)
    result["source"]["execution_blueprint_id"] = str(
        plan["source"]["execution_blueprint_id"]
    )
    result["source"]["execution_blueprint_sha256"] = str(
        plan["source"]["execution_blueprint_sha256"]
    )
    result["invariants"].update(
        {
            "manifest_v2_required_for_every_dictionary": True,
            "exact_sample_contract_required_for_every_dictionary": True,
            "all_planned_hyperparameters_required_for_every_dictionary": True,
            "explicit_compute_authorization_required_after_scientific_release": True,
            "release_receipt_alone_cannot_start_compute": True,
        }
    )
    result["scientific_boundary"]["compute_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result_body = {
        key: value
        for key, value in result.items()
        if key != "training_release_receipt_id"
    }
    result["training_release_receipt_id"] = content_sha256(result_body)
    return result


def verify_training_release_receipt_v2(
    path: Path,
    *,
    release_gate_plan_path: Path,
    training_plan_path: Path,
) -> dict[str, Any]:
    receipt = _load_addressed(
        path, schema=RECEIPT_SCHEMA, id_field="training_release_receipt_id"
    )
    gate = load_release_gate_plan_v2(release_gate_plan_path)
    plan = load_training_plan_v2(training_plan_path)
    source = receipt.get("source", {})
    if (
        str(source.get("release_gate_plan_id"))
        != str(gate["release_gate_plan_id"])
        or str(source.get("release_gate_plan_sha256"))
        != sha256_file(release_gate_plan_path)
        or str(source.get("training_plan_id")) != str(plan["training_plan_id"])
        or str(source.get("training_plan_sha256"))
        != sha256_file(training_plan_path)
        or str(source.get("execution_blueprint_id"))
        != str(plan["source"]["execution_blueprint_id"])
    ):
        raise ValueError("training release receipt V2 source differs")
    if receipt.get("status") != RECEIPT_STATUS:
        raise ValueError("training release receipt V2 status differs")
    if receipt.get("scientific_boundary", {}).get("compute_authorized") is not False:
        raise ValueError("scientific release receipt improperly authorizes compute")
    invariants = receipt.get("invariants", {})
    for field in (
        "manifest_v2_required_for_every_dictionary",
        "exact_sample_contract_required_for_every_dictionary",
        "all_planned_hyperparameters_required_for_every_dictionary",
        "explicit_compute_authorization_required_after_scientific_release",
        "release_receipt_alone_cannot_start_compute",
    ):
        if invariants.get(field) is not True:
            raise ValueError("training release receipt V2 weakens its invariants")
    predecessor = json.loads(json.dumps(receipt))
    predecessor["schema"] = v1.RECEIPT_SCHEMA
    predecessor["status"] = v1.RECEIPT_STATUS
    predecessor["source"] = json.loads(
        json.dumps(source.get("predecessor_v1_source", {}))
    )
    for field in (
        "manifest_v2_required_for_every_dictionary",
        "exact_sample_contract_required_for_every_dictionary",
        "all_planned_hyperparameters_required_for_every_dictionary",
        "explicit_compute_authorization_required_after_scientific_release",
        "release_receipt_alone_cannot_start_compute",
    ):
        predecessor["invariants"].pop(field, None)
    predecessor["scientific_boundary"].pop("execution_authorized", None)
    predecessor["training_release_receipt_id"] = str(
        source.get("predecessor_v1_release_receipt_id", "")
    )
    predecessor_body = {
        key: value
        for key, value in predecessor.items()
        if key != "training_release_receipt_id"
    }
    if (
        content_sha256(predecessor_body)
        != predecessor["training_release_receipt_id"]
        or content_sha256(predecessor)
        != str(source.get("predecessor_v1_release_receipt_content_sha256", ""))
        or predecessor.get("scientific_boundary", {}).get("compute_authorized")
        is not False
        or predecessor.get("released_partition", {}).get("specific_failure_count")
        != 0
    ):
        raise ValueError("training release receipt V2 predecessor is invalid")
    return {
        "schema": "sch.vpd-compartment-sae-training-release-verification.v2",
        "status": "pass",
        "training_release_receipt_id": str(
            receipt["training_release_receipt_id"]
        ),
        "training_release_receipt_sha256": sha256_file(path),
        "release_gate_plan_id": str(gate["release_gate_plan_id"]),
        "training_plan_id": str(plan["training_plan_id"]),
        "compute_authorized": False,
    }


def verify_explicit_compute_authorization(
    path: Path,
    *,
    release_verification: dict[str, Any],
    release_gate_plan_path: Path,
    training_plan_path: Path,
    required_job_id: str,
    required_action: str,
) -> dict[str, Any]:
    authorization = _load_addressed(
        path,
        schema=COMPUTE_AUTHORIZATION_SCHEMA,
        id_field="compute_authorization_id",
    )
    gate = load_release_gate_plan_v2(release_gate_plan_path)
    plan = load_training_plan_v2(training_plan_path)
    source = authorization.get("source", {})
    if (
        str(source.get("training_plan_id")) != str(plan["training_plan_id"])
        or str(source.get("training_plan_sha256"))
        != sha256_file(training_plan_path)
        or str(source.get("release_gate_plan_id"))
        != str(gate["release_gate_plan_id"])
        or str(source.get("release_gate_plan_sha256"))
        != sha256_file(release_gate_plan_path)
        or str(source.get("training_release_receipt_id"))
        != str(release_verification["training_release_receipt_id"])
        or str(source.get("training_release_receipt_sha256"))
        != str(release_verification["training_release_receipt_sha256"])
    ):
        raise ValueError("explicit compute authorization source differs")
    if authorization.get("status") != "authorized_by_user":
        raise ValueError("explicit compute authorization did not authorize")
    scope = authorization.get("scope", {})
    if (
        required_action not in set(scope.get("authorized_actions", []))
        or required_job_id not in set(scope.get("authorized_job_ids", []))
        or scope.get("real_actions_authorized") is not False
        or scope.get("base_model_weights_mutable") is not False
        or scope.get("sae_dictionary_weight_training_authorized") is not True
    ):
        raise ValueError("explicit compute authorization scope is insufficient")
    guarded_ids = {
        str(row["source_job_id"])
        for row in gate["guarded_execution"]["guarded_jobs"]
    }
    if not set(scope.get("authorized_job_ids", [])).issubset(guarded_ids):
        raise ValueError("explicit compute authorization contains an unguarded job")
    return {
        "schema": "sch.vpd-explicit-sae-compute-authorization-verification.v1",
        "status": "pass",
        "compute_authorization_id": str(
            authorization["compute_authorization_id"]
        ),
        "compute_authorization_sha256": sha256_file(path),
        "required_job_id": required_job_id,
        "required_action": required_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit or reproduce the compartment SAE scientific release V2."
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
    observed = build_release_receipt_v2(**kwargs)
    if args.verify:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("training release receipt V2 is not reproducible")
    else:
        write_json(args.output, observed)
    print(
        json.dumps(
            {
                "schema": "sch.vpd-compartment-sae-training-release-cli.v2",
                "status": observed["status"],
                "training_release_receipt_id": observed[
                    "training_release_receipt_id"
                ],
                "compute_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
