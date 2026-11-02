from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.attest_partition_audit_effect_receipt import (
    load_gate_plan,
    verify_attested_partition_audit_receipt,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.finalize_partition_audit_pack_v1 import (
    verify_partition_audit_manifest,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    merge_effect_shards,
    verify_effect_merge,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import (
    load_registry,
    verify_signature,
)


SUMMARY_SCHEMA = "sch.vpd-partition-audit-signed-receipt-summary.v1"


def verify_capture_packet_bundle(
    *, bundle_dir: Path, plan: dict[str, Any]
) -> dict[str, str]:
    source = plan.get("source", {}).get("current_bundle_evidence", {})
    expected_manifest_sha256 = str(source.get("manifest_sha256", ""))
    expected_component_set_sha256 = str(
        source.get("component_id_set_sha256", "")
    )
    manifest_path = bundle_dir / "manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("partition-audit capture packet bundle differs from gate plan")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_component_set_sha256 = str(
        manifest.get("invariants", {}).get("component_id_set_sha256", "")
    )
    if observed_component_set_sha256 != expected_component_set_sha256:
        raise ValueError("partition-audit capture packet component census differs")
    return {
        "manifest_sha256": expected_manifest_sha256,
        "component_id_set_sha256": expected_component_set_sha256,
    }


def validate_signed_partition_audit_receipts(
    *,
    receipt_paths: list[Path],
    registry_path: Path,
    pack_dir: Path,
    gate_plan_path: Path,
    require_complete: bool,
) -> dict[str, Any]:
    if not receipt_paths:
        raise ValueError("partition-audit merge has no signed receipts")
    workers = load_registry(registry_path)
    plan = load_gate_plan(gate_plan_path)
    expected_work_units = {
        str(row["work_unit_id"]): row for row in plan["capture_work_units"]
    }
    expected_packets = {
        str(row["packet_id"]): str(row["work_unit_id"])
        for row in plan["capture_work_units"]
    }
    rows: list[dict[str, Any]] = []
    seen_results: set[str] = set()
    seen_work_units: set[str] = set()
    seen_packets: set[str] = set()
    measurement_contract_ids: set[str] = set()
    for path in sorted(receipt_paths, key=lambda item: item.as_posix()):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        verification = verify_attested_partition_audit_receipt(
            receipt,
            pack_dir=pack_dir,
            gate_plan_path=gate_plan_path,
            require_signed=True,
        )
        worker_id = str(receipt.get("worker_id", ""))
        worker = workers.get(worker_id)
        if worker is None or not bool(worker.get("enabled", False)):
            raise ValueError("partition-audit receipt worker is absent or disabled")
        verify_signature(receipt, worker)
        result_id = str(receipt["result_id"])
        work_unit_id = str(verification["work_unit_id"])
        packet_id = str(verification["packet_id"])
        if result_id in seen_results:
            raise ValueError("partition-audit result ID is duplicated")
        if work_unit_id in seen_work_units or packet_id in seen_packets:
            raise ValueError("partition-audit work unit or packet is duplicated")
        if work_unit_id not in expected_work_units:
            raise ValueError("partition-audit receipt uses an unknown work unit")
        if expected_packets.get(packet_id) != work_unit_id:
            raise ValueError("partition-audit receipt packet/work-unit mapping differs")
        seen_results.add(result_id)
        seen_work_units.add(work_unit_id)
        seen_packets.add(packet_id)
        measurement_contract_id = verification.get("measurement_contract_id")
        if measurement_contract_id is not None:
            measurement_contract_ids.add(str(measurement_contract_id))
        rows.append(
            {
                "receipt_path": path.as_posix(),
                "receipt_sha256": sha256_file(path),
                "result_id": result_id,
                "worker_id": worker_id,
                "packet_id": packet_id,
                "work_unit_id": work_unit_id,
                "provenance_id": verification["provenance_id"],
                "measurement_contract_id": measurement_contract_id,
                "signature_verified": True,
            }
        )
    missing_work_units = sorted(set(expected_work_units) - seen_work_units)
    if require_complete and missing_work_units:
        raise ValueError(
            f"partition-audit signed receipt coverage is incomplete: {len(missing_work_units)}"
        )
    if plan.get("schema") == "sch.vpd-partition-audit-gate-plan.v2":
        if len(measurement_contract_ids) != 1:
            raise ValueError(
                "partition-audit signed receipts differ on measurement contract"
            )
        measurement_contract_id: str | None = next(iter(measurement_contract_ids))
    else:
        if measurement_contract_ids:
            raise ValueError("legacy partition-audit receipts claim a measurement contract")
        measurement_contract_id = None
    body = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete" if not missing_work_units else "incomplete",
        "pack_id": verify_partition_audit_manifest(pack_dir)["pack_id"],
        "gate_plan_id": plan["gate_plan_id"],
        "worker_registry_sha256": sha256_file(registry_path),
        "expected_work_units": len(expected_work_units),
        "verified_work_units": len(seen_work_units),
        "missing_work_unit_ids": missing_work_units,
        "signed_receipts": rows,
        "signed_receipt_sha256_set_sha256": content_sha256(
            sorted(row["receipt_sha256"] for row in rows)
        ),
        "all_signatures_verified": True,
        "measurement_contract_id": measurement_contract_id,
        "identity_message_and_semantic_template_disjointness": "pass",
        "final_evaluation_consumed": False,
    }
    return {**body, "summary_id": content_sha256(body)}


def merge_partition_audit_effect_shards(
    *,
    bundle_dir: Path,
    registry_path: Path,
    receipt_paths: list[Path],
    shard_dir: Path,
    output_dir: Path,
    pack_dir: Path,
    gate_plan_path: Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    plan = load_gate_plan(gate_plan_path)
    capture_packet_bundle = verify_capture_packet_bundle(
        bundle_dir=bundle_dir, plan=plan
    )
    summary = validate_signed_partition_audit_receipts(
        receipt_paths=receipt_paths,
        registry_path=registry_path,
        pack_dir=pack_dir,
        gate_plan_path=gate_plan_path,
        require_complete=require_complete,
    )
    manifest = merge_effect_shards(
        bundle_dir=bundle_dir,
        registry_path=registry_path,
        receipt_paths=receipt_paths,
        shard_dir=shard_dir,
        output_dir=output_dir,
        require_complete=require_complete,
    )
    summary_path = output_dir / "partition_audit_signed_receipts.json"
    write_json(summary_path, summary)
    body = copy.deepcopy(manifest)
    body.pop("merge_id")
    body["partition_audit_provenance"] = {
        "schema": "sch.vpd-partition-audit-effect-merge-provenance.v1",
        "role": "partition_audit",
        "pack_id": summary["pack_id"],
        "pack_manifest_sha256": sha256_file(pack_dir / "artifact_manifest.json"),
        "gate_plan_id": summary["gate_plan_id"],
        "gate_plan_sha256": sha256_file(gate_plan_path),
        "measurement_contract_id": summary["measurement_contract_id"],
        "capture_packet_bundle_manifest_sha256": capture_packet_bundle[
            "manifest_sha256"
        ],
        "capture_packet_bundle_component_id_set_sha256": capture_packet_bundle[
            "component_id_set_sha256"
        ],
        "signed_receipt_summary_id": summary["summary_id"],
        "signed_receipt_summary_sha256": sha256_file(summary_path),
        "receipt_attestation_adapter_sha256": sha256_file(
            Path(__file__).with_name("attest_partition_audit_effect_receipt.py")
        ),
        "merge_adapter_sha256": sha256_file(Path(__file__)),
        "signed_exact_work_unit_coverage": summary["status"] == "complete",
        "threshold_cluster_or_component_selection_performed": False,
        "shared_to_specific_promotion_performed": False,
        "final_evaluation_consumed": False,
    }
    body["artifacts"]["partition_audit_signed_receipts"] = {
        "path": summary_path.relative_to(output_dir).as_posix(),
        "sha256": sha256_file(summary_path),
    }
    result = {**body, "merge_id": content_sha256(body)}
    write_json(output_dir / "effect_merge_manifest.json", result)
    return result


def verify_partition_audit_effect_merge(
    output_dir: Path,
    *,
    pack_dir: Path,
    gate_plan_path: Path,
) -> dict[str, Any]:
    base = verify_effect_merge(output_dir)
    manifest_path = output_dir / "effect_merge_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = copy.deepcopy(manifest)
    merge_id = str(body.pop("merge_id", ""))
    if content_sha256(body) != merge_id:
        raise ValueError("partition-audit effect merge content address mismatch")
    provenance = manifest.get("partition_audit_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("partition-audit effect merge provenance is absent")
    summary_path = output_dir / manifest["artifacts"][
        "partition_audit_signed_receipts"
    ]["path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_body = copy.deepcopy(summary)
    summary_id = str(summary_body.pop("summary_id", ""))
    if content_sha256(summary_body) != summary_id:
        raise ValueError("partition-audit signed receipt summary address mismatch")
    pack = verify_partition_audit_manifest(pack_dir)
    plan = load_gate_plan(gate_plan_path)
    expected = {
        "schema": "sch.vpd-partition-audit-effect-merge-provenance.v1",
        "role": "partition_audit",
        "pack_id": pack["pack_id"],
        "pack_manifest_sha256": sha256_file(pack_dir / "artifact_manifest.json"),
        "gate_plan_id": plan["gate_plan_id"],
        "gate_plan_sha256": sha256_file(gate_plan_path),
        "measurement_contract_id": summary["measurement_contract_id"],
        "capture_packet_bundle_manifest_sha256": plan["source"][
            "current_bundle_evidence"
        ]["manifest_sha256"],
        "capture_packet_bundle_component_id_set_sha256": plan["source"][
            "current_bundle_evidence"
        ]["component_id_set_sha256"],
        "signed_receipt_summary_id": summary_id,
        "signed_receipt_summary_sha256": sha256_file(summary_path),
        "receipt_attestation_adapter_sha256": sha256_file(
            Path(__file__).with_name("attest_partition_audit_effect_receipt.py")
        ),
        "merge_adapter_sha256": sha256_file(Path(__file__)),
        "signed_exact_work_unit_coverage": summary["status"] == "complete",
        "threshold_cluster_or_component_selection_performed": False,
        "shared_to_specific_promotion_performed": False,
        "final_evaluation_consumed": False,
    }
    if provenance != expected:
        raise ValueError("partition-audit effect merge provenance differs")
    if summary["pack_id"] != pack["pack_id"] or summary["gate_plan_id"] != plan[
        "gate_plan_id"
    ]:
        raise ValueError("partition-audit signed receipt summary lineage differs")
    return {
        "schema": "sch.vpd-partition-audit-effect-merge-verification.v1",
        "status": "pass",
        "merge_id": merge_id,
        "merge_status": base["merge_status"],
        "components": base["components"],
        "activation_categories": base["activation_categories"],
        "signed_receipts": len(summary["signed_receipts"]),
        "pack_id": pack["pack_id"],
        "gate_plan_id": plan["gate_plan_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and merge signed fresh-pack partition-audit effect shards."
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--receipts-dir", type=Path)
    parser.add_argument("--shards-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--gate-plan", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        result = verify_partition_audit_effect_merge(
            args.verify_output,
            pack_dir=args.pack_dir,
            gate_plan_path=args.gate_plan,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    required = (
        args.bundle,
        args.registry,
        args.receipts_dir,
        args.shards_dir,
        args.output_dir,
    )
    if any(value is None for value in required):
        parser.error(
            "merge requires --bundle, --registry, --receipts-dir, --shards-dir, "
            "and --output-dir"
        )
    receipt_paths = sorted(args.receipts_dir.glob("*.json"))
    result = merge_partition_audit_effect_shards(
        bundle_dir=args.bundle,
        registry_path=args.registry,
        receipt_paths=receipt_paths,
        shard_dir=args.shards_dir,
        output_dir=args.output_dir,
        pack_dir=args.pack_dir,
        gate_plan_path=args.gate_plan,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
