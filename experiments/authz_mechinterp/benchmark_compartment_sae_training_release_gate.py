from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.activation_categories import (
    COMPONENT_COMPARTMENT_PROTOCOLS,
    category_contract,
)
from experiments.authz_mechinterp.admit_compartment_sae_training_release import (
    evaluate_release_contract,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


def _readdress(document: dict[str, Any], id_field: str) -> dict[str, Any]:
    body = copy.deepcopy(document)
    body.pop(id_field, None)
    return {**body, id_field: content_sha256(body)}


def _fixtures(project_root: Path) -> dict[str, Any]:
    release_gate_plan_path = (
        project_root
        / "artifacts/vpd_compartment_sae_training_release_gate_v1/plan.json"
    )
    training_plan_path = (
        project_root / "artifacts/vpd_compartment_sae_training_plan_v3/plan.json"
    )
    protocol_path = (
        project_root / "configs/evals/vpd_activation_category_compartments_v6.json"
    )
    partition_gate_path = (
        project_root / "artifacts/vpd_partition_audit_gate_v2/plan.json"
    )
    pack_manifest_path = (
        project_root / "artifacts/vpd_partition_audit_pack_v1/artifact_manifest.json"
    )
    release_gate_plan = json.loads(
        release_gate_plan_path.read_text(encoding="utf-8")
    )
    training_plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    partition_gate = json.loads(partition_gate_path.read_text(encoding="utf-8"))
    pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    categories = sorted(
        name for name in COMPONENT_COMPARTMENT_PROTOCOLS if name != "shared_cross_category"
    )
    component_count = 4032
    component_set = "0cab606cc4ab7f092b57ab9b02a58544511be7c11009872afaebd84621f38aa2"
    bundle_sha = "a" * 64
    bundle_manifest = {
        "schema": "sch.vpd-sae-cluster-manifest.v1",
        "status": "complete",
        "algorithm": {
            "activation_category_compartments": {
                "enabled": True,
                "contract_sha256": content_sha256(category_contract()),
                "categories": categories,
            }
        },
        "source": {
            "component_inventory_sha256": "1" * 64,
        },
        "invariants": {
            "component_count": component_count,
            "assigned_component_count": component_count,
            "unique_assigned_component_count": component_count,
            "exact_partition": True,
            "component_id_set_sha256": component_set,
        },
    }
    bundle_verification = {
        "schema": "sch.vpd-sae-cluster-verification.v1",
        "status": "pass",
        "components": component_count,
        "clusters": 16,
        "packets": 16,
        "exact_partition": True,
    }
    measurement_body = {
        "schema": "sch.vpd-partition-audit-cross-archive-measurement-identity.v1",
        "status": "pass",
        "shared_identity": {
            "model_id": "openai/bonsai-8b",
            "revision": "synthetic-frozen-revision",
            "adapter_sha256": "c" * 64,
            "effect_space_id": training_plan["profile"]["activation_key"],
            "phase": "pre_gate",
            "generated_tokens": 0,
        },
        "dataset_identity_intentionally_distinct": True,
        "split_role_identity_intentionally_distinct": True,
    }
    measurement = {
        **measurement_body,
        "identity_id": content_sha256(measurement_body),
    }
    audit_body = {
        "schema": "sch.vpd-frozen-category-partition-audit.v1",
        "status": "pass_partition_audit",
        "source": {
            "bundle_manifest_sha256": bundle_sha,
            "partition_audit_gate_plan_id": partition_gate["gate_plan_id"],
            "partition_audit_pack_id": pack_manifest["pack_id"],
            "cross_archive_measurement_identity": measurement,
        },
        "policy": {
            "mode": "synthetic_release_contract_fixture",
        },
        "category_names": categories,
        "component_id_set_sha256": component_set,
        "summary": {
            "component_count": component_count,
            "frozen_specific_component_count": 8,
            "frozen_shared_component_count": component_count - 8,
            "specific_pass_count": 8,
            "specific_failure_count": 0,
            "specific_failure_reason_counts": {},
            "eligible_for_category_specific_sae_training": True,
        },
        "decisions": [],
        "scientific_boundary": {
            "model_outcomes_used": False,
            "partition_mutated": False,
            "selection_performed_on_audit": False,
        },
    }
    audit_report = {
        **audit_body,
        "partition_audit_id": content_sha256(audit_body),
    }
    audit_verification = {
        "schema": "sch.vpd-production-partition-audit-verification.v1",
        "status": "pass",
        "partition_audit_id": audit_report["partition_audit_id"],
        "audit_decision": "pass_partition_audit",
        "component_count": component_count,
        "specific_failure_count": 0,
        "eligible_for_category_specific_sae_training": True,
    }
    return {
        "release_gate_plan": release_gate_plan,
        "release_gate_plan_sha256": sha256_file(release_gate_plan_path),
        "training_plan": training_plan,
        "training_plan_sha256": sha256_file(training_plan_path),
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "bundle_manifest": bundle_manifest,
        "bundle_manifest_sha256": bundle_sha,
        "bundle_verification": bundle_verification,
        "partition_audit_report": audit_report,
        "partition_audit_report_sha256": "b" * 64,
        "partition_audit_verification": audit_verification,
        "partition_audit_gate_plan": partition_gate,
        "partition_audit_gate_plan_sha256": sha256_file(partition_gate_path),
        "partition_audit_pack_manifest": pack_manifest,
        "partition_audit_pack_manifest_sha256": sha256_file(pack_manifest_path),
    }


def _reject_report(mutator: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
    def apply(inputs: dict[str, Any]) -> None:
        report = copy.deepcopy(inputs["partition_audit_report"])
        mutator(report)
        report = _readdress(report, "partition_audit_id")
        inputs["partition_audit_report"] = report
        inputs["partition_audit_verification"]["partition_audit_id"] = report[
            "partition_audit_id"
        ]

    return apply


def build_benchmark_report(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[2]

    def reject_audit(report: dict[str, Any]) -> None:
        report["status"] = "reject_partition_audit"
        report["summary"]["specific_failure_count"] = 1
        report["summary"]["specific_pass_count"] = 7
        report["summary"]["eligible_for_category_specific_sae_training"] = False

    def mismatch_census(report: dict[str, Any]) -> None:
        report["summary"]["component_count"] += 1

    def stale_gate(report: dict[str, Any]) -> None:
        report["source"]["partition_audit_gate_plan_id"] = "d" * 64

    def stale_pack(report: dict[str, Any]) -> None:
        report["source"]["partition_audit_pack_id"] = "e" * 64

    def shared_only(report: dict[str, Any]) -> None:
        report["summary"]["frozen_specific_component_count"] = 0
        report["summary"]["frozen_shared_component_count"] = report["summary"][
            "component_count"
        ]

    def wrong_effect_space(report: dict[str, Any]) -> None:
        report["source"]["cross_archive_measurement_identity"]["shared_identity"][
            "effect_space_id"
        ] = "layer_34__proposal_start"

    scenarios: list[
        tuple[str, Callable[[dict[str, Any]], None] | None, str | None]
    ] = [
        ("clean_release", None, None),
        (
            "missing_partition_audit",
            lambda inputs: inputs.__setitem__("partition_audit_report", {}),
            "partition audit report content address mismatch",
        ),
        (
            "rejected_partition_audit",
            _reject_report(reject_audit),
            "partition audit did not release category-specific SAE training",
        ),
        (
            "wrong_bundle",
            lambda inputs: inputs.__setitem__("bundle_manifest_sha256", "e" * 64),
            "partition audit lineage differs",
        ),
        (
            "component_census_mismatch",
            _reject_report(mismatch_census),
            "partition audit did not release category-specific SAE training",
        ),
        (
            "stale_gate_plan",
            _reject_report(stale_gate),
            "partition audit lineage differs",
        ),
        (
            "stale_fresh_pack",
            _reject_report(stale_pack),
            "partition audit lineage differs",
        ),
        (
            "shared_only_candidate",
            _reject_report(shared_only),
            "partition audit did not release category-specific SAE training",
        ),
        (
            "wrong_effect_space",
            _reject_report(wrong_effect_space),
            "partition audit activation identity differs from SAE training",
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, mutation, expected_error in scenarios:
        inputs = copy.deepcopy(_fixtures(root))
        if mutation is not None:
            mutation(inputs)
        try:
            receipt = evaluate_release_contract(**inputs)
            if expected_error is not None:
                raise AssertionError(f"scenario unexpectedly passed: {name}")
            results.append(
                {
                    "name": name,
                    "status": "pass",
                    "observed_stage": "training_release_receipt",
                    "training_release_receipt_id": receipt[
                        "training_release_receipt_id"
                    ],
                    "compute_authorized": receipt["scientific_boundary"][
                        "compute_authorized"
                    ],
                }
            )
        except ValueError as exc:
            if expected_error is None or expected_error not in str(exc):
                raise
            results.append(
                {
                    "name": name,
                    "status": "pass",
                    "observed_stage": "pre_release_rejection",
                    "error": str(exc),
                }
            )
    body = {
        "schema": "sch.vpd-compartment-sae-training-release-gate-benchmark.v1",
        "status": "pass",
        "scenarios": results,
        "summary": {
            "scenarios": len(results),
            "passed": sum(row["status"] == "pass" for row in results),
            "released": sum(
                row["observed_stage"] == "training_release_receipt"
                for row in results
            ),
            "rejected_before_release": sum(
                row["observed_stage"] == "pre_release_rejection"
                for row in results
            ),
        },
        "scientific_boundary": {
            "synthetic_only": True,
            "model_inference_performed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, project_root: Path | None = None) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("training release gate benchmark content address mismatch")
    expected = build_benchmark_report(project_root)
    if observed != expected:
        raise ValueError("training release gate benchmark is not reproducible")
    return {
        "schema": "sch.vpd-compartment-sae-training-release-gate-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "scenarios": observed["summary"]["scenarios"],
        "released": observed["summary"]["released"],
        "rejected_before_release": observed["summary"]["rejected_before_release"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the model-free compartment SAE training release joins."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_benchmark(args.output)
    else:
        result = build_benchmark_report()
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
