from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_partition_audit_capture_pipeline import (
    verify_benchmark as verify_capture_pipeline_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v5 import (
    verify_protocol as verify_v5_protocol,
)
from experiments.authz_mechinterp.derive_vpd_partition_audit_gate_plan_v2 import (
    verify_gate_plan_v2,
)
from experiments.authz_mechinterp.evaluate_vpd_partition_audit import (
    MEASUREMENT_IDENTITY_KEYS,
)


SCHEMA = "sch.vpd-activation-category-protocol.v6"


def build_protocol(
    *,
    predecessor_path: Path,
    v4_protocol_path: Path,
    pack_dir: Path,
    benchmark_path: Path,
    predecessor_gate_plan_path: Path,
    gate_plan_path: Path,
    capture_pipeline_benchmark_path: Path,
    bundle_dir: Path,
    discovery_campaign_plan_path: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    predecessor_verification = verify_v5_protocol(
        predecessor_path,
        predecessor_path=v4_protocol_path,
        pack_dir=pack_dir,
        benchmark_path=benchmark_path,
        gate_plan_path=predecessor_gate_plan_path,
        bundle_dir=bundle_dir,
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v5":
        raise ValueError("capture-adapter protocol predecessor is not V5")
    gate_verification = verify_gate_plan_v2(
        gate_plan_path,
        predecessor_gate_plan_path=predecessor_gate_plan_path,
        predecessor_protocol_path=v4_protocol_path,
        pack_dir=pack_dir,
        benchmark_path=benchmark_path,
        metadata_bundle_dir=bundle_dir,
        discovery_campaign_plan_path=discovery_campaign_plan_path,
    )
    gate = json.loads(gate_plan_path.read_text(encoding="utf-8"))
    capture_pipeline_benchmark = verify_capture_pipeline_benchmark(
        capture_pipeline_benchmark_path, project_root=project_root
    )
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = predecessor_path.resolve().relative_to(
        project_root.resolve()
    ).as_posix()
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    audit = result["component_compartments"]["post_freeze_partition_audit"]
    audit["gate_plan_path"] = gate_plan_path.resolve().relative_to(
        project_root.resolve()
    ).as_posix()
    audit["gate_plan_id"] = gate_verification["gate_plan_id"]
    audit["capture_adapter"] = {
        "status": gate_verification["capture_adapter_status"],
        "attestation_before_signature_required": True,
        "fresh_pack_and_exact_work_unit_provenance_required": True,
        "five_role_split_provenance_forbidden": True,
    }
    audit["signed_merge_front_end_required"] = True
    audit["production_evaluator_required"] = True
    audit["same_frozen_model_revision_adapter_required"] = True
    audit["same_measurement_identity_fields"] = list(MEASUREMENT_IDENTITY_KEYS)
    audit["dataset_and_split_role_identity_expected_to_differ"] = True
    audit["signed_campaign_measurement_contract_required"] = True
    audit["packet_specific_compute_command_hash_required"] = True
    campaign = result["effect_capture"]["partition_audit_campaign"]
    campaign["gate_plan_path"] = audit["gate_plan_path"]
    campaign["gate_plan_id"] = gate_verification["gate_plan_id"]
    campaign["capture_adapter_status"] = gate_verification[
        "capture_adapter_status"
    ]
    campaign["signed_attested_receipts_required"] = True
    campaign["dedicated_partition_audit_merge_required"] = True
    campaign["production_evaluation_driver_required"] = True
    campaign["cross_archive_measurement_identity_validation_required"] = True
    campaign["worker_scoped_signing_required"] = True
    campaign["assigned_workers"] = sorted(
        gate["execution_plan"]["worker_signing"]["workers"]
    )
    campaign["global_worker_or_private_key_placeholder_forbidden"] = True
    campaign["verified_discovery_capture_contract"] = gate["source"][
        "discovery_effect_capture_campaign"
    ]
    campaign["audit_numeric_capture_parameters_derived_from_discovery"] = True
    campaign["model_load_dtype_is_not_runtime_configurable"] = True
    campaign["receipt_level_campaign_measurement_conformance_required"] = True
    campaign["shared_measurement_contract_id_required_across_signed_merge"] = True
    campaign["packet_specific_compute_command_hash_required"] = True
    campaign["model_load_dtype_evidence_boundary"] = (
        "content_addressed_gate_command_and_worker_hash_not_worker_observed_field"
    )
    campaign["prepared_job_groups"] = gate_verification["job_groups"]
    campaign["model_free_capture_pipeline_benchmark"] = {
        "path": capture_pipeline_benchmark_path.resolve().relative_to(
            project_root.resolve()
        ).as_posix(),
        "sha256": sha256_file(capture_pipeline_benchmark_path),
        "benchmark_id": capture_pipeline_benchmark["benchmark_id"],
        "scenarios": capture_pipeline_benchmark["scenarios"],
        "status": capture_pipeline_benchmark["status"],
    }
    campaign["execution_authorized"] = False
    training_gate = result["sae_training"]["partition_audit_gate"]
    training_gate["gate_plan_id"] = gate_verification["gate_plan_id"]
    training_gate["current_status"] = (
        "capture_path_implemented_not_run_no_empirical_category_bundle"
    )
    training_gate["execution_authorized"] = False
    result["unified_lineage"]["capture_adapter_implemented_and_hash_bound"] = True
    result["unified_lineage"]["signed_partition_audit_merge_required"] = True
    result["unified_lineage"]["production_partition_audit_driver_required"] = True
    result["unified_lineage"][
        "discovery_and_audit_measurement_identity_must_match"
    ] = True
    result["unified_lineage"]["packet_assigned_worker_signing_required"] = True
    result["unified_lineage"][
        "verified_discovery_campaign_capture_parameters_bound"
    ] = True
    result["unified_lineage"][
        "signed_receipt_campaign_measurement_contract_required"
    ] = True
    result["unified_lineage"]["model_inference_performed"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        "predecessor_verification": predecessor_verification,
        "gate_plan_v2_verification": gate_verification,
        "capture_pipeline_benchmark_verification": capture_pipeline_benchmark,
        "implementation_hashes": gate["implementation"],
        "research_protocol_v6_is_distinct_from_historical_experiment_v6": True,
    }
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = build_protocol(**kwargs)
    if observed != expected:
        raise ValueError("V6 VPD capture-adapter protocol is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v6",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "capture_adapter_status": observed["effect_capture"][
            "partition_audit_campaign"
        ]["capture_adapter_status"],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Path]:
    artifacts = project_root / "artifacts"
    configs = project_root / "configs" / "evals"
    return {
        "predecessor_path": configs / "vpd_activation_category_compartments_v5.json",
        "v4_protocol_path": configs / "vpd_activation_category_compartments_v4.json",
        "pack_dir": artifacts / "vpd_partition_audit_pack_v1",
        "benchmark_path": artifacts
        / "vpd_frozen_partition_audit_benchmark_v1"
        / "report.json",
        "predecessor_gate_plan_path": artifacts
        / "vpd_partition_audit_gate_v1"
        / "plan.json",
        "gate_plan_path": artifacts / "vpd_partition_audit_gate_v2" / "plan.json",
        "capture_pipeline_benchmark_path": artifacts
        / "vpd_partition_audit_capture_pipeline_benchmark_v1"
        / "report.json",
        "bundle_dir": artifacts / "bonsai8b_authz_v8" / "vpd_sae_clusters",
        "discovery_campaign_plan_path": artifacts
        / "vpd_effect_capture_campaign_v4"
        / "plan.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the capture-adapter research protocol V6."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_protocol(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_protocol(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
