from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.plan_vpd_partition_audit_gate import (
    verify_gate_plan as verify_v1_gate_plan,
)
from experiments.authz_mechinterp.plan_vpd_effect_capture_campaign import (
    SCHEMA as DISCOVERY_CAMPAIGN_SCHEMA,
    verify_campaign_plan,
)
from experiments.authz_mechinterp.plan_vpd_effect_capture_campaign_v2 import (
    SCHEMA as DISCOVERY_CAMPAIGN_V2_SCHEMA,
    canonical_inputs as discovery_campaign_v2_inputs,
    verify_campaign_plan_v2,
)


SCHEMA = "sch.vpd-partition-audit-gate-plan.v2"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("partition-audit V2 source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def _module_command(module: str, *arguments: str) -> list[str]:
    return ["{PYTHON}", "-m", module, *arguments]


def _worker_private_key_name(worker_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", worker_id).strip("_").upper()
    if not suffix:
        raise ValueError("partition-audit packet has an invalid assigned worker")
    return f"WORKER_PRIVATE_KEY_{suffix}"


def _verified_discovery_campaign(
    path: Path, *, project_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    campaign_schema = campaign.get("schema")
    if campaign_schema not in {
        DISCOVERY_CAMPAIGN_SCHEMA,
        DISCOVERY_CAMPAIGN_V2_SCHEMA,
    }:
        raise ValueError("partition-audit discovery campaign schema differs")
    body = copy.deepcopy(campaign)
    campaign_plan_id = str(body.pop("campaign_plan_id", ""))
    if content_sha256(body) != campaign_plan_id:
        raise ValueError("partition-audit discovery campaign content address mismatch")

    if campaign_schema == DISCOVERY_CAMPAIGN_V2_SCHEMA:
        verification = verify_campaign_plan_v2(
            path, **discovery_campaign_v2_inputs(project_root)
        )
        if verification.get("status") != "pass":
            raise ValueError("partition-audit discovery campaign verification failed")
        return campaign, verification

    source = campaign.get("source", {})

    def source_path(*keys: str) -> Path:
        value: Any = source
        for key in keys:
            value = value[key]
        resolved = (project_root / str(value)).resolve()
        if not resolved.is_relative_to(project_root.resolve()):
            raise ValueError("partition-audit discovery campaign source escapes project")
        return resolved

    split_manifest_path = source_path("split", "path")
    verification = verify_campaign_plan(
        path,
        bundle_dir=source_path("bundle", "path"),
        split_dir=split_manifest_path.parent,
        protocol_config_path=source_path("protocol", "path"),
        hierarchy_plan_path=source_path("hierarchy_plan", "path"),
    )
    if verification.get("status") != "pass":
        raise ValueError("partition-audit discovery campaign verification failed")
    return campaign, verification


def build_gate_plan_v2(
    *,
    predecessor_gate_plan_path: Path,
    predecessor_protocol_path: Path,
    pack_dir: Path,
    benchmark_path: Path,
    metadata_bundle_dir: Path,
    discovery_campaign_plan_path: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    discovery_campaign, discovery_campaign_verification = (
        _verified_discovery_campaign(
            discovery_campaign_plan_path, project_root=project_root
        )
    )
    campaign_parameters = discovery_campaign["parameters"]
    campaign_worker_modules = {
        str(row["arguments"][1])
        for row in discovery_campaign.get("compute_jobs", [])
        if row.get("arguments", [])[:1] == ["-m"]
    }
    if len(campaign_worker_modules) != 1:
        raise ValueError("partition-audit discovery campaign worker module differs")
    campaign_worker_module = next(iter(campaign_worker_modules))
    required_campaign_parameters = {
        "batch_size",
        "dtype",
        "effect_space_id",
        "epsilon",
        "estimator",
        "expected_hidden_size",
        "generated_tokens",
        "maximum_episodes",
        "selection_seed",
        "uncertainty_blocks",
        "uncertainty_sketch_dimension",
    }
    if set(campaign_parameters) != required_campaign_parameters:
        raise ValueError("partition-audit discovery capture parameter contract differs")
    if (
        int(campaign_parameters["batch_size"]) < 1
        or str(campaign_parameters["dtype"])
        not in {"float16", "bfloat16", "float32"}
        or not 0.0 < float(campaign_parameters["epsilon"]) < 1.0
        or int(campaign_parameters["uncertainty_blocks"]) < 1
        or int(campaign_parameters["uncertainty_sketch_dimension"]) < 1
        or int(campaign_parameters["maximum_episodes"]) != 0
        or int(campaign_parameters["generated_tokens"]) != 0
        or int(campaign_parameters["expected_hidden_size"]) != 4096
        or str(campaign_parameters["estimator"])
        != "mean_central_difference_lora_rank_multiplier"
    ):
        raise ValueError("partition-audit discovery capture parameters are invalid")
    predecessor_verification = verify_v1_gate_plan(
        predecessor_gate_plan_path,
        predecessor_protocol_path=predecessor_protocol_path,
        pack_dir=pack_dir,
        benchmark_path=benchmark_path,
        bundle_dir=metadata_bundle_dir,
    )
    predecessor = json.loads(
        predecessor_gate_plan_path.read_text(encoding="utf-8")
    )
    if predecessor.get("schema") != "sch.vpd-partition-audit-gate-plan.v1":
        raise ValueError("partition-audit V2 predecessor is not V1")
    if predecessor["resources"]["batch_size"] != int(
        campaign_parameters["batch_size"]
    ):
        raise ValueError("partition-audit and discovery campaign batch sizes differ")

    manifest = json.loads(
        (metadata_bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )
    packet_by_id = {
        str(row["packet_id"]): row for row in manifest["artifacts"]["packets"]
    }
    dataset_path = pack_dir / "gated_episodes.jsonl"
    gate_plan_runtime_path = "artifacts/vpd_partition_audit_gate_v2/plan.json"
    job_groups = []
    assigned_worker_jobs: dict[str, list[str]] = {}
    for unit in predecessor["capture_work_units"]:
        work_unit_id = str(unit["work_unit_id"])
        packet_id = str(unit["packet_id"])
        packet = packet_by_id.get(packet_id)
        if packet is None:
            raise ValueError("partition-audit work unit packet is absent")
        packet_path = _relative(
            metadata_bundle_dir / str(packet["packet_path"]), project_root
        )
        packet_payload = json.loads(
            (metadata_bundle_dir / str(packet["packet_path"])).read_text(
                encoding="utf-8"
            )
        )
        packet_effect_space_id = str(
            packet_payload.get("effect_refinement", {}).get("effect_space_id", "")
        )
        if packet_effect_space_id != str(campaign_parameters["effect_space_id"]):
            raise ValueError(
                "partition-audit packet and discovery campaign effect spaces differ"
            )
        assigned_worker = str(
            packet_payload.get("distribution", {}).get("assigned_worker", "")
        )
        if not assigned_worker:
            raise ValueError("partition-audit packet has no assigned worker")
        private_key_name = _worker_private_key_name(assigned_worker)
        private_key_placeholder = "{" + private_key_name + "}"
        assigned_worker_jobs.setdefault(assigned_worker, []).append(work_unit_id)
        stem = f"{{RUN_DIR}}/{work_unit_id}"
        compute = _module_command(
            campaign_worker_module,
            "--packet",
            packet_path,
            "--expected-packet-sha256",
            str(unit["packet_sha256"]),
            "--dataset",
            _relative(dataset_path, project_root),
            "--model",
            "{MODEL}",
            "--revision",
            "{MODEL_REVISION}",
            "--adapter",
            "{ADAPTER}",
            "--expected-adapter-sha256",
            "{ADAPTER_SHA256}",
            "--output-shard",
            f"{{SHARDS_DIR}}/{work_unit_id}.npz",
            "--output-receipt",
            f"{stem}/receipt.unsigned.base.json",
            "--worker-id",
            assigned_worker,
            "--run-id",
            "{RUN_ID}",
            "--maximum-episodes",
            str(int(campaign_parameters["maximum_episodes"])),
            "--selection-seed",
            str(int(campaign_parameters["selection_seed"])),
            "--epsilon",
            str(float(campaign_parameters["epsilon"])),
            "--batch-size",
            str(int(campaign_parameters["batch_size"])),
            "--uncertainty-blocks",
            str(int(campaign_parameters["uncertainty_blocks"])),
            "--uncertainty-sketch-dimension",
            str(int(campaign_parameters["uncertainty_sketch_dimension"])),
            "--minimum-condition-coverage",
            "0.75",
            "--minimum-condition-direction-cosine",
            "0.5",
            "--minimum-condition-relative-norm",
            "0.25",
            "--minimum-template-group-coverage",
            "0.75",
            "--minimum-template-group-direction-cosine",
            "0.5",
            "--minimum-template-group-relative-norm",
            "0.25",
            "--device",
            "{DEVICE}",
            "--dtype",
            str(campaign_parameters["dtype"]),
        )
        attest = _module_command(
            "experiments.authz_mechinterp.attest_partition_audit_effect_receipt",
            "--base-receipt",
            f"{stem}/receipt.unsigned.base.json",
            "--pack-dir",
            _relative(pack_dir, project_root),
            "--gate-plan",
            gate_plan_runtime_path,
            "--work-unit-id",
            work_unit_id,
            "--output",
            f"{stem}/receipt.unsigned.attested.json",
        )
        sign = _module_command(
            "experiments.authz_mechinterp.merge_vpd_effect_shards",
            "--sign-receipt",
            f"{stem}/receipt.unsigned.attested.json",
            "--private-key",
            private_key_placeholder,
            "--signed-output",
            f"{{SIGNED_RECEIPTS_DIR}}/{work_unit_id}.json",
        )
        job_groups.append(
            {
                "schema": "sch.vpd-partition-audit-capture-job-group.v2",
                "work_unit_id": work_unit_id,
                "packet_id": packet_id,
                "cluster_id": str(unit["cluster_id"]),
                "assigned_worker": assigned_worker,
                "private_key_placeholder": private_key_name,
                "steps": [
                    {"name": "compute_unsigned_base_receipt", "command": compute},
                    {"name": "attest_fresh_pack_before_signing", "command": attest},
                    {"name": "sign_attested_receipt", "command": sign},
                ],
                "execution_authorized": False,
            }
        )

    merge_command = _module_command(
        "experiments.authz_mechinterp.merge_vpd_partition_audit_effect_shards",
        "--bundle",
        "{CAPTURE_PACKET_BUNDLE}",
        "--registry",
        "{WORKER_REGISTRY}",
        "--receipts-dir",
        "{SIGNED_RECEIPTS_DIR}",
        "--shards-dir",
        "{SHARDS_DIR}",
        "--output-dir",
        "{AUDIT_MERGE_DIR}",
        "--pack-dir",
        _relative(pack_dir, project_root),
        "--gate-plan",
        gate_plan_runtime_path,
    )
    evaluate_command = _module_command(
        "experiments.authz_mechinterp.evaluate_vpd_partition_audit",
        "--bundle-dir",
        "{CATEGORY_AWARE_BUNDLE}",
        "--hierarchy-plan",
        "{COMPONENT_INVENTORY}",
        "--discovery-effects",
        "{DISCOVERY_EFFECTS}",
        "--discovery-merge-manifest",
        "{DISCOVERY_MERGE_MANIFEST}",
        "--audit-merge-dir",
        "{AUDIT_MERGE_DIR}",
        "--pack-dir",
        _relative(pack_dir, project_root),
        "--gate-plan",
        gate_plan_runtime_path,
        "--output",
        "{PARTITION_AUDIT_REPORT}",
    )

    body = copy.deepcopy(predecessor)
    body.pop("gate_plan_id", None)
    body["schema"] = SCHEMA
    body["status"] = (
        "locally_verified_capture_attestation_signing_merge_and_evaluation_ready_"
        "requires_category_aware_bundle_credentials_and_compute_authorization"
    )
    body["predecessor_gate_plan"] = {
        "path": _relative(predecessor_gate_plan_path, project_root),
        "sha256": sha256_file(predecessor_gate_plan_path),
        "gate_plan_id": predecessor_verification["gate_plan_id"],
    }
    body["source"]["discovery_effect_capture_campaign"] = {
        "path": _relative(discovery_campaign_plan_path, project_root),
        "sha256": sha256_file(discovery_campaign_plan_path),
        "campaign_plan_id": discovery_campaign_verification["campaign_plan_id"],
        "verification": discovery_campaign_verification,
        "capture_parameters": copy.deepcopy(campaign_parameters),
    }
    body["implementation"] = {
        "preserved_effect_worker": {
            "path": "experiments/authz_mechinterp/compute_vpd_effect_shard.py",
            "sha256": sha256_file(
                project_root / "experiments/authz_mechinterp/compute_vpd_effect_shard.py"
            ),
            "modified_for_partition_audit": False,
        },
        "effect_worker_execution_wrapper": copy.deepcopy(
            discovery_campaign["source"]["implementation"].get(
                "effect_worker_wrapper",
                {
                    "path": "experiments/authz_mechinterp/compute_vpd_effect_shard.py",
                    "sha256": sha256_file(
                        project_root
                        / "experiments/authz_mechinterp/compute_vpd_effect_shard.py"
                    ),
                    "preserved_worker_modified": False,
                    "correction": "none_legacy_campaign",
                },
            )
        ),
        "base_merge_and_signer": {
            "path": "experiments/authz_mechinterp/merge_vpd_effect_shards.py",
            "sha256": sha256_file(
                project_root / "experiments/authz_mechinterp/merge_vpd_effect_shards.py"
            ),
            "modified_for_partition_audit": False,
        },
        "capture_adapter": {
            "path": "experiments/authz_mechinterp/attest_partition_audit_effect_receipt.py",
            "sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/attest_partition_audit_effect_receipt.py"
            ),
            "status": "implemented_locally_not_executed",
            "attestation_occurs_before_signature": True,
        },
        "signed_merge_front_end": {
            "path": "experiments/authz_mechinterp/merge_vpd_partition_audit_effect_shards.py",
            "sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/merge_vpd_partition_audit_effect_shards.py"
            ),
            "status": "implemented_locally_not_executed",
        },
        "production_evaluator": {
            "path": "experiments/authz_mechinterp/evaluate_vpd_partition_audit.py",
            "sha256": sha256_file(
                project_root / "experiments/authz_mechinterp/evaluate_vpd_partition_audit.py"
            ),
            "status": "implemented_locally_not_executed",
        },
        "core_reject_only_evaluator": predecessor["implementation"]["evaluator"],
    }
    body["execution_plan"] = {
        "schema": "sch.vpd-partition-audit-execution-plan.v2",
        "required_placeholders": [
            "PYTHON",
            "RUN_DIR",
            "MODEL",
            "MODEL_REVISION",
            "ADAPTER",
            "ADAPTER_SHA256",
            "RUN_ID",
            *[
                _worker_private_key_name(worker_id)
                for worker_id in sorted(assigned_worker_jobs)
            ],
            "WORKER_REGISTRY",
            "SIGNED_RECEIPTS_DIR",
            "SHARDS_DIR",
            "CAPTURE_PACKET_BUNDLE",
            "CATEGORY_AWARE_BUNDLE",
            "COMPONENT_INVENTORY",
            "DISCOVERY_EFFECTS",
            "DISCOVERY_MERGE_MANIFEST",
            "AUDIT_MERGE_DIR",
            "PARTITION_AUDIT_REPORT",
            "DEVICE",
        ],
        "job_groups": job_groups,
        "bundle_roles": {
            "capture_packet_bundle": {
                "purpose": "stable_exact_component_packetization_only",
                "manifest_sha256": predecessor["source"][
                    "current_bundle_evidence"
                ]["manifest_sha256"],
                "component_id_set_sha256": predecessor["source"][
                    "current_bundle_evidence"
                ]["component_id_set_sha256"],
                "activation_category_assignment_claimed": False,
            },
            "frozen_category_aware_bundle": {
                "purpose": "frozen_assignment_discovery_lineage_and_audit_evaluation",
                "resolved_only_after_empirical_candidate_freeze": True,
                "same_component_id_set_as_capture_packet_bundle_required": True,
                "same_packet_ids_as_capture_packet_bundle_required": False,
            },
        },
        "worker_signing": {
            "schema": "sch.vpd-partition-audit-worker-signing-plan.v1",
            "workers": {
                worker_id: {
                    "private_key_placeholder": _worker_private_key_name(worker_id),
                    "work_unit_ids": sorted(work_unit_ids),
                    "registry_entry_must_be_enabled": True,
                    "private_key_must_match_registry_public_key": True,
                }
                for worker_id, work_unit_ids in sorted(assigned_worker_jobs.items())
            },
            "one_assigned_worker_per_packet_required": True,
        },
        "merge_step": {"command": merge_command, "execution_authorized": False},
        "evaluation_step": {
            "command": evaluate_command,
            "execution_authorized": False,
        },
        "preflight_requirements": [
            "explicit_user_compute_authorization",
            "frozen_category_aware_bundle_passes_verify_bundle",
            "capture_packet_bundle_manifest_matches_gate_plan_exactly",
            "category_aware_bundle_component_census_matches_capture_packet_bundle",
            "model_revision_and_adapter_hash_are_pinned",
            "audit_model_revision_adapter_and_measurement_identity_match_discovery_merge",
            "worker_registry_and_private_keys_are_provisioned",
            "each_packet_worker_matches_its_enabled_registry_entry_and_private_key",
            "all_output_directories_are_new_and_empty",
        ],
        "execution_authorized": False,
    }
    body["invariants"]["capture_adapter_is_implemented_and_hash_bound"] = True
    body["invariants"]["every_capture_receipt_is_attested_before_signing"] = True
    body["invariants"]["dedicated_signed_merge_is_required"] = True
    body["invariants"]["production_evaluation_is_reproducible"] = True
    body["invariants"][
        "capture_packetization_is_not_category_assignment_evidence"
    ] = True
    body["invariants"][
        "category_aware_bundle_packet_ids_need_not_match_capture_packet_ids"
    ] = True
    body["invariants"][
        "audit_and_discovery_share_frozen_model_adapter_and_measurement_identity"
    ] = True
    body["invariants"]["every_job_uses_its_packet_assigned_worker"] = True
    body["invariants"]["signing_keys_are_worker_scoped"] = True
    body["invariants"][
        "audit_numeric_capture_parameters_equal_verified_discovery_campaign"
    ] = True
    body["invariants"][
        "model_load_dtype_is_frozen_and_not_a_runtime_placeholder"
    ] = True
    body["invariants"][
        "audit_uses_the_verified_discovery_campaign_worker_module"
    ] = True
    body["scientific_boundary"]["capture_adapter_executed"] = False
    body["scientific_boundary"]["signed_receipts_created"] = False
    body["scientific_boundary"]["partition_audit_report_created"] = False
    return {**body, "gate_plan_id": content_sha256(body)}


def verify_gate_plan_v2(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    plan_id = str(body.pop("gate_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("partition-audit V2 gate plan content address mismatch")
    if build_gate_plan_v2(**kwargs) != observed:
        raise ValueError("partition-audit V2 gate plan is not reproducible")
    return {
        "schema": "sch.vpd-partition-audit-gate-plan-verification.v2",
        "status": "pass",
        "gate_plan_id": plan_id,
        "work_units": len(observed["capture_work_units"]),
        "job_groups": len(observed["execution_plan"]["job_groups"]),
        "episode_forward_equivalents": observed["resources"][
            "episode_forward_equivalents"
        ],
        "capture_adapter_status": observed["implementation"]["capture_adapter"][
            "status"
        ],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the executable-boundary partition-audit gate V2."
    )
    parser.add_argument("--predecessor-gate-plan", type=Path, required=True)
    parser.add_argument("--predecessor-protocol", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--metadata-bundle-dir", type=Path, required=True)
    parser.add_argument("--discovery-campaign-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = {
        "predecessor_gate_plan_path": args.predecessor_gate_plan,
        "predecessor_protocol_path": args.predecessor_protocol,
        "pack_dir": args.pack_dir,
        "benchmark_path": args.benchmark,
        "metadata_bundle_dir": args.metadata_bundle_dir,
        "discovery_campaign_plan_path": args.discovery_campaign_plan,
    }
    if args.verify is not None:
        result = verify_gate_plan_v2(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_gate_plan_v2(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
