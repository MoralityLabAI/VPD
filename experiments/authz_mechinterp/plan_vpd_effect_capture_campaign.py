from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import categorize_episodes
from experiments.authz_mechinterp.activation_category_split_contract import (
    verify_activation_category_split_role,
)
from experiments.authz_mechinterp.benchmark_vpd_effect_campaign_stage_gates import (
    verify_benchmark as verify_stage_gate_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    verify_bundle,
    write_json,
)
from experiments.authz_mechinterp.compute_vpd_effect_shard import iter_jsonl


SCHEMA = "sch.vpd-effect-capture-campaign-plan.v1"
EFFECT_ROLES = ("discovery", "calibration", "cluster_validation")
RUNTIME_PLACEHOLDERS = (
    "FROZEN_BASE_MODEL",
    "FROZEN_BASE_REVISION",
    "FROZEN_ADAPTER_PATH",
    "FROZEN_ADAPTER_SHA256",
    "EFFECT_CAPTURE_DEVICE",
)


def _relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"campaign source escapes the project: {path}")
    return resolved.relative_to(root).as_posix()


def _addressed(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: content_sha256(body)}


def _role_resources(
    rows: list[dict[str, Any]], *, component_count: int, batch_size: int,
    sketch_dimension: int, uncertainty_blocks: int, hidden_size: int
) -> dict[str, Any]:
    categorized = categorize_episodes(rows)
    category_counts = {key: len(value) for key, value in sorted(categorized.items())}
    conditions = sorted(
        {
            (category, str(row["condition"]))
            for category, category_rows in categorized.items()
            for row in category_rows
        }
    )
    templates = sorted(
        {
            (category, str(row["template_group"]))
            for category, category_rows in categorized.items()
            for row in category_rows
        }
    )
    floats_per_component = (
        hidden_size
        + len(categorized) * hidden_size
        + len(categorized) * uncertainty_blocks * sketch_dimension
        + len(conditions) * sketch_dimension
        + len(templates) * sketch_dimension
    )
    batch_forwards_per_component = 2 * sum(
        math.ceil(count / batch_size) for count in category_counts.values()
    )
    return {
        "episodes": len(rows),
        "pairs": len({str(row["pair_id"]) for row in rows}),
        "categories": len(categorized),
        "expected_hidden_size": hidden_size,
        "category_episode_counts": category_counts,
        "condition_cells": len(conditions),
        "template_cells": len(templates),
        "signed_perturbation_lanes": 2,
        "episode_forward_equivalents": 2 * component_count * len(rows),
        "batch_forwards_at_planned_batch_size": (
            component_count * batch_forwards_per_component
        ),
        "estimated_float32_effect_payload_bytes": (
            component_count * floats_per_component * 4
        ),
        "estimate_excludes_npz_container_ids_counters_model_and_runtime_overhead": True,
    }


def build_campaign_plan(
    *,
    bundle_dir: Path,
    split_dir: Path,
    protocol_config_path: Path,
    hierarchy_plan_path: Path,
    batch_size: int = 4,
    dtype: str = "float16",
    epsilon: float = 0.10,
    selection_seed: int = 1707,
    uncertainty_blocks: int = 8,
    sketch_dimension: int = 256,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("effect campaign batch size must be positive")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("effect campaign dtype is unsupported")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("effect campaign epsilon must be between zero and one")
    if uncertainty_blocks != 8 or sketch_dimension < 256:
        raise ValueError("effect campaign robustness contract differs from production")

    project_root = Path(__file__).resolve().parents[2]
    bundle_verification = verify_bundle(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_config_path.read_text(encoding="utf-8"))
    stage_gate_benchmark_relative_path = str(
        protocol.get("effect_capture", {})
        .get("campaign_plan", {})
        .get(
            "stage_gate_benchmark_path",
            "artifacts/vpd_effect_campaign_stage_gate_benchmark_v1/report.json",
        )
    )
    stage_gate_benchmark_path = (
        project_root / stage_gate_benchmark_relative_path
    ).resolve()
    if (
        not stage_gate_benchmark_relative_path.startswith("artifacts/")
        or not stage_gate_benchmark_path.is_relative_to(project_root.resolve())
    ):
        raise ValueError("effect campaign stage-gate benchmark escapes the project")
    stage_gate_benchmark_verification = verify_stage_gate_benchmark(
        stage_gate_benchmark_path,
        protocol_path=protocol_config_path,
    )
    split_path = split_dir / "split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "sch.vpd-activation-category-split-manifest.v3":
        raise ValueError("effect campaign requires the five-role V3 split")
    if protocol.get("data_pack", {}).get("split_id") != split.get("split_id"):
        raise ValueError("effect campaign protocol and split IDs differ")
    if protocol.get("effect_capture", {}).get("receipt_schema") != (
        "sch.vpd-effect-shard-receipt.v5"
    ):
        raise ValueError("effect campaign protocol receipt schema differs")
    if protocol.get("effect_capture", {}).get("generated_tokens") != 0:
        raise ValueError("effect campaign must remain prompt-only")
    campaign_contract = protocol.get("effect_capture", {}).get("campaign_plan", {})
    configured_plan_path = str(campaign_contract.get("path", ""))
    configured_plan_target = (project_root / configured_plan_path).resolve()
    if (
        not configured_plan_path.startswith("artifacts/")
        or configured_plan_target.suffix != ".json"
        or not configured_plan_target.is_relative_to(project_root.resolve())
    ):
        raise ValueError("effect campaign protocol plan path is not project-confined JSON")
    required_campaign_contract = {
        "schema": SCHEMA,
        "status": "locally_verified_requires_frozen_identities_signing_registry_and_compute_authorization",
        "effect_roles": list(EFFECT_ROLES),
        "packets_per_role": 16,
        "compute_jobs": 48,
        "signing_jobs": 48,
        "merge_jobs": 3,
        "stage_gate_jobs": 2,
        "stage_order": ["discovery", "calibration", "cluster_validation"],
        "later_stage_requires_prior_passed_gate": True,
        "plan_does_not_authorize_execution": True,
    }
    if any(
        campaign_contract.get(key) != value
        for key, value in required_campaign_contract.items()
    ):
        raise ValueError("effect campaign protocol launch contract differs")
    sketch_contract = protocol["effect_capture"]["sketch_fidelity_admission"]
    if int(sketch_contract.get("selected_countsketch_dimension", 0)) != sketch_dimension:
        raise ValueError("effect campaign sketch dimension differs from protocol admission")
    if manifest.get("invariants", {}).get("exact_partition") is not True:
        raise ValueError("effect campaign source bundle is not an exact partition")
    if manifest.get("source", {}).get("source_kind") != "hierarchy_plan_exact_partition":
        raise ValueError("effect campaign requires the recovered hierarchy partition")
    if sha256_file(hierarchy_plan_path) != manifest["source"]["component_inventory_sha256"]:
        raise ValueError("effect campaign hierarchy plan differs from the V8 bundle")

    component_count = int(manifest["invariants"]["component_count"])
    packet_entries = sorted(
        manifest["artifacts"]["packets"], key=lambda row: str(row["cluster_id"])
    )
    packet_rows: list[dict[str, Any]] = []
    all_component_ids: list[str] = []
    worker_components: Counter[str] = Counter()
    worker_packets: Counter[str] = Counter()
    for entry in packet_entries:
        packet_path = bundle_dir / str(entry["packet_path"])
        if sha256_file(packet_path) != str(entry["packet_sha256"]):
            raise ValueError("effect campaign packet hash differs")
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        if packet.get("packet_id") != entry.get("packet_id"):
            raise ValueError("effect campaign packet identity differs")
        ids = sorted(str(value) for value in packet["component_ids"])
        if content_sha256(ids) != packet.get("cluster_component_set_sha256"):
            raise ValueError("effect campaign packet component address differs")
        worker = str(packet["distribution"].get("assigned_worker", ""))
        if not worker:
            raise ValueError("effect campaign packet has no assigned worker")
        effect_space = str(packet["effect_refinement"].get("effect_space_id", ""))
        if effect_space != "layer_35__proposal_start":
            raise ValueError("effect campaign packet uses another effect space")
        all_component_ids.extend(ids)
        worker_components[worker] += len(ids)
        worker_packets[worker] += 1
        packet_rows.append(
            {
                "cluster_id": str(packet["cluster_id"]),
                "packet_id": str(packet["packet_id"]),
                "path": _relative(packet_path, project_root),
                "sha256": str(entry["packet_sha256"]),
                "component_count": len(ids),
                "component_id_set_sha256": str(packet["cluster_component_set_sha256"]),
                "assigned_worker": worker,
                "effect_space_id": effect_space,
            }
        )
    if len(all_component_ids) != component_count or len(set(all_component_ids)) != component_count:
        raise ValueError("effect campaign packets are not an exact component partition")
    if content_sha256(sorted(all_component_ids)) != manifest["invariants"][
        "component_id_set_sha256"
    ]:
        raise ValueError("effect campaign component inventory differs")

    role_rows: dict[str, list[dict[str, Any]]] = {}
    role_sources: dict[str, dict[str, Any]] = {}
    for role in EFFECT_ROLES:
        entry = split["splits"][role]
        dataset_path = split_dir / str(entry["artifact"]["path"])
        provenance = verify_activation_category_split_role(
            dataset_path, split_path, role
        )
        rows = list(iter_jsonl(dataset_path))
        role_rows[role] = rows
        role_sources[role] = {
            "path": _relative(dataset_path, project_root),
            "sha256": sha256_file(dataset_path),
            "rows": len(rows),
            "pairs": len({str(row["pair_id"]) for row in rows}),
            "episode_id_set_sha256": provenance[
                "activation_category_split_episode_id_set_sha256"
            ],
            "pair_id_set_sha256": provenance[
                "activation_category_split_pair_id_set_sha256"
            ],
            "template_group_set_sha256": provenance[
                "activation_category_split_template_group_set_sha256"
            ],
        }
    template_hashes = {
        row["template_group_set_sha256"] for row in role_sources.values()
    }
    pair_hashes = {row["pair_id_set_sha256"] for row in role_sources.values()}
    if len(template_hashes) != len(EFFECT_ROLES) or len(pair_hashes) != len(EFFECT_ROLES):
        raise ValueError("effect campaign role semantics are not disjoint")

    implementation = {
        name: {
            "path": _relative(path, project_root),
            "sha256": sha256_file(path),
        }
        for name, path in {
            "planner": Path(__file__),
            "effect_worker": Path(__file__).with_name("compute_vpd_effect_shard.py"),
            "effect_merge": Path(__file__).with_name("merge_vpd_effect_shards.py"),
            "stage_gate": Path(__file__).with_name(
                "gate_vpd_effect_campaign_stage.py"
            ),
            "cluster_builder": Path(__file__).with_name("cluster_vpd_for_sae.py"),
        }.items()
    }
    execution_parameters = {
        "effect_space_id": "layer_35__proposal_start",
        "expected_hidden_size": 4096,
        "estimator": "mean_central_difference_lora_rank_multiplier",
        "epsilon": epsilon,
        "maximum_episodes": 0,
        "selection_seed": selection_seed,
        "batch_size": batch_size,
        "uncertainty_blocks": uncertainty_blocks,
        "uncertainty_sketch_dimension": sketch_dimension,
        "dtype": dtype,
        "generated_tokens": 0,
    }
    source = {
        "bundle": {
            "path": _relative(bundle_dir, project_root),
            "manifest_sha256": sha256_file(manifest_path),
            "verification": bundle_verification,
            "component_count": component_count,
            "component_id_set_sha256": manifest["invariants"][
                "component_id_set_sha256"
            ],
        },
        "hierarchy_plan": {
            "path": _relative(hierarchy_plan_path, project_root),
            "sha256": sha256_file(hierarchy_plan_path),
        },
        "split": {
            "path": _relative(split_path, project_root),
            "sha256": sha256_file(split_path),
            "schema": split["schema"],
            "split_id": split["split_id"],
            "roles": role_sources,
        },
        "protocol": {
            "path": _relative(protocol_config_path, project_root),
            "sha256": sha256_file(protocol_config_path),
        },
        "stage_gate_benchmark": {
            "path": _relative(stage_gate_benchmark_path, project_root),
            "sha256": sha256_file(stage_gate_benchmark_path),
            "verification": stage_gate_benchmark_verification,
        },
        "implementation": implementation,
    }
    execution_identity = {
        "source": source,
        "parameters": execution_parameters,
        "runtime_identity_requirements": list(RUNTIME_PLACEHOLDERS),
    }
    execution_id = content_sha256(execution_identity)
    run_ids = {
        role: content_sha256({"execution_id": execution_id, "role": role})
        for role in EFFECT_ROLES
    }

    compute_jobs: list[dict[str, Any]] = []
    signing_jobs: list[dict[str, Any]] = []
    merge_jobs: list[dict[str, Any]] = []
    gate_jobs: list[dict[str, Any]] = []
    role_resources: dict[str, dict[str, Any]] = {}

    def add_role_stage(
        role: str, *, depends_on_gate_job_id: str | None
    ) -> dict[str, Any]:
        resources = _role_resources(
            role_rows[role],
            component_count=component_count,
            batch_size=batch_size,
            sketch_dimension=sketch_dimension,
            uncertainty_blocks=uncertainty_blocks,
            hidden_size=int(execution_parameters["expected_hidden_size"]),
        )
        role_resources[role] = resources
        stage_compute_jobs: list[dict[str, Any]] = []
        for packet in packet_rows:
            shard_name = f"{packet['cluster_id']}--effect-shard.npz"
            receipt_name = f"{packet['cluster_id']}--effect-shard-receipt.json"
            shard_path = f"runs/{role}/shards/{shard_name}"
            receipt_path = f"runs/{role}/unsigned_receipts/{receipt_name}"
            job_body = {
                "schema": "sch.vpd-effect-capture-job.v1",
                "role": role,
                "run_id": run_ids[role],
                "packet": packet,
                "split_role": role_sources[role],
                "output": {
                    "shard": shard_path,
                    "unsigned_receipt": receipt_path,
                },
                "arguments": [
                    "-m", "experiments.authz_mechinterp.compute_vpd_effect_shard",
                    "--packet", packet["path"],
                    "--expected-packet-sha256", packet["sha256"],
                    "--dataset", role_sources[role]["path"],
                    "--activation-category-split-manifest", source["split"]["path"],
                    "--activation-category-split-role", role,
                    "--model", "${FROZEN_BASE_MODEL}",
                    "--revision", "${FROZEN_BASE_REVISION}",
                    "--adapter", "${FROZEN_ADAPTER_PATH}",
                    "--expected-adapter-sha256", "${FROZEN_ADAPTER_SHA256}",
                    "--output-shard", shard_path,
                    "--output-receipt", receipt_path,
                    "--worker-id", packet["assigned_worker"],
                    "--run-id", run_ids[role],
                    "--maximum-episodes", "0",
                    "--selection-seed", str(selection_seed),
                    "--epsilon", str(epsilon),
                    "--batch-size", str(batch_size),
                    "--uncertainty-blocks", str(uncertainty_blocks),
                    "--uncertainty-sketch-dimension", str(sketch_dimension),
                    "--minimum-condition-coverage", "0.75",
                    "--minimum-condition-direction-cosine", "0.5",
                    "--minimum-condition-relative-norm", "0.25",
                    "--minimum-template-group-coverage", "0.75",
                    "--minimum-template-group-direction-cosine", "0.5",
                    "--minimum-template-group-relative-norm", "0.25",
                    "--device", "${EFFECT_CAPTURE_DEVICE}",
                    "--dtype", dtype,
                ],
                "runtime_placeholders": list(RUNTIME_PLACEHOLDERS),
                "estimated_episode_forward_equivalents": (
                    2 * packet["component_count"] * len(role_rows[role])
                ),
                "estimated_batch_forwards": (
                    packet["component_count"]
                    * resources["batch_forwards_at_planned_batch_size"]
                    // component_count
                ),
            }
            if depends_on_gate_job_id is not None:
                job_body["depends_on_passed_stage_gate_job_id"] = (
                    depends_on_gate_job_id
                )
            job = _addressed(job_body, "job_id")
            compute_jobs.append(job)
            stage_compute_jobs.append(job)

        stage_signing_jobs: list[dict[str, Any]] = []
        for job in stage_compute_jobs:
            worker_key_placeholder = (
                str(job["packet"]["assigned_worker"])
                .upper()
                .replace("-", "_")
                + "_PRIVATE_KEY"
            )
            signed_name = (
                f"{job['packet']['cluster_id']}--effect-shard-receipt.signed.json"
            )
            signed_path = f"runs/{role}/signed_receipts/{signed_name}"
            signing_body = {
                "schema": "sch.vpd-effect-receipt-signing-job.v1",
                "role": role,
                "packet_id": job["packet"]["packet_id"],
                "worker_id": job["packet"]["assigned_worker"],
                "depends_on_compute_job_id": job["job_id"],
                "arguments": [
                    "-m", "experiments.authz_mechinterp.merge_vpd_effect_shards",
                    "--sign-receipt", job["output"]["unsigned_receipt"],
                    "--private-key", f"${{{worker_key_placeholder}}}",
                    "--signed-output", signed_path,
                ],
                "runtime_placeholders": [worker_key_placeholder],
                "output": signed_path,
                "signed_receipt_must_bind_exact_unsigned_receipt": True,
            }
            signing_job = _addressed(signing_body, "job_id")
            signing_jobs.append(signing_job)
            stage_signing_jobs.append(signing_job)

        dependencies = sorted(
            row["job_id"] for row in stage_signing_jobs
        )
        merge_body = {
            "schema": "sch.vpd-effect-merge-job.v1",
            "role": role,
            "depends_on_signing_job_ids": dependencies,
            "arguments": [
                "-m", "experiments.authz_mechinterp.merge_vpd_effect_shards",
                "--bundle", source["bundle"]["path"],
                "--registry", "${FROZEN_WORKER_REGISTRY}",
                "--receipts-dir", f"runs/{role}/signed_receipts",
                "--shards-dir", f"runs/{role}/shards",
                "--output-dir", f"merged/{role}",
            ],
            "runtime_placeholders": ["FROZEN_WORKER_REGISTRY"],
            "expected_outputs": {
                "effects": f"merged/{role}/activation_effects.npz",
                "merge_manifest": f"merged/{role}/effect_merge_manifest.json",
                "components": component_count,
                "activation_categories": 8,
                "hidden_size": int(execution_parameters["expected_hidden_size"]),
            },
            "all_receipts_must_be_signed_and_exact_packet_coverage_required": True,
        }
        merge_job = _addressed(merge_body, "job_id")
        merge_jobs.append(merge_job)
        return merge_job

    discovery_merge = add_role_stage(
        "discovery", depends_on_gate_job_id=None
    )
    discovery_gate_body = {
        "schema": "sch.vpd-effect-campaign-stage-gate-job.v1",
        "stage": "discovery",
        "depends_on_merge_job_ids": [discovery_merge["job_id"]],
        "arguments": [
            "-m", "experiments.authz_mechinterp.gate_vpd_effect_campaign_stage",
            "--stage", "discovery",
            "--hierarchy-plan", source["hierarchy_plan"]["path"],
            "--protocol-config", source["protocol"]["path"],
            "--discovery-effects", "merged/discovery/activation_effects.npz",
            "--discovery-merge-manifest", "merged/discovery/effect_merge_manifest.json",
            "--output", "gates/discovery-continuation.json",
        ],
        "output": "gates/discovery-continuation.json",
        "required_pass_status": "pass_continue_calibration",
        "reject_status_stops_all_remaining_effect_compute": "reject_stop_campaign",
    }
    discovery_gate_job = _addressed(discovery_gate_body, "job_id")
    gate_jobs.append(discovery_gate_job)

    calibration_merge = add_role_stage(
        "calibration", depends_on_gate_job_id=discovery_gate_job["job_id"]
    )
    calibration_gate_body = {
        "schema": "sch.vpd-effect-campaign-stage-gate-job.v1",
        "stage": "calibration",
        "depends_on_gate_job_ids": [discovery_gate_job["job_id"]],
        "depends_on_merge_job_ids": [
            discovery_merge["job_id"],
            calibration_merge["job_id"],
        ],
        "arguments": [
            "-m", "experiments.authz_mechinterp.gate_vpd_effect_campaign_stage",
            "--stage", "calibration",
            "--hierarchy-plan", source["hierarchy_plan"]["path"],
            "--protocol-config", source["protocol"]["path"],
            "--discovery-effects", "merged/discovery/activation_effects.npz",
            "--discovery-merge-manifest", "merged/discovery/effect_merge_manifest.json",
            "--calibration-effects", "merged/calibration/activation_effects.npz",
            "--calibration-merge-manifest", "merged/calibration/effect_merge_manifest.json",
            "--discovery-gate-receipt", "gates/discovery-continuation.json",
            "--output", "gates/calibration-continuation.json",
        ],
        "output": "gates/calibration-continuation.json",
        "required_pass_status": "pass_continue_cluster_validation",
        "reject_status_stops_all_remaining_effect_compute": "reject_stop_campaign",
    }
    calibration_gate_job = _addressed(calibration_gate_body, "job_id")
    gate_jobs.append(calibration_gate_job)

    cluster_validation_merge = add_role_stage(
        "cluster_validation",
        depends_on_gate_job_id=calibration_gate_job["job_id"],
    )

    clustering = protocol["component_compartments"]
    count_contract = clustering["within_category_cluster_count_selection"]
    threshold = clustering["threshold_calibration"]
    cluster_body = {
        "schema": "sch.vpd-effect-campaign-cluster-job.v1",
        "depends_on_gate_job_ids": [calibration_gate_job["job_id"]],
        "depends_on_merge_job_ids": sorted(row["job_id"] for row in merge_jobs),
        "arguments": [
            "-m", "experiments.authz_mechinterp.cluster_vpd_for_sae",
            "--hierarchy-plan", source["hierarchy_plan"]["path"],
            "--effects", "merged/discovery/activation_effects.npz",
            "--effects-merge-manifest", "merged/discovery/effect_merge_manifest.json",
            "--calibration-effects", "merged/calibration/activation_effects.npz",
            "--calibration-effects-merge-manifest", "merged/calibration/effect_merge_manifest.json",
            "--cluster-validation-effects", "merged/cluster_validation/activation_effects.npz",
            "--cluster-validation-effects-merge-manifest", "merged/cluster_validation/effect_merge_manifest.json",
            "--workers", ",".join(sorted(worker_components)),
            "--seed", str(selection_seed),
            "--projection-dim", "64",
            "--effect-space-id", "layer_35__proposal_start",
            "--minimum-dominant-category-share", str(threshold["candidate_grid"]["minimum_dominant_share"][0]),
            "--minimum-category-dominance-margin", str(threshold["candidate_grid"]["minimum_dominance_margin"][0]),
            "--maximum-category-normalized-entropy", str(threshold["candidate_grid"]["maximum_normalized_entropy"][-1]),
            "--minimum-category-replication-rate", str(threshold["minimum_replication_rate"]),
            "--minimum-category-direction-cosine", str(threshold["minimum_signed_direction_cosine"]),
            "--minimum-category-strength-ratio", str(threshold["minimum_dominant_strength_ratio"]),
            "--minimum-stable-specific-components", str(threshold["minimum_stable_specific_components"]),
            "--minimum-category-calibration-episodes", str(threshold["minimum_episodes_per_category_per_archive"]),
            "--minimum-category-jackknife-stability", str(clustering["within_split_uncertainty_screen"]["minimum_same_compartment_stability"]),
            "--minimum-category-pair-block-direction-coverage", str(clustering["within_category_pair_block_direction_robustness_screen"]["minimum_pair_block_coverage"]),
            "--minimum-pair-block-direction-cosine", str(clustering["within_category_pair_block_direction_robustness_screen"]["minimum_pair_block_direction_cosine"]),
            "--minimum-pair-block-relative-norm", str(clustering["within_category_pair_block_direction_robustness_screen"]["minimum_pair_block_relative_norm"]),
            "--minimum-category-condition-coverage", str(clustering["within_category_condition_robustness_screen"]["minimum_condition_coverage"]),
            "--minimum-condition-direction-cosine", str(clustering["within_category_condition_robustness_screen"]["minimum_condition_direction_cosine"]),
            "--minimum-condition-relative-norm", str(clustering["within_category_condition_robustness_screen"]["minimum_condition_relative_norm"]),
            "--minimum-category-template-coverage", str(clustering["within_category_template_robustness_screen"]["minimum_template_group_coverage"]),
            "--minimum-template-direction-cosine", str(clustering["within_category_template_robustness_screen"]["minimum_template_group_direction_cosine"]),
            "--minimum-template-relative-norm", str(clustering["within_category_template_robustness_screen"]["minimum_template_group_relative_norm"]),
            "--require-category-uncertainty",
            "--require-category-pair-block-direction-robustness",
            "--require-category-condition-robustness",
            "--require-category-template-robustness",
            "--select-category-cluster-counts-by-projection",
            "--require-category-cluster-sampling-stability",
            "--require-category-cluster-cross-archive-stability",
            "--category-cluster-projection-replicates", str(count_contract["discovery_projection_replicates"]),
            "--maximum-clusters-per-category", str(count_contract["maximum_clusters_per_category"]),
            "--minimum-category-cluster-projection-stability", str(count_contract["minimum_replicate_adjusted_rand"]),
            "--minimum-category-cluster-silhouette", str(count_contract["minimum_replicate_silhouette"]),
            "--minimum-category-cluster-size", str(count_contract["minimum_cluster_size"]),
            "--minimum-category-robustness-sketch-dimension", str(sketch_dimension),
            "--output-dir", "category_clusters",
        ],
        "expected_output": "category_clusters/manifest.json",
        "production_fail_closed_gates_enabled": True,
    }
    cluster_job = _addressed(cluster_body, "job_id")

    total_resources = {
        "roles": len(EFFECT_ROLES),
        "packets_per_role": len(packet_rows),
        "compute_jobs": len(compute_jobs),
        "signing_jobs": len(signing_jobs),
        "merge_jobs": len(merge_jobs),
        "stage_gate_jobs": len(gate_jobs),
        "cluster_jobs": 1,
        "components_per_role": component_count,
        "episode_forward_equivalents": sum(
            row["episode_forward_equivalents"] for row in role_resources.values()
        ),
        "batch_forwards_at_planned_batch_size": sum(
            row["batch_forwards_at_planned_batch_size"]
            for row in role_resources.values()
        ),
        "estimated_float32_effect_payload_bytes": sum(
            row["estimated_float32_effect_payload_bytes"]
            for row in role_resources.values()
        ),
        "worker_component_load_per_role": dict(sorted(worker_components.items())),
        "worker_packet_load_per_role": dict(sorted(worker_packets.items())),
        "tokenized_sequence_lengths_model_memory_wall_time_and_cost_not_estimated": True,
    }
    expected_campaign_contract = {
        **required_campaign_contract,
        "path": configured_plan_path,
        "episode_forward_equivalents": total_resources[
            "episode_forward_equivalents"
        ],
        "batch_forwards_at_planned_batch_size": total_resources[
            "batch_forwards_at_planned_batch_size"
        ],
        **(
            {"stage_gate_benchmark_path": stage_gate_benchmark_relative_path}
            if "stage_gate_benchmark_path" in campaign_contract
            else {}
        ),
    }
    if campaign_contract != expected_campaign_contract:
        raise ValueError("effect campaign protocol resource contract differs")
    body = {
        "schema": SCHEMA,
        "status": "locally_verified_requires_frozen_identities_signing_registry_and_compute_authorization",
        "execution_id": execution_id,
        "source": source,
        "parameters": execution_parameters,
        "run_ids_by_role": run_ids,
        "packets": packet_rows,
        "compute_jobs": compute_jobs,
        "signing_jobs": signing_jobs,
        "merge_jobs": merge_jobs,
        "stage_gate_jobs": gate_jobs,
        "cluster_job": cluster_job,
        "resources": {"by_role": role_resources, "total": total_resources},
        "invariants": {
            "three_semantically_disjoint_effect_roles": True,
            "exact_packet_coverage_in_every_role": True,
            "exact_component_partition_in_every_role": True,
            "merged_hidden_size_must_equal_4096": True,
            "full_verified_role_capture_required": True,
            "generated_tokens": 0,
            "final_evaluation_excluded": True,
            "sae_training_excluded": True,
            "signed_merge_receipts_required": True,
            "unique_flat_shard_and_receipt_basenames_per_role": True,
            "calibration_compute_requires_passed_discovery_gate": True,
            "cluster_validation_compute_requires_passed_calibration_gate": True,
            "reject_gate_stops_all_later_effect_compute": True,
            "production_32_view_cluster_count_gate_required": True,
            "plan_does_not_authorize_execution": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "adapter_components_would_be_temporarily_perturbed_only_after_authorization": True,
            "factor_restoration_must_be_verified_by_every_worker_receipt": True,
            "campaign_plan_is_not_empirical_clustering_evidence": True,
            "v7_authorization_release_decision_unchanged": True,
        },
    }
    return _addressed(body, "campaign_plan_id")


def verify_campaign_plan(
    path: Path,
    *,
    bundle_dir: Path,
    split_dir: Path,
    protocol_config_path: Path,
    hierarchy_plan_path: Path,
) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported effect capture campaign plan schema")
    body = dict(plan)
    plan_id = str(body.pop("campaign_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("effect capture campaign plan content address mismatch")
    parameters = plan["parameters"]
    expected = build_campaign_plan(
        bundle_dir=bundle_dir,
        split_dir=split_dir,
        protocol_config_path=protocol_config_path,
        hierarchy_plan_path=hierarchy_plan_path,
        batch_size=int(parameters["batch_size"]),
        dtype=str(parameters["dtype"]),
        epsilon=float(parameters["epsilon"]),
        selection_seed=int(parameters["selection_seed"]),
        uncertainty_blocks=int(parameters["uncertainty_blocks"]),
        sketch_dimension=int(parameters["uncertainty_sketch_dimension"]),
    )
    if expected != plan:
        raise ValueError("effect capture campaign plan is not reproducible")
    return {
        "schema": "sch.vpd-effect-capture-campaign-verification.v1",
        "status": "pass",
        "campaign_plan_id": plan_id,
        "roles": len(EFFECT_ROLES),
        "compute_jobs": len(plan["compute_jobs"]),
        "components_per_role": plan["resources"]["total"]["components_per_role"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the fail-closed three-role VPD effect campaign."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--hierarchy-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--selection-seed", type=int, default=1707)
    parser.add_argument("--uncertainty-blocks", type=int, default=8)
    parser.add_argument("--sketch-dimension", type=int, default=256)
    args = parser.parse_args()
    common = {
        "bundle_dir": args.bundle_dir,
        "split_dir": args.split_dir,
        "protocol_config_path": args.protocol_config,
        "hierarchy_plan_path": args.hierarchy_plan,
    }
    if args.verify is not None:
        result = verify_campaign_plan(args.verify, **common)
    else:
        if args.output is None:
            parser.error("campaign generation requires --output")
        result = build_campaign_plan(
            **common,
            batch_size=args.batch_size,
            dtype=args.dtype,
            epsilon=args.epsilon,
            selection_seed=args.selection_seed,
            uncertainty_blocks=args.uncertainty_blocks,
            sketch_dimension=args.sketch_dimension,
        )
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
