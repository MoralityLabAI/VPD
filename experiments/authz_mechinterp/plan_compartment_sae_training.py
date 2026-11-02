from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import (
    COMPONENT_COMPARTMENT_PROTOCOLS,
    category_contract,
)
from experiments.authz_mechinterp.benchmark_compartment_sae_quality import (
    verify_benchmark as verify_quality_benchmark,
)
from experiments.authz_mechinterp.benchmark_compartment_sae_screening import (
    verify_benchmark as verify_screening_benchmark,
)
from experiments.authz_mechinterp.benchmark_activation_capture_semantics import (
    verify_benchmark as verify_capture_semantic_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.evaluate_compartment_sae_quality import (
    DEFAULT_QUALITY_SAMPLING_POLICY,
    DEFAULT_SAE_DATA_SELECTION_SEED,
    streaming_quality_resource_bounds,
)
from experiments.authz_mechinterp.plan_compartment_capture_resources import (
    verify_capture_resource_plan,
)
DEFAULT_INITIALIZATION_SEEDS = (17, 29)
FROZEN_DATA_SELECTION_SEED = DEFAULT_SAE_DATA_SELECTION_SEED
DEFAULT_STEPS = 2000
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_ACTIVATION_KEY = "layer_35__proposal_start"


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _job(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "job_id": content_sha256(body)}


def build_training_plan(
    *,
    capacity_plan_path: Path,
    capture_resource_plan_path: Path,
    pack_dir: Path,
    split_dir: Path | None = None,
    protocol_config_path: Path,
    quality_benchmark_path: Path,
    initialization_seeds: list[int] | None = None,
    profile_id: str = "bounded_context_pilot",
    activation_key: str = DEFAULT_ACTIVATION_KEY,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    seeds = list(DEFAULT_INITIALIZATION_SEEDS if initialization_seeds is None else initialization_seeds)
    if seeds != list(DEFAULT_INITIALIZATION_SEEDS):
        raise ValueError("compartment SAE plan initialization seeds differ from V1")
    if steps < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("compartment SAE training hyperparameters are invalid")
    resource_verification = verify_capture_resource_plan(
        capture_resource_plan_path,
        capacity_plan_path=capacity_plan_path,
        pack_dir=pack_dir,
        split_dir=split_dir,
    )
    capacity = json.loads(capacity_plan_path.read_text(encoding="utf-8"))
    resource = json.loads(capture_resource_plan_path.read_text(encoding="utf-8"))
    if capacity.get("capacity_plan_id") != resource_verification[
        "capacity_plan_id"
    ]:
        raise ValueError("training and capture resource capacity plans differ")
    profiles = {str(row["id"]): row for row in capacity["profiles"]}
    if profile_id not in profiles or profiles[profile_id].get("status") != "ready":
        raise ValueError("training plan requires a support-ready capacity profile")
    profile = profiles[profile_id]
    if resource["profile"]["id"] != profile_id:
        raise ValueError("training and capture resource profiles differ")
    quality_verification = verify_quality_benchmark(quality_benchmark_path)
    protocol = json.loads(protocol_config_path.read_text(encoding="utf-8"))
    semantic_contract = protocol.get("sae_training", {}).get(
        "capture_semantic_admission", {}
    )
    semantic_benchmark_path = (
        project_root / str(semantic_contract.get("synthetic_benchmark_path", ""))
    ).resolve()
    if not semantic_benchmark_path.is_relative_to(project_root.resolve()):
        raise ValueError("protocol capture semantic benchmark escapes the project")
    semantic_benchmark_verification = verify_capture_semantic_benchmark(
        semantic_benchmark_path
    )
    if semantic_contract.get("synthetic_benchmark_id") != (
        semantic_benchmark_verification["benchmark_id"]
    ) or not all(
        semantic_contract.get(field) is True
        for field in (
            "source_episode_order_and_content_address_required",
            "observable_activation_category_recomputed_from_source",
            "prompt_window_coordinates_reproduced_without_model_inference",
            "required_before_dictionary_training",
            "required_before_dictionary_bundle_admission",
            "required_before_calibration_quality",
        )
    ):
        raise ValueError("protocol capture semantic admission contract differs")
    quality_gate = protocol.get("sae_training", {}).get("quality_gate", {})
    if quality_gate.get("synthetic_benchmark_id") != quality_verification[
        "benchmark_id"
    ]:
        raise ValueError("protocol and quality benchmark identities differ")
    admission_contract = quality_gate.get("dictionary_bundle_admission", {})
    if (
        admission_contract.get("schema")
        != "sch.vpd-compartment-sae-dictionary-bundle.v1"
        or admission_contract.get("runner")
        != "experiments/authz_mechinterp/admit_compartment_sae_dictionary_bundle.py"
        or admission_contract.get("receipt")
        != "dictionary_bundle/dictionary_bundle_receipt.json"
        or admission_contract.get("exact_planned_job_coverage_required") is not True
        or admission_contract.get("required_before_quality_scoring") is not True
        or admission_contract.get("quality_claimed_by_admission") is not False
    ):
        raise ValueError("protocol dictionary bundle admission contract differs")
    screening_contract = protocol.get("focused_evaluation", {}).get(
        "screening_endpoint", {}
    )
    screening_path = (
        project_root / str(screening_contract.get("synthetic_benchmark_path", ""))
    ).resolve()
    if not screening_path.is_relative_to(project_root.resolve()):
        raise ValueError("protocol screening benchmark escapes the project")
    try:
        screening_verification = verify_screening_benchmark(screening_path)
        screening_reproducibility = "current_runner_reproduced"
    except ValueError as exc:
        if (
            protocol.get("schema") != "sch.vpd-activation-category-protocol.v2"
            or "not reproducible" not in str(exc)
        ):
            raise
        historical_report = json.loads(screening_path.read_text(encoding="utf-8"))
        historical_body = dict(historical_report)
        historical_id = str(historical_body.pop("benchmark_id", ""))
        if (
            content_sha256(historical_body) != historical_id
            or historical_report.get("status") != "pass"
        ):
            raise ValueError(
                "historical compartment SAE screening benchmark is invalid"
            ) from exc
        screening_verification = {
            "status": "pass_historical_content_address_only",
            "benchmark_id": historical_id,
            "scenarios": len(historical_report.get("scenarios", [])),
        }
        screening_reproducibility = "historical_content_address_only"
    if (
        screening_contract.get("schema")
        != "sch.vpd-focused-sae-screen-endpoint.v1"
        or screening_contract.get("specific_compartment")
        != "observable_activation_category_membership"
        or screening_contract.get(
            "specific_model_outcome_label_used_for_selection"
        )
        is not False
        or screening_contract.get("shared_compartment") != "behavior_label"
        or screening_contract.get("heldout_intervention_endpoint")
        != "signed_authorization_decision_margin"
        or screening_contract.get("synthetic_benchmark_id")
        != screening_verification["benchmark_id"]
    ):
        raise ValueError("protocol focused screening contract differs")
    replica_contract = protocol.get("sae_training", {}).get(
        "replica_training_plan", {}
    )
    if replica_contract.get("initialization_seeds") != seeds or int(
        replica_contract.get("data_selection_seed", -1)
    ) != FROZEN_DATA_SELECTION_SEED or int(
        replica_contract.get("replicas_per_compartment", -1)
    ) != len(seeds):
        raise ValueError("protocol and replica training plan differ")
    effective_split_dir = split_dir or (pack_dir / "splits")
    split_manifest_path = effective_split_dir / "split_manifest.json"
    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split.get("split_id") != capacity["source"]["split_id"]:
        raise ValueError("training plan pack and capacity split IDs differ")
    unified_lineage = protocol.get("unified_lineage")
    if unified_lineage is not None:
        if (
            not isinstance(unified_lineage, dict)
            or unified_lineage.get("schema")
            != "sch.vpd-unified-compartment-lineage.v1"
            or unified_lineage.get(
                "all_effect_sae_and_evaluation_roles_share_one_exact_split"
            )
            is not True
            or unified_lineage.get("cross_pack_role_reuse_forbidden") is not True
            or unified_lineage.get("execution_authorized") is not False
        ):
            raise ValueError("training plan unified lineage contract differs")
        data_pack = protocol.get("data_pack", {})
        if (
            str(data_pack.get("split_id")) != str(split["split_id"])
            or str(unified_lineage.get("split_id")) != str(split["split_id"])
            or str(unified_lineage.get("split_manifest_sha256"))
            != sha256_file(split_manifest_path)
            or str(data_pack.get("pack_id")) != str(capacity["source"]["pack_id"])
        ):
            raise ValueError("training plan does not share the unified protocol lineage")
        configured_roles = data_pack.get("roles", {})
        if set(configured_roles) != set(split["splits"]):
            raise ValueError("training plan unified role coverage differs")
        for role, record in split["splits"].items():
            configured = configured_roles[role]
            if (
                int(configured.get("rows", -1)) != int(record["rows"])
                or int(configured.get("pairs", -1)) != int(record["pairs"])
                or sorted(configured.get("template_groups", []))
                != sorted(record["template_group_counts"])
            ):
                raise ValueError(f"training plan unified role differs: {role}")
    role_paths = {
        role: effective_split_dir / str(split["splits"][role]["artifact"]["path"])
        for role in ("sae_training", "calibration", "final_evaluation")
    }
    for role, path in role_paths.items():
        if sha256_file(path) != split["splits"][role]["artifact"]["sha256"]:
            raise ValueError(f"training plan split artifact differs: {role}")

    implementation_paths = {
        "capture_worker": project_root / "experiments/authz_mechinterp/capture_activations.py",
        "sae_trainer": project_root / "experiments/authz_mechinterp/train_sae.py",
        "capture_semantic_verifier": project_root / "experiments/authz_mechinterp/verify_activation_capture_semantics.py",
        "dictionary_bundle_admission": project_root / "experiments/authz_mechinterp/admit_compartment_sae_dictionary_bundle.py",
        "quality_runner": project_root / "experiments/authz_mechinterp/run_compartment_sae_quality.py",
        "quality_evaluator": project_root / "experiments/authz_mechinterp/evaluate_compartment_sae_quality.py",
    }
    capture_jobs: list[dict[str, Any]] = []
    for role in ("sae_training", "calibration"):
        role_resources = resource["capture"].get("roles", {}).get(role)
        if not isinstance(role_resources, dict) or int(role_resources["episodes"]) != int(
            split["splits"][role]["rows"]
        ) or int(role_resources["activation_rows"]) != int(
            split["splits"][role]["rows"]
        ) * int(profile["activation_rows_per_episode"]):
            raise ValueError(f"capture resource plan differs for role: {role}")
        output_dir = f"captures/{role}"
        body = {
            "schema": "sch.vpd-compartment-sae-capture-job.v1",
            "role": role,
            "input": {
                "path": _relative(role_paths[role], project_root),
                "sha256": sha256_file(role_paths[role]),
                "split_id": split["split_id"],
                "split_manifest_path": _relative(split_manifest_path, project_root),
                "split_manifest_sha256": sha256_file(split_manifest_path),
                "episodes": int(split["splits"][role]["rows"]),
                "template_group_set_sha256": split["splits"][role][
                    "template_group_set_sha256"
                ],
            },
            "output_dir": output_dir,
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.capture_activations",
                "--input",
                _relative(role_paths[role], project_root),
                "--activation-category-split-manifest",
                _relative(split_manifest_path, project_root),
                "--activation-category-split-role",
                role,
                "--model",
                "${FROZEN_BASE_MODEL}",
                "--revision",
                "${FROZEN_BASE_REVISION}",
                "--layers",
                "35",
                "--phases",
                "pre_gate",
                "--sae-training-window",
                str(profile["activation_rows_per_episode"]),
                "--capture-resource-plan",
                _relative(capture_resource_plan_path, project_root),
                "--activation-shard-rows",
                str(resource["sharding"]["rows_per_full_shard"]),
                "--output-dir",
                output_dir,
            ],
            "runtime_placeholders": ["FROZEN_BASE_MODEL", "FROZEN_BASE_REVISION"],
            "generated_tokens": 0,
            "model_variant": "base",
            "adapter_sha256": None,
            "complete_role_required": True,
            "expected_capture_resources": dict(role_resources),
        }
        capture_jobs.append(_job(body))

    compartments = sorted(COMPONENT_COMPARTMENT_PROTOCOLS)
    training_jobs: list[dict[str, Any]] = []
    training_capture_id = next(
        row["job_id"] for row in capture_jobs if row["role"] == "sae_training"
    )
    for compartment in compartments:
        support = (
            profile["shared_compartment"]
            if compartment == "shared_cross_category"
            else profile["specific_compartments"][compartment]
        )
        if support.get("status") != "ready":
            raise ValueError(f"training support is insufficient for {compartment}")
        for initialization_seed in seeds:
            output_dir = f"dictionaries/{compartment}/seed-{initialization_seed}"
            selection_args = (
                ["--balanced-across-activation-categories"]
                if compartment == "shared_cross_category"
                else ["--activation-category", compartment]
            )
            body = {
                "schema": "sch.vpd-compartment-sae-training-job.v1",
                "compartment": compartment,
                "protocol": COMPONENT_COMPARTMENT_PROTOCOLS[compartment],
                "category_contract_sha256": content_sha256(category_contract()),
                "initialization_seed": initialization_seed,
                "data_selection_seed": FROZEN_DATA_SELECTION_SEED,
                "depends_on_capture_job_id": training_capture_id,
                "output_dir": output_dir,
                "expected_support": {
                    "activation_rows": int(support["total_observations"]),
                    "episodes": int(support["episodes"]),
                    "unique_pairs": int(support["unique_pairs"]),
                    "status": "ready",
                },
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.train_sae",
                    "--activations",
                    "captures/sae_training",
                    "--capture-manifest",
                    "captures/sae_training/capture_manifest.json",
                    "--metadata",
                    "captures/sae_training/metadata.jsonl",
                    "--source-dataset",
                    _relative(role_paths["sae_training"], project_root),
                    "--activation-category-split-manifest",
                    _relative(split_manifest_path, project_root),
                    *selection_args,
                    "--capacity-plan",
                    _relative(capacity_plan_path, project_root),
                    "--capacity-profile",
                    profile_id,
                    "--key",
                    activation_key,
                    "--expansion",
                    str(profile["expansion"]),
                    "--top-k",
                    str(profile["top_k"]),
                    "--steps",
                    str(steps),
                    "--batch-size",
                    str(batch_size),
                    "--learning-rate",
                    str(learning_rate),
                    "--seed",
                    str(initialization_seed),
                    "--split-seed",
                    str(FROZEN_DATA_SELECTION_SEED),
                    "--device",
                    "${SAE_TRAINING_DEVICE}",
                    "--output-dir",
                    output_dir,
                ],
                "runtime_placeholders": ["SAE_TRAINING_DEVICE"],
            }
            training_jobs.append(_job(body))

    calibration_capture_id = next(
        row["job_id"] for row in capture_jobs if row["role"] == "calibration"
    )
    training_plan_relative_path = str(replica_contract.get("path", ""))
    training_plan_target = (project_root / training_plan_relative_path).resolve()
    if (
        not training_plan_relative_path.startswith("artifacts/")
        or training_plan_target.suffix != ".json"
        or not training_plan_target.is_relative_to(project_root.resolve())
    ):
        raise ValueError("compartment SAE protocol plan path is not project-confined JSON")
    dictionary_bundle_receipt = str(admission_contract["receipt"])
    admission_job = _job(
        {
            "schema": "sch.vpd-compartment-sae-dictionary-bundle-admission-job.v1",
            "depends_on_capture_job_id": training_capture_id,
            "depends_on_training_job_ids": sorted(
                row["job_id"] for row in training_jobs
            ),
            "output": dictionary_bundle_receipt,
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.admit_compartment_sae_dictionary_bundle",
                "--run-dir",
                ".",
                "--project-root",
                ".",
                "--training-plan",
                training_plan_relative_path,
                "--output",
                dictionary_bundle_receipt,
            ],
            "required_receipt_status": "pass_ready_for_calibration_quality",
            "does_not_claim_dictionary_quality": True,
        }
    )
    quality_arguments = [
        "-m",
        "experiments.authz_mechinterp.run_compartment_sae_quality",
        "--capture-dir",
        "captures/calibration",
        "--capture-manifest",
        "captures/calibration/capture_manifest.json",
        "--metadata",
        "captures/calibration/metadata.jsonl",
        "--source-dataset",
        _relative(role_paths["calibration"], project_root),
        "--activation-category-split-manifest",
        _relative(split_manifest_path, project_root),
        "--activation-key",
        activation_key,
        "--training-plan",
        training_plan_relative_path,
        "--dictionary-bundle-receipt",
        dictionary_bundle_receipt,
    ]
    for row in sorted(training_jobs, key=lambda item: (item["compartment"], item["initialization_seed"])):
        quality_arguments.extend(["--dictionary-dir", row["output_dir"]])
    quality_arguments.extend(
        [
            "--batch-rows",
            str(quality_gate["execution"]["activation_batch_rows"]),
            "--episodes-per-category",
            str(DEFAULT_QUALITY_SAMPLING_POLICY["episodes_per_category"]),
            "--sampling-seed",
            str(DEFAULT_QUALITY_SAMPLING_POLICY["seed"]),
            "--output",
            "quality/compartment_sae_quality_report.json",
        ]
    )
    quality_job = _job(
        {
            "schema": "sch.vpd-compartment-sae-quality-job.v1",
            "depends_on_capture_job_id": calibration_capture_id,
            "depends_on_training_job_ids": sorted(row["job_id"] for row in training_jobs),
            "depends_on_admission_job_id": admission_job["job_id"],
            "output": "quality/compartment_sae_quality_report.json",
            "arguments": quality_arguments,
            "sampling_policy": dict(DEFAULT_QUALITY_SAMPLING_POLICY),
            "required_report_status": "pass",
            "does_not_authorize_focused_interventions": True,
        }
    )

    hidden_size = int(capacity["model_space"]["hidden_size"])
    dictionary_features = int(profile["dictionary_features"])
    minimum_dictionary_numeric_bytes = (
        2 * dictionary_features * hidden_size * 4
        + (dictionary_features + hidden_size) * 4
    )
    quality_bounds = streaming_quality_resource_bounds(
        hidden_size=hidden_size,
        maximum_features=dictionary_features,
        categories=8,
        batch_rows=int(quality_gate["execution"]["activation_batch_rows"]),
    )
    body = {
        "schema": "sch.vpd-compartment-sae-training-plan.v1",
        "status": "locally_verified_requires_model_identity_and_compute_authorization",
        "source": {
            "pack_id": capacity["source"]["pack_id"],
            "pack_manifest_sha256": sha256_file(pack_dir / "artifact_manifest.json"),
            "split_id": split["split_id"],
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "capacity_plan_id": capacity["capacity_plan_id"],
            "capacity_plan_sha256": sha256_file(capacity_plan_path),
            "capture_resource_plan_id": resource["resource_plan_id"],
            "capture_resource_plan_sha256": sha256_file(capture_resource_plan_path),
            "protocol_config_sha256": sha256_file(protocol_config_path),
            "unified_lineage": (
                None if unified_lineage is None else dict(unified_lineage)
            ),
            "quality_benchmark_id": quality_verification["benchmark_id"],
            "quality_benchmark_sha256": sha256_file(quality_benchmark_path),
            "focused_screening_benchmark_id": screening_verification[
                "benchmark_id"
            ],
            "focused_screening_benchmark_sha256": sha256_file(screening_path),
            "focused_screening_benchmark_reproducibility": screening_reproducibility,
            "capture_semantic_benchmark_id": semantic_benchmark_verification[
                "benchmark_id"
            ],
            "capture_semantic_benchmark_sha256": sha256_file(
                semantic_benchmark_path
            ),
            "planner_sha256": sha256_file(Path(__file__)),
            "implementation": {
                name: {
                    "path": _relative(path, project_root),
                    "sha256": sha256_file(path),
                }
                for name, path in implementation_paths.items()
            },
        },
        "profile": {
            "id": profile_id,
            "activation_key": activation_key,
            "hidden_size": hidden_size,
            "dictionary_features": dictionary_features,
            "expansion": int(profile["expansion"]),
            "top_k": int(profile["top_k"]),
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "initialization_seeds": seeds,
            "data_selection_seed": FROZEN_DATA_SELECTION_SEED,
            "activation_rows_per_episode": int(profile["activation_rows_per_episode"]),
        },
        "capture_jobs": capture_jobs,
        "training_jobs": training_jobs,
        "admission_job": admission_job,
        "quality_job": quality_job,
        "resource_summary": {
            "capture_roles": 2,
            "capture_rows_by_role": {
                role: int(record["activation_rows"])
                for role, record in sorted(resource["capture"]["roles"].items())
            },
            "capture_bytes_by_role": {
                role: int(record["array_bytes_per_layer"])
                for role, record in sorted(resource["capture"]["roles"].items())
            },
            "capture_rows_total": sum(
                int(record["activation_rows"])
                for record in resource["capture"]["roles"].values()
            ),
            "compartments": len(compartments),
            "replicas_per_compartment": len(seeds),
            "training_jobs": len(training_jobs),
            "dictionary_bundle_admission_jobs": 1,
            "training_activation_row_visits": sum(
                int(row["expected_support"]["activation_rows"])
                for row in training_jobs
            ),
            "minimum_numeric_dictionary_artifact_bytes_each": minimum_dictionary_numeric_bytes,
            "minimum_numeric_dictionary_artifact_bytes_total": (
                minimum_dictionary_numeric_bytes * len(training_jobs)
            ),
            "quality_selected_rows": int(DEFAULT_QUALITY_SAMPLING_POLICY["selected_rows"]),
            "quality_activation_row_visits": int(
                DEFAULT_QUALITY_SAMPLING_POLICY["selected_rows"]
            )
            * len(training_jobs),
            "quality_numeric_resource_bounds": quality_bounds,
            "checkpoint_optimizer_and_framework_overhead_not_estimated": True,
        },
        "invariants": {
            "exact_compartment_partition": True,
            "same_training_capture_for_every_replica": True,
            "same_data_selection_seed_for_every_replica": True,
            "distinct_initialization_seeds_within_every_compartment": True,
            "calibration_capture_disjoint_from_dictionary_training": True,
            "final_evaluation_not_used": True,
            "quality_pass_required_before_focused_interventions": True,
            "dictionary_bundle_admission_required_before_quality": True,
            "capture_semantic_admission_required_before_training_and_quality": True,
            "plan_does_not_authorize_capture_training_or_interventions": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_compartment_quality_claimed": False,
            "v7_negative_release_decision_unchanged": True,
        },
    }
    return {**body, "training_plan_id": content_sha256(body)}


def _argument_value(arguments: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"training plan argument is missing or duplicated: {flag}")
    return str(arguments[positions[0] + 1])


def validate_training_plan_structure(plan: dict[str, Any]) -> None:
    if plan.get("schema") != "sch.vpd-compartment-sae-training-plan.v1":
        raise ValueError("unsupported compartment SAE training plan schema")
    capture_jobs = plan.get("capture_jobs")
    training_jobs = plan.get("training_jobs")
    admission_job = plan.get("admission_job")
    quality_job = plan.get("quality_job")
    if not isinstance(capture_jobs, list) or not isinstance(training_jobs, list) or not isinstance(
        quality_job, dict
    ) or not isinstance(admission_job, dict):
        raise ValueError("compartment SAE training plan jobs are malformed")
    for job in [*capture_jobs, *training_jobs, admission_job, quality_job]:
        body = dict(job)
        job_id = str(body.pop("job_id", ""))
        if content_sha256(body) != job_id:
            raise ValueError("compartment SAE training job content address mismatch")
    if {str(row["role"]) for row in capture_jobs} != {"sae_training", "calibration"}:
        raise ValueError("compartment SAE capture job roles differ")
    if len({str(row["job_id"]) for row in capture_jobs}) != len(capture_jobs):
        raise ValueError("compartment SAE capture job IDs are duplicated")
    capture_by_role = {str(row["role"]): row for row in capture_jobs}
    for role, job in capture_by_role.items():
        if _argument_value(job["arguments"], "--activation-category-split-role") != role:
            raise ValueError("compartment SAE capture arguments changed their role")
        if int(_argument_value(job["arguments"], "--sae-training-window")) != 128:
            raise ValueError("compartment SAE capture window differs")
        if "--adapter" in job["arguments"] or int(job["generated_tokens"]) != 0:
            raise ValueError("compartment SAE capture crossed the base prompt-only boundary")
        capture_input = job.get("input", {})
        expected_resources = job.get("expected_capture_resources", {})
        if (
            _argument_value(job["arguments"], "--input")
            != str(capture_input.get("path"))
            or _argument_value(
                job["arguments"], "--activation-category-split-manifest"
            )
            != str(capture_input.get("split_manifest_path"))
            or len(str(capture_input.get("sha256", ""))) != 64
            or len(str(capture_input.get("split_manifest_sha256", ""))) != 64
            or str(capture_input.get("split_id")) != str(plan["source"]["split_id"])
            or int(expected_resources.get("episodes", -1))
            != int(capture_input.get("episodes", -2))
            or int(expected_resources.get("activation_rows", -1))
            != int(capture_input.get("episodes", -2))
            * int(plan["profile"]["activation_rows_per_episode"])
        ):
            raise ValueError("compartment SAE capture semantic inputs differ")

    profile = plan["profile"]
    seeds = [int(value) for value in profile["initialization_seeds"]]
    expected_grid = {
        (compartment, seed)
        for compartment in COMPONENT_COMPARTMENT_PROTOCOLS
        for seed in seeds
    }
    observed_grid = {
        (str(row["compartment"]), int(row["initialization_seed"]))
        for row in training_jobs
    }
    if observed_grid != expected_grid or len(training_jobs) != len(expected_grid):
        raise ValueError("compartment SAE training job grid differs")
    if len({str(row["job_id"]) for row in training_jobs}) != len(training_jobs):
        raise ValueError("compartment SAE training job IDs are duplicated")
    outputs: set[str] = set()
    for job in training_jobs:
        compartment = str(job["compartment"])
        seed = int(job["initialization_seed"])
        arguments = [str(value) for value in job["arguments"]]
        if int(_argument_value(arguments, "--seed")) != seed:
            raise ValueError("compartment SAE initialization seed differs from its job")
        if int(_argument_value(arguments, "--split-seed")) != int(
            profile["data_selection_seed"]
        ) or int(job["data_selection_seed"]) != int(profile["data_selection_seed"]):
            raise ValueError("compartment SAE replicas use different data selections")
        if job["depends_on_capture_job_id"] != capture_by_role["sae_training"]["job_id"]:
            raise ValueError("compartment SAE training uses a different capture")
        if _argument_value(arguments, "--output-dir") != str(job["output_dir"]):
            raise ValueError("compartment SAE training output differs from its job")
        if _argument_value(arguments, "--source-dataset") != str(
            capture_by_role["sae_training"]["input"]["path"]
        ) or _argument_value(
            arguments, "--activation-category-split-manifest"
        ) != str(
            capture_by_role["sae_training"]["input"]["split_manifest_path"]
        ):
            raise ValueError("compartment SAE training semantic inputs differ")
        if compartment == "shared_cross_category":
            if "--balanced-across-activation-categories" not in arguments or (
                "--activation-category" in arguments
            ):
                raise ValueError("shared SAE job changed its balance selection")
        elif _argument_value(arguments, "--activation-category") != compartment:
            raise ValueError("specific SAE job changed its activation category")
        output = str(job["output_dir"])
        if output in outputs:
            raise ValueError("compartment SAE training outputs are duplicated")
        outputs.add(output)

    if admission_job["depends_on_capture_job_id"] != capture_by_role["sae_training"][
        "job_id"
    ] or set(admission_job["depends_on_training_job_ids"]) != {
        str(row["job_id"]) for row in training_jobs
    }:
        raise ValueError("compartment SAE bundle admission dependencies are incomplete")
    admission_arguments = [str(value) for value in admission_job["arguments"]]
    planned_path = _argument_value(admission_arguments, "--training-plan")
    if (
        _argument_value(admission_arguments, "--run-dir") != "."
        or _argument_value(admission_arguments, "--project-root") != "."
        or not planned_path.startswith("artifacts/")
        or not planned_path.endswith("/plan.json")
        or _argument_value(admission_arguments, "--output")
        != str(admission_job["output"])
    ):
        raise ValueError("compartment SAE bundle admission arguments differ")

    quality_body = dict(quality_job)
    if quality_job["depends_on_capture_job_id"] != capture_by_role["calibration"][
        "job_id"
    ]:
        raise ValueError("compartment SAE quality uses a different capture")
    if set(quality_job["depends_on_training_job_ids"]) != {
        str(row["job_id"]) for row in training_jobs
    }:
        raise ValueError("compartment SAE quality dependencies are incomplete")
    if quality_job.get("depends_on_admission_job_id") != admission_job["job_id"]:
        raise ValueError("compartment SAE quality lacks bundle admission dependency")
    quality_arguments = [str(value) for value in quality_job["arguments"]]
    if _argument_value(quality_arguments, "--training-plan") != _argument_value(
        admission_arguments, "--training-plan"
    ) or _argument_value(
        quality_arguments, "--dictionary-bundle-receipt"
    ) != str(admission_job["output"]):
        raise ValueError("compartment SAE quality bundle receipt differs")
    if _argument_value(quality_arguments, "--source-dataset") != str(
        capture_by_role["calibration"]["input"]["path"]
    ) or _argument_value(
        quality_arguments, "--activation-category-split-manifest"
    ) != str(
        capture_by_role["calibration"]["input"]["split_manifest_path"]
    ):
        raise ValueError("compartment SAE quality semantic inputs differ")
    dictionary_dirs = [
        quality_arguments[index + 1]
        for index, value in enumerate(quality_arguments[:-1])
        if value == "--dictionary-dir"
    ]
    if len(dictionary_dirs) != len(training_jobs) or set(dictionary_dirs) != outputs:
        raise ValueError("compartment SAE quality dictionary coverage differs")
    if int(_argument_value(quality_arguments, "--episodes-per-category")) != int(
        DEFAULT_QUALITY_SAMPLING_POLICY["episodes_per_category"]
    ) or int(_argument_value(quality_arguments, "--sampling-seed")) != int(
        DEFAULT_QUALITY_SAMPLING_POLICY["seed"]
    ):
        raise ValueError("compartment SAE quality sampling differs")
    if "final_evaluation" in json.dumps(
        {
            "capture_jobs": capture_jobs,
            "training_jobs": training_jobs,
            "admission_job": admission_job,
            "quality_job": quality_body,
        },
        sort_keys=True,
    ):
        raise ValueError("compartment SAE jobs reference final evaluation")
    boundary = plan.get("scientific_boundary", {})
    if not all(
        boundary.get(field) is expected
        for field, expected in (
            ("model_inference_performed", False),
            ("sae_training_performed", False),
            ("weights_loaded_or_mutated", False),
            ("empirical_compartment_quality_claimed", False),
            ("v7_negative_release_decision_unchanged", True),
        )
    ):
        raise ValueError("compartment SAE training plan weakens its scientific boundary")


def verify_training_plan(
    path: Path,
    *,
    capacity_plan_path: Path,
    capture_resource_plan_path: Path,
    pack_dir: Path,
    split_dir: Path | None = None,
    protocol_config_path: Path,
    quality_benchmark_path: Path,
) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    body = dict(plan)
    plan_id = str(body.pop("training_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("compartment SAE training plan content address mismatch")
    validate_training_plan_structure(plan)
    profile = plan["profile"]
    expected = build_training_plan(
        capacity_plan_path=capacity_plan_path,
        capture_resource_plan_path=capture_resource_plan_path,
        pack_dir=pack_dir,
        split_dir=split_dir,
        protocol_config_path=protocol_config_path,
        quality_benchmark_path=quality_benchmark_path,
        initialization_seeds=[int(value) for value in profile["initialization_seeds"]],
        profile_id=str(profile["id"]),
        activation_key=str(profile["activation_key"]),
        steps=int(profile["steps"]),
        batch_size=int(profile["batch_size"]),
        learning_rate=float(profile["learning_rate"]),
    )
    if expected != plan:
        raise ValueError("compartment SAE training plan is not reproducible")
    compartments = {row["compartment"] for row in plan["training_jobs"]}
    if compartments != set(COMPONENT_COMPARTMENT_PROTOCOLS):
        raise ValueError("compartment SAE training plan partition differs")
    return {
        "schema": "sch.vpd-compartment-sae-training-plan-verification.v1",
        "status": "pass",
        "training_plan_id": plan_id,
        "capture_jobs": len(plan["capture_jobs"]),
        "training_jobs": len(plan["training_jobs"]),
        "compartments": len(compartments),
        "replicas_per_compartment": len(profile["initialization_seeds"]),
        "quality_job_id": plan["quality_job"]["job_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or verify the complete compartment SAE replica experiment."
    )
    parser.add_argument("--capacity-plan", type=Path, required=True)
    parser.add_argument("--capture-resource-plan", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--quality-benchmark", type=Path, required=True)
    parser.add_argument("--initialization-seeds", default="17,29")
    parser.add_argument("--profile", default="bounded_context_pilot")
    parser.add_argument("--activation-key", default=DEFAULT_ACTIVATION_KEY)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    common = {
        "capacity_plan_path": args.capacity_plan,
        "capture_resource_plan_path": args.capture_resource_plan,
        "pack_dir": args.pack_dir,
        "split_dir": args.split_dir,
        "protocol_config_path": args.protocol_config,
        "quality_benchmark_path": args.quality_benchmark,
    }
    if args.verify is not None:
        result = verify_training_plan(args.verify, **common)
    else:
        if args.output is None:
            parser.error("planning requires --output")
        result = build_training_plan(
            **common,
            initialization_seeds=[
                int(value) for value in args.initialization_seeds.split(",")
            ],
            profile_id=args.profile,
            activation_key=args.activation_key,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
