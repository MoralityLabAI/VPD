from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_shards import (
    SHARDED_ACTIVATION_SCHEMA,
    verify_sharded_activation_artifact,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.plan_compartment_sae_training import (
    validate_training_plan_structure,
)
from experiments.authz_mechinterp.verify_activation_capture_semantics import (
    validate_semantic_verification_receipt,
    verify_prompt_window_capture_semantics,
)


def _load_content_addressed(
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


def _confined(directory: Path, relative_path: str) -> Path:
    root = directory.resolve()
    path = (directory / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("dictionary artifact escapes its planned output directory")
    return path


def validate_bundle_receipt_structure(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != "sch.vpd-compartment-sae-dictionary-bundle.v1":
        raise ValueError("unsupported compartment SAE dictionary bundle schema")
    body = dict(receipt)
    bundle_id = str(body.pop("dictionary_bundle_id", ""))
    if content_sha256(body) != bundle_id:
        raise ValueError("compartment SAE dictionary bundle content address mismatch")
    rows = receipt.get("dictionaries")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError("compartment SAE dictionary bundle must contain 18 dictionaries")
    if len({str(row["planned_job_id"]) for row in rows}) != len(rows):
        raise ValueError("compartment SAE dictionary bundle repeats planned jobs")
    if len({str(row["dictionary_id"]) for row in rows}) != len(rows):
        raise ValueError("compartment SAE dictionary bundle repeats dictionary IDs")
    grid = {
        (str(row["compartment"]), int(row["initialization_seed"])) for row in rows
    }
    compartments = {compartment for compartment, _ in grid}
    if len(grid) != 18 or len(compartments) != 9:
        raise ValueError("compartment SAE dictionary bundle grid is incomplete")
    for compartment in compartments:
        seeds = {seed for name, seed in grid if name == compartment}
        if seeds != {17, 29}:
            raise ValueError("compartment SAE dictionary bundle replica seeds differ")
    if {int(row["data_selection_seed"]) for row in rows} != {17}:
        raise ValueError("compartment SAE dictionary bundle data selections differ")
    for row in rows:
        for field in (
            "planned_job_id",
            "dictionary_id",
            "dictionary_manifest_sha256",
            "dictionary_sha256",
            "train_validation_split_sha256",
        ):
            if len(str(row.get(field, ""))) != 64:
                raise ValueError("compartment SAE dictionary bundle has an invalid digest")
        if row.get("status") != "pass":
            raise ValueError("compartment SAE dictionary bundle contains a failed output")
    coverage = receipt.get("coverage", {})
    if int(coverage.get("training_jobs", -1)) != 18 or int(
        coverage.get("admitted_dictionaries", -1)
    ) != 18 or int(coverage.get("compartments", -1)) != 9:
        raise ValueError("compartment SAE dictionary bundle coverage differs")
    if receipt.get("status") != "pass_ready_for_calibration_quality":
        raise ValueError("compartment SAE dictionary bundle status differs")
    source = receipt.get("source", {})
    for field in (
        "training_plan_id",
        "training_plan_sha256",
        "quality_job_id",
        "admission_runner_sha256",
    ):
        if len(str(source.get(field, ""))) != 64:
            raise ValueError("compartment SAE dictionary bundle source is incomplete")
    training_capture = source.get("training_capture", {})
    for field in (
        "capture_id",
        "capture_manifest_sha256",
        "activation_capture_sha256",
        "metadata_sha256",
        "split_id",
    ):
        if len(str(training_capture.get(field, ""))) != 64:
            raise ValueError("compartment SAE dictionary bundle capture is incomplete")
    if training_capture.get("model", {}).get("adapter_sha256") is not None:
        raise ValueError("compartment SAE dictionary bundle is not base-model only")
    semantic = training_capture.get("semantic_verification", {})
    validate_semantic_verification_receipt(semantic)
    boundary = receipt.get("scientific_boundary", {})
    if not all(
        boundary.get(field) is expected
        for field, expected in (
            ("model_outcomes_evaluated", False),
            ("quality_claimed", False),
            ("focused_interventions_authorized", False),
            ("v7_negative_release_decision_unchanged", True),
        )
    ):
        raise ValueError("compartment SAE dictionary bundle weakens its boundary")


def build_bundle_receipt(
    *,
    training_plan: dict[str, Any],
    training_plan_path: Path,
    capture_identity: dict[str, Any],
    dictionary_records: list[dict[str, Any]],
    admission_runner_path: Path | None = None,
) -> dict[str, Any]:
    validate_training_plan_structure(training_plan)
    plan_jobs = {str(row["job_id"]): row for row in training_plan["training_jobs"]}
    records = sorted(
        (dict(row) for row in dictionary_records),
        key=lambda row: (str(row["compartment"]), int(row["initialization_seed"])),
    )
    if {str(row.get("planned_job_id")) for row in records} != set(plan_jobs):
        raise ValueError("dictionary records do not cover the exact training plan")
    for record in records:
        job = plan_jobs[str(record["planned_job_id"])]
        if str(record.get("compartment")) != str(job["compartment"]) or int(
            record.get("initialization_seed", -1)
        ) != int(job["initialization_seed"]):
            raise ValueError("dictionary record differs from its planned job")
        if int(record.get("data_selection_seed", -1)) != int(
            job["data_selection_seed"]
        ):
            raise ValueError("dictionary record changed its planned data selection")
        if record.get("status") != "pass":
            raise ValueError("dictionary record did not pass admission")
    for field in (
        "capture_id",
        "capture_manifest_sha256",
        "activation_capture_sha256",
        "metadata_sha256",
        "split_id",
    ):
        if len(str(capture_identity.get(field, ""))) != 64:
            raise ValueError("training capture identity is incomplete")
    semantic = capture_identity.get("semantic_verification", {})
    validate_semantic_verification_receipt(semantic)
    runner_path = Path(__file__) if admission_runner_path is None else admission_runner_path
    body = {
        "schema": "sch.vpd-compartment-sae-dictionary-bundle.v1",
        "status": "pass_ready_for_calibration_quality",
        "source": {
            "training_plan_id": str(training_plan["training_plan_id"]),
            "training_plan_sha256": sha256_file(training_plan_path),
            "quality_job_id": str(training_plan["quality_job"]["job_id"]),
            "admission_runner_sha256": sha256_file(runner_path),
            "training_capture": dict(capture_identity),
        },
        "dictionaries": records,
        "coverage": {
            "training_jobs": len(plan_jobs),
            "admitted_dictionaries": len(records),
            "compartments": len({str(row["compartment"]) for row in records}),
            "replicas_per_compartment": 2,
            "initialization_seeds": [17, 29],
            "data_selection_seed": 17,
        },
        "invariants": {
            "exact_planned_job_coverage": True,
            "dictionary_manifest_and_artifact_hashes_verified": True,
            "same_training_capture_for_every_dictionary": True,
            "same_base_model_for_every_dictionary": True,
            "same_pair_atomic_data_selection_for_replicas": True,
            "distinct_initialization_seeds_per_compartment": True,
            "bundle_receipt_does_not_establish_dictionary_quality": True,
        },
        "scientific_boundary": {
            "model_outcomes_evaluated": False,
            "quality_claimed": False,
            "focused_interventions_authorized": False,
            "v7_negative_release_decision_unchanged": True,
        },
    }
    receipt = {**body, "dictionary_bundle_id": content_sha256(body)}
    validate_bundle_receipt_structure(receipt)
    return receipt


def admit_run_directory(
    *,
    run_dir: Path,
    project_root: Path,
    training_plan_path: Path,
    output_path: Path | None,
) -> dict[str, Any]:
    plan = _load_content_addressed(
        training_plan_path,
        schema="sch.vpd-compartment-sae-training-plan.v1",
        id_field="training_plan_id",
    )
    validate_training_plan_structure(plan)
    capture_dir = run_dir / "captures" / "sae_training"
    capture_manifest_path = capture_dir / "capture_manifest.json"
    capture = _load_content_addressed(
        capture_manifest_path,
        schema="sch.vpd-activation-capture-manifest.v1",
        id_field="capture_id",
    )
    if str(capture.get("source", {}).get("runner_sha256")) != str(
        plan["source"]["implementation"]["capture_worker"]["sha256"]
    ):
        raise ValueError("training capture used a different planned worker")
    capture_jobs = [
        row for row in plan["capture_jobs"] if row.get("role") == "sae_training"
    ]
    if len(capture_jobs) != 1:
        raise ValueError("training plan does not identify one SAE capture job")
    capture_input = capture_jobs[0].get("input", {})
    dataset_path = project_root / str(capture_input.get("path", ""))
    split_manifest_path = project_root / str(
        capture_input.get("split_manifest_path", "")
    )
    if sha256_file(dataset_path) != str(capture_input.get("sha256", "")) or sha256_file(
        split_manifest_path
    ) != str(capture_input.get("split_manifest_sha256", "")):
        raise ValueError("training capture planned semantic inputs differ")
    semantic_verification = verify_prompt_window_capture_semantics(
        capture_dir=capture_dir,
        capture_manifest_path=capture_manifest_path,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        expected_role="sae_training",
    )
    split = capture.get("source", {}).get("activation_category_split", {})
    if split.get("activation_category_split_role") != "sae_training" or split.get(
        "activation_category_split_verification_status"
    ) != "pass" or str(split.get("activation_category_split_id")) != str(
        plan["source"]["split_id"]
    ):
        raise ValueError("training capture split provenance differs from the plan")
    if capture.get("model", {}).get("adapter_sha256") is not None:
        raise ValueError("dictionary bundle training capture is not base-model only")
    if str(plan["profile"]["activation_key"]) not in {
        str(value) for value in capture.get("capture", {}).get("activation_keys", [])
    }:
        raise ValueError("dictionary bundle training capture lacks the planned activation")
    sampling = capture["capture"].get("sae_training_sampling", {})
    if sampling.get("mode") != "prompt_context_window" or int(
        sampling.get("generated_tokens", -1)
    ) != 0 or int(sampling.get("activation_rows_per_episode", -1)) != int(
        plan["profile"]["activation_rows_per_episode"]
        if "activation_rows_per_episode" in plan["profile"]
        else 128
    ):
        raise ValueError("dictionary bundle training capture sampling differs")
    resource = capture.get("source", {}).get("capture_resource_plan", {})
    if str(resource.get("resource_plan_id")) != str(
        plan["source"]["capture_resource_plan_id"]
    ):
        raise ValueError("dictionary bundle capture resource plan differs")
    artifact = capture["artifacts"]["activations"]
    if artifact.get("schema") != SHARDED_ACTIVATION_SCHEMA:
        raise ValueError("dictionary bundle training capture is not sharded")
    verify_sharded_activation_artifact(capture_dir, artifact)
    metadata_path = capture_dir / str(capture["artifacts"]["metadata"]["path"])
    if sha256_file(metadata_path) != str(capture["artifacts"]["metadata"]["sha256"]):
        raise ValueError("dictionary bundle training metadata differs")
    capture_identity = {
        "capture_id": str(capture["capture_id"]),
        "capture_manifest_sha256": sha256_file(capture_manifest_path),
        "activation_capture_sha256": content_sha256(artifact),
        "metadata_sha256": sha256_file(metadata_path),
        "split_id": str(split["activation_category_split_id"]),
        "model": dict(capture["model"]),
        "semantic_verification": semantic_verification,
    }

    records: list[dict[str, Any]] = []
    for job in plan["training_jobs"]:
        directory = run_dir / str(job["output_dir"])
        manifest_path = directory / "sae_dictionary_manifest.json"
        manifest = _load_content_addressed(
            manifest_path,
            schema="sch.vpd-fixed-sae-dictionary-manifest.v1",
            id_field="dictionary_id",
        )
        dictionary = manifest["dictionary"]
        source = manifest["source"]
        compartment = dictionary.get("activation_compartment", {})
        if manifest.get("model") != capture.get("model"):
            raise ValueError("dictionary bundle contains a different base model")
        if str(source.get("capture_id")) != str(capture["capture_id"]) or str(
            source.get("activation_capture_sha256")
        ) != content_sha256(artifact) or str(source.get("capture_manifest_sha256")) != (
            sha256_file(capture_manifest_path)
        ):
            raise ValueError("dictionary bundle contains a different training capture")
        if source.get("capture_semantic_verification") != semantic_verification:
            raise ValueError("dictionary bundle capture semantic verification differs")
        if str(source.get("runner_sha256")) != str(
            plan["source"]["implementation"]["sae_trainer"]["sha256"]
        ) or str(source.get("capacity_plan_sha256")) != str(
            plan["source"]["capacity_plan_sha256"]
        ):
            raise ValueError("dictionary bundle used a different SAE trainer")
        training_split = source.get("activation_category_split", {})
        if training_split.get("activation_category_split_role") != "sae_training" or str(
            training_split.get("activation_category_split_id")
        ) != str(plan["source"]["split_id"]):
            raise ValueError("dictionary bundle split provenance differs")
        capacity = source.get("capacity_authorization", {})
        if capacity.get("status") != "pass" or str(
            capacity.get("capacity_plan_id")
        ) != str(plan["source"]["capacity_plan_id"]) or str(
            capacity.get("profile_id")
        ) != str(plan["profile"]["id"]):
            raise ValueError("dictionary bundle capacity authorization differs")
        expected_support = job["expected_support"]
        if any(
            int(capacity.get(field, -1)) != int(expected_support[expected])
            for field, expected in (
                ("selected_activation_rows", "activation_rows"),
                ("selected_episodes", "episodes"),
                ("selected_unique_pairs", "unique_pairs"),
            )
        ):
            raise ValueError("dictionary bundle support differs from the plan")
        expected_fields = {
            "activation_key": plan["profile"]["activation_key"],
            "input_dim": plan["profile"]["hidden_size"],
            "dict_dim": plan["profile"]["dictionary_features"],
            "top_k": plan["profile"]["top_k"],
            "expansion": plan["profile"]["expansion"],
            "steps": plan["profile"]["steps"],
            "seed": job["initialization_seed"],
            "data_selection_seed": job["data_selection_seed"],
        }
        if any(dictionary.get(field) != value for field, value in expected_fields.items()):
            raise ValueError("dictionary bundle hyperparameters differ from the plan")
        if str(compartment.get("activation_category")) != str(job["compartment"]) or str(
            compartment.get("protocol")
        ) != str(job["protocol"]) or str(
            compartment.get("category_contract_sha256")
        ) != str(job["category_contract_sha256"]) or compartment.get(
            "capacity_authorization"
        ) != capacity:
            raise ValueError("dictionary bundle compartment receipt differs")
        split_receipt = dictionary.get("train_validation_split", {})
        if int(split_receipt.get("seed", -1)) != int(job["data_selection_seed"]) or not all(
            split_receipt.get(field) is True
            for field in (
                "pair_sets_disjoint",
                "row_sets_disjoint",
                "exact_selected_row_partition",
            )
        ):
            raise ValueError("dictionary bundle pair-atomic split differs")
        artifact_record = manifest["artifacts"]["dictionary"]
        artifact_path = _confined(directory, str(artifact_record["path"]))
        if sha256_file(artifact_path) != str(artifact_record["sha256"]):
            raise ValueError("dictionary bundle artifact differs from its manifest")
        records.append(
            {
                "status": "pass",
                "planned_job_id": str(job["job_id"]),
                "compartment": str(job["compartment"]),
                "initialization_seed": int(job["initialization_seed"]),
                "data_selection_seed": int(job["data_selection_seed"]),
                "dictionary_id": str(manifest["dictionary_id"]),
                "dictionary_manifest_sha256": sha256_file(manifest_path),
                "dictionary_sha256": sha256_file(artifact_path),
                "train_validation_split_sha256": content_sha256(split_receipt),
                "capacity_authorization": capacity,
                "planned_output_dir": str(job["output_dir"]),
            }
        )
    receipt = build_bundle_receipt(
        training_plan=plan,
        training_plan_path=training_plan_path,
        capture_identity=capture_identity,
        dictionary_records=records,
    )
    if output_path is not None:
        write_json(output_path, receipt)
    return receipt


def verify_bundle_for_quality(
    *,
    receipt_path: Path,
    training_plan_path: Path,
    dictionary_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_bundle_receipt_structure(receipt)
    plan = _load_content_addressed(
        training_plan_path,
        schema="sch.vpd-compartment-sae-training-plan.v1",
        id_field="training_plan_id",
    )
    validate_training_plan_structure(plan)
    if str(receipt["source"]["training_plan_id"]) != str(plan["training_plan_id"]) or str(
        receipt["source"]["training_plan_sha256"]
    ) != sha256_file(training_plan_path):
        raise ValueError("dictionary bundle and training plan differ")
    if str(receipt["source"].get("quality_job_id")) != str(
        plan["quality_job"]["job_id"]
    ) or str(receipt["source"].get("admission_runner_sha256")) != sha256_file(
        Path(__file__)
    ):
        raise ValueError("dictionary bundle source implementation differs")
    if str(receipt["source"]["training_capture"].get("split_id")) != str(
        plan["source"]["split_id"]
    ):
        raise ValueError("dictionary bundle training capture split differs")
    plan_jobs = {str(row["job_id"]): row for row in plan["training_jobs"]}
    receipt_by_job = {
        str(row["planned_job_id"]): row for row in receipt["dictionaries"]
    }
    if set(receipt_by_job) != set(plan_jobs):
        raise ValueError("dictionary bundle planned job coverage differs")
    for job_id, admitted in receipt_by_job.items():
        job = plan_jobs[job_id]
        if (
            str(admitted.get("compartment")) != str(job["compartment"])
            or int(admitted.get("initialization_seed", -1))
            != int(job["initialization_seed"])
            or int(admitted.get("data_selection_seed", -1))
            != int(job["data_selection_seed"])
            or str(admitted.get("planned_output_dir")) != str(job["output_dir"])
        ):
            raise ValueError("dictionary bundle differs from its planned job")
    receipt_rows = {str(row["dictionary_id"]): row for row in receipt["dictionaries"]}
    source_rows = {str(row["dictionary_id"]): row for row in dictionary_sources}
    if set(receipt_rows) != set(source_rows):
        raise ValueError("dictionary bundle and quality sources differ")
    for dictionary_id, source in source_rows.items():
        admitted = receipt_rows[dictionary_id]
        for field in (
            "compartment",
            "initialization_seed",
            "data_selection_seed",
            "dictionary_manifest_sha256",
            "dictionary_sha256",
        ):
            if admitted.get(field) != source.get(field):
                raise ValueError("dictionary bundle source identity differs")
    return {
        "schema": "sch.vpd-compartment-sae-dictionary-bundle-authorization.v1",
        "status": "pass",
        "training_plan_id": str(plan["training_plan_id"]),
        "training_plan_sha256": sha256_file(training_plan_path),
        "dictionary_bundle_id": str(receipt["dictionary_bundle_id"]),
        "dictionary_bundle_sha256": sha256_file(receipt_path),
        "training_capture": dict(receipt["source"]["training_capture"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Admit exact planned compartment SAE outputs before quality scoring."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        observed = admit_run_directory(
            run_dir=args.run_dir,
            project_root=args.project_root,
            training_plan_path=args.training_plan,
            output_path=None,
        )
        if observed != expected:
            raise ValueError("dictionary bundle receipt is not reproducible")
        result = observed
    else:
        if args.output is None:
            parser.error("admission requires --output")
        result = admit_run_directory(
            run_dir=args.run_dir,
            project_root=args.project_root,
            training_plan_path=args.training_plan,
            output_path=args.output,
        )
    print(
        json.dumps(
            {
                "schema": "sch.vpd-compartment-sae-dictionary-bundle-cli-result.v1",
                "status": result["status"],
                "dictionary_bundle_id": result["dictionary_bundle_id"],
                "dictionaries": len(result["dictionaries"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
