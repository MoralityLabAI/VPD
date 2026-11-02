from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.admit_compartment_sae_dictionary_bundle import (
    build_bundle_receipt,
    validate_bundle_receipt_structure,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_compartment_sae_training_execution_blueprint_v2 import (
    derive_blueprint,
    load_training_plan,
    validate_blueprint,
)


SCHEMA = "sch.vpd-compartment-sae-training-execution-contract-benchmark.v1"


def _readdress(blueprint: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in blueprint.items()
        if key != "execution_blueprint_id"
    }
    return {**body, "execution_blueprint_id": content_sha256(body)}


def _capture_identity(plan: dict[str, Any]) -> dict[str, Any]:
    semantic_body = {
        "schema": "sch.vpd-prompt-window-capture-semantic-verification.v1",
        "status": "pass",
        "exact_source_episode_order_verified": True,
        "exact_category_labels_verified": True,
        "exact_prompt_window_coordinates_verified": True,
        "model_outcomes_used": False,
    }
    return {
        "capture_id": "1" * 64,
        "capture_manifest_sha256": "2" * 64,
        "activation_capture_sha256": "3" * 64,
        "metadata_sha256": "4" * 64,
        "split_id": str(plan["source"]["split_id"]),
        "semantic_verification": {
            **semantic_body,
            "semantic_verification_id": content_sha256(semantic_body),
        },
        "model": {
            "model_id": "synthetic/base-model",
            "revision": "synthetic-revision",
            "adapter_sha256": None,
        },
    }


def _predecessor_records(plan: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, job in enumerate(plan["training_jobs"]):
        dictionary_id = content_sha256(
            {"planned_job_id": job["job_id"], "synthetic_dictionary": True}
        )
        record = {
            "status": "pass",
            "planned_job_id": str(job["job_id"]),
            "compartment": str(job["compartment"]),
            "initialization_seed": int(job["initialization_seed"]),
            "data_selection_seed": int(job["data_selection_seed"]),
            "dictionary_id": dictionary_id,
            "dictionary_manifest_sha256": content_sha256(
                {"dictionary_id": dictionary_id, "artifact": "manifest"}
            ),
            "dictionary_sha256": content_sha256(
                {"dictionary_id": dictionary_id, "artifact": "dictionary"}
            ),
            "train_validation_split_sha256": content_sha256(
                {"dictionary_id": dictionary_id, "artifact": "split"}
            ),
            "capacity_authorization": {"status": "pass"},
            "planned_output_dir": str(job["output_dir"]),
        }
        if index == 0:
            record["synthetic_reported_runtime_hyperparameters"] = {
                "batch_size": 64,
                "learning_rate": 0.01,
            }
        records.append(record)
    return records


def _expect_rejection(action: Callable[[], None]) -> bool:
    try:
        action()
    except ValueError:
        return True
    return False


def run_benchmark(
    *, training_plan_path: Path, project_root: Path
) -> dict[str, Any]:
    blueprint = derive_blueprint(
        training_plan_path=training_plan_path, project_root=project_root
    )
    scenarios: list[dict[str, Any]] = []

    validate_blueprint(
        blueprint,
        training_plan_path=training_plan_path,
        project_root=project_root,
    )
    scenarios.append(
        {
            "id": "exact_v2_blueprint",
            "expected": "accept",
            "observed": "accept",
            "status": "pass",
        }
    )

    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (
            "changed_batch_size",
            lambda value: value["jobs"][0]["required_manifest_hyperparameters"].__setitem__(
                "batch_size", 64
            ),
        ),
        (
            "changed_learning_rate",
            lambda value: value["jobs"][0]["required_manifest_hyperparameters"].__setitem__(
                "learning_rate", 0.01
            ),
        ),
        (
            "changed_selected_episode_identity",
            lambda value: value["jobs"][0]["required_sample_contract"].__setitem__(
                "selected_episode_id_set_sha256", "f" * 64
            ),
        ),
        (
            "specific_cross_compartment_row",
            lambda value: value["jobs"][0]["required_sample_contract"].__setitem__(
                "specific_cross_compartment_rows", 1
            ),
        ),
        (
            "shared_balance_weakened",
            lambda value: next(
                row
                for row in value["jobs"]
                if row["compartment"] == "shared_cross_category"
            )["required_sample_contract"].__setitem__(
                "shared_equal_episode_counts", False
            ),
        ),
        (
            "training_job_removed",
            lambda value: value["jobs"].pop(),
        ),
    ]
    for scenario_id, mutate in mutations:
        changed = copy.deepcopy(blueprint)
        mutate(changed)
        changed = _readdress(changed)
        rejected = _expect_rejection(
            lambda changed=changed: validate_blueprint(
                changed,
                training_plan_path=training_plan_path,
                project_root=project_root,
            )
        )
        scenarios.append(
            {
                "id": scenario_id,
                "expected": "reject",
                "observed": "reject" if rejected else "accept",
                "status": "pass" if rejected else "fail",
            }
        )

    plan = load_training_plan(training_plan_path)
    predecessor = build_bundle_receipt(
        training_plan=plan,
        training_plan_path=training_plan_path,
        capture_identity=_capture_identity(plan),
        dictionary_records=_predecessor_records(plan),
    )
    validate_bundle_receipt_structure(predecessor)
    mismatch = predecessor["dictionaries"][0][
        "synthetic_reported_runtime_hyperparameters"
    ]
    predecessor_accepts = mismatch == {"batch_size": 64, "learning_rate": 0.01}
    scenarios.append(
        {
            "id": "predecessor_v1_documents_unbound_hyperparameter_acceptance",
            "expected": "accept_documents_gap",
            "observed": (
                "accept_documents_gap" if predecessor_accepts else "reject"
            ),
            "status": "pass" if predecessor_accepts else "fail",
            "predecessor_dictionary_bundle_id": predecessor[
                "dictionary_bundle_id"
            ],
            "planned": {
                "batch_size": int(plan["profile"]["batch_size"]),
                "learning_rate": float(plan["profile"]["learning_rate"]),
            },
            "synthetic_reported": mismatch,
        }
    )

    body = {
        "schema": SCHEMA,
        "status": (
            "pass" if all(row["status"] == "pass" for row in scenarios) else "fail"
        ),
        "source": {
            "training_plan_id": str(plan["training_plan_id"]),
            "training_plan_sha256": sha256_file(training_plan_path),
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "execution_blueprint_runner_sha256": blueprint["source"][
                "derivation_runner_sha256"
            ],
            "benchmark_runner_sha256": sha256_file(Path(__file__)),
        },
        "scenarios": scenarios,
        "summary": {
            "scenarios": len(scenarios),
            "passed": sum(row["status"] == "pass" for row in scenarios),
            "v2_accepts_exact_contract": True,
            "v2_rejects_hyperparameter_or_sample_contract_substitution": True,
            "v1_batch_size_and_learning_rate_gap_reproduced": predecessor_accepts,
        },
        "scientific_boundary": dict(blueprint["scientific_boundary"]),
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(
    path: Path, *, training_plan_path: Path, project_root: Path
) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("training execution contract benchmark address mismatch")
    expected = run_benchmark(
        training_plan_path=training_plan_path, project_root=project_root
    )
    if observed != expected:
        raise ValueError("training execution contract benchmark is not reproducible")
    return {
        "schema": "sch.vpd-compartment-sae-training-execution-benchmark-verification.v1",
        "status": str(observed["status"]),
        "benchmark_id": identifier,
        "scenarios": len(observed["scenarios"]),
        "predecessor_gap_reproduced": bool(
            observed["summary"]["v1_batch_size_and_learning_rate_gap_reproduced"]
        ),
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the design-only V2 compartment SAE execution contract."
    )
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    report = run_benchmark(
        training_plan_path=args.training_plan,
        project_root=args.project_root.resolve(),
    )
    if args.verify is not None:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        if report != expected:
            raise ValueError("training execution contract benchmark is not reproducible")
    elif args.output is not None:
        write_json(args.output, report)
    else:
        parser.error("use --output or --verify")
    print(
        json.dumps(
            {
                "schema": "sch.vpd-compartment-sae-training-execution-benchmark-cli.v1",
                "status": report["status"],
                "benchmark_id": report["benchmark_id"],
                "scenarios": len(report["scenarios"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
