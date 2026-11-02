from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.admit_compartment_sae_dictionary_bundle_v2 import (
    BUNDLE_SCHEMA,
    validate_bundle_receipt_structure_v2,
    verify_bundle_for_quality_v2,
)
from experiments.authz_mechinterp.admit_compartment_sae_training_release_v2 import (
    COMPUTE_AUTHORIZATION_SCHEMA,
    verify_explicit_compute_authorization,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_compartment_sae_capture_job_v2 import (
    upgrade_capture_manifest,
    validate_capture_execution,
)
from experiments.authz_mechinterp.run_compartment_sae_training_job_v2 import (
    MANIFEST_V2_SCHEMA,
    upgrade_dictionary_manifest_v2,
    validate_dictionary_manifest_v2,
)


SCHEMA = "sch.vpd-compartment-sae-runtime-benchmark.v2"


def _release(job_id: str, action: str) -> tuple[dict[str, Any], dict[str, Any]]:
    release = {
        "schema": "sch.vpd-compartment-sae-training-release-verification.v2",
        "status": "pass",
        "training_release_receipt_id": "a" * 64,
        "training_release_receipt_sha256": "b" * 64,
        "compute_authorized": False,
    }
    compute = {
        "schema": "sch.vpd-explicit-sae-compute-authorization-verification.v1",
        "status": "pass",
        "compute_authorization_id": "c" * 64,
        "compute_authorization_sha256": "d" * 64,
        "required_job_id": job_id,
        "required_action": action,
    }
    return release, compute


def _predecessor_manifest(plan: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    sample = job["required_sample_contract"]
    partition = sample["pair_atomic_partition"]
    body = {
        "schema": "sch.vpd-fixed-sae-dictionary-manifest.v1",
        "source": {
            "runner_sha256": plan["source"]["implementation"][
                "predecessor_sae_trainer"
            ]["sha256"]
        },
        "model": {"model_id": "synthetic/base", "adapter_sha256": None},
        "dictionary": {
            "steps": job["required_manifest_hyperparameters"]["steps"],
            "seed": job["initialization_seed"],
            "data_selection_seed": job["data_selection_seed"],
            "activation_compartment": {
                "activation_category": job["compartment"],
                "capacity_authorization": {
                    "selected_activation_rows": sample["selected_activation_rows"],
                    "selected_episodes": sample["selected_episodes"],
                    "selected_unique_pairs": sample["selected_unique_pairs"],
                },
            },
            "train_validation_split": {
                key: partition[key]
                for key in (
                    "training_pairs",
                    "validation_pairs",
                    "training_rows",
                    "validation_rows",
                    "training_pair_id_set_sha256",
                    "validation_pair_id_set_sha256",
                )
            },
        },
        "artifacts": {
            "dictionary": {"path": "sae_dictionary.npz", "sha256": "1" * 64}
        },
    }
    return {**body, "dictionary_id": content_sha256(body)}


def _upgrade_dictionary(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    blueprint: dict[str, Any],
    blueprint_path: Path,
    job: dict[str, Any],
    compute_job_id: str | None = None,
) -> dict[str, Any]:
    release, compute = _release(
        compute_job_id or str(job["job_id"]), "compartment_sae_training"
    )
    return upgrade_dictionary_manifest_v2(
        predecessor_manifest=_predecessor_manifest(plan, job),
        predecessor_manifest_sha256="2" * 64,
        plan=plan,
        training_plan_path=plan_path,
        job=job,
        blueprint=blueprint,
        execution_blueprint_path=blueprint_path,
        release_verification=release,
        compute_authorization_verification=compute,
    )


def _readdress(value: dict[str, Any], id_field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != id_field}
    return {**body, id_field: content_sha256(body)}


def _rejects(action: Callable[[], None]) -> bool:
    try:
        action()
    except ValueError:
        return True
    return False


def _bundle_fixture(
    *, plan: dict[str, Any], plan_path: Path, blueprint: dict[str, Any], blueprint_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index, job in enumerate(plan["training_jobs"]):
        dictionary_id = content_sha256(
            {"synthetic_runtime_v2_dictionary": index, "job_id": job["job_id"]}
        )
        row = {
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
            "training_execution_receipt_id": content_sha256(
                {"dictionary_id": dictionary_id, "artifact": "execution"}
            ),
            "capacity_authorization": {"status": "pass"},
            "planned_output_dir": str(job["output_dir"]),
            "manifest_schema": MANIFEST_V2_SCHEMA,
        }
        rows.append(row)
        sources.append(
            {
                key: row[key]
                for key in (
                    "dictionary_id",
                    "dictionary_manifest_sha256",
                    "dictionary_sha256",
                    "compartment",
                    "initialization_seed",
                    "data_selection_seed",
                    "training_execution_receipt_id",
                )
            }
        )
    body = {
        "schema": BUNDLE_SCHEMA,
        "status": "pass_ready_for_calibration_quality_v2",
        "source": {
            "training_plan_id": str(plan["training_plan_id"]),
            "training_plan_sha256": sha256_file(plan_path),
            "execution_blueprint_id": str(blueprint["execution_blueprint_id"]),
            "execution_blueprint_sha256": sha256_file(blueprint_path),
            "training_release_receipt_id": "a" * 64,
            "training_release_receipt_sha256": "b" * 64,
            "compute_authorization_id": "c" * 64,
            "compute_authorization_sha256": "d" * 64,
            "admission_runner_sha256": sha256_file(
                Path(__file__).with_name("admit_compartment_sae_dictionary_bundle_v2.py")
            ),
            "predecessor_v1_admission_runner_sha256": "e" * 64,
            "training_capture": {"synthetic": True},
        },
        "dictionaries": rows,
        "coverage": {
            "training_jobs": 18,
            "admitted_dictionaries": 18,
            "compartments": 9,
        },
        "invariants": {
            "all_dictionary_manifests_are_v2": True,
            "all_training_execution_receipts_reverified": True,
            "all_planned_hyperparameters_bound": True,
            "all_exact_sample_contracts_bound": True,
            "one_release_receipt_shared_by_every_dictionary": True,
            "predecessor_v1_bundle_preserved": True,
        },
        "scientific_boundary": {
            "quality_claimed": False,
            "focused_interventions_authorized": False,
        },
    }
    return {**body, "dictionary_bundle_id": content_sha256(body)}, sources


def run_benchmark(
    *,
    training_plan_path: Path,
    release_gate_path: Path,
    execution_blueprint_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    gate = json.loads(release_gate_path.read_text(encoding="utf-8"))
    blueprint = json.loads(execution_blueprint_path.read_text(encoding="utf-8"))
    job = plan["training_jobs"][0]
    scenarios: list[dict[str, Any]] = []

    exact = _upgrade_dictionary(
        plan=plan,
        plan_path=training_plan_path,
        blueprint=blueprint,
        blueprint_path=execution_blueprint_path,
        job=job,
    )
    validate_dictionary_manifest_v2(
        manifest=exact,
        plan=plan,
        training_plan_path=training_plan_path,
        job=job,
        blueprint=blueprint,
        execution_blueprint_path=execution_blueprint_path,
    )
    scenarios.append({"id": "exact_manifest_v2", "status": "pass"})

    for field, changed_value in (("batch_size", 64), ("learning_rate", 0.01)):
        changed = copy.deepcopy(exact)
        changed["dictionary"][field] = changed_value
        changed = _readdress(changed, "dictionary_id")
        rejected = _rejects(
            lambda changed=changed: validate_dictionary_manifest_v2(
                manifest=changed,
                plan=plan,
                training_plan_path=training_plan_path,
                job=job,
                blueprint=blueprint,
                execution_blueprint_path=execution_blueprint_path,
            )
        )
        scenarios.append(
            {
                "id": f"changed_{field}",
                "status": "pass" if rejected else "fail",
            }
        )

    changed_sample = copy.deepcopy(exact)
    changed_sample["dictionary"]["exact_sample_contract"][
        "selected_episode_id_set_sha256"
    ] = "f" * 64
    changed_sample = _readdress(changed_sample, "dictionary_id")
    sample_rejected = _rejects(
        lambda: validate_dictionary_manifest_v2(
            manifest=changed_sample,
            plan=plan,
            training_plan_path=training_plan_path,
            job=job,
            blueprint=blueprint,
            execution_blueprint_path=execution_blueprint_path,
        )
    )
    scenarios.append(
        {
            "id": "changed_exact_sample_contract",
            "status": "pass" if sample_rejected else "fail",
        }
    )

    wrong_compute_rejected = _rejects(
        lambda: _upgrade_dictionary(
            plan=plan,
            plan_path=training_plan_path,
            blueprint=blueprint,
            blueprint_path=execution_blueprint_path,
            job=job,
            compute_job_id="0" * 64,
        )
    )
    scenarios.append(
        {
            "id": "wrong_job_compute_authorization",
            "status": "pass" if wrong_compute_rejected else "fail",
        }
    )

    capture_job = plan["capture_jobs"][0]
    release, compute = _release(
        str(capture_job["job_id"]), "base_activation_capture"
    )
    capture_body = {
        "schema": "sch.vpd-activation-capture-manifest.v1",
        "source": {
            "runner_sha256": plan["source"]["implementation"][
                "predecessor_capture_worker"
            ]["sha256"]
        },
        "capture": {"generated_tokens": 0},
        "artifacts": {},
    }
    capture = upgrade_capture_manifest(
        predecessor_manifest={
            **capture_body,
            "capture_id": content_sha256(capture_body),
        },
        plan=plan,
        training_plan_path=training_plan_path,
        job=capture_job,
        release_verification=release,
        compute_verification=compute,
    )
    validate_capture_execution(
        manifest=capture,
        plan=plan,
        training_plan_path=training_plan_path,
        job=capture_job,
    )
    scenarios.append({"id": "exact_capture_execution", "status": "pass"})

    bundle, sources = _bundle_fixture(
        plan=plan,
        plan_path=training_plan_path,
        blueprint=blueprint,
        blueprint_path=execution_blueprint_path,
    )
    validate_bundle_receipt_structure_v2(bundle)
    with tempfile.TemporaryDirectory() as directory:
        bundle_path = Path(directory) / "bundle.json"
        write_json(bundle_path, bundle)
        verify_bundle_for_quality_v2(
            receipt_path=bundle_path,
            training_plan_path=training_plan_path,
            execution_blueprint_path=execution_blueprint_path,
            dictionary_sources=sources,
        )
    scenarios.append({"id": "exact_bundle_v2_quality_handoff", "status": "pass"})

    wrong_job_bundle = copy.deepcopy(bundle)
    wrong_job_bundle["dictionaries"][0]["planned_job_id"] = "0" * 64
    wrong_job_bundle = _readdress(wrong_job_bundle, "dictionary_bundle_id")
    with tempfile.TemporaryDirectory() as directory:
        wrong_path = Path(directory) / "wrong-bundle.json"
        write_json(wrong_path, wrong_job_bundle)
        bundle_rejected = _rejects(
            lambda: verify_bundle_for_quality_v2(
                receipt_path=wrong_path,
                training_plan_path=training_plan_path,
                execution_blueprint_path=execution_blueprint_path,
                dictionary_sources=sources,
            )
        )
    scenarios.append(
        {
            "id": "changed_bundle_job_coverage",
            "status": "pass" if bundle_rejected else "fail",
        }
    )

    v1_bundle = copy.deepcopy(bundle)
    v1_bundle["dictionaries"][0]["manifest_schema"] = (
        "sch.vpd-fixed-sae-dictionary-manifest.v1"
    )
    v1_bundle = _readdress(v1_bundle, "dictionary_bundle_id")
    v1_rejected = _rejects(lambda: validate_bundle_receipt_structure_v2(v1_bundle))
    scenarios.append(
        {
            "id": "predecessor_manifest_v1_in_bundle",
            "status": "pass" if v1_rejected else "fail",
        }
    )

    authorization_body = {
        "schema": COMPUTE_AUTHORIZATION_SCHEMA,
        "status": "authorized_by_user",
        "source": {
            "training_plan_id": plan["training_plan_id"],
            "training_plan_sha256": sha256_file(training_plan_path),
            "release_gate_plan_id": gate["release_gate_plan_id"],
            "release_gate_plan_sha256": sha256_file(release_gate_path),
            "training_release_receipt_id": "a" * 64,
            "training_release_receipt_sha256": "b" * 64,
        },
        "scope": {
            "authorized_actions": ["compartment_sae_training"],
            "authorized_job_ids": [str(job["job_id"])],
            "real_actions_authorized": False,
            "base_model_weights_mutable": False,
            "sae_dictionary_weight_training_authorized": True,
        },
    }
    authorization = {
        **authorization_body,
        "compute_authorization_id": content_sha256(authorization_body),
    }
    with tempfile.TemporaryDirectory() as directory:
        auth_path = Path(directory) / "authorization.json"
        write_json(auth_path, authorization)
        auth_verification = verify_explicit_compute_authorization(
            auth_path,
            release_verification={
                "training_release_receipt_id": "a" * 64,
                "training_release_receipt_sha256": "b" * 64,
            },
            release_gate_plan_path=release_gate_path,
            training_plan_path=training_plan_path,
            required_job_id=str(job["job_id"]),
            required_action="compartment_sae_training",
        )
    scenarios.append(
        {
            "id": "external_exact_job_compute_authorization",
            "status": "pass" if auth_verification["status"] == "pass" else "fail",
        }
    )

    coverage_pass = (
        len(gate["guarded_execution"]["guarded_jobs"]) == 22
        and gate["guarded_execution"]["release_receipt_alone_cannot_start_any_job"]
        is True
        and gate["guarded_execution"][
            "explicit_compute_authorization_exact_job_scope_required"
        ]
        is True
        and gate["scientific_boundary"]["compute_authorized"] is False
    )
    scenarios.append(
        {
            "id": "all_22_jobs_dual_gated",
            "status": "pass" if coverage_pass else "fail",
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
            "release_gate_plan_id": str(gate["release_gate_plan_id"]),
            "release_gate_plan_sha256": sha256_file(release_gate_path),
            "execution_blueprint_id": str(blueprint["execution_blueprint_id"]),
            "execution_blueprint_sha256": sha256_file(execution_blueprint_path),
            "benchmark_runner_sha256": sha256_file(Path(__file__)),
        },
        "scenarios": scenarios,
        "summary": {
            "scenarios": len(scenarios),
            "passed": sum(row["status"] == "pass" for row in scenarios),
            "manifest_v2_hyperparameters_and_samples_bound": True,
            "capture_training_bundle_and_quality_handoffs_bound": True,
            "scientific_release_and_external_compute_authorization_both_required": True,
            "repository_generated_compute_authorization": False,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("runtime V2 benchmark address mismatch")
    expected = run_benchmark(**kwargs)
    if observed != expected:
        raise ValueError("runtime V2 benchmark is not reproducible")
    return {
        "schema": "sch.vpd-compartment-sae-runtime-benchmark-verification.v2",
        "status": str(observed["status"]),
        "benchmark_id": identifier,
        "scenarios": len(observed["scenarios"]),
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "training_plan_path": project_root
        / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json",
        "release_gate_path": project_root
        / "artifacts/vpd_compartment_sae_training_release_gate_v2/plan.json",
        "execution_blueprint_path": project_root
        / "artifacts/vpd_compartment_sae_training_execution_blueprint_v2/blueprint.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the executable-but-not-run compartment SAE runtime V2."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_benchmark(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = run_benchmark(**inputs)
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": str(result["schema"]),
                "status": str(result["status"]),
                "benchmark_id": str(result["benchmark_id"]),
                "scenarios": int(result["summary"]["scenarios"])
                if "summary" in result
                else int(result["scenarios"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
