from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1 import (
    _synthetic_materialization,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1 import (
    build_execution_blueprint,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    REQUIRED_ACTION,
    SIMULATION_COMPUTE_SCHEMA,
    SIMULATION_RELEASE_SCHEMA,
    SIMULATION_STATUS,
    ClusterAwareSAETrainingLedger,
    ClusterAwareSAETrainingRuntime,
    reserve_cluster_aware_sae_job_before_runtime_attach,
    validate_execution_blueprint_content,
)


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def _release(blueprint: dict[str, Any], *, real: bool = False) -> dict[str, Any]:
    return {
        "schema": SIMULATION_RELEASE_SCHEMA,
        "status": "pass" if real else SIMULATION_STATUS,
        "scientific_release_receipt_id": content_sha256(
            {"fixture": "cluster-aware-sae-release-v1"}
        ),
        "scientific_release_receipt_sha256": content_sha256(
            {"fixture": "cluster-aware-sae-release-bytes-v1"}
        ),
        "execution_blueprint_id": blueprint["execution_blueprint_id"],
        "materialization_id": blueprint["source"]["materialization_id"],
        "cluster_aware_sae_plan_id": blueprint["source"][
            "cluster_aware_sae_plan_id"
        ],
        "global_pair_split_id": blueprint["source"]["global_pair_split_id"],
        "compute_authorized": False,
    }


def _compute(
    blueprint: dict[str, Any],
    job: dict[str, Any],
    release: dict[str, Any],
    *,
    real: bool = False,
) -> dict[str, Any]:
    return {
        "schema": SIMULATION_COMPUTE_SCHEMA,
        "status": "pass" if real else SIMULATION_STATUS,
        "compute_authorization_id": content_sha256(
            {"fixture": "cluster-aware-sae-compute-v1", "job": job["execution_job_id"]}
        ),
        "compute_authorization_sha256": content_sha256(
            {
                "fixture": "cluster-aware-sae-compute-bytes-v1",
                "job": job["execution_job_id"],
            }
        ),
        "execution_blueprint_id": blueprint["execution_blueprint_id"],
        "required_job_id": job["execution_job_id"],
        "required_action": REQUIRED_ACTION,
        "scientific_release_receipt_id": release[
            "scientific_release_receipt_id"
        ],
        "base_model_weights_mutable": False,
        "real_actions_authorized": False,
        "sae_dictionary_weight_training_authorized": real,
        "execution_authorized": real,
    }


def _inert_runtime(
    callback: Callable[[dict[str, Any]], dict[str, Any]], state: dict[str, Any]
) -> ClusterAwareSAETrainingRuntime:
    state["loader_calls"] = state.get("loader_calls", 0) + 1

    def close() -> None:
        state["close_calls"] = state.get("close_calls", 0) + 1

    return ClusterAwareSAETrainingRuntime(
        train_job=callback,
        close=close,
        runtime_kind="synthetic_callback_no_model_framework_no_training",
        actual_model_loaded=False,
        actual_training_enabled=False,
        persistent_output_enabled=False,
    )


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    materialization = _synthetic_materialization(project_root)
    blueprint = build_execution_blueprint(materialization=materialization)
    blueprint_check = validate_execution_blueprint_content(blueprint)
    jobs = blueprint["execution_jobs"]
    job = jobs[0]
    release = _release(blueprint)
    compute = _compute(blueprint, job, release)
    ledger = ClusterAwareSAETrainingLedger(
        execution_blueprint_id=blueprint["execution_blueprint_id"]
    )
    state: dict[str, Any] = {
        "loader_calls": 0,
        "close_calls": 0,
        "callback_calls": 0,
        "callback_snapshots": [],
    }
    reservation = reserve_cluster_aware_sae_job_before_runtime_attach(
        blueprint=blueprint,
        execution_job_id=job["execution_job_id"],
        release_verification=release,
        compute_authorization_verification=compute,
        ledger=ledger,
        simulation_only=True,
    )
    after_reservation = ledger.snapshot()

    def callback(observed_job: dict[str, Any]) -> dict[str, Any]:
        state["callback_calls"] += 1
        state["callback_snapshots"].append(ledger.snapshot())
        state["observed_job"] = observed_job
        return {
            "status": "synthetic_completed_no_training",
            "execution_job_id": observed_job["execution_job_id"],
            "model_loaded": False,
            "weights_mutated": False,
            "persistent_output_written": False,
            "dictionary_id": None,
        }

    session = reservation.attach_runtime(_inert_runtime(callback, state))
    receipt = session.run()
    session.close()
    session.close()

    replay_rejected = _rejects(
        lambda: reserve_cluster_aware_sae_job_before_runtime_attach(
            blueprint=blueprint,
            execution_job_id=job["execution_job_id"],
            release_verification=release,
            compute_authorization_verification=compute,
            ledger=ledger,
            simulation_only=True,
        )
    )
    second_attach_rejected = _rejects(
        lambda: reservation.attach_runtime(_inert_runtime(callback, {}))
    )
    second_run_rejected = _rejects(session.run)

    failure_job = jobs[1]
    failure_release = _release(blueprint)
    failure_compute = _compute(blueprint, failure_job, failure_release)
    failure_ledger = ClusterAwareSAETrainingLedger(
        execution_blueprint_id=blueprint["execution_blueprint_id"]
    )
    failure_state: dict[str, Any] = {"callback_calls": 0}
    failure_reservation = reserve_cluster_aware_sae_job_before_runtime_attach(
        blueprint=blueprint,
        execution_job_id=failure_job["execution_job_id"],
        release_verification=failure_release,
        compute_authorization_verification=failure_compute,
        ledger=failure_ledger,
        simulation_only=True,
    )

    def failed_callback(_job: dict[str, Any]) -> dict[str, Any]:
        failure_state["callback_calls"] += 1
        raise RuntimeError("synthetic trainer callback failure")

    failed_session = failure_reservation.attach_runtime(
        _inert_runtime(failed_callback, failure_state)
    )
    callback_failure_rejected = _rejects(failed_session.run)

    no_attach_job = jobs[2]
    no_attach_release = _release(blueprint)
    no_attach_compute = _compute(blueprint, no_attach_job, no_attach_release)
    no_attach_ledger = ClusterAwareSAETrainingLedger(
        execution_blueprint_id=blueprint["execution_blueprint_id"]
    )
    no_attach_reservation = reserve_cluster_aware_sae_job_before_runtime_attach(
        blueprint=blueprint,
        execution_job_id=no_attach_job["execution_job_id"],
        release_verification=no_attach_release,
        compute_authorization_verification=no_attach_compute,
        ledger=no_attach_ledger,
        simulation_only=True,
    )

    wrong_job = jobs[3]
    wrong_release = _release(blueprint)
    wrong_compute = _compute(blueprint, wrong_job, wrong_release)
    wrong_compute["required_job_id"] = jobs[4]["execution_job_id"]
    wrong_ledger = ClusterAwareSAETrainingLedger(
        execution_blueprint_id=blueprint["execution_blueprint_id"]
    )
    wrong_authority_rejected = _rejects(
        lambda: reserve_cluster_aware_sae_job_before_runtime_attach(
            blueprint=blueprint,
            execution_job_id=wrong_job["execution_job_id"],
            release_verification=wrong_release,
            compute_authorization_verification=wrong_compute,
            ledger=wrong_ledger,
            simulation_only=True,
        )
    )

    real_release = _release(blueprint, real=True)
    real_compute = _compute(blueprint, jobs[5], real_release, real=True)
    production_from_synthetic_rejected = _rejects(
        lambda: reserve_cluster_aware_sae_job_before_runtime_attach(
            blueprint=blueprint,
            execution_job_id=jobs[5]["execution_job_id"],
            release_verification=real_release,
            compute_authorization_verification=real_compute,
            ledger=ClusterAwareSAETrainingLedger(
                execution_blueprint_id=blueprint["execution_blueprint_id"]
            ),
            simulation_only=False,
        )
    )

    active_runtime_state: dict[str, Any] = {}
    active_runtime_rejected = _rejects(
        lambda: no_attach_reservation.attach_runtime(
            ClusterAwareSAETrainingRuntime(
                train_job=callback,
                close=lambda: None,
                runtime_kind="forbidden-active-synthetic-runtime",
                actual_model_loaded=True,
                actual_training_enabled=True,
                persistent_output_enabled=True,
            )
        )
    )
    del active_runtime_state

    tampered_blueprint = copy.deepcopy(blueprint)
    tampered_blueprint["execution_jobs"][0]["exact_sample_contract"][
        "training_pair_ids"
    ] = []
    outer_body = {
        key: value
        for key, value in tampered_blueprint.items()
        if key != "execution_blueprint_id"
    }
    tampered_blueprint["execution_blueprint_id"] = content_sha256(outer_body)
    blueprint_job_tamper_rejected = _rejects(
        lambda: validate_execution_blueprint_content(tampered_blueprint)
    )

    result_job = jobs[6]
    result_release = _release(blueprint)
    result_compute = _compute(blueprint, result_job, result_release)
    result_ledger = ClusterAwareSAETrainingLedger(
        execution_blueprint_id=blueprint["execution_blueprint_id"]
    )
    result_reservation = reserve_cluster_aware_sae_job_before_runtime_attach(
        blueprint=blueprint,
        execution_job_id=result_job["execution_job_id"],
        release_verification=result_release,
        compute_authorization_verification=result_compute,
        ledger=result_ledger,
        simulation_only=True,
    )
    bad_result_session = result_reservation.attach_runtime(
        _inert_runtime(
            lambda _job: {
                "status": "synthetic_completed_no_training",
                "execution_job_id": "f" * 64,
                "model_loaded": False,
                "weights_mutated": False,
                "persistent_output_written": False,
                "dictionary_id": None,
            },
            {},
        )
    )
    callback_result_tamper_rejected = _rejects(bad_result_session.run)

    receipt_body = {
        key: value for key, value in receipt.items() if key != "runtime_worker_receipt_id"
    }
    controls = {
        "blueprint_content_verifies_without_model_framework": blueprint_check[
            "status"
        ]
        == "pass",
        "all_34_execution_jobs_are_runtime_validated": blueprint_check[
            "execution_jobs"
        ]
        == 34,
        "job_reserved_before_runtime_loader": after_reservation["jobs_reserved"]
        == 1
        and state["loader_calls"] == 1
        and reservation.before_reservation["jobs_reserved"] == 0,
        "planned_steps_and_rows_charged_at_reservation": after_reservation[
            "planned_steps"
        ]
        == int(job["profile"]["steps"])
        and after_reservation["planned_activation_rows"]
        == int(job["exact_sample_contract"]["sae_activation_rows"]),
        "reservation_object_contains_no_runtime": not hasattr(
            reservation, "runtime"
        ),
        "runtime_attachment_is_single_use": second_attach_rejected,
        "trainer_callback_is_single_use": second_run_rejected,
        "callback_observes_preincremented_started_state": state[
            "callback_snapshots"
        ][0]["trainer_calls"]
        == 1
        and state["callback_snapshots"][0]["reservations"][
            job["execution_job_id"]
        ]
        == "started",
        "callback_receives_exact_job_contract": state["observed_job"] == job,
        "successful_job_finishes_completed": ledger.reservations[
            job["execution_job_id"]
        ]
        == "completed",
        "failed_callback_remains_charged": callback_failure_rejected
        and failure_state["callback_calls"] == 1
        and failure_ledger.reservations[failure_job["execution_job_id"]] == "failed"
        and failure_ledger.planned_steps == int(failure_job["profile"]["steps"]),
        "reservation_can_stop_without_runtime_attach": no_attach_reservation.runtime_attached
        is False
        and no_attach_ledger.reservations[no_attach_job["execution_job_id"]]
        == "reserved",
        "replay_is_rejected": replay_rejected,
        "wrong_job_authority_rejected_before_reservation": wrong_authority_rejected
        and wrong_ledger.reservations == {},
        "synthetic_blueprint_rejected_for_production": production_from_synthetic_rejected,
        "active_runtime_rejected_in_simulation": active_runtime_rejected,
        "blueprint_inner_job_tamper_rejected": blueprint_job_tamper_rejected,
        "callback_result_identity_tamper_rejected_and_charged": callback_result_tamper_rejected
        and result_ledger.reservations[result_job["execution_job_id"]] == "failed",
        "runtime_close_delegates_exactly_once": state["close_calls"] == 1,
        "receipt_is_content_addressed": receipt[
            "runtime_worker_receipt_id"
        ]
        == content_sha256(receipt_body),
        "simulation_receipt_remains_non_authorizing": receipt[
            "scientific_boundary"
        ]
        == {
            "simulation_only": True,
            "model_loaded": False,
            "training_performed": False,
            "persistent_output_written": False,
            "base_model_weights_mutated": False,
            "real_actions": 0,
            "receipt_authorizes_additional_execution": False,
        },
        "worker_imports_no_model_frameworks": before_frameworks
        == (MODEL_FRAMEWORKS & set(sys.modules)),
    }
    if not all(controls.values()):
        raise ValueError(
            "cluster-aware SAE runtime worker benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-runtime-worker-benchmark.v1",
        "status": "pass",
        "source": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "execution_blueprint_builder_sha256": sha256_file(
                Path(__file__).with_name(
                    "build_cluster_aware_sae_execution_blueprint_v1.py"
                )
            ),
            "runtime_worker_sha256": sha256_file(
                Path(__file__).with_name("run_cluster_aware_sae_training_job_v1.py")
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "worker_core_implemented": True,
            "runtime_is_injected_after_reservation": True,
            "simulation_only_benchmark": True,
            "production_cli_implemented": False,
            "real_release_verifier_implemented": False,
            "real_compute_authorization_verifier_implemented": False,
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "execution_jobs": blueprint_check["execution_jobs"],
            "synthetic_runtime_worker_receipt_id": receipt[
                "runtime_worker_receipt_id"
            ],
            "successful_job_id": job["execution_job_id"],
            "failed_job_id": failure_job["execution_job_id"],
        },
        "scientific_boundary": {
            "synthetic_callbacks_only": True,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "persistent_output_created": False,
            "weights_mutated": False,
            "empirical_blueprint_present": False,
            "scientific_release_receipt_present": False,
            "explicit_compute_authorization_present": False,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster-aware SAE runtime worker benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("cluster-aware SAE runtime worker benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster-aware SAE runtime worker benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-aware-sae-runtime-worker-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": int(observed["summary"]["controls"]),
        "execution_jobs": int(observed["results"]["execution_jobs"]),
        "worker_core_implemented": True,
        "production_cli_implemented": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the cluster-aware SAE runtime worker V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
