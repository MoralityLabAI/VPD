from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1 import (
    COMPUTE_ACTION,
    RELEASE_ACTION,
    sign_receipt,
    verify_scientific_release_receipt,
)
from experiments.authz_mechinterp.attach_cluster_aware_sae_inert_runtime_v1 import (
    CLAIM_DIR_NAME,
    claim_inert_runtime_attachment,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_admission_v1 import (
    AS_OF,
    _compute_body,
    _registry,
    _release_body,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1 import (
    _synthetic_materialization,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_transition_v1 import (
    _reserve,
    _write_inputs,
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
from experiments.authz_mechinterp.execute_cluster_aware_sae_inert_callback_v1 import (
    EXECUTION_CLAIM_DIR_NAME,
    FAILURE_RECEIPT_NAME,
    SUCCESS_RECEIPT_NAME,
    SyntheticCallbackExecutionError,
    execute_claimed_inert_callback,
    verify_synthetic_callback_execution,
    verify_synthetic_callback_failure,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import crypto_types
from experiments.authz_mechinterp.reserve_cluster_aware_sae_training_transition_v1 import (
    JOB_RECEIPT_NAME,
    RESERVATION_PARENT,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    ClusterAwareSAETrainingRuntime,
)


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError):
        return True
    return False


def _valid_result(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "synthetic_completed_no_training",
        "execution_job_id": job["execution_job_id"],
        "model_loaded": False,
        "weights_mutated": False,
        "persistent_output_written": False,
        "dictionary_id": None,
    }


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    materialization = _synthetic_materialization(project_root)
    blueprint = build_execution_blueprint(materialization=materialization)
    Ed25519PrivateKey, _, _, _ = crypto_types()
    release_key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
    compute_key = Ed25519PrivateKey.from_private_bytes(bytes(range(33, 65)))
    release_registry = _registry(
        release_key,
        authority_id="synthetic-release-authority",
        action=RELEASE_ACTION,
    )
    compute_registry = _registry(
        compute_key,
        authority_id="synthetic-compute-authority",
        action=COMPUTE_ACTION,
    )
    with tempfile.TemporaryDirectory(prefix="vpd-sae-inert-callback-v1-") as temporary:
        root = Path(temporary)
        (root / RESERVATION_PARENT).mkdir(parents=True)
        blueprint_hash_source = root / "blueprint-hash-source.json"
        write_json(blueprint_hash_source, blueprint)
        blueprint_sha256 = sha256_file(blueprint_hash_source)
        release_body = _release_body(
            blueprint=blueprint, blueprint_sha256=blueprint_sha256
        )
        release_receipt = sign_receipt(release_body, release_key)
        release_verification = verify_scientific_release_receipt(
            receipt=release_receipt,
            registry=release_registry,
            blueprint=blueprint,
            blueprint_sha256=blueprint_sha256,
            as_of=AS_OF,
        )

        def reserve_job(index: int, nonce_digit: str) -> tuple[dict[str, Any], Path]:
            job = blueprint["execution_jobs"][index]
            compute_body = _compute_body(
                blueprint=blueprint,
                job=job,
                release_verification=release_verification,
            )
            compute_body["nonce"] = nonce_digit * 64
            compute_receipt = sign_receipt(compute_body, compute_key)
            paths = _write_inputs(
                root=root,
                blueprint=blueprint,
                release_receipt=release_receipt,
                release_registry=release_registry,
                compute_receipt=compute_receipt,
                compute_registry=compute_registry,
                suffix=f"-callback-{index}",
            )
            transition = _reserve(
                root=root, paths=paths, job_id=job["execution_job_id"]
            )
            transition_path = (
                Path(transition["job"]["reserved_job_root"]) / JOB_RECEIPT_NAME
            )
            return transition, transition_path

        success_transition, success_transition_path = reserve_job(0, "7")
        success_claim_root = success_transition_path.parent / CLAIM_DIR_NAME
        success_execution_root = success_claim_root / EXECUTION_CLAIM_DIR_NAME
        success_state = {
            "callback_calls": 0,
            "close_calls": 0,
            "execution_claim_exists_at_callback": False,
        }

        def success_callback(job: dict[str, Any]) -> dict[str, Any]:
            success_state["callback_calls"] += 1
            success_state["execution_claim_exists_at_callback"] = (
                success_execution_root.is_dir()
            )
            return _valid_result(job)

        def success_factory() -> ClusterAwareSAETrainingRuntime:
            def close() -> None:
                success_state["close_calls"] += 1

            return ClusterAwareSAETrainingRuntime(
                train_job=success_callback,
                close=close,
                runtime_kind="synthetic_inert_callback_no_model_no_training",
                actual_model_loaded=False,
                actual_training_enabled=False,
                persistent_output_enabled=False,
            )

        successful_claim = claim_inert_runtime_attachment(
            transition_receipt_path=success_transition_path,
            project_root=root,
            runtime_factory=success_factory,
        )
        success_receipt = execute_claimed_inert_callback(
            claimed=successful_claim, project_root=root
        )
        success_receipt_path = success_execution_root / SUCCESS_RECEIPT_NAME
        success_verification = verify_synthetic_callback_execution(
            success_receipt_path, project_root=root
        )
        success_replay_rejected = _rejects(
            lambda: execute_claimed_inert_callback(
                claimed=successful_claim, project_root=root
            )
        )

        failed_transition, failed_transition_path = reserve_job(1, "8")
        failed_claim_root = failed_transition_path.parent / CLAIM_DIR_NAME
        failed_execution_root = failed_claim_root / EXECUTION_CLAIM_DIR_NAME
        failure_state = {
            "callback_calls": 0,
            "close_calls": 0,
            "execution_claim_exists_at_callback": False,
        }

        def failed_callback(_job: dict[str, Any]) -> dict[str, Any]:
            failure_state["callback_calls"] += 1
            failure_state["execution_claim_exists_at_callback"] = (
                failed_execution_root.is_dir()
            )
            raise RuntimeError("synthetic callback benchmark failure")

        def failed_factory() -> ClusterAwareSAETrainingRuntime:
            def close() -> None:
                failure_state["close_calls"] += 1

            return ClusterAwareSAETrainingRuntime(
                train_job=failed_callback,
                close=close,
                runtime_kind="synthetic_inert_callback_failure",
                actual_model_loaded=False,
                actual_training_enabled=False,
                persistent_output_enabled=False,
            )

        failed_claim = claim_inert_runtime_attachment(
            transition_receipt_path=failed_transition_path,
            project_root=root,
            runtime_factory=failed_factory,
        )
        failure_raised_with_receipt = False
        try:
            execute_claimed_inert_callback(claimed=failed_claim, project_root=root)
        except SyntheticCallbackExecutionError as exc:
            failure_raised_with_receipt = (
                exc.receipt_path == failed_execution_root / FAILURE_RECEIPT_NAME
                and exc.cause_type == "RuntimeError"
            )
        failure_receipt_path = failed_execution_root / FAILURE_RECEIPT_NAME
        failure_verification = verify_synthetic_callback_failure(
            failure_receipt_path, project_root=root
        )
        failure_replay_rejected = _rejects(
            lambda: execute_claimed_inert_callback(
                claimed=failed_claim, project_root=root
            )
        )

        invalid_transition, invalid_transition_path = reserve_job(2, "9")
        invalid_claim_root = invalid_transition_path.parent / CLAIM_DIR_NAME
        invalid_execution_root = invalid_claim_root / EXECUTION_CLAIM_DIR_NAME
        invalid_state = {"callback_calls": 0, "close_calls": 0}

        def invalid_callback(job: dict[str, Any]) -> dict[str, Any]:
            invalid_state["callback_calls"] += 1
            result = _valid_result(job)
            result["weights_mutated"] = True
            return result

        def invalid_factory() -> ClusterAwareSAETrainingRuntime:
            def close() -> None:
                invalid_state["close_calls"] += 1

            return ClusterAwareSAETrainingRuntime(
                train_job=invalid_callback,
                close=close,
                runtime_kind="synthetic_inert_invalid_result",
                actual_model_loaded=False,
                actual_training_enabled=False,
                persistent_output_enabled=False,
            )

        invalid_claim = claim_inert_runtime_attachment(
            transition_receipt_path=invalid_transition_path,
            project_root=root,
            runtime_factory=invalid_factory,
        )
        invalid_result_rejected = _rejects(
            lambda: execute_claimed_inert_callback(
                claimed=invalid_claim, project_root=root
            )
        )
        invalid_failure_verification = verify_synthetic_callback_failure(
            invalid_execution_root / FAILURE_RECEIPT_NAME, project_root=root
        )

        original_success_text = success_receipt_path.read_text(encoding="utf-8")
        changed_success = json.loads(original_success_text)
        changed_success["scientific_boundary"]["training_performed"] = True
        write_json(success_receipt_path, changed_success)
        success_tamper_rejected = _rejects(
            lambda: verify_synthetic_callback_execution(
                success_receipt_path, project_root=root
            )
        )
        success_receipt_path.write_text(original_success_text, encoding="utf-8")

        original_failure_text = failure_receipt_path.read_text(encoding="utf-8")
        changed_failure = json.loads(original_failure_text)
        changed_failure["ledger_after"]["trainer_calls"] = 0
        write_json(failure_receipt_path, changed_failure)
        failure_tamper_rejected = _rejects(
            lambda: verify_synthetic_callback_failure(
                failure_receipt_path, project_root=root
            )
        )
        failure_receipt_path.write_text(original_failure_text, encoding="utf-8")

        original_claim_text = successful_claim.claim_receipt_path.read_text(
            encoding="utf-8"
        )
        changed_claim = json.loads(original_claim_text)
        changed_claim["runtime"]["actual_model_loaded"] = True
        write_json(successful_claim.claim_receipt_path, changed_claim)
        attachment_tamper_rejected = _rejects(
            lambda: verify_synthetic_callback_execution(
                success_receipt_path, project_root=root
            )
        )
        successful_claim.claim_receipt_path.write_text(
            original_claim_text, encoding="utf-8"
        )

        controls = {
            "execution_claim_precedes_success_callback": success_state[
                "execution_claim_exists_at_callback"
            ]
            is True,
            "success_callback_invoked_exactly_once": success_state["callback_calls"]
            == 1,
            "success_runtime_closed_exactly_once": success_state["close_calls"] == 1,
            "success_receipt_is_durable_and_verifiable": success_receipt_path.is_file()
            and success_verification["status"] == "pass",
            "success_worker_receipt_is_content_addressed": success_verification[
                "runtime_worker_receipt_id"
            ]
            == success_receipt["runtime_worker_receipt"]["runtime_worker_receipt_id"],
            "success_charge_is_completed_once": success_verification[
                "callback_charge_persisted"
            ]
            is True
            and success_receipt["ledger_after"]["trainer_calls"] == 1,
            "success_replay_rejected_before_second_callback": success_replay_rejected
            and success_state["callback_calls"] == 1,
            "execution_claim_precedes_failed_callback": failure_state[
                "execution_claim_exists_at_callback"
            ]
            is True,
            "callback_exception_emits_terminal_failure_receipt": failure_raised_with_receipt
            and failure_receipt_path.is_file(),
            "failed_callback_is_charged": failure_verification[
                "callback_charge_persisted"
            ]
            is True
            and failure_verification["callback_failed"] is True,
            "failed_callback_invoked_exactly_once": failure_state["callback_calls"]
            == 1,
            "failed_runtime_closed_exactly_once": failure_state["close_calls"] == 1,
            "failure_replay_rejected_before_second_callback": failure_replay_rejected
            and failure_state["callback_calls"] == 1,
            "boundary_weakening_result_becomes_charged_failure": invalid_result_rejected
            and invalid_failure_verification["callback_failed"] is True
            and invalid_state["callback_calls"] == 1
            and invalid_state["close_calls"] == 1,
            "success_receipt_tamper_rejected": success_tamper_rejected,
            "failure_receipt_tamper_rejected": failure_tamper_rejected,
            "attachment_claim_tamper_rejected_transitively": attachment_tamper_rejected,
            "callback_execution_imports_no_model_frameworks": before_frameworks
            == (MODEL_FRAMEWORKS & set(sys.modules)),
            "callbacks_create_no_dictionary_output": all(
                not Path(transition["job"]["reserved_exact_output_path"]).exists()
                for transition in (
                    success_transition,
                    failed_transition,
                    invalid_transition,
                )
            ),
            "callback_receipts_remain_non_authorizing": all(
                success_receipt["scientific_boundary"][field_name] in (False, 0)
                for field_name in (
                    "model_loaded",
                    "model_inference_performed",
                    "training_performed",
                    "persistent_dictionary_output_created",
                    "weights_loaded_or_mutated",
                    "production_runtime_attached",
                    "execution_authorized_by_receipt",
                    "sae_training_authorized_by_receipt",
                    "real_actions_authorized",
                )
            ),
            "all_execution_artifacts_are_temporary": all(
                path.is_relative_to(root)
                for path in (
                    success_receipt_path,
                    failure_receipt_path,
                    invalid_execution_root / FAILURE_RECEIPT_NAME,
                )
            ),
        }
    if not all(controls.values()):
        raise ValueError(
            "cluster-aware SAE inert callback benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-inert-callback-benchmark.v1",
        "status": "pass",
        "source": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "inert_attachment_sha256": sha256_file(
                Path(__file__).with_name(
                    "attach_cluster_aware_sae_inert_runtime_v1.py"
                )
            ),
            "inert_callback_execution_sha256": sha256_file(
                Path(__file__).with_name(
                    "execute_cluster_aware_sae_inert_callback_v1.py"
                )
            ),
            "runtime_worker_sha256": sha256_file(
                Path(__file__).with_name("run_cluster_aware_sae_training_job_v1.py")
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "durable_execution_claim_before_callback": True,
            "synthetic_success_receipt": True,
            "terminal_charged_failure_receipt": True,
            "callback_single_use": True,
            "production_callback_invocation_implemented": False,
            "trainer_or_model_implementation_present": False,
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "synthetic_success_callback_invocations": 1,
            "synthetic_failed_callback_invocations": 2,
            "charged_callback_invocations": 3,
            "persistent_callback_receipts_created": False,
        },
        "scientific_boundary": {
            "temporary_simulation_only": True,
            "synthetic_keys_receipts_runtime_and_callbacks_only": True,
            "empirical_blueprint_present": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "persistent_dictionary_output_created": False,
            "weights_loaded_or_mutated": False,
            "production_runtime_attached": False,
            "production_callback_invoked": False,
            "synthetic_callback_invoked": True,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster-aware SAE inert callback benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("cluster-aware SAE inert callback benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster-aware SAE inert callback benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-aware-sae-inert-callback-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": int(observed["summary"]["controls"]),
        "durable_execution_claim_implemented": True,
        "synthetic_callback_invoked": True,
        "failure_charge_persisted": True,
        "production_callback_invoked": False,
        "model_loaded": False,
        "training_performed": False,
        "persistent_output_written": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cluster-aware SAE inert callback execution V1."
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
