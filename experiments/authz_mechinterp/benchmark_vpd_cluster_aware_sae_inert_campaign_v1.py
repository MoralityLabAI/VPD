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
    ClaimedInertRuntimeSession,
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
    SUCCESS_RECEIPT_NAME,
)
from experiments.authz_mechinterp.execute_cluster_aware_sae_inert_campaign_v1 import (
    FAILURE_RECEIPT_NAME_CAMPAIGN,
    SUCCESS_RECEIPT_NAME_CAMPAIGN,
    SyntheticCampaignExecutionError,
    execute_synthetic_inert_campaign,
    verify_synthetic_inert_campaign,
    verify_synthetic_inert_campaign_failure,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import crypto_types
from experiments.authz_mechinterp.reserve_cluster_aware_sae_training_transition_v1 import (
    JOB_RECEIPT_NAME,
    RESERVATION_PARENT,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    ClusterAwareSAETrainingRuntime,
)


FAIL_INDEX = 7


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
    with tempfile.TemporaryDirectory(prefix="vpd-sae-inert-campaign-v1-") as temporary:
        root = Path(temporary)
        (root / RESERVATION_PARENT).mkdir(parents=True)
        blueprint_path = root / "blueprint.json"
        write_json(blueprint_path, blueprint)
        blueprint_sha256 = sha256_file(blueprint_path)
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

        def build_claims(
            *,
            nonce_offset: int,
            suffix: str,
            campaign_root: Path,
            fail_index: int | None,
        ) -> tuple[
            list[ClaimedInertRuntimeSession],
            list[dict[str, Any]],
            dict[str, dict[str, Any]],
        ]:
            claims: list[ClaimedInertRuntimeSession] = []
            transitions: list[dict[str, Any]] = []
            states: dict[str, dict[str, Any]] = {}
            for index, job in enumerate(blueprint["execution_jobs"]):
                compute_body = _compute_body(
                    blueprint=blueprint,
                    job=job,
                    release_verification=release_verification,
                )
                compute_body["nonce"] = f"{nonce_offset + index:064x}"
                compute_receipt = sign_receipt(compute_body, compute_key)
                paths = _write_inputs(
                    root=root,
                    blueprint=blueprint,
                    release_receipt=release_receipt,
                    release_registry=release_registry,
                    compute_receipt=compute_receipt,
                    compute_registry=compute_registry,
                    suffix=f"-{suffix}-{index:02d}",
                )
                transition = _reserve(
                    root=root, paths=paths, job_id=job["execution_job_id"]
                )
                transition_path = (
                    Path(transition["job"]["reserved_job_root"]) / JOB_RECEIPT_NAME
                )
                execution_claim_root = (
                    transition_path.parent
                    / CLAIM_DIR_NAME
                    / EXECUTION_CLAIM_DIR_NAME
                )
                state = {
                    "callback_calls": 0,
                    "close_calls": 0,
                    "campaign_claim_exists_at_callback": False,
                    "execution_claim_exists_at_callback": False,
                }
                states[str(job["execution_job_id"])] = state

                def callback(
                    callback_job: dict[str, Any],
                    *,
                    state: dict[str, Any] = state,
                    index: int = index,
                    execution_claim_root: Path = execution_claim_root,
                ) -> dict[str, Any]:
                    state["callback_calls"] += 1
                    state["campaign_claim_exists_at_callback"] = campaign_root.is_dir()
                    state["execution_claim_exists_at_callback"] = (
                        execution_claim_root.is_dir()
                    )
                    if fail_index is not None and index == fail_index:
                        raise RuntimeError("synthetic campaign benchmark callback failure")
                    return _valid_result(callback_job)

                def factory(
                    *, state: dict[str, Any] = state, index: int = index
                ) -> ClusterAwareSAETrainingRuntime:
                    def close() -> None:
                        state["close_calls"] += 1

                    return ClusterAwareSAETrainingRuntime(
                        train_job=callback,
                        close=close,
                        runtime_kind=(
                            "synthetic_inert_campaign_failure"
                            if fail_index is not None and index == fail_index
                            else "synthetic_inert_campaign_success"
                        ),
                        actual_model_loaded=False,
                        actual_training_enabled=False,
                        persistent_output_enabled=False,
                    )

                claims.append(
                    claim_inert_runtime_attachment(
                        transition_receipt_path=transition_path,
                        project_root=root,
                        runtime_factory=factory,
                    )
                )
                transitions.append(transition)
            return claims, transitions, states

        success_root = root / "campaign-success"
        success_claims, success_transitions, success_states = build_claims(
            nonce_offset=1,
            suffix="success",
            campaign_root=success_root,
            fail_index=None,
        )

        missing_root = root / "campaign-missing"
        missing_job_rejected_before_claim = _rejects(
            lambda: execute_synthetic_inert_campaign(
                blueprint_path=blueprint_path,
                claimed_sessions=success_claims[:-1],
                campaign_root=missing_root,
                project_root=root,
            )
        ) and not missing_root.exists()
        duplicate_root = root / "campaign-duplicate"
        duplicate_job_rejected_before_claim = _rejects(
            lambda: execute_synthetic_inert_campaign(
                blueprint_path=blueprint_path,
                claimed_sessions=success_claims[:-1] + [success_claims[0]],
                campaign_root=duplicate_root,
                project_root=root,
            )
        ) and not duplicate_root.exists()

        tampered_claim_path = success_claims[0].claim_receipt_path
        original_claim_text = tampered_claim_path.read_text(encoding="utf-8")
        changed_claim = json.loads(original_claim_text)
        changed_claim["scientific_boundary"]["model_loaded"] = True
        write_json(tampered_claim_path, changed_claim)
        tampered_claim_root = root / "campaign-tampered-claim"
        claim_tamper_rejected_before_campaign_claim = _rejects(
            lambda: execute_synthetic_inert_campaign(
                blueprint_path=blueprint_path,
                claimed_sessions=success_claims,
                campaign_root=tampered_claim_root,
                project_root=root,
            )
        ) and not tampered_claim_root.exists()
        tampered_claim_path.write_text(original_claim_text, encoding="utf-8")

        success_receipt = execute_synthetic_inert_campaign(
            blueprint_path=blueprint_path,
            claimed_sessions=success_claims,
            campaign_root=success_root,
            project_root=root,
        )
        success_receipt_path = success_root / SUCCESS_RECEIPT_NAME_CAMPAIGN
        success_verification = verify_synthetic_inert_campaign(
            success_receipt_path, project_root=root
        )
        success_replay_rejected = _rejects(
            lambda: execute_synthetic_inert_campaign(
                blueprint_path=blueprint_path,
                claimed_sessions=success_claims,
                campaign_root=success_root,
                project_root=root,
            )
        )

        failure_root = root / "campaign-failure"
        failure_claims, failure_transitions, failure_states = build_claims(
            nonce_offset=1000,
            suffix="failure",
            campaign_root=failure_root,
            fail_index=FAIL_INDEX,
        )
        failure_raised_with_receipt = False
        try:
            execute_synthetic_inert_campaign(
                blueprint_path=blueprint_path,
                claimed_sessions=failure_claims,
                campaign_root=failure_root,
                project_root=root,
            )
        except SyntheticCampaignExecutionError as exc:
            failure_raised_with_receipt = (
                exc.receipt_path == failure_root / FAILURE_RECEIPT_NAME_CAMPAIGN
                and exc.failed_execution_job_id
                == blueprint["execution_jobs"][FAIL_INDEX]["execution_job_id"]
            )
        failure_receipt_path = failure_root / FAILURE_RECEIPT_NAME_CAMPAIGN
        failure_verification = verify_synthetic_inert_campaign_failure(
            failure_receipt_path, project_root=root
        )
        failure_replay_rejected = _rejects(
            lambda: execute_synthetic_inert_campaign(
                blueprint_path=blueprint_path,
                claimed_sessions=failure_claims,
                campaign_root=failure_root,
                project_root=root,
            )
        )

        original_success_text = success_receipt_path.read_text(encoding="utf-8")
        changed_success = json.loads(original_success_text)
        changed_success["summary"]["charged_callbacks"] -= 1
        write_json(success_receipt_path, changed_success)
        success_receipt_tamper_rejected = _rejects(
            lambda: verify_synthetic_inert_campaign(
                success_receipt_path, project_root=root
            )
        )
        success_receipt_path.write_text(original_success_text, encoding="utf-8")

        first_callback_path = (
            root
            / success_receipt["jobs"][0]["success_receipt_relative_path"]
        ).resolve()
        original_callback_text = first_callback_path.read_text(encoding="utf-8")
        changed_callback = json.loads(original_callback_text)
        changed_callback["scientific_boundary"]["training_performed"] = True
        write_json(first_callback_path, changed_callback)
        callback_receipt_tamper_rejected_transitively = _rejects(
            lambda: verify_synthetic_inert_campaign(
                success_receipt_path, project_root=root
            )
        )
        first_callback_path.write_text(original_callback_text, encoding="utf-8")

        original_blueprint_text = blueprint_path.read_text(encoding="utf-8")
        blueprint_path.write_text(original_blueprint_text + " ", encoding="utf-8")
        blueprint_tamper_rejected_transitively = _rejects(
            lambda: verify_synthetic_inert_campaign(
                success_receipt_path, project_root=root
            )
        )
        blueprint_path.write_text(original_blueprint_text, encoding="utf-8")

        original_failure_text = failure_receipt_path.read_text(encoding="utf-8")
        changed_failure = json.loads(original_failure_text)
        changed_failure["summary"]["retry_authorized"] = True
        write_json(failure_receipt_path, changed_failure)
        failure_receipt_tamper_rejected = _rejects(
            lambda: verify_synthetic_inert_campaign_failure(
                failure_receipt_path, project_root=root
            )
        )
        failure_receipt_path.write_text(original_failure_text, encoding="utf-8")

        remaining_claim_path = failure_claims[FAIL_INDEX + 1].claim_receipt_path
        original_remaining_claim = remaining_claim_path.read_text(encoding="utf-8")
        changed_remaining_claim = json.loads(original_remaining_claim)
        changed_remaining_claim["runtime"]["actual_training_enabled"] = True
        write_json(remaining_claim_path, changed_remaining_claim)
        remaining_claim_tamper_rejected_transitively = _rejects(
            lambda: verify_synthetic_inert_campaign_failure(
                failure_receipt_path, project_root=root
            )
        )
        remaining_claim_path.write_text(original_remaining_claim, encoding="utf-8")

        all_success_callbacks_once = all(
            state["callback_calls"] == 1 for state in success_states.values()
        )
        all_success_closed_once = all(
            state["close_calls"] == 1 for state in success_states.values()
        )
        all_success_claims_precede_callbacks = all(
            state["campaign_claim_exists_at_callback"]
            and state["execution_claim_exists_at_callback"]
            for state in success_states.values()
        )
        ordered_failure_states = [
            failure_states[str(job["execution_job_id"])]
            for job in blueprint["execution_jobs"]
        ]
        failure_stops_at_exact_job = all(
            state["callback_calls"] == (1 if index <= FAIL_INDEX else 0)
            for index, state in enumerate(ordered_failure_states)
        )
        all_failure_runtimes_closed_once = all(
            state["close_calls"] == 1 for state in ordered_failure_states
        )
        invoked_failure_claims_precede_callbacks = all(
            state["campaign_claim_exists_at_callback"]
            and state["execution_claim_exists_at_callback"]
            for state in ordered_failure_states[: FAIL_INDEX + 1]
        )
        all_transitions = success_transitions + failure_transitions
        no_dictionary_outputs = all(
            not Path(transition["job"]["reserved_exact_output_path"]).exists()
            for transition in all_transitions
        )
        controls = {
            "exact_34_job_coverage_required_before_campaign_claim": missing_job_rejected_before_claim,
            "duplicate_job_rejected_before_campaign_claim": duplicate_job_rejected_before_claim,
            "claim_tamper_rejected_before_campaign_claim": claim_tamper_rejected_before_campaign_claim,
            "campaign_and_per_job_claims_precede_all_success_callbacks": all_success_claims_precede_callbacks,
            "all_34_success_callbacks_invoked_exactly_once": all_success_callbacks_once,
            "all_34_success_runtimes_closed_exactly_once": all_success_closed_once,
            "success_campaign_receipt_is_durable_and_verifiable": success_receipt_path.is_file()
            and success_verification["status"] == "pass",
            "success_campaign_covers_8_categories": success_verification["categories"]
            == 8,
            "success_campaign_covers_17_compartments": success_verification[
                "compartments"
            ]
            == 17,
            "success_campaign_charges_exactly_34_callbacks": success_verification[
                "charged_callbacks"
            ]
            == 34,
            "success_campaign_preserves_34_unique_output_dirs": success_verification[
                "unique_output_dirs"
            ]
            == 34,
            "success_campaign_replay_rejected": success_replay_rejected
            and all_success_callbacks_once,
            "failure_campaign_emits_terminal_receipt": failure_raised_with_receipt
            and failure_receipt_path.is_file(),
            "failure_campaign_stops_at_exact_job": failure_stops_at_exact_job,
            "failure_campaign_charges_successes_plus_failure": failure_verification[
                "charged_callbacks"
            ]
            == FAIL_INDEX + 1,
            "failure_campaign_leaves_remaining_uninvoked": failure_verification[
                "uninvoked_callbacks"
            ]
            == 34 - FAIL_INDEX - 1,
            "failure_campaign_closes_all_runtimes_once": all_failure_runtimes_closed_once,
            "campaign_and_per_job_claims_precede_invoked_failure_path": invoked_failure_claims_precede_callbacks,
            "failure_campaign_replay_rejected": failure_replay_rejected
            and failure_stops_at_exact_job,
            "success_campaign_receipt_tamper_rejected": success_receipt_tamper_rejected,
            "callback_receipt_tamper_rejected_transitively": callback_receipt_tamper_rejected_transitively,
            "blueprint_tamper_rejected_transitively": blueprint_tamper_rejected_transitively,
            "failure_campaign_receipt_tamper_rejected": failure_receipt_tamper_rejected,
            "remaining_claim_tamper_rejected_transitively": remaining_claim_tamper_rejected_transitively,
            "campaign_imports_no_model_frameworks": before_frameworks
            == (MODEL_FRAMEWORKS & set(sys.modules)),
            "campaigns_create_no_dictionary_output": no_dictionary_outputs,
            "campaign_receipts_remain_non_authorizing": all(
                success_receipt["scientific_boundary"][field_name] in (False, 0)
                for field_name in (
                    "model_loaded",
                    "model_inference_performed",
                    "training_performed",
                    "persistent_dictionary_output_created",
                    "weights_loaded_or_mutated",
                    "production_runtime_attached",
                    "production_trainer_callback_invoked",
                    "execution_authorized_by_receipt",
                    "sae_training_authorized_by_receipt",
                    "real_actions_authorized",
                )
            ),
            "all_campaign_artifacts_are_temporary": all(
                path.is_relative_to(root)
                for path in (
                    success_receipt_path,
                    failure_receipt_path,
                    first_callback_path,
                )
            ),
        }
        failure_campaign_id = json.loads(
            failure_receipt_path.read_text(encoding="utf-8")
        )["synthetic_campaign_failure_id"]
    if not all(controls.values()):
        raise ValueError(
            "cluster-aware SAE inert campaign benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-inert-campaign-benchmark.v1",
        "status": "pass",
        "source": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "inert_callback_execution_sha256": sha256_file(
                Path(__file__).with_name(
                    "execute_cluster_aware_sae_inert_callback_v1.py"
                )
            ),
            "inert_campaign_execution_sha256": sha256_file(
                Path(__file__).with_name(
                    "execute_cluster_aware_sae_inert_campaign_v1.py"
                )
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "exact_activation_categories": 8,
            "exact_compartments": 17,
            "exact_specific_compartments": 16,
            "exact_shared_fallback_compartments": 1,
            "exact_execution_jobs": 34,
            "unique_output_directories": 34,
            "one_global_pair_split": True,
            "durable_campaign_claim_before_callbacks": True,
            "durable_per_job_claim_before_each_callback": True,
            "production_campaign_execution_implemented": False,
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "synthetic_success_campaign_id": success_receipt[
                "synthetic_campaign_id"
            ],
            "synthetic_failure_campaign_id": failure_campaign_id,
            "success_callbacks_charged": 34,
            "failure_campaign_callbacks_charged": FAIL_INDEX + 1,
            "persistent_campaign_receipts_created": False,
        },
        "scientific_boundary": {
            "temporary_simulation_only": True,
            "synthetic_keys_receipts_runtimes_and_callbacks_only": True,
            "empirical_blueprint_present": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "persistent_dictionary_output_created": False,
            "weights_loaded_or_mutated": False,
            "production_runtime_attached": False,
            "production_trainer_callback_invoked": False,
            "synthetic_campaign_callbacks_invoked": True,
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
        raise ValueError("cluster-aware SAE inert campaign benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("cluster-aware SAE inert campaign benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster-aware SAE inert campaign benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-aware-sae-inert-campaign-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": int(observed["summary"]["controls"]),
        "categories": int(observed["design"]["exact_activation_categories"]),
        "compartments": int(observed["design"]["exact_compartments"]),
        "jobs": int(observed["design"]["exact_execution_jobs"]),
        "unique_output_dirs": int(observed["design"]["unique_output_directories"]),
        "success_callbacks_charged": int(
            observed["results"]["success_callbacks_charged"]
        ),
        "failure_campaign_callbacks_charged": int(
            observed["results"]["failure_campaign_callbacks_charged"]
        ),
        "production_campaign_executed": False,
        "model_loaded": False,
        "training_performed": False,
        "persistent_output_written": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact cluster-aware SAE inert 34-job campaigns V1."
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
