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
    CLAIM_RECEIPT_NAME,
    claim_inert_runtime_attachment,
    verify_inert_runtime_attachment_claim,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_admission_v1 import (
    AS_OF,
    DEVICE,
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
    with tempfile.TemporaryDirectory(prefix="vpd-sae-inert-attachment-v1-") as temporary:
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

        def reserve_job(index: int, nonce_digit: str) -> tuple[dict[str, Any], dict[str, Path]]:
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
                suffix=f"-attach-{index}",
            )
            transition = _reserve(
                root=root, paths=paths, job_id=job["execution_job_id"]
            )
            return transition, paths

        transition, _ = reserve_job(0, "2")
        transition_path = (
            Path(transition["job"]["reserved_job_root"]) / JOB_RECEIPT_NAME
        )
        claim_root = transition_path.parent / CLAIM_DIR_NAME
        state: dict[str, Any] = {
            "factory_calls": 0,
            "close_calls": 0,
            "callback_calls": 0,
            "claim_exists_at_factory": False,
        }

        def callback(_job: dict[str, Any]) -> dict[str, Any]:
            state["callback_calls"] += 1
            raise RuntimeError("inert attachment benchmark must never invoke callback")

        def inert_factory() -> ClusterAwareSAETrainingRuntime:
            state["factory_calls"] += 1
            state["claim_exists_at_factory"] = claim_root.is_dir()

            def close() -> None:
                state["close_calls"] += 1

            return ClusterAwareSAETrainingRuntime(
                train_job=callback,
                close=close,
                runtime_kind="synthetic_inert_no_model_no_training",
                actual_model_loaded=False,
                actual_training_enabled=False,
                persistent_output_enabled=False,
            )

        claimed = claim_inert_runtime_attachment(
            transition_receipt_path=transition_path,
            project_root=root,
            runtime_factory=inert_factory,
        )
        claim_verification = verify_inert_runtime_attachment_claim(
            claimed.claim_receipt_path, project_root=root
        )
        callback_still_absent_before_close = claimed.callback_invoked is False
        claimed.close()
        claimed.close()
        second_claim_rejected = _rejects(
            lambda: claim_inert_runtime_attachment(
                transition_receipt_path=transition_path,
                project_root=root,
                runtime_factory=inert_factory,
            )
        )
        factory_not_recalled_on_replay = state["factory_calls"] == 1

        failed_transition, _ = reserve_job(1, "3")
        failed_transition_path = (
            Path(failed_transition["job"]["reserved_job_root"]) / JOB_RECEIPT_NAME
        )
        failed_claim_root = failed_transition_path.parent / CLAIM_DIR_NAME
        failure_state = {"factory_calls": 0}

        def failed_factory() -> ClusterAwareSAETrainingRuntime:
            failure_state["factory_calls"] += 1
            if not failed_claim_root.is_dir():
                raise ValueError("claim did not precede factory")
            raise RuntimeError("synthetic inert runtime factory failure")

        factory_failure_rejected = _rejects(
            lambda: claim_inert_runtime_attachment(
                transition_receipt_path=failed_transition_path,
                project_root=root,
                runtime_factory=failed_factory,
            )
        )
        failed_claim_is_terminal = failed_claim_root.is_dir() and not (
            failed_claim_root / CLAIM_RECEIPT_NAME
        ).exists()
        retry_after_factory_failure_rejected = _rejects(
            lambda: claim_inert_runtime_attachment(
                transition_receipt_path=failed_transition_path,
                project_root=root,
                runtime_factory=failed_factory,
            )
        )
        failure_factory_not_recalled = failure_state["factory_calls"] == 1

        active_transition, _ = reserve_job(2, "4")
        active_transition_path = (
            Path(active_transition["job"]["reserved_job_root"]) / JOB_RECEIPT_NAME
        )
        active_claim_root = active_transition_path.parent / CLAIM_DIR_NAME
        active_state = {"factory_calls": 0, "close_calls": 0}

        def active_factory() -> ClusterAwareSAETrainingRuntime:
            active_state["factory_calls"] += 1

            def close() -> None:
                active_state["close_calls"] += 1

            return ClusterAwareSAETrainingRuntime(
                train_job=callback,
                close=close,
                runtime_kind="forbidden_active_runtime",
                actual_model_loaded=True,
                actual_training_enabled=True,
                persistent_output_enabled=True,
            )

        active_runtime_rejected = _rejects(
            lambda: claim_inert_runtime_attachment(
                transition_receipt_path=active_transition_path,
                project_root=root,
                runtime_factory=active_factory,
            )
        )
        active_claim_is_terminal = active_claim_root.is_dir() and not (
            active_claim_root / CLAIM_RECEIPT_NAME
        ).exists()

        source_transition, source_paths = reserve_job(3, "5")
        source_transition_path = (
            Path(source_transition["job"]["reserved_job_root"]) / JOB_RECEIPT_NAME
        )
        source_claim_root = source_transition_path.parent / CLAIM_DIR_NAME
        original_blueprint = source_paths["blueprint"].read_text(encoding="utf-8")
        source_paths["blueprint"].write_text(
            original_blueprint + " ", encoding="utf-8"
        )
        source_tamper_rejected_before_claim = _rejects(
            lambda: claim_inert_runtime_attachment(
                transition_receipt_path=source_transition_path,
                project_root=root,
                runtime_factory=inert_factory,
            )
        )
        source_paths["blueprint"].write_text(original_blueprint, encoding="utf-8")
        source_tamper_creates_no_claim = not source_claim_root.exists()

        ledger_transition, _ = reserve_job(4, "6")
        ledger_transition_path = (
            Path(ledger_transition["job"]["reserved_job_root"]) / JOB_RECEIPT_NAME
        )
        ledger_claim_root = ledger_transition_path.parent / CLAIM_DIR_NAME
        original_ledger_transition = ledger_transition_path.read_text(encoding="utf-8")
        changed_transition = json.loads(original_ledger_transition)
        changed_transition["worker_ledger_after_reservation"]["planned_steps"] += 1
        changed_body = {
            key: value
            for key, value in changed_transition.items()
            if key != "transition_reservation_id"
        }
        changed_transition["transition_reservation_id"] = content_sha256(changed_body)
        write_json(ledger_transition_path, changed_transition)
        ledger_tamper_rejected_before_claim = _rejects(
            lambda: claim_inert_runtime_attachment(
                transition_receipt_path=ledger_transition_path,
                project_root=root,
                runtime_factory=inert_factory,
            )
        )
        ledger_transition_path.write_text(
            original_ledger_transition, encoding="utf-8"
        )
        ledger_tamper_creates_no_claim = not ledger_claim_root.exists()

        original_claim_text = claimed.claim_receipt_path.read_text(encoding="utf-8")
        changed_claim = json.loads(original_claim_text)
        changed_claim["scientific_boundary"]["trainer_callback_invoked"] = True
        write_json(claimed.claim_receipt_path, changed_claim)
        claim_tamper_rejected = _rejects(
            lambda: verify_inert_runtime_attachment_claim(
                claimed.claim_receipt_path, project_root=root
            )
        )
        claimed.claim_receipt_path.write_text(original_claim_text, encoding="utf-8")

        original_transition_text = transition_path.read_text(encoding="utf-8")
        transition_path.write_text(original_transition_text + " ", encoding="utf-8")
        transition_source_tamper_rejected = _rejects(
            lambda: verify_inert_runtime_attachment_claim(
                claimed.claim_receipt_path, project_root=root
            )
        )
        transition_path.write_text(original_transition_text, encoding="utf-8")

        claim_body = {
            key: value
            for key, value in claimed.claim.items()
            if key != "attachment_claim_id"
        }
        controls = {
            "durable_claim_precedes_runtime_factory": state[
                "claim_exists_at_factory"
            ]
            is True,
            "transition_and_signed_admission_reconstruct_exactly": claim_verification[
                "ledger_reconstructed_exactly"
            ]
            is True,
            "inert_runtime_attaches_after_claim": claim_verification[
                "inert_runtime_attached"
            ]
            is True,
            "production_runtime_remains_absent": claim_verification[
                "production_runtime_attached"
            ]
            is False,
            "trainer_callback_never_invoked": state["callback_calls"] == 0
            and callback_still_absent_before_close
            and claim_verification["trainer_callback_invoked"] is False,
            "runtime_close_delegates_exactly_once": state["close_calls"] == 1,
            "attachment_claim_is_single_use_across_reconstruction": second_claim_rejected
            and factory_not_recalled_on_replay,
            "factory_failure_leaves_terminal_claim_tombstone": factory_failure_rejected
            and failed_claim_is_terminal,
            "factory_failure_retry_rejected_before_factory": retry_after_factory_failure_rejected
            and failure_factory_not_recalled,
            "active_runtime_rejected_and_closed": active_runtime_rejected
            and active_state["factory_calls"] == 1
            and active_state["close_calls"] == 1,
            "active_runtime_rejection_leaves_terminal_claim": active_claim_is_terminal,
            "source_tamper_rejected_before_claim": source_tamper_rejected_before_claim
            and source_tamper_creates_no_claim,
            "ledger_tamper_rejected_before_claim": ledger_tamper_rejected_before_claim
            and ledger_tamper_creates_no_claim,
            "claim_receipt_tamper_rejected": claim_tamper_rejected,
            "transition_byte_tamper_rejected_after_claim": transition_source_tamper_rejected,
            "claim_receipt_is_content_addressed": claimed.claim[
                "attachment_claim_id"
            ]
            == content_sha256(claim_body),
            "inert_attachment_imports_no_model_frameworks": before_frameworks
            == (MODEL_FRAMEWORKS & set(sys.modules)),
            "inert_attachment_creates_no_dictionary_output": not Path(
                transition["job"]["reserved_exact_output_path"]
            ).exists(),
            "inert_attachment_remains_non_authorizing": all(
                claimed.claim["scientific_boundary"][field_name] in (False, 0)
                for field_name in (
                    "model_loaded",
                    "model_inference_performed",
                    "persistent_dictionary_output_created",
                    "weights_loaded_or_mutated",
                    "production_runtime_attached",
                    "trainer_callback_invoked",
                    "execution_authorized_by_claim",
                    "sae_training_authorized_by_claim",
                    "real_actions_authorized",
                )
            ),
            "all_claim_artifacts_are_temporary": all(
                path.is_relative_to(root)
                for path in (
                    claimed.claim_receipt_path,
                    failed_claim_root,
                    active_claim_root,
                )
            ),
        }
    if not all(controls.values()):
        raise ValueError(
            "cluster-aware SAE inert attachment benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-inert-attachment-benchmark.v1",
        "status": "pass",
        "source": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "durable_transition_sha256": sha256_file(
                Path(__file__).with_name(
                    "reserve_cluster_aware_sae_training_transition_v1.py"
                )
            ),
            "inert_attachment_sha256": sha256_file(
                Path(__file__).with_name(
                    "attach_cluster_aware_sae_inert_runtime_v1.py"
                )
            ),
            "runtime_worker_sha256": sha256_file(
                Path(__file__).with_name("run_cluster_aware_sae_training_job_v1.py")
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "durable_attachment_claim_before_factory": True,
            "exact_ledger_reconstruction": True,
            "synthetic_inert_runtime_only": True,
            "terminal_claim_tombstones": True,
            "production_runtime_attach_implemented": False,
            "trainer_callback_invocation_implemented": False,
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "successful_execution_job_id": transition["job"]["execution_job_id"],
            "temporary_claim_receipt_content_addressed": True,
            "factory_failure_claim_terminal": True,
            "active_runtime_claim_terminal": True,
        },
        "scientific_boundary": {
            "temporary_simulation_only": True,
            "persistent_workspace_claim_created": False,
            "synthetic_keys_receipts_and_runtime_only": True,
            "empirical_blueprint_present": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "persistent_dictionary_output_created": False,
            "weights_loaded_or_mutated": False,
            "inert_runtime_attached": True,
            "production_runtime_attached": False,
            "trainer_callback_invoked": False,
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
        raise ValueError("cluster-aware SAE inert attachment benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("cluster-aware SAE inert attachment benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster-aware SAE inert attachment benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-aware-sae-inert-attachment-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": int(observed["summary"]["controls"]),
        "durable_attachment_claim_implemented": True,
        "ledger_reconstruction_implemented": True,
        "inert_runtime_attached": True,
        "production_runtime_attached": False,
        "trainer_callback_invoked": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cluster-aware SAE inert runtime attachment V1."
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
