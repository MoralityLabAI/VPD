from __future__ import annotations

import argparse
import copy
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
    CAMPAIGN_RECEIPT_NAME,
    JOB_RECEIPT_NAME,
    RESERVATION_PARENT,
    reserve_training_transition,
    verify_training_transition,
)


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return True
    return False


def _write_inputs(
    *,
    root: Path,
    blueprint: dict[str, Any],
    release_receipt: dict[str, Any],
    release_registry: dict[str, Any],
    compute_receipt: dict[str, Any],
    compute_registry: dict[str, Any],
    suffix: str = "",
) -> dict[str, Path]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    paths = {
        "blueprint": inputs / "blueprint.json",
        "release_receipt": inputs / f"release{suffix}.json",
        "release_registry": inputs / "release-registry.json",
        "compute_receipt": inputs / f"compute{suffix}.json",
        "compute_registry": inputs / "compute-registry.json",
    }
    write_json(paths["blueprint"], blueprint)
    write_json(paths["release_receipt"], release_receipt)
    write_json(paths["release_registry"], release_registry)
    write_json(paths["compute_receipt"], compute_receipt)
    write_json(paths["compute_registry"], compute_registry)
    return paths


def _reserve(
    *,
    root: Path,
    paths: dict[str, Path],
    job_id: str,
    simulation_only: bool = True,
) -> dict[str, Any]:
    return reserve_training_transition(
        project_root=root,
        blueprint_path=paths["blueprint"],
        execution_job_id=job_id,
        device=DEVICE,
        release_receipt_path=paths["release_receipt"],
        release_registry_path=paths["release_registry"],
        compute_receipt_path=paths["compute_receipt"],
        compute_registry_path=paths["compute_registry"],
        as_of=AS_OF,
        simulation_only=simulation_only,
    )


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
    persistent_paths_after: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vpd-sae-transition-v1-") as temporary:
        root = Path(temporary)
        reservation_parent = root / RESERVATION_PARENT
        reservation_parent.mkdir(parents=True)
        blueprint_path_for_hash = root / "blueprint-hash-source.json"
        write_json(blueprint_path_for_hash, blueprint)
        blueprint_sha256 = sha256_file(blueprint_path_for_hash)
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
        first_job = blueprint["execution_jobs"][0]
        compute_body = _compute_body(
            blueprint=blueprint,
            job=first_job,
            release_verification=release_verification,
        )
        compute_receipt = sign_receipt(compute_body, compute_key)
        paths = _write_inputs(
            root=root,
            blueprint=blueprint,
            release_receipt=release_receipt,
            release_registry=release_registry,
            compute_receipt=compute_receipt,
            compute_registry=compute_registry,
        )
        transition = _reserve(
            root=root, paths=paths, job_id=first_job["execution_job_id"]
        )
        first_job_root = Path(transition["job"]["reserved_job_root"])
        first_receipt_path = first_job_root / JOB_RECEIPT_NAME
        first_verification = verify_training_transition(
            first_receipt_path, project_root=root
        )
        campaign_path = Path(transition["campaign"]["campaign_receipt_path"])
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))

        replay_rejected = _rejects(
            lambda: _reserve(
                root=root, paths=paths, job_id=first_job["execution_job_id"]
            )
        )
        same_nonce_different_job_rejected = _rejects(
            lambda: _reserve(
                root=root,
                paths=paths,
                job_id=blueprint["execution_jobs"][1]["execution_job_id"],
            )
        )

        second_job = blueprint["execution_jobs"][1]
        second_compute_body = _compute_body(
            blueprint=blueprint,
            job=second_job,
            release_verification=release_verification,
        )
        second_compute_body["nonce"] = "3" * 64
        second_compute_receipt = sign_receipt(second_compute_body, compute_key)
        second_paths = _write_inputs(
            root=root,
            blueprint=blueprint,
            release_receipt=release_receipt,
            release_registry=release_registry,
            compute_receipt=second_compute_receipt,
            compute_registry=compute_registry,
            suffix="-second",
        )
        second_transition = _reserve(
            root=root,
            paths=second_paths,
            job_id=second_job["execution_job_id"],
        )
        second_verification = verify_training_transition(
            Path(second_transition["job"]["reserved_job_root"]) / JOB_RECEIPT_NAME,
            project_root=root,
        )

        collision_release_body = copy.deepcopy(release_body)
        collision_release_body["explicit_release_statement"] = (
            "different synthetic release with deliberately reused nonce"
        )
        collision_release_receipt = sign_receipt(collision_release_body, release_key)
        collision_release_verification = verify_scientific_release_receipt(
            receipt=collision_release_receipt,
            registry=release_registry,
            blueprint=blueprint,
            blueprint_sha256=blueprint_sha256,
            as_of=AS_OF,
        )
        collision_compute_body = _compute_body(
            blueprint=blueprint,
            job=blueprint["execution_jobs"][2],
            release_verification=collision_release_verification,
        )
        collision_compute_body["nonce"] = "4" * 64
        collision_compute_receipt = sign_receipt(collision_compute_body, compute_key)
        collision_paths = _write_inputs(
            root=root,
            blueprint=blueprint,
            release_receipt=collision_release_receipt,
            release_registry=release_registry,
            compute_receipt=collision_compute_receipt,
            compute_registry=compute_registry,
            suffix="-collision",
        )
        reused_release_nonce_different_receipt_rejected = _rejects(
            lambda: _reserve(
                root=root,
                paths=collision_paths,
                job_id=blueprint["execution_jobs"][2]["execution_job_id"],
            )
        )
        collision_compute_root_absent = not (
            reservation_parent / f"compute-nonce-{'4' * 64}"
        ).exists()

        tombstone_release_body = copy.deepcopy(release_body)
        tombstone_release_body["nonce"] = "5" * 64
        tombstone_release_receipt = sign_receipt(tombstone_release_body, release_key)
        tombstone_release_verification = verify_scientific_release_receipt(
            receipt=tombstone_release_receipt,
            registry=release_registry,
            blueprint=blueprint,
            blueprint_sha256=blueprint_sha256,
            as_of=AS_OF,
        )
        tombstone_compute_body = _compute_body(
            blueprint=blueprint,
            job=blueprint["execution_jobs"][3],
            release_verification=tombstone_release_verification,
        )
        tombstone_compute_body["nonce"] = "6" * 64
        tombstone_compute_receipt = sign_receipt(tombstone_compute_body, compute_key)
        tombstone_paths = _write_inputs(
            root=root,
            blueprint=blueprint,
            release_receipt=tombstone_release_receipt,
            release_registry=release_registry,
            compute_receipt=tombstone_compute_receipt,
            compute_registry=compute_registry,
            suffix="-tombstone",
        )
        (reservation_parent / f"release-nonce-{'5' * 64}").mkdir()
        incomplete_campaign_tombstone_rejected = _rejects(
            lambda: _reserve(
                root=root,
                paths=tombstone_paths,
                job_id=blueprint["execution_jobs"][3]["execution_job_id"],
            )
        )
        tombstone_compute_root_absent = not (
            reservation_parent / f"compute-nonce-{'6' * 64}"
        ).exists()

        real_mode_from_synthetic_rejected = _rejects(
            lambda: reserve_training_transition(
                project_root=root,
                blueprint_path=paths["blueprint"],
                execution_job_id=blueprint["execution_jobs"][4][
                    "execution_job_id"
                ],
                device=DEVICE,
                release_receipt_path=paths["release_receipt"],
                release_registry_path=paths["release_registry"],
                compute_receipt_path=paths["compute_receipt"],
                compute_registry_path=paths["compute_registry"],
                as_of=AS_OF,
                simulation_only=False,
            )
        )

        original_transition_text = first_receipt_path.read_text(encoding="utf-8")
        tampered_transition = json.loads(original_transition_text)
        tampered_transition["scientific_boundary"]["runtime_attached"] = True
        write_json(first_receipt_path, tampered_transition)
        transition_tamper_rejected = _rejects(
            lambda: verify_training_transition(first_receipt_path, project_root=root)
        )
        first_receipt_path.write_text(original_transition_text, encoding="utf-8")

        original_campaign_text = campaign_path.read_text(encoding="utf-8")
        tampered_campaign = json.loads(original_campaign_text)
        tampered_campaign["trainer_callback_invoked"] = True
        write_json(campaign_path, tampered_campaign)
        campaign_tamper_rejected = _rejects(
            lambda: verify_training_transition(first_receipt_path, project_root=root)
        )
        campaign_path.write_text(original_campaign_text, encoding="utf-8")

        original_blueprint_text = paths["blueprint"].read_text(encoding="utf-8")
        paths["blueprint"].write_text(
            original_blueprint_text + " ", encoding="utf-8"
        )
        source_tamper_rejected = _rejects(
            lambda: verify_training_transition(first_receipt_path, project_root=root)
        )
        paths["blueprint"].write_text(original_blueprint_text, encoding="utf-8")

        campaign_body = {
            key: value
            for key, value in campaign.items()
            if key != "campaign_reservation_id"
        }
        transition_body = {
            key: value
            for key, value in transition.items()
            if key != "transition_reservation_id"
        }
        controls = {
            "signed_admission_reserves_transition": first_verification["status"]
            == "pass",
            "release_nonce_creates_durable_campaign": campaign_path.is_file()
            and campaign["release_nonce_consumed"] is True,
            "compute_nonce_atomically_reserves_job_root": first_job_root.is_dir()
            and first_job_root.name
            == f"compute-nonce-{compute_receipt['nonce']}",
            "transition_receipt_is_inside_reserved_job_root": first_receipt_path.is_file()
            and first_receipt_path.parent == first_job_root,
            "exact_dictionary_output_path_is_reserved_but_not_created": not Path(
                transition["job"]["reserved_exact_output_path"]
            ).exists()
            and Path(transition["job"]["reserved_exact_output_path"]).is_relative_to(
                first_job_root
            ),
            "worker_ledger_reserved_before_runtime": transition[
                "worker_ledger_after_reservation"
            ]["reservations"][first_job["execution_job_id"]]
            == "reserved"
            and transition["transition"]["runtime_attached"] is False,
            "same_compute_nonce_replay_rejected": replay_rejected,
            "same_compute_nonce_cannot_switch_jobs": same_nonce_different_job_rejected,
            "one_release_campaign_can_bind_multiple_unique_jobs": second_transition[
                "campaign"
            ]["campaign_reservation_id"]
            == transition["campaign"]["campaign_reservation_id"]
            and second_transition["campaign"]["campaign_created_by_this_transition"]
            is False
            and second_verification["status"] == "pass",
            "reused_release_nonce_cannot_switch_receipts": reused_release_nonce_different_receipt_rejected
            and collision_compute_root_absent,
            "incomplete_campaign_tombstone_is_terminal": incomplete_campaign_tombstone_rejected
            and tombstone_compute_root_absent,
            "synthetic_transition_rejected_in_real_mode": real_mode_from_synthetic_rejected,
            "transition_receipt_tamper_rejected": transition_tamper_rejected,
            "campaign_receipt_tamper_rejected": campaign_tamper_rejected,
            "source_byte_tamper_rejected": source_tamper_rejected,
            "campaign_receipt_is_content_addressed": campaign[
                "campaign_reservation_id"
            ]
            == content_sha256(campaign_body),
            "job_transition_is_content_addressed": transition[
                "transition_reservation_id"
            ]
            == content_sha256(transition_body),
            "partial_failure_policy_is_fail_closed_no_rollback": transition[
                "transition"
            ]["partial_failure_leaves_terminal_tombstone"]
            is True
            and transition["transition"]["rollback_or_resume_implemented"] is False,
            "transition_imports_no_model_frameworks": before_frameworks
            == (MODEL_FRAMEWORKS & set(sys.modules)),
            "transition_never_attaches_or_invokes_or_trains": all(
                transition["scientific_boundary"][field_name] in (False, 0)
                for field_name in (
                    "model_loaded",
                    "model_inference_performed",
                    "persistent_dictionary_output_created",
                    "weights_loaded_or_mutated",
                    "runtime_attached",
                    "trainer_callback_invoked",
                    "execution_authorized_by_transition",
                    "sae_training_authorized_by_transition",
                    "real_actions_authorized",
                )
            ),
        }
        persistent_paths_after = [
            str(path) for path in reservation_parent.iterdir() if path.exists()
        ]
    if not all(controls.values()):
        raise ValueError(
            "cluster-aware SAE transition benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-transition-benchmark.v1",
        "status": "pass",
        "source": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "signed_admission_verifier_sha256": sha256_file(
                Path(__file__).with_name("admit_cluster_aware_sae_training_v1.py")
            ),
            "transition_sha256": sha256_file(
                Path(__file__).with_name(
                    "reserve_cluster_aware_sae_training_transition_v1.py"
                )
            ),
            "runtime_worker_sha256": sha256_file(
                Path(__file__).with_name("run_cluster_aware_sae_training_job_v1.py")
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "release_campaign_nonce_directory": True,
            "atomic_compute_nonce_and_job_root_directory": True,
            "worker_ledger_reserved_before_runtime": True,
            "terminal_tombstones_no_rollback": True,
            "runtime_attach_implemented": False,
            "trainer_invocation_implemented": False,
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "first_execution_job_id": first_job["execution_job_id"],
            "second_execution_job_id": second_job["execution_job_id"],
            "campaign_reservation_id": transition["campaign"][
                "campaign_reservation_id"
            ],
            "temporary_transition_receipts_content_addressed": True,
            "temporary_reservation_entries_created": len(persistent_paths_after),
        },
        "scientific_boundary": {
            "temporary_simulation_only": True,
            "persistent_workspace_output_created": False,
            "synthetic_keys_and_receipts_only": True,
            "empirical_blueprint_present": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "persistent_dictionary_output_created": False,
            "weights_loaded_or_mutated": False,
            "runtime_attached": False,
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
        raise ValueError("cluster-aware SAE transition benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("cluster-aware SAE transition benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster-aware SAE transition benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-aware-sae-transition-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": int(observed["summary"]["controls"]),
        "release_nonce_consumption_implemented": True,
        "atomic_compute_nonce_and_job_root_reservation_implemented": True,
        "worker_ledger_reservation_implemented": True,
        "runtime_attached": False,
        "trainer_callback_invoked": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the cluster-aware SAE durable transition V1."
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
