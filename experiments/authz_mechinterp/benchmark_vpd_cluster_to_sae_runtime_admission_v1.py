from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_capture_authorization_v1 import (
    AS_OF,
    _receipt_body,
    _registry,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    OUTPUT_PARENT,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 import (
    build_preflight,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_runtime_admission_v1 import (
    _atomic_reserve_output_root,
    build_runtime_admission,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import crypto_types
from experiments.authz_mechinterp.verify_vpd_cluster_to_sae_bound_content_v1 import (
    build_model_files_manifest,
)
from experiments.authz_mechinterp.verify_vpd_cluster_to_sae_capture_authorization_v1 import (
    sign_authorization_receipt,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def _address_registry(registry: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in registry.items() if key != "registry_id"}
    return {**body, "registry_id": content_sha256(body)}


def build_report(project_root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vpd-runtime-admission-") as temporary:
        root = Path(temporary)
        model_root = root / "model"
        adapter_root = root / "adapter"
        model_root.mkdir()
        adapter_root.mkdir()
        (model_root / "config.json").write_text(
            '{"model_type":"synthetic"}\n', encoding="utf-8"
        )
        (model_root / "model.safetensors").write_bytes(b"synthetic-model-bytes")
        weights_path = adapter_root / "adapter_model.safetensors"
        config_path = adapter_root / "adapter_config.json"
        weights_path.write_bytes(b"synthetic-adapter-bytes")
        config_path.write_text('{"r":8}\n', encoding="utf-8")
        manifest_path = root / "model-manifest.json"
        write_json(
            manifest_path,
            build_model_files_manifest(
                model_root=model_root,
                model_id="synthetic/model-not-loaded",
                model_revision="synthetic-revision-not-loaded",
            ),
        )
        output_root = (
            project_root
            / OUTPUT_PARENT
            / "synthetic-runtime-admission-do-not-create"
        ).resolve()
        bindings = {
            "model_id": "synthetic/model-not-loaded",
            "model_revision": "synthetic-revision-not-loaded",
            "model_files_manifest_sha256": sha256_file(manifest_path),
            "adapter_path": str(adapter_root.resolve()),
            "adapter_weights_sha256": sha256_file(weights_path),
            "adapter_config_sha256": sha256_file(config_path),
            "device": "cpu",
            "wall_time_ceiling_seconds": 3600,
            "cost_ceiling_usd": 1.0,
            "output_root": str(output_root),
        }
        preflight = build_preflight(project_root=project_root, bindings=bindings)
        preflight_path = root / "preflight.json"
        write_json(preflight_path, preflight)
        Ed25519PrivateKey, _, _, _ = crypto_types()
        key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
        registry = _registry(key)
        registry_path = root / "registry.json"
        write_json(registry_path, registry)
        receipt = sign_authorization_receipt(_receipt_body(preflight), key)
        receipt_path = root / "receipt.json"
        write_json(receipt_path, receipt)
        kwargs = {
            "preflight_path": preflight_path,
            "model_manifest_path": manifest_path,
            "authorization_receipt_path": receipt_path,
            "authority_registry_path": registry_path,
            "as_of": AS_OF,
            "project_root": project_root,
        }
        clean = build_runtime_admission(**kwargs)

        original_weight = weights_path.read_bytes()
        weights_path.write_bytes(original_weight + b"tamper")
        mutated_content_rejected = _rejects(lambda: build_runtime_admission(**kwargs))
        weights_path.write_bytes(original_weight)

        consumed_registry = json.loads(json.dumps(registry))
        consumed_registry["consumed_nonces"] = [receipt["nonce"]]
        write_json(registry_path, _address_registry(consumed_registry))
        consumed_nonce_rejected = _rejects(lambda: build_runtime_admission(**kwargs))
        write_json(registry_path, registry)

        wrong_body = _receipt_body(preflight)
        wrong_body["source"]["preflight_id"] = "f" * 64
        write_json(receipt_path, sign_authorization_receipt(wrong_body, key))
        wrong_preflight_rejected = _rejects(lambda: build_runtime_admission(**kwargs))
        write_json(receipt_path, receipt)

        expired_body = _receipt_body(preflight)
        expired_body["expires_at"] = "2030-01-01T00:15:00Z"
        write_json(receipt_path, sign_authorization_receipt(expired_body, key))
        expired_receipt_rejected = _rejects(lambda: build_runtime_admission(**kwargs))
        write_json(receipt_path, receipt)

        reservation_project = root / "reservation-project"
        reservation_parent = reservation_project / OUTPUT_PARENT
        reservation_parent.mkdir(parents=True)
        reservation_root = reservation_parent / "synthetic-reservation-run"
        reservation = _atomic_reserve_output_root(
            output_root=reservation_root,
            project_root=reservation_project,
        )
        double_reservation_rejected = _rejects(
            lambda: _atomic_reserve_output_root(
                output_root=reservation_root,
                project_root=reservation_project,
            )
        )
        outside_reservation_rejected = _rejects(
            lambda: _atomic_reserve_output_root(
                output_root=reservation_project / "outside-run",
                project_root=reservation_project,
            )
        )
        controls = {
            "four_proofs_join_same_preflight": (
                clean["joined_checks"]["all_checks_join_same_preflight"]
                and clean["joined_checks"]["preflight_reproducible"]
                and clean["joined_checks"]["canonical_bindings_complete"]
                and clean["joined_checks"]["exact_model_tree_verified"]
                and clean["joined_checks"]["exact_adapter_bytes_verified"]
                and clean["joined_checks"]["signature_valid"]
                and clean["joined_checks"]["scope_exact"]
            ),
            "synthetic_authorization_blocks_reservation": (
                clean["status"] == "blocked_authenticated_synthetic_fixture"
                and clean["reservation"]["reservation_performed"] is False
                and clean["scientific_boundary"]["execution_authorized"] is False
                and not output_root.exists()
            ),
            "content_mutation_rejected_at_join": mutated_content_rejected,
            "consumed_nonce_rejected_at_join": consumed_nonce_rejected,
            "wrong_preflight_receipt_rejected_at_join": wrong_preflight_rejected,
            "expired_receipt_rejected_at_join": expired_receipt_rejected,
            "atomic_reservation_creates_empty_root_without_capture": (
                reservation["status"]
                == "reserved_empty_output_root_capture_not_started"
                and reservation_root.is_dir()
                and not any(reservation_root.iterdir())
                and reservation["capture_started"] is False
                and reservation["execution_authorized"] is False
            ),
            "double_reservation_rejected_fail_closed": double_reservation_rejected,
            "outside_reservation_rejected": outside_reservation_rejected,
            "reservation_helper_does_not_overwrite_or_resume": (
                reservation["preexisting_content_overwritten"] is False
                and reservation["resume_authorized"] is False
            ),
            "admission_join_does_not_start_capture": (
                clean["scientific_boundary"]["capture_started"] is False
                and clean["scientific_boundary"]["model_inference_performed"] is False
                and clean["scientific_boundary"]["weights_mutated"] is False
            ),
            "all_paths_remain_non_authorizing": all(
                clean["scientific_boundary"][field] is False
                for field in (
                    "output_root_created",
                    "capture_started",
                    "execution_authorized",
                    "sae_training_authorized",
                    "final_evaluation_authorized",
                    "real_actions_authorized",
                )
            ),
        }
    if not all(controls.values()):
        raise ValueError(
            "cluster-to-SAE runtime admission benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-to-sae-runtime-admission-benchmark.v1",
        "status": "pass",
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "source": {
            "runtime_admission_builder_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/build_vpd_cluster_to_sae_runtime_admission_v1.py"
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "synthetic_temporary_inputs_only": True,
            "real_model_or_adapter_content_opened": False,
            "real_authorization_receipt_present": False,
            "persistent_output_root_created": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_mutated": False,
            "capture_started": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = str(observed.get("benchmark_id", ""))
    body = {key: value for key, value in observed.items() if key != "benchmark_id"}
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("cluster-to-SAE runtime admission benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-to-sae-runtime-admission-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "persistent_output_root_created": False,
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the read-only cluster-to-SAE runtime admission join."
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
