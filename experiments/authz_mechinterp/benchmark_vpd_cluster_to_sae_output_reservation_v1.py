from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
    OUTPUT_PARENT,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.vpd_cluster_to_sae_output_reservation_v1 import (
    RECEIPT_NAME,
    reserve_admitted_output_root,
    verify_output_reservation,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (FileNotFoundError, TypeError, ValueError):
        return True
    return False


def _admission(
    *, source_paths: dict[str, Path], output_root: Path
) -> dict[str, Any]:
    source = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            **(
                {"preflight_id": "1" * 64}
                if name == "preflight"
                else {}
            ),
        }
        for name, path in source_paths.items()
    }
    body = {
        "schema": "sch.vpd-cluster-to-sae-runtime-admission.v1",
        "status": "blocked_authenticated_synthetic_fixture",
        "as_of": "2030-01-01T00:00:00Z",
        "source": source,
        "joined_checks": {
            "preflight_reproducible": True,
            "canonical_bindings_complete": True,
            "output_root_nonexistent": True,
            "exact_model_tree_verified": True,
            "exact_adapter_bytes_verified": True,
            "signature_valid": True,
            "scope_exact": True,
            "authorization_active": True,
            "nonce_unconsumed": True,
            "all_checks_join_same_preflight": True,
        },
        "reservation": {
            "output_root": str(output_root.resolve()),
            "output_root_exists": False,
            "atomic_create_required": True,
            "reservation_performed": False,
            "existing_root_is_terminal_fail_closed": True,
            "reservation_itself_does_not_start_capture": True,
        },
        "stop_reason": "synthetic_authorization_fixture_cannot_authorize_reservation",
        "scientific_boundary": {
            "synthetic_authorization_fixture_only": True,
            "readonly_admission_join_only": True,
            "model_framework_imported": False,
            "model_loaded": False,
            "adapter_loaded": False,
            "model_inference_performed": False,
            "weights_mutated": False,
            "output_root_created": False,
            "capture_started": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "final_evaluation_authorized": False,
            "real_actions_authorized": False,
        },
    }
    return {**body, "admission_id": content_sha256(body)}


def build_report(project_root: Path) -> dict[str, Any]:
    del project_root
    before = MODEL_FRAMEWORKS & set(sys.modules)
    with tempfile.TemporaryDirectory(prefix="vpd-output-reservation-") as temporary:
        root = Path(temporary)
        parent = root / OUTPUT_PARENT
        parent.mkdir(parents=True)
        source_paths = {
            "preflight": root / "preflight.json",
            "model_manifest": root / "model-manifest.json",
            "authorization_receipt": root / "authorization.json",
            "authority_registry": root / "registry.json",
        }
        for index, path in enumerate(source_paths.values()):
            path.write_text(json.dumps({"synthetic": index}) + "\n", encoding="utf-8")
        output_root = parent / "synthetic-reservation"
        admission_path = root / "admission.json"
        admission = _admission(source_paths=source_paths, output_root=output_root)
        write_json(admission_path, admission)
        real_mode_rejected = _rejects(
            lambda: reserve_admitted_output_root(
                admission_path=admission_path,
                project_root=root,
                simulation_only=False,
            )
        )
        root_absent_after_real_reject = not output_root.exists()
        receipt = reserve_admitted_output_root(
            admission_path=admission_path,
            project_root=root,
            simulation_only=True,
        )
        receipt_path = output_root / RECEIPT_NAME
        verification = verify_output_reservation(receipt_path, project_root=root)
        double_reservation_rejected = _rejects(
            lambda: reserve_admitted_output_root(
                admission_path=admission_path,
                project_root=root,
                simulation_only=True,
            )
        )
        original_receipt = receipt_path.read_text(encoding="utf-8")
        changed_receipt = json.loads(original_receipt)
        changed_receipt["capture_started"] = True
        write_json(receipt_path, changed_receipt)
        receipt_tamper_rejected = _rejects(
            lambda: verify_output_reservation(receipt_path, project_root=root)
        )
        receipt_path.write_text(original_receipt, encoding="utf-8")
        original_source = source_paths["model_manifest"].read_text(encoding="utf-8")
        source_paths["model_manifest"].write_text(
            original_source + "tamper\n", encoding="utf-8"
        )
        source_tamper_rejected = _rejects(
            lambda: verify_output_reservation(receipt_path, project_root=root)
        )
        source_paths["model_manifest"].write_text(original_source, encoding="utf-8")
        original_admission = admission_path.read_text(encoding="utf-8")
        changed_admission = json.loads(original_admission)
        changed_admission["joined_checks"]["scope_exact"] = False
        write_json(admission_path, changed_admission)
        admission_tamper_rejected = _rejects(
            lambda: verify_output_reservation(receipt_path, project_root=root)
        )
        admission_path.write_text(original_admission, encoding="utf-8")
        controls = {
            "synthetic_admission_rejected_in_real_mode": real_mode_rejected,
            "real_mode_reject_creates_no_root": root_absent_after_real_reject,
            "simulation_reservation_is_atomic_direct_child": (
                output_root.is_dir()
                and output_root.parent == parent
                and receipt["reservation"]["atomic_create"] is True
            ),
            "reservation_receipt_is_inside_reserved_root": (
                receipt_path.is_file() and receipt_path.parent == output_root
            ),
            "reservation_transition_verifies": verification["status"] == "pass",
            "reservation_binds_admission_and_preflight": (
                verification["admission_id"] == admission["admission_id"]
                and verification["preflight_id"] == "1" * 64
            ),
            "reservation_does_not_start_capture": (
                verification["capture_started"] is False
                and verification["execution_authorized"] is False
            ),
            "double_reservation_rejected": double_reservation_rejected,
            "receipt_tamper_rejected": receipt_tamper_rejected,
            "admission_source_tamper_rejected": source_tamper_rejected,
            "admission_receipt_tamper_rejected": admission_tamper_rejected,
            "reservation_imports_no_model_frameworks": before
            == (MODEL_FRAMEWORKS & set(sys.modules)),
            "reservation_remains_non_authorizing": all(
                receipt[field] in (False, 0)
                for field in (
                    "capture_started",
                    "model_loaded",
                    "model_inference_performed",
                    "execution_authorized",
                    "sae_training_authorized",
                )
            ),
        }
    if not all(controls.values()):
        raise ValueError(
            "output reservation benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-to-sae-output-reservation-benchmark.v1",
        "status": "pass",
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "synthetic_admission_content_address_verified": True,
            "reservation_transition_content_address_verified": True,
            "simulation_only": True,
        },
        "source": {
            "reservation_transition_sha256": sha256_file(
                Path(__file__).with_name("vpd_cluster_to_sae_output_reservation_v1.py")
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "temporary_simulation_only": True,
            "persistent_output_created": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "capture_started": False,
            "weights_mutated": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in observed.items() if key != "benchmark_id"}
    if observed.get("benchmark_id") != content_sha256(body):
        raise ValueError("output reservation benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("output reservation benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-to-sae-output-reservation-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": observed["benchmark_id"],
        "controls": observed["summary"]["controls"],
        "persistent_output_created": False,
        "capture_started": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark admitted output reservation.")
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
