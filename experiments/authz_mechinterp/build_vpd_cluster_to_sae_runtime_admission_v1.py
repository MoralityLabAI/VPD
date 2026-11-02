from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    OUTPUT_PARENT,
    _canonical_absolute,
    _existing_ancestor_has_reparse_point,
    audit_completed_bindings,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 import (
    verify_preflight,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.verify_vpd_cluster_to_sae_bound_content_v1 import (
    verify_bound_content,
)
from experiments.authz_mechinterp.verify_vpd_cluster_to_sae_capture_authorization_v1 import (
    verify_authorization_receipt,
)


SCHEMA = "sch.vpd-cluster-to-sae-runtime-admission.v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_reserve_output_root(
    *, output_root: Path, project_root: Path
) -> dict[str, Any]:
    root = output_root.resolve()
    parent = (project_root / OUTPUT_PARENT).resolve()
    if root.parent != parent:
        raise ValueError("runtime reservation output is not a direct child")
    if not parent.is_dir():
        raise ValueError("runtime reservation parent does not exist")
    if _existing_ancestor_has_reparse_point(parent, stop=project_root):
        raise ValueError("runtime reservation path crosses a reparse point")
    try:
        root.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("runtime reservation output already exists") from exc
    body = {
        "schema": "sch.vpd-cluster-to-sae-output-root-reservation.v1",
        "status": "reserved_empty_output_root_capture_not_started",
        "output_root": str(root),
        "atomic_create": True,
        "preexisting_content_overwritten": False,
        "resume_authorized": False,
        "capture_started": False,
        "execution_authorized": False,
    }
    return {**body, "reservation_id": content_sha256(body)}


def build_runtime_admission(
    *,
    preflight_path: Path,
    model_manifest_path: Path,
    authorization_receipt_path: Path,
    authority_registry_path: Path,
    as_of: str,
    project_root: Path,
) -> dict[str, Any]:
    preflight_verification = verify_preflight(
        preflight_path, project_root=project_root
    )
    preflight = _load(preflight_path)
    bindings = preflight["immutable_bindings"]
    binding_audit = audit_completed_bindings(
        project_root=project_root, bindings=bindings
    )
    content_verification = verify_bound_content(
        preflight_path=preflight_path,
        model_manifest_path=model_manifest_path,
    )
    receipt = _load(authorization_receipt_path)
    registry = _load(authority_registry_path)
    authorization_verification = verify_authorization_receipt(
        receipt=receipt,
        registry=registry,
        preflight=preflight,
        as_of=as_of,
    )
    preflight_id = preflight["preflight_id"]
    if any(
        observed != preflight_id
        for observed in (
            preflight_verification["preflight_id"],
            binding_audit["preflight_id"],
            content_verification["preflight_id"],
            authorization_verification["preflight_id"],
        )
    ):
        raise ValueError("runtime admission preflight join differs")
    if content_verification["model_manifest_file_sha256"] != bindings[
        "model_files_manifest_sha256"
    ]:
        raise ValueError("runtime admission model manifest binding differs")
    if content_verification["adapter_weights_sha256"] != bindings[
        "adapter_weights_sha256"
    ]:
        raise ValueError("runtime admission adapter weight binding differs")
    if content_verification["adapter_config_sha256"] != bindings[
        "adapter_config_sha256"
    ]:
        raise ValueError("runtime admission adapter configuration binding differs")
    synthetic = authorization_verification["synthetic_fixture_only"]
    output_root = _canonical_absolute(bindings["output_root"], field="output_root")
    body = {
        "schema": SCHEMA,
        "status": (
            "blocked_authenticated_synthetic_fixture"
            if synthetic
            else "readonly_checks_pass_real_receipt_requires_atomic_reservation"
        ),
        "as_of": as_of,
        "source": {
            "preflight": {
                "path": str(preflight_path.resolve()),
                "sha256": sha256_file(preflight_path),
                "preflight_id": preflight_id,
            },
            "model_manifest": {
                "path": str(model_manifest_path.resolve()),
                "sha256": sha256_file(model_manifest_path),
                "manifest_id": content_verification["model_manifest_id"],
            },
            "authorization_receipt": {
                "path": str(authorization_receipt_path.resolve()),
                "sha256": sha256_file(authorization_receipt_path),
                "receipt_id": authorization_verification[
                    "authorization_receipt_id"
                ],
            },
            "authority_registry": {
                "path": str(authority_registry_path.resolve()),
                "sha256": sha256_file(authority_registry_path),
                "registry_id": authorization_verification["authority_registry_id"],
            },
        },
        "joined_checks": {
            "preflight_reproducible": True,
            "canonical_bindings_complete": binding_audit["bindings_complete"],
            "output_root_nonexistent": binding_audit["output_root_nonexistent"],
            "exact_model_tree_verified": content_verification["model_files"] > 0,
            "exact_adapter_bytes_verified": True,
            "signature_valid": authorization_verification["signature_valid"],
            "scope_exact": authorization_verification["scope_exact"],
            "authorization_active": authorization_verification["active_at_as_of"],
            "nonce_unconsumed": authorization_verification["nonce_unconsumed"],
            "all_checks_join_same_preflight": True,
        },
        "reservation": {
            "output_root": str(output_root),
            "output_root_exists": False,
            "atomic_create_required": True,
            "reservation_performed": False,
            "existing_root_is_terminal_fail_closed": True,
            "reservation_itself_does_not_start_capture": True,
        },
        "stop_reason": (
            "synthetic_authorization_fixture_cannot_authorize_reservation"
            if synthetic
            else "explicit_atomic_reservation_step_not_performed"
        ),
        "scientific_boundary": {
            "synthetic_authorization_fixture_only": synthetic,
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


def verify_runtime_admission(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    identifier = str(observed.get("admission_id", ""))
    body = {key: value for key, value in observed.items() if key != "admission_id"}
    if identifier != content_sha256(body) or observed != build_runtime_admission(**kwargs):
        raise ValueError("cluster-to-SAE runtime admission is not reproducible")
    return {
        "schema": "sch.vpd-cluster-to-sae-runtime-admission-verification.v1",
        "status": "pass",
        "admission_id": identifier,
        "synthetic_authorization_fixture_only": observed["scientific_boundary"][
            "synthetic_authorization_fixture_only"
        ],
        "reservation_performed": False,
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify a read-only cluster-to-SAE runtime admission receipt."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--authority-registry", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    kwargs = {
        "preflight_path": args.preflight,
        "model_manifest_path": args.model_manifest,
        "authorization_receipt_path": args.authorization_receipt,
        "authority_registry_path": args.authority_registry,
        "as_of": args.as_of,
        "project_root": Path(__file__).resolve().parents[2],
    }
    if args.output is not None:
        result = build_runtime_admission(**kwargs)
        write_json(args.output, result)
    else:
        result = verify_runtime_admission(args.verify, **kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
