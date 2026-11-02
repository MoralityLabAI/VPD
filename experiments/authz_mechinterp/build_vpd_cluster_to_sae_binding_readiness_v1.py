from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 import (
    REQUIRED_EXECUTION_BINDINGS,
    build_preflight,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-cluster-to-sae-binding-readiness.v1"
OUTPUT_PARENT = "artifacts/vpd_cluster_to_sae_empirical_capture_runs"
DEVICE_RE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)")
RUN_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,127}")
MODEL_FRAMEWORKS = {"torch", "transformers", "peft", "accelerate"}

BINDING_CONTRACTS: dict[str, dict[str, Any]] = {
    "model_id": {
        "kind": "nonempty_string",
        "completion_evidence": "exact configured model identifier",
        "content_opened_by_this_check": False,
    },
    "model_revision": {
        "kind": "nonempty_string",
        "completion_evidence": "immutable revision or commit identifier",
        "content_opened_by_this_check": False,
    },
    "model_files_manifest_sha256": {
        "kind": "sha256",
        "completion_evidence": "digest of a separately produced model-files manifest",
        "content_opened_by_this_check": False,
    },
    "adapter_path": {
        "kind": "canonical_absolute_path",
        "completion_evidence": "canonical absolute adapter path",
        "content_opened_by_this_check": False,
    },
    "adapter_weights_sha256": {
        "kind": "sha256",
        "completion_evidence": "adapter weight-file digest",
        "content_opened_by_this_check": False,
    },
    "adapter_config_sha256": {
        "kind": "sha256",
        "completion_evidence": "adapter configuration digest",
        "content_opened_by_this_check": False,
    },
    "device": {
        "kind": "cpu_or_cuda_device",
        "completion_evidence": "literal cpu, cuda, or cuda:N device",
        "content_opened_by_this_check": False,
    },
    "wall_time_ceiling_seconds": {
        "kind": "positive_integer",
        "completion_evidence": "explicit operator ceiling later bounded by signed authority",
        "content_opened_by_this_check": False,
    },
    "cost_ceiling_usd": {
        "kind": "positive_finite_number",
        "completion_evidence": "explicit operator ceiling later bounded by signed authority",
        "content_opened_by_this_check": False,
    },
    "output_root": {
        "kind": "canonical_absolute_new_direct_child",
        "completion_evidence": (
            "nonexistent direct child of the dedicated empirical-capture parent"
        ),
        "content_opened_by_this_check": False,
    },
}


def _canonical_absolute(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"binding readiness {field} is absent")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"binding readiness {field} is not absolute")
    resolved = path.resolve()
    if value != str(resolved):
        raise ValueError(f"binding readiness {field} is not canonical")
    return resolved


def _existing_ancestor_has_reparse_point(path: Path, *, stop: Path) -> bool:
    current = path
    boundary = stop.resolve()
    while current.is_relative_to(boundary):
        if current.exists():
            if current.is_symlink():
                return True
            is_junction = getattr(current, "is_junction", None)
            if callable(is_junction) and is_junction():
                return True
        if current == boundary:
            break
        current = current.parent
    return False


def audit_completed_bindings(
    *,
    project_root: Path,
    bindings: dict[str, Any],
    output_root_exists: bool | None = None,
) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    preflight = build_preflight(project_root=project_root, bindings=bindings)
    if preflight["missing_required_bindings"]:
        raise ValueError("binding readiness requires all ten execution bindings")

    adapter_path = _canonical_absolute(bindings["adapter_path"], field="adapter_path")
    if DEVICE_RE.fullmatch(str(bindings["device"])) is None:
        raise ValueError("binding readiness device is not cpu, cuda, or cuda:N")

    output_root = _canonical_absolute(bindings["output_root"], field="output_root")
    allowed_parent = (project_root / OUTPUT_PARENT).resolve()
    if output_root.parent != allowed_parent:
        raise ValueError(
            "binding readiness output root is not a direct child of the dedicated parent"
        )
    if RUN_NAME_RE.fullmatch(output_root.name) is None:
        raise ValueError("binding readiness output run name is invalid")
    observed_exists = output_root.exists() if output_root_exists is None else output_root_exists
    if observed_exists:
        raise ValueError("binding readiness output root already exists")
    if _existing_ancestor_has_reparse_point(output_root.parent, stop=project_root):
        raise ValueError("binding readiness output path crosses a reparse point")

    after = MODEL_FRAMEWORKS & set(sys.modules)
    if before != after:
        raise ValueError("binding readiness imported a model framework")
    return {
        "schema": "sch.vpd-cluster-to-sae-completed-binding-audit.v1",
        "status": "pass_identity_syntax_and_output_isolation_only",
        "preflight_id": preflight["preflight_id"],
        "binding_fields": list(REQUIRED_EXECUTION_BINDINGS),
        "bindings_complete": True,
        "adapter_path_canonical": str(adapter_path) == bindings["adapter_path"],
        "adapter_content_opened": False,
        "model_content_opened": False,
        "device_syntax_valid": True,
        "output_root": str(output_root),
        "output_root_parent": str(allowed_parent),
        "output_root_nonexistent": True,
        "output_root_direct_child": True,
        "output_path_reparse_point_free": True,
        "model_framework_imported": False,
        "authorization_receipt_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def build_readiness(
    *, project_root: Path, bindings: dict[str, Any] | None = None
) -> dict[str, Any]:
    supplied = dict(bindings or {})
    preflight = build_preflight(project_root=project_root, bindings=supplied)
    missing = preflight["missing_required_bindings"]
    completed_audit = None
    if not missing:
        completed_audit = audit_completed_bindings(
            project_root=project_root, bindings=supplied
        )
    rows = [
        {
            "field": field,
            "value": supplied.get(field),
            "status": "missing" if field in missing else "supplied_unopened",
            **BINDING_CONTRACTS[field],
        }
        for field in REQUIRED_EXECUTION_BINDINGS
    ]
    body = {
        "schema": SCHEMA,
        "status": (
            "blocked_missing_execution_bindings"
            if missing
            else "identity_complete_output_isolated_awaiting_signed_authorization"
        ),
        "source": {
            "preflight_v2": {
                "path": "artifacts/vpd_cluster_to_sae_empirical_preflight_v2/preflight.json",
                "template_id": json.loads(
                    (
                        project_root
                        / "artifacts/vpd_cluster_to_sae_empirical_preflight_v2/preflight.json"
                    ).read_text(encoding="utf-8")
                )["preflight_id"],
                "bound_preflight_id": preflight["preflight_id"],
                "builder_sha256": sha256_file(
                    project_root
                    / "experiments/authz_mechinterp/build_vpd_cluster_to_sae_empirical_preflight_v2.py"
                ),
            },
            "readiness_builder_sha256": sha256_file(Path(__file__)),
        },
        "binding_rows": rows,
        "summary": {
            "required_bindings": len(REQUIRED_EXECUTION_BINDINGS),
            "supplied_bindings": len(REQUIRED_EXECUTION_BINDINGS) - len(missing),
            "missing_bindings": missing,
            "identity_syntax_complete": not missing,
            "model_or_adapter_content_verified": False,
            "output_isolation_verified": completed_audit is not None,
        },
        "completed_binding_audit": completed_audit,
        "runtime_isolation_contract": {
            "allowed_output_parent": str((project_root / OUTPUT_PARENT).resolve()),
            "output_root_must_be_canonical_absolute": True,
            "output_root_must_be_new_direct_child": True,
            "output_root_must_not_preexist": True,
            "existing_reparse_points_forbidden": True,
            "runtime_must_create_output_root_atomically": True,
            "runtime_must_refuse_overwrite_or_resume": True,
            "writes_outside_output_root_forbidden": True,
            "model_or_adapter_files_opened_by_readiness_builder": False,
        },
        "authorization_join": {
            "preflight_path": None,
            "authority_registry_path": None,
            "signed_receipt_path": None,
            "verification_as_of_utc": None,
            "status": "not_attempted_real_inputs_absent",
            "signature_validation_is_necessary_but_not_sufficient": True,
            "runtime_identity_and_output_checks_are_separate": True,
        },
        "resource_ceiling": preflight["resource_ceiling"],
        "stop_conditions": [
            "any_binding_missing_or_noncanonical",
            "model_or_adapter_digest_not_independently_verified_at_runtime",
            "output_root_exists_is_outside_dedicated_parent_or_crosses_reparse_point",
            "signed_authorization_receipt_absent_invalid_inactive_revoked_or_consumed",
            "authorization_scope_or_resource_ceiling_differs",
            "runtime_attempts_generation_training_final_evaluation_or_real_actions",
        ],
        "scientific_boundary": {
            "worksheet_only": True,
            "model_or_adapter_content_opened": False,
            "model_framework_imported": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "authorization_receipt_present": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "final_evaluation_authorized": False,
            "real_actions_authorized": False,
        },
    }
    return {**body, "readiness_id": content_sha256(body)}


def verify_readiness(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = str(observed.get("readiness_id", ""))
    body = {key: value for key, value in observed.items() if key != "readiness_id"}
    bindings = {
        row["field"]: row["value"]
        for row in observed["binding_rows"]
        if row["value"] is not None
    }
    if identifier != content_sha256(body):
        raise ValueError("cluster-to-SAE binding readiness address differs")
    if observed != build_readiness(project_root=project_root, bindings=bindings):
        raise ValueError("cluster-to-SAE binding readiness is not reproducible")
    return {
        "schema": "sch.vpd-cluster-to-sae-binding-readiness-verification.v1",
        "status": "pass",
        "readiness_id": identifier,
        "required_bindings": observed["summary"]["required_bindings"],
        "missing_bindings": len(observed["summary"]["missing_bindings"]),
        "output_isolation_verified": observed["summary"]["output_isolation_verified"],
        "model_or_adapter_content_opened": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the non-authorizing cluster-to-SAE binding worksheet."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--bindings", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        bindings = (
            json.loads(args.bindings.read_text(encoding="utf-8"))
            if args.bindings is not None
            else {}
        )
        result = build_readiness(project_root=project_root, bindings=bindings)
        write_json(args.output, result)
    else:
        if args.bindings is not None:
            raise ValueError("--bindings is only valid with --output")
        result = verify_readiness(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
