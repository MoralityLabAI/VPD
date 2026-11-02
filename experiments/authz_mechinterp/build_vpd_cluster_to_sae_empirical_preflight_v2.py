from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp import (
    build_vpd_cluster_to_sae_empirical_preflight_v1 as v1,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-cluster-to-sae-empirical-preflight.v2"
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_EXECUTION_BINDINGS = (
    "model_id",
    "model_revision",
    "model_files_manifest_sha256",
    "adapter_path",
    "adapter_weights_sha256",
    "adapter_config_sha256",
    "device",
    "wall_time_ceiling_seconds",
    "cost_ceiling_usd",
    "output_root",
)
DEFERRED_AUTHORIZATION_FIELDS = (
    "authorization_receipt_id",
    "authorization_receipt_sha256",
    "explicit_user_authorization_statement",
)


def _validate_execution_bindings(bindings: dict[str, Any]) -> list[str]:
    unknown = sorted(set(bindings) - set(REQUIRED_EXECUTION_BINDINGS))
    if unknown:
        raise ValueError(
            "empirical preflight V2 has unknown bindings: " + ", ".join(unknown)
        )
    missing: list[str] = []
    for field in REQUIRED_EXECUTION_BINDINGS:
        value = bindings.get(field)
        if value is None or value == "":
            missing.append(field)
            continue
        if field.endswith("_sha256"):
            if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
                raise ValueError(f"empirical preflight V2 binding {field} is not a digest")
        elif field == "wall_time_ceiling_seconds":
            if type(value) is not int or value <= 0:
                raise ValueError("empirical preflight V2 wall-time ceiling is invalid")
        elif field == "cost_ceiling_usd":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("empirical preflight V2 cost ceiling is invalid")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"empirical preflight V2 binding {field} is invalid")
    return missing


def build_preflight(
    *, project_root: Path, bindings: dict[str, Any] | None = None
) -> dict[str, Any]:
    supplied = dict(bindings or {})
    missing = _validate_execution_bindings(supplied)
    predecessor_bindings = {
        **supplied,
        "authorization_receipt_id": "0" * 64,
        "authorization_receipt_sha256": "0" * 64,
        "explicit_user_authorization_statement": (
            "deferred to one-way signed authorization receipt in preflight V2"
        ),
    }
    predecessor = v1.build_preflight(
        project_root=project_root,
        bindings=predecessor_bindings,
    )
    result = json.loads(json.dumps(predecessor))
    result.pop("preflight_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "blocked_missing_execution_identity_bindings"
        if missing
        else "identity_complete_awaiting_signed_authorization_receipt"
    )
    result["predecessor"] = {
        "schema": v1.SCHEMA,
        "path": "artifacts/vpd_cluster_to_sae_empirical_preflight_v1/preflight.json",
        "source_sha256": sha256_file(Path(v1.__file__)),
        "gap": (
            "V1 required the future authorization receipt ID and hash inside the "
            "preflight that receipt must sign, creating a content-address cycle"
        ),
        "execution_authorized": False,
    }
    result["immutable_bindings"] = {
        field: supplied.get(field) for field in REQUIRED_EXECUTION_BINDINGS
    }
    result["missing_required_bindings"] = missing
    result["authorization_handoff"] = {
        "status": "absent_pending_separate_signed_receipt",
        "receipt_schema": (
            "sch.vpd-cluster-to-sae-capture-authorization-receipt.v1"
        ),
        "authority_registry_schema": (
            "sch.vpd-cluster-to-sae-capture-authority-registry.v1"
        ),
        "receipt_must_bind_preflight_id": True,
        "receipt_must_bind_preflight_digest": True,
        "receipt_id_is_not_an_input_to_this_preflight": True,
        "receipt_digest_is_not_an_input_to_this_preflight": True,
        "explicit_authorization_statement_belongs_in_receipt": True,
        "signature_alone_does_not_authorize_runtime": True,
    }
    result["stop_conditions"] = [
        condition
        for condition in result["stop_conditions"]
        if condition != "any_required_binding_is_missing"
    ] + [
        "any_execution_identity_binding_is_missing",
        "signed_authorization_receipt_is_absent_invalid_inactive_revoked_or_consumed",
        "authorization_receipt_scope_or_resource_ceiling_differs",
    ]
    result["scientific_boundary"]["authorization_receipt_present"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["sae_training_authorized"] = False
    return {**result, "preflight_id": content_sha256(result)}


def verify_preflight(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = observed.get("preflight_id")
    body = {key: value for key, value in observed.items() if key != "preflight_id"}
    if identifier != content_sha256(body):
        raise ValueError("cluster-to-SAE preflight V2 address differs")
    bindings = {
        key: value
        for key, value in observed["immutable_bindings"].items()
        if value is not None
    }
    if observed != build_preflight(project_root=project_root, bindings=bindings):
        raise ValueError("cluster-to-SAE preflight V2 is not reproducible")
    return {
        "schema": "sch.vpd-cluster-to-sae-empirical-preflight-verification.v2",
        "status": "pass",
        "preflight_id": identifier,
        "readiness": observed["status"],
        "missing_required_bindings": len(observed["missing_required_bindings"]),
        "authorization_receipt_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify cycle-free cluster-to-SAE empirical preflight V2."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--bindings", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    bindings = (
        json.loads(args.bindings.read_text(encoding="utf-8"))
        if args.bindings is not None
        else {}
    )
    if args.output is not None:
        result = build_preflight(project_root=project_root, bindings=bindings)
        write_json(args.output, result)
    else:
        if args.bindings is not None:
            raise ValueError("--bindings is only valid with --output")
        result = verify_preflight(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
