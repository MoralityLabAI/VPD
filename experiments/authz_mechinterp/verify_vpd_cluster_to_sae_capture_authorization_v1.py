from __future__ import annotations

import argparse
import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    canonical_bytes,
    content_sha256,
    sha256_file,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import crypto_types


RECEIPT_SCHEMA = "sch.vpd-cluster-to-sae-capture-authorization-receipt.v1"
REGISTRY_SCHEMA = "sch.vpd-cluster-to-sae-capture-authority-registry.v1"
ACTION = "vpd_four_role_zero_action_effect_capture"
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
CAPTURE_ROLES = ["discovery", "calibration", "cluster_validation", "sae_training"]


def receipt_body(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key not in {"authorization_receipt_id", "signature"}
    }


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"authorization {field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"authorization {field} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"authorization {field} is not UTC")
    return parsed


def _verify_addressed(value: dict[str, Any], *, identifier_field: str, schema: str) -> str:
    if value.get("schema") != schema:
        raise ValueError(f"unsupported {schema} artifact")
    body = dict(value)
    identifier = str(body.pop(identifier_field, ""))
    if DIGEST_RE.fullmatch(identifier) is None or identifier != content_sha256(body):
        raise ValueError(f"{schema} content address differs")
    return identifier


def sign_authorization_receipt(body: dict[str, Any], private_key: Any) -> dict[str, Any]:
    if body.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("authorization receipt body schema differs")
    if "authorization_receipt_id" in body or "signature" in body:
        raise ValueError("authorization receipt body already contains addressed fields")
    identifier = content_sha256(body)
    signature = private_key.sign(canonical_bytes(body))
    return {
        **body,
        "authorization_receipt_id": identifier,
        "signature": {
            "algorithm": "ed25519",
            "value_base64": base64.b64encode(signature).decode("ascii"),
        },
    }


def _required_scope(preflight: dict[str, Any]) -> dict[str, Any]:
    ceiling = preflight["resource_ceiling"]
    output_root = preflight["immutable_bindings"]["output_root"]
    return {
        "authorized_action": ACTION,
        "capture_roles": CAPTURE_ROLES,
        "stage_order": preflight["stage_order"],
        "output_root": output_root,
        "maximum_compute_jobs": ceiling["maximum_compute_jobs"],
        "maximum_batch_forwards_at_batch_size_4": ceiling[
            "maximum_batch_forwards_at_batch_size_4"
        ],
        "maximum_episode_forward_equivalents": ceiling[
            "maximum_episode_forward_equivalents"
        ],
        "maximum_float32_effect_payload_bytes": ceiling[
            "estimated_float32_effect_payload_bytes"
        ],
        "generated_tokens_authorized": False,
        "temporary_adapter_parameter_perturbation_authorized": True,
        "adapter_weight_persistence_authorized": False,
        "base_model_weight_mutation_authorized": False,
        "sae_dictionary_training_authorized": False,
        "final_evaluation_authorized": False,
        "real_actions_authorized": False,
    }


def verify_authorization_receipt(
    *,
    receipt: dict[str, Any],
    registry: dict[str, Any],
    preflight: dict[str, Any],
    as_of: str,
) -> dict[str, Any]:
    preflight_id = _verify_addressed(
        preflight,
        identifier_field="preflight_id",
        schema="sch.vpd-cluster-to-sae-empirical-preflight.v2",
    )
    if preflight.get("missing_required_bindings"):
        raise ValueError("authorization requires an identity-complete preflight")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("unsupported authorization receipt schema")
    receipt_id = str(receipt.get("authorization_receipt_id", ""))
    if (
        DIGEST_RE.fullmatch(receipt_id) is None
        or receipt_id != content_sha256(receipt_body(receipt))
    ):
        raise ValueError("authorization receipt content address differs")
    registry_id = _verify_addressed(
        registry,
        identifier_field="registry_id",
        schema=REGISTRY_SCHEMA,
    )
    synthetic = bool(receipt.get("synthetic_fixture_only"))
    expected_status = "synthetic_fixture_only" if synthetic else "authorized_by_user"
    if receipt.get("status") != expected_status:
        raise ValueError("authorization receipt status differs")
    source = receipt.get("source", {})
    if source.get("preflight_id") != preflight_id:
        raise ValueError("authorization receipt preflight ID differs")
    if source.get("preflight_sha256") != content_sha256(preflight):
        raise ValueError("authorization receipt preflight digest differs")
    if receipt.get("identity_bindings") != preflight["immutable_bindings"]:
        raise ValueError("authorization receipt identity bindings differ")
    if receipt.get("scope") != _required_scope(preflight):
        raise ValueError("authorization receipt scope differs")

    issued = _parse_utc(receipt.get("issued_at"), field="issued_at")
    not_before = _parse_utc(receipt.get("not_before"), field="not_before")
    expires = _parse_utc(receipt.get("expires_at"), field="expires_at")
    observed_time = _parse_utc(as_of, field="as_of")
    if not issued <= not_before < expires:
        raise ValueError("authorization receipt time interval differs")
    if not not_before <= observed_time < expires:
        raise ValueError("authorization receipt is inactive or expired")
    nonce = str(receipt.get("nonce", ""))
    if DIGEST_RE.fullmatch(nonce) is None:
        raise ValueError("authorization receipt nonce is invalid")
    if receipt_id in registry.get("revoked_receipt_ids", []):
        raise ValueError("authorization receipt is revoked")
    if nonce in registry.get("consumed_nonces", []):
        raise ValueError("authorization receipt nonce was already consumed")

    authorities = {
        str(row["authority_id"]): row for row in registry.get("authorities", [])
    }
    authority_id = str(receipt.get("authority_id", ""))
    authority = authorities.get(authority_id)
    if authority is None:
        raise ValueError("authorization receipt authority is not registered")
    if authority.get("enabled") is not True:
        raise ValueError("authorization receipt authority is disabled")
    if authority.get("algorithm") != "ed25519":
        raise ValueError("authorization receipt authority algorithm differs")
    if bool(authority.get("synthetic_fixture_only")) != synthetic:
        raise ValueError("authorization receipt authority environment differs")
    if authority.get("allowed_actions") != [ACTION]:
        raise ValueError("authorization receipt authority action differs")
    authority_ceiling = authority.get("maximum_scope", {})
    scope = receipt["scope"]
    for field in (
        "maximum_compute_jobs",
        "maximum_batch_forwards_at_batch_size_4",
        "maximum_episode_forward_equivalents",
        "maximum_float32_effect_payload_bytes",
    ):
        if type(authority_ceiling.get(field)) is not int or scope[field] > authority_ceiling[field]:
            raise ValueError(f"authorization receipt exceeds authority ceiling: {field}")

    signature = receipt.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        raise ValueError("authorization receipt signature is absent")
    try:
        signature_bytes = base64.b64decode(
            str(signature["value_base64"]), validate=True
        )
        public_bytes = base64.b64decode(
            str(authority["public_key_base64"]), validate=True
        )
        if len(public_bytes) != 32:
            raise ValueError("authorization public key length differs")
        _, Ed25519PublicKey, _, _ = crypto_types()
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(
            signature_bytes, canonical_bytes(receipt_body(receipt))
        )
    except Exception as exc:
        raise ValueError("authorization receipt signature verification failed") from exc

    return {
        "schema": "sch.vpd-cluster-to-sae-capture-authorization-verification.v1",
        "status": (
            "pass_authenticated_synthetic_fixture_not_authorized"
            if synthetic
            else "pass_authenticated_scope_exact_requires_runtime_preflight"
        ),
        "authorization_receipt_id": receipt_id,
        "authority_registry_id": registry_id,
        "authority_id": authority_id,
        "preflight_id": preflight_id,
        "signature_valid": True,
        "scope_exact": True,
        "active_at_as_of": True,
        "nonce_unconsumed": True,
        "synthetic_fixture_only": synthetic,
        "execution_authorized": False,
        "sae_training_authorized": False,
        "final_evaluation_authorized": False,
        "real_actions_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a signed cluster-to-SAE capture authorization receipt."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    result = verify_authorization_receipt(
        receipt=receipt,
        registry=registry,
        preflight=preflight,
        as_of=args.as_of,
    )
    result["receipt_sha256"] = sha256_file(args.receipt)
    result["registry_sha256"] = sha256_file(args.registry)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
