from __future__ import annotations

import copy
import hashlib
import hmac
import json
from typing import Any, Mapping


def _normalize_numbers(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not value.is_integer():
            return value
        return int(value)
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_numbers(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_numbers(item) for key, item in value.items()}
    return value


def canonical_json(value: Any) -> bytes:
    """Canonical JSON shared with the TypeScript demo signer.

    Integral floats are normalized to integers to match JSON.stringify. Keep signed
    authorization fields to ordinary finite decimal magnitudes; a deployed protocol
    should adopt a full RFC 8785 implementation and asymmetric signatures.
    """
    return json.dumps(
        _normalize_numbers(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def grant_signing_payload(grant: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(grant))
    signature = dict(payload.get("signature", {}))
    signature["value"] = ""
    payload["signature"] = signature
    return payload


def sign_grant(grant: Mapping[str, Any], secret: bytes) -> str:
    return hmac.new(
        secret,
        canonical_json(grant_signing_payload(grant)),
        hashlib.sha256,
    ).hexdigest()


def verify_grant_signature(grant: Mapping[str, Any], secret: bytes) -> bool:
    signature = grant.get("signature", {})
    if signature.get("alg") != "HS256":
        return False
    supplied = str(signature.get("value", ""))
    expected = sign_grant(grant, secret)
    return hmac.compare_digest(supplied, expected)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
