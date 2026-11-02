"""Deterministic authorization activation-category contract.

Categories are model-visible authorization states, not model outcomes.  They
partition effect-capture episodes so downstream clustering can preserve which
kind of observable state produced each activation effect.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


CATEGORY_PROTOCOLS = {
    "valid_authority": "valid_authority_retention",
    "missing_or_untrusted_authority": "provenance_rejection",
    "signature_invalid": "cryptographic_validity_rejection",
    "time_inactive": "temporal_validity_rejection",
    "revoked": "revocation_state_rejection",
    "replayed": "nonce_freshness_rejection",
    "scope_invalid": "scope_and_resource_rejection",
    "other_invalid": "conservative_request_rejection",
}

COMPONENT_COMPARTMENT_PROTOCOLS = {
    **CATEGORY_PROTOCOLS,
    "shared_cross_category": "balanced_cross_category_dictionary",
}

EXACT_CONDITION_CATEGORIES = {
    "valid_visible": "valid_authority",
    "none": "missing_or_untrusted_authority",
    "user_pasted_valid": "missing_or_untrusted_authority",
    "tool_claim_valid": "missing_or_untrusted_authority",
    "assistant_claim_valid": "missing_or_untrusted_authority",
    "prose_only": "missing_or_untrusted_authority",
    "structured_user_paste": "missing_or_untrusted_authority",
    "tool_delegation_claim": "missing_or_untrusted_authority",
    "self_authorized_claim": "missing_or_untrusted_authority",
    "valid_hidden": "missing_or_untrusted_authority",
    "compacted_no_grant": "missing_or_untrusted_authority",
    "compacted_valid_hidden": "missing_or_untrusted_authority",
    "forged_host_visible": "signature_invalid",
    "forged_signature": "signature_invalid",
    "expired_host_visible": "time_inactive",
    "expired_visible": "time_inactive",
    "revoked_host_visible": "revoked",
    "revoked_after_visible": "revoked",
    "replayed_host_visible": "replayed",
    "one_shot_replay_probe": "replayed",
    "scope_mismatch_visible": "scope_invalid",
    "scope_overflow": "scope_invalid",
}


def activation_category(row: dict[str, Any]) -> str:
    explicit = row.get("activation_category")
    if explicit is not None:
        category = str(explicit)
        if category not in CATEGORY_PROTOCOLS:
            raise ValueError(f"unknown explicit activation category: {category}")
        return category
    condition = str(row.get("condition", ""))
    known = EXACT_CONDITION_CATEGORIES.get(condition)
    if known is not None:
        return known
    lowered = condition.lower()
    pattern_categories = (
        (("forg", "signature"), "signature_invalid"),
        (("expir", "inactive", "time"), "time_inactive"),
        (("revok",), "revoked"),
        (("replay", "nonce", "consumed"), "replayed"),
        (("scope", "resource", "budget"), "scope_invalid"),
        (("missing", "hidden", "untrusted", "paste", "claim"), "missing_or_untrusted_authority"),
    )
    for needles, category in pattern_categories:
        if any(needle in lowered for needle in needles):
            return category
    if str(row.get("expected_gate_decision", "deny")) == "allow":
        return "valid_authority"
    return "other_invalid"


def categorize_episodes(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[activation_category(row)].append(row)
    return {
        category: sorted(values, key=lambda row: str(row["episode_id"]))
        for category, values in sorted(grouped.items())
    }


def category_contract() -> dict[str, Any]:
    return {
        "schema": "sch.vpd-activation-category-contract.v1",
        "basis": "model_visible_authorization_state_not_model_outcome",
        "categories": [
            {"id": category, "sae_protocol": protocol}
            for category, protocol in CATEGORY_PROTOCOLS.items()
        ],
        "unknown_condition_rule": "allow_maps_to_valid_authority_else_other_invalid",
        "explicit_category_must_be_known": True,
    }
