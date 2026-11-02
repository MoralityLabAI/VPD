from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from .crypto import sign_grant


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_contract(
    *,
    now: datetime,
    subject: str = "agent:bitagent-small",
    max_notional: float = 50.0,
) -> dict[str, Any]:
    return {
        "version": "sch.run-contract.v1",
        "contract_id": f"contract-{uuid4().hex}",
        "subject": subject,
        "objective": {
            "description": "Allocate a bounded amount to the simulated starter strategy.",
            "success_predicates": [
                "receipt.decision == allow",
                "ledger.positions.starter.status == open",
            ],
        },
        "action_scope": [
            {
                "tool": "tradelayer",
                "verb": "allocate_strategy",
                "constraints": {
                    "strategy_id": "starter",
                    "network": "regtest",
                    "max_notional": max_notional,
                },
            }
        ],
        "resources": {
            "max_calls": 1,
            "max_notional": max_notional,
            "max_fee": 1.0,
            "max_compute_seconds": 30.0,
            "network_allowlist": ["regtest"],
            "writable_paths": ["artifacts/bitagent/**"],
        },
        "invariants": [
            "commit authority remains in the host gate",
            "a simulation receipt must precede execution",
            "no wallet withdrawal is allowed",
        ],
        "evidence": [
            {"kind": "receipt", "value": "decision_receipt.json"},
            {"kind": "state_predicate", "value": "positions.starter.status == open"},
        ],
        "created_at": _iso(now - timedelta(minutes=1)),
        "expires_at": _iso(now + timedelta(hours=1)),
        "metadata": {"owner_address": "owner-regtest-001"},
    }


def make_unsigned_grant(
    *,
    contract: dict[str, Any],
    now: datetime,
    max_notional: float = 50.0,
    expires_delta: timedelta = timedelta(minutes=30),
    one_shot: bool = True,
) -> dict[str, Any]:
    return {
        "version": "sch.authorization-grant.v1",
        "grant_id": f"grant-{uuid4().hex}",
        "contract_id": contract["contract_id"],
        "issuer": "host:user-approval-service",
        "subject": contract["subject"],
        "audience": "host:bitagent-executor",
        "allowed_actions": [
            {
                "tool": "tradelayer",
                "verb": "allocate_strategy",
                "constraints": {
                    "strategy_id": "starter",
                    "network": "regtest",
                    "max_notional": max_notional,
                },
            }
        ],
        "limits": {
            "max_calls": 1,
            "max_notional": max_notional,
            "max_fee": 1.0,
            "max_compute_seconds": 30.0,
        },
        "issued_at": _iso(now - timedelta(seconds=5)),
        "expires_at": _iso(now + expires_delta),
        "nonce": f"nonce-{uuid4().hex}",
        "one_shot": one_shot,
        "parent_grant_id": None,
        "signature": {"alg": "HS256", "key_id": "local-dev-v1", "value": ""},
        "metadata": {"confirmation_surface": "typed-confirmation"},
    }


def sign_fixture_grant(grant: dict[str, Any], secret: bytes) -> dict[str, Any]:
    signed = deepcopy(grant)
    signed["signature"]["value"] = sign_grant(signed, secret)
    return signed


def make_proposal(
    *,
    contract: dict[str, Any],
    grant: dict[str, Any] | None,
    now: datetime,
    amount: float = 25.0,
) -> dict[str, Any]:
    return {
        "version": "sch.action-proposal.v1",
        "proposal_id": f"proposal-{uuid4().hex}",
        "contract_id": contract["contract_id"],
        "grant_id": grant["grant_id"] if grant else None,
        "tool": "tradelayer",
        "verb": "allocate_strategy",
        "args": {
            "strategy_id": "starter",
            "asset": "USDT",
            "amount": amount,
            "network": "regtest",
        },
        "estimated_effects": {
            "notional": amount,
            "fee": 0.25,
            "compute_seconds": 2.0,
            "network": "regtest",
        },
        "justification": "The proposal matches the bounded starter-strategy contract.",
        "model_context_hash": None,
        "created_at": _iso(now),
        "metadata": {
            "subject": contract["subject"],
            "simulation_receipt_hash": "simulated-receipt-hash",
        },
    }
