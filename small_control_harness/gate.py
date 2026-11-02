from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from uuid import uuid4

from .crypto import stable_hash, verify_grant_signature
from .models import ActionProposal, AuthorizationGrant, RunContract, ScopedAction, parse_time, utc_now_iso


ReasonCode = str
InvariantCheck = Callable[[RunContract, ActionProposal, Mapping[str, Any]], tuple[bool, str | None]]


@dataclass
class GrantBudget:
    calls_used: int = 0
    notional_used: float = 0.0
    fee_used: float = 0.0
    compute_seconds_used: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "calls_used": self.calls_used,
            "notional_used": self.notional_used,
            "fee_used": self.fee_used,
            "compute_seconds_used": self.compute_seconds_used,
        }


@dataclass
class GateState:
    used_nonces: set[str] = field(default_factory=set)
    revoked_grant_ids: set[str] = field(default_factory=set)
    budgets: dict[str, GrantBudget] = field(default_factory=dict)

    def budget_for(self, grant_id: str) -> GrantBudget:
        return self.budgets.setdefault(grant_id, GrantBudget())


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason_codes: tuple[ReasonCode, ...]
    checks: Mapping[str, bool]
    budget_before: Mapping[str, Any]
    budget_after: Mapping[str, Any]
    receipt: Mapping[str, Any]


class ContractGate:
    """Deterministic gate for model-proposed consequential actions.

    The reference signer uses HMAC for a local simulated harness. A production host
    should swap the key resolver for a platform-backed signing service while keeping
    the canonical payload and decision semantics stable.
    """

    def __init__(
        self,
        *,
        key_resolver: Callable[[str], bytes | None],
        expected_issuer: str,
        expected_audience: str,
        invariant_checks: tuple[InvariantCheck, ...] = (),
    ) -> None:
        self._key_resolver = key_resolver
        self._expected_issuer = expected_issuer
        self._expected_audience = expected_audience
        self._invariant_checks = invariant_checks

    def evaluate(
        self,
        *,
        contract: RunContract,
        grant: AuthorizationGrant | None,
        proposal: ActionProposal,
        state: GateState,
        world_state: Mapping[str, Any],
        now: datetime | None = None,
    ) -> GateDecision:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        checks["proposal_contract_match"] = proposal.contract_id == contract.contract_id
        checks["contract_subject_match"] = contract.subject == proposal.metadata.get(
            "subject", contract.subject
        )
        checks["contract_active"] = parse_time(contract.created_at) <= now <= parse_time(
            contract.expires_at
        )
        checks["contract_action_match"] = self._action_matches(
            contract.action_scope, proposal
        )
        checks["contract_resource_match"] = self._resource_match(
            contract.resources, proposal
        )

        if grant is None:
            checks.update(
                {
                    "grant_present": False,
                    "grant_id_match": False,
                    "grant_contract_match": False,
                    "grant_subject_match": False,
                    "grant_issuer_match": False,
                    "grant_audience_match": False,
                    "grant_active": False,
                    "grant_signature_valid": False,
                    "grant_action_match": False,
                    "grant_resource_match": False,
                    "grant_nonce_fresh": False,
                    "grant_not_revoked": False,
                }
            )
            budget_before: dict[str, Any] = {}
            budget_after: dict[str, Any] = {}
        else:
            budget = state.budget_for(grant.grant_id)
            budget_before = budget.to_dict()
            checks["grant_present"] = True
            checks["grant_id_match"] = proposal.grant_id == grant.grant_id
            checks["grant_contract_match"] = grant.contract_id == contract.contract_id
            checks["grant_subject_match"] = grant.subject == contract.subject
            checks["grant_issuer_match"] = grant.issuer == self._expected_issuer
            checks["grant_audience_match"] = grant.audience == self._expected_audience
            checks["grant_active"] = parse_time(grant.issued_at) <= now <= parse_time(
                grant.expires_at
            )
            key = self._key_resolver(str(grant.signature.get("key_id", "")))
            checks["grant_signature_valid"] = bool(
                key and verify_grant_signature(grant.to_dict(), key)
            )
            checks["grant_action_match"] = self._action_matches(
                grant.allowed_actions, proposal
            )
            checks["grant_resource_match"] = self._grant_budget_match(
                grant, proposal, budget
            )
            checks["grant_nonce_fresh"] = grant.nonce not in state.used_nonces
            checks["grant_not_revoked"] = grant.grant_id not in state.revoked_grant_ids
            budget_after = budget.to_dict()

        invariant_results: list[bool] = []
        for index, invariant_check in enumerate(self._invariant_checks):
            ok, code = invariant_check(contract, proposal, world_state)
            checks[f"invariant_{index}"] = ok
            invariant_results.append(ok)
            if not ok and code:
                reasons.append(code)
        checks["invariants_hold"] = all(invariant_results) if invariant_results else True

        for name, ok in checks.items():
            if ok:
                continue
            if grant is None and name.startswith("grant_") and name != "grant_present":
                continue
            reasons.append(self._reason_for_check(name))

        allowed = all(checks.values())
        if allowed and grant is not None:
            budget = state.budget_for(grant.grant_id)
            effects = proposal.estimated_effects
            budget.calls_used += 1
            budget.notional_used += float(effects.get("notional", 0.0) or 0.0)
            budget.fee_used += float(effects.get("fee", 0.0) or 0.0)
            budget.compute_seconds_used += float(
                effects.get("compute_seconds", 0.0) or 0.0
            )
            if grant.one_shot:
                state.used_nonces.add(grant.nonce)
            budget_after = budget.to_dict()

        receipt = {
            "version": "sch.decision-receipt.v1",
            "receipt_id": f"receipt-{uuid4().hex}",
            "contract_id": contract.contract_id,
            "proposal_id": proposal.proposal_id,
            "grant_id": grant.grant_id if grant else None,
            "decision": "allow" if allowed else "deny",
            "reason_codes": sorted(set(reasons)),
            "checks": checks,
            "budget_before": budget_before,
            "budget_after": budget_after,
            "executed": False,
            "tool_result_hash": None,
            "state_before_hash": stable_hash(world_state),
            "state_after_hash": None,
            "created_at": utc_now_iso(),
            "metadata": {
                "proposal_hash": stable_hash(proposal.to_dict()),
                "contract_hash": stable_hash(contract.to_dict()),
                "grant_hash": stable_hash(grant.to_dict()) if grant else None,
            },
        }
        return GateDecision(
            allowed=allowed,
            reason_codes=tuple(sorted(set(reasons))),
            checks=checks,
            budget_before=budget_before,
            budget_after=budget_after,
            receipt=receipt,
        )

    @staticmethod
    def _reason_for_check(check: str) -> str:
        mapping = {
            "proposal_contract_match": "PROPOSAL_CONTRACT_MISMATCH",
            "contract_subject_match": "CONTRACT_SUBJECT_MISMATCH",
            "contract_active": "CONTRACT_INACTIVE",
            "contract_action_match": "CONTRACT_SCOPE_MISMATCH",
            "contract_resource_match": "CONTRACT_RESOURCE_EXCEEDED",
            "grant_present": "GRANT_MISSING",
            "grant_id_match": "GRANT_ID_MISMATCH",
            "grant_contract_match": "GRANT_CONTRACT_MISMATCH",
            "grant_subject_match": "GRANT_SUBJECT_MISMATCH",
            "grant_issuer_match": "GRANT_ISSUER_UNTRUSTED",
            "grant_audience_match": "GRANT_AUDIENCE_MISMATCH",
            "grant_active": "GRANT_INACTIVE",
            "grant_signature_valid": "GRANT_SIGNATURE_INVALID",
            "grant_action_match": "GRANT_SCOPE_MISMATCH",
            "grant_resource_match": "GRANT_BUDGET_EXCEEDED",
            "grant_nonce_fresh": "GRANT_REPLAYED",
            "grant_not_revoked": "GRANT_REVOKED",
            "invariants_hold": "CONTRACT_INVARIANT_FAILED",
        }
        return mapping.get(check, f"CHECK_FAILED:{check}")

    @classmethod
    def _action_matches(
        cls, allowed_actions: tuple[ScopedAction, ...], proposal: ActionProposal
    ) -> bool:
        return any(
            action.tool == proposal.tool
            and action.verb == proposal.verb
            and cls._constraints_match(action.constraints, proposal)
            for action in allowed_actions
        )

    @staticmethod
    def _constraints_match(
        constraints: Mapping[str, Any], proposal: ActionProposal
    ) -> bool:
        args = proposal.args
        effects = proposal.estimated_effects
        for key, expected in constraints.items():
            if key.startswith("max_"):
                field_name = key.removeprefix("max_")
                observed = effects.get(field_name, args.get(field_name))
                if observed is None or float(observed) > float(expected):
                    return False
            elif key.startswith("min_"):
                field_name = key.removeprefix("min_")
                observed = effects.get(field_name, args.get(field_name))
                if observed is None or float(observed) < float(expected):
                    return False
            elif key.endswith("_allowlist"):
                field_name = key.removesuffix("_allowlist")
                observed = effects.get(field_name, args.get(field_name))
                if observed not in expected:
                    return False
            else:
                observed = args.get(key, effects.get(key))
                if isinstance(expected, list):
                    if observed not in expected:
                        return False
                elif observed != expected:
                    return False
        return True

    @classmethod
    def _resource_match(
        cls, resources: Mapping[str, Any], proposal: ActionProposal
    ) -> bool:
        effects = proposal.estimated_effects
        comparisons = {
            "max_notional": "notional",
            "max_fee": "fee",
            "max_compute_seconds": "compute_seconds",
        }
        for limit_name, effect_name in comparisons.items():
            if limit_name in resources:
                if float(effects.get(effect_name, 0.0) or 0.0) > float(
                    resources[limit_name]
                ):
                    return False
        network_allowlist = resources.get("network_allowlist")
        if network_allowlist and effects.get("network") not in network_allowlist:
            return False
        return True

    @classmethod
    def _grant_budget_match(
        cls,
        grant: AuthorizationGrant,
        proposal: ActionProposal,
        budget: GrantBudget,
    ) -> bool:
        limits = grant.limits
        effects = proposal.estimated_effects
        if budget.calls_used + 1 > int(limits.get("max_calls", 0)):
            return False
        projected = {
            "max_notional": budget.notional_used
            + float(effects.get("notional", 0.0) or 0.0),
            "max_fee": budget.fee_used + float(effects.get("fee", 0.0) or 0.0),
            "max_compute_seconds": budget.compute_seconds_used
            + float(effects.get("compute_seconds", 0.0) or 0.0),
        }
        for key, observed in projected.items():
            if key in limits and observed > float(limits[key]):
                return False
        return True
