from __future__ import annotations

import unittest
from copy import deepcopy
from datetime import datetime, timezone

from small_control_harness.environment import BitAgentContractEnvironment
from small_control_harness.fixtures import (
    make_contract,
    make_proposal,
    make_unsigned_grant,
    sign_fixture_grant,
)
from small_control_harness.gate import ContractGate
from small_control_harness.invariants import require_simulation_receipt
from small_control_harness.models import ActionProposal, AuthorizationGrant, RunContract
from small_control_harness.perturbations import corrupt_signature, expire_grant, widen_proposal_scope


SECRET = b"local-development-key"
NOW = datetime(2026, 8, 8, 20, 0, tzinfo=timezone.utc)


def make_env() -> BitAgentContractEnvironment:
    gate = ContractGate(
        key_resolver=lambda key_id: SECRET if key_id == "local-dev-v1" else None,
        expected_issuer="host:user-approval-service",
        expected_audience="host:bitagent-executor",
        invariant_checks=(require_simulation_receipt,),
    )
    return BitAgentContractEnvironment(gate)


def fixture() -> tuple[dict, dict, dict]:
    contract = make_contract(now=NOW)
    grant = sign_fixture_grant(make_unsigned_grant(contract=contract, now=NOW), SECRET)
    proposal = make_proposal(contract=contract, grant=grant, now=NOW)
    return contract, grant, proposal


class ContractGateTests(unittest.TestCase):
    def run_case(self, contract: dict, grant: dict | None, proposal: dict):
        env = make_env()
        return env.step(
            contract=RunContract.from_dict(contract),
            grant=AuthorizationGrant.from_dict(grant) if grant else None,
            proposal=ActionProposal.from_dict(proposal),
            now=NOW,
        )

    def test_valid_exact_grant_executes(self) -> None:
        contract, grant, proposal = fixture()
        decision = self.run_case(contract, grant, proposal)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.receipt["executed"])
        self.assertEqual(decision.receipt["decision"], "allow")

    def test_missing_grant_denies(self) -> None:
        contract, _grant, proposal = fixture()
        proposal["grant_id"] = None
        decision = self.run_case(contract, None, proposal)
        self.assertFalse(decision.allowed)
        self.assertIn("GRANT_MISSING", decision.reason_codes)

    def test_forged_signature_denies(self) -> None:
        contract, grant, proposal = fixture()
        forged = corrupt_signature(grant)
        proposal["grant_id"] = forged["grant_id"]
        decision = self.run_case(contract, forged, proposal)
        self.assertFalse(decision.allowed)
        self.assertIn("GRANT_SIGNATURE_INVALID", decision.reason_codes)

    def test_expired_grant_denies(self) -> None:
        contract, grant, proposal = fixture()
        expired = expire_grant(grant, now=NOW, secret=SECRET)
        proposal["grant_id"] = expired["grant_id"]
        env = make_env()
        decision = env.gate.evaluate(
            contract=RunContract.from_dict(contract),
            grant=AuthorizationGrant.from_dict(expired),
            proposal=ActionProposal.from_dict(proposal),
            state=env.gate_state,
            world_state=env.ledger.snapshot(),
            now=NOW,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("GRANT_INACTIVE", decision.reason_codes)

    def test_scope_overflow_denies(self) -> None:
        contract, grant, proposal = fixture()
        widened = widen_proposal_scope(proposal, amount=75.0)
        decision = self.run_case(contract, grant, widened)
        self.assertFalse(decision.allowed)
        self.assertTrue(
            {"CONTRACT_SCOPE_MISMATCH", "GRANT_SCOPE_MISMATCH", "GRANT_BUDGET_EXCEEDED"}
            & set(decision.reason_codes)
        )

    def test_one_shot_replay_denies_second_use(self) -> None:
        contract, grant, proposal = fixture()
        env = make_env()
        parsed_contract = RunContract.from_dict(contract)
        parsed_grant = AuthorizationGrant.from_dict(grant)
        first = env.step(
            contract=parsed_contract,
            grant=parsed_grant,
            proposal=ActionProposal.from_dict(proposal),
            now=NOW,
        )
        self.assertTrue(first.allowed)
        second_proposal = deepcopy(proposal)
        second_proposal["proposal_id"] += "-second"
        second = env.gate.evaluate(
            contract=parsed_contract,
            grant=parsed_grant,
            proposal=ActionProposal.from_dict(second_proposal),
            state=env.gate_state,
            world_state=env.ledger.snapshot(),
            now=NOW,
        )
        self.assertFalse(second.allowed)
        self.assertIn("GRANT_REPLAYED", second.reason_codes)

    def test_missing_simulation_receipt_denies(self) -> None:
        contract, grant, proposal = fixture()
        proposal["metadata"].pop("simulation_receipt_hash")
        decision = self.run_case(contract, grant, proposal)
        self.assertFalse(decision.allowed)
        self.assertIn("SIMULATION_RECEIPT_MISSING", decision.reason_codes)

    def test_revoked_grant_denies(self) -> None:
        contract, grant, proposal = fixture()
        env = make_env()
        env.gate_state.revoked_grant_ids.add(grant["grant_id"])
        decision = env.gate.evaluate(
            contract=RunContract.from_dict(contract),
            grant=AuthorizationGrant.from_dict(grant),
            proposal=ActionProposal.from_dict(proposal),
            state=env.gate_state,
            world_state=env.ledger.snapshot(),
            now=NOW,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("GRANT_REVOKED", decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
