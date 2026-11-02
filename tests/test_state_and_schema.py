from __future__ import annotations

import unittest
from datetime import datetime, timezone

from small_control_harness.fixtures import (
    make_contract,
    make_proposal,
    make_unsigned_grant,
    sign_fixture_grant,
)
from small_control_harness.environment import BitAgentContractEnvironment
from small_control_harness.gate import ContractGate
from small_control_harness.invariants import require_simulation_receipt
from small_control_harness.models import ActionProposal, AuthorizationGrant, RunContract
from small_control_harness.state_machine import RunLifecycle, RunPhase
from small_control_harness.validate import validate_document


NOW = datetime(2026, 8, 8, 20, 0, tzinfo=timezone.utc)
SECRET = b"local-development-key"


class StateAndSchemaTests(unittest.TestCase):
    def test_fixture_documents_validate(self) -> None:
        contract = make_contract(now=NOW)
        grant = sign_fixture_grant(make_unsigned_grant(contract=contract, now=NOW), SECRET)
        proposal = make_proposal(contract=contract, grant=grant, now=NOW)
        validate_document(contract)
        validate_document(grant)
        validate_document(proposal)
        gate = ContractGate(
            key_resolver=lambda key_id: SECRET if key_id == "local-dev-v1" else None,
            expected_issuer="host:user-approval-service",
            expected_audience="host:bitagent-executor",
            invariant_checks=(require_simulation_receipt,),
        )
        environment = BitAgentContractEnvironment(gate)
        decision = environment.step(
            contract=RunContract.from_dict(contract),
            grant=AuthorizationGrant.from_dict(grant),
            proposal=ActionProposal.from_dict(proposal),
            now=NOW,
        )
        validate_document(decision.receipt)

    def test_lifecycle_requires_evidence_and_valid_transition(self) -> None:
        lifecycle = RunLifecycle()
        lifecycle.transition(RunPhase.SIMULATED, evidence_id="simulation-receipt-1")
        lifecycle.transition(RunPhase.AUTHORIZED, evidence_id="grant-1")
        lifecycle.transition(RunPhase.PROPOSED, evidence_id="proposal-1")
        lifecycle.transition(RunPhase.GATE_ALLOWED, evidence_id="decision-1")
        lifecycle.transition(RunPhase.EXECUTED, evidence_id="tool-result-1")
        lifecycle.transition(RunPhase.VERIFIED, evidence_id="verification-1")
        self.assertEqual(lifecycle.phase, RunPhase.VERIFIED)
        with self.assertRaises(ValueError):
            lifecycle.transition(RunPhase.PROPOSED, evidence_id="late-proposal")


if __name__ == "__main__":
    unittest.main()
