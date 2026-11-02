from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from .crypto import stable_hash
from .gate import ContractGate, GateDecision, GateState
from .ledger import SimulatedLedger
from .models import ActionProposal, AuthorizationGrant, RunContract, utc_now_iso


class BitAgentContractEnvironment:
    """Simulated consequential-action environment with external commit authority."""

    def __init__(self, gate: ContractGate, ledger: SimulatedLedger | None = None) -> None:
        self.gate = gate
        self.ledger = ledger or SimulatedLedger()
        self.gate_state = GateState()
        self.receipts: list[dict[str, Any]] = []

    def step(
        self,
        *,
        contract: RunContract,
        grant: AuthorizationGrant | None,
        proposal: ActionProposal,
        now: datetime | None = None,
    ) -> GateDecision:
        before = self.ledger.snapshot()
        decision = self.gate.evaluate(
            contract=contract,
            grant=grant,
            proposal=proposal,
            state=self.gate_state,
            world_state=before,
            now=now,
        )
        receipt = deepcopy(dict(decision.receipt))
        if decision.allowed:
            result = self.ledger.execute(proposal.tool, proposal.verb, proposal.args)
            after = self.ledger.snapshot()
            receipt.update(
                {
                    "executed": True,
                    "tool_result_hash": stable_hash(result),
                    "state_after_hash": stable_hash(after),
                    "metadata": {
                        **dict(receipt.get("metadata", {})),
                        "tool_result": result,
                    },
                    "created_at": utc_now_iso(),
                }
            )
        else:
            receipt["state_after_hash"] = stable_hash(before)
        self.receipts.append(receipt)
        return GateDecision(
            allowed=decision.allowed,
            reason_codes=decision.reason_codes,
            checks=decision.checks,
            budget_before=decision.budget_before,
            budget_after=decision.budget_after,
            receipt=receipt,
        )
