from __future__ import annotations

from typing import Any, Mapping

from .models import ActionProposal, RunContract


def forbid_self_transfer(
    contract: RunContract,
    proposal: ActionProposal,
    world_state: Mapping[str, Any],
) -> tuple[bool, str | None]:
    _ = world_state
    owner = str(contract.metadata.get("owner_address", ""))
    recipient = str(proposal.args.get("recipient", ""))
    if proposal.tool == "wallet" and proposal.verb == "transfer" and recipient == owner:
        return False, "FORBIDDEN_BENEFICIARY"
    return True, None


def require_simulation_receipt(
    contract: RunContract,
    proposal: ActionProposal,
    world_state: Mapping[str, Any],
) -> tuple[bool, str | None]:
    _ = (contract, world_state)
    receipt_hash = proposal.metadata.get("simulation_receipt_hash")
    if proposal.verb in {"allocate_strategy", "transfer"} and not receipt_hash:
        return False, "SIMULATION_RECEIPT_MISSING"
    return True, None
