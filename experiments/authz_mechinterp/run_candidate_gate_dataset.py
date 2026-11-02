from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from small_control_harness.crypto import stable_hash
from small_control_harness.gate import ContractGate, GateState
from small_control_harness.models import (
    ActionProposal,
    AuthorizationGrant,
    RunContract,
    parse_time,
)


LOCAL_SECRET = b"local-development-key"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate-only authorization fixtures without an execution path."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate = ContractGate(
        key_resolver=lambda key_id: LOCAL_SECRET if key_id == "local-dev-v1" else None,
        expected_issuer="host:user-approval-service",
        expected_audience="host:bitagent-executor",
        invariant_checks=(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for episode in iter_jsonl(args.input):
            contract = RunContract.from_dict(episode["contract"])
            grant = (
                AuthorizationGrant.from_dict(episode["host_grant"])
                if episode.get("host_grant")
                else None
            )
            proposal = ActionProposal.from_dict(episode["proposal_template"])
            state = GateState()
            if episode.get("replay_preconsume_nonce") and grant is not None:
                state.used_nonces.add(grant.nonce)
            if episode.get("revoke_preconsume_grant") and grant is not None:
                state.revoked_grant_ids.add(grant.grant_id)
            world_state = {
                "mode": "candidate_only",
                "effect": "none",
                "execution_tools_exposed": False,
            }
            decision = gate.evaluate(
                contract=contract,
                grant=grant,
                proposal=proposal,
                state=state,
                world_state=world_state,
                now=parse_time(proposal.created_at),
            )
            receipt = dict(decision.receipt)
            receipt["state_after_hash"] = stable_hash(world_state)
            receipt["metadata"] = {
                **dict(receipt.get("metadata", {})),
                "candidate_only": True,
                "effect": "none",
                "execution_tools_exposed": False,
            }
            observed = str(receipt["decision"])
            expected = str(episode["expected_gate_decision"])
            if observed != expected:
                raise RuntimeError(
                    f"fixture mismatch for {episode['episode_id']}: "
                    f"expected {expected}, got {observed}; "
                    f"reasons={receipt['reason_codes']}"
                )
            if receipt.get("executed") is not False:
                raise RuntimeError("candidate-only gate receipt must record executed=false")
            enriched = dict(episode)
            enriched.update(
                {
                    "gate_receipt": receipt,
                    "observed_gate_decision": observed,
                    "actual_execution": False,
                    "execution_path_exposed": False,
                }
            )
            out.write(json.dumps(enriched, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
