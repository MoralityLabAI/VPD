from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from small_control_harness.environment import BitAgentContractEnvironment
from small_control_harness.gate import ContractGate
from small_control_harness.invariants import require_simulation_receipt
from small_control_harness.models import ActionProposal, AuthorizationGrant, RunContract, parse_time


LOCAL_SECRET = b"local-development-key"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate = ContractGate(
        key_resolver=lambda key_id: LOCAL_SECRET if key_id == "local-dev-v1" else None,
        expected_issuer="host:user-approval-service",
        expected_audience="host:bitagent-executor",
        invariant_checks=(require_simulation_receipt,),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for episode in iter_jsonl(args.input):
            env = BitAgentContractEnvironment(gate)
            contract = RunContract.from_dict(episode["contract"])
            grant = (
                AuthorizationGrant.from_dict(episode["host_grant"])
                if episode.get("host_grant")
                else None
            )
            proposal = ActionProposal.from_dict(episode["proposal_template"])
            if episode.get("replay_preconsume_nonce") and grant is not None:
                env.gate_state.used_nonces.add(grant.nonce)
            if episode.get("revoke_preconsume_grant") and grant is not None:
                env.gate_state.revoked_grant_ids.add(grant.grant_id)
            decision = env.step(
                contract=contract,
                grant=grant,
                proposal=proposal,
                now=parse_time(proposal.created_at),
            )
            receipt = dict(decision.receipt)
            expected = episode["expected_gate_decision"]
            observed = receipt["decision"]
            if observed != expected:
                raise RuntimeError(
                    f"fixture mismatch for {episode['episode_id']}: expected {expected}, got {observed}"
                )
            enriched = dict(episode)
            enriched["gate_receipt"] = receipt
            enriched["messages_post_gate"] = [
                *episode["messages"],
                {
                    "role": "tool",
                    "content": (
                        f"<GATE_RESULT>{json.dumps(receipt, sort_keys=True)}"
                        f"</GATE_RESULT>"
                    ),
                },
            ]
            enriched["observed_gate_decision"] = observed
            enriched["actual_execution"] = bool(receipt["executed"])
            out.write(json.dumps(enriched, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
