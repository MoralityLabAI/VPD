import { randomUUID } from "node:crypto";
import type {
  ActionProposal,
  AuthorizationGrant,
  BitAgentStrategyIntent,
  RunContract,
} from "./types";

export function compileStrategyContract(
  intent: BitAgentStrategyIntent,
  subject = "agent:bitagent-small",
): RunContract {
  const now = new Date();
  const expiry = new Date(now.getTime() + 60 * 60 * 1000);
  return {
    version: "sch.run-contract.v1",
    contract_id: `contract-${randomUUID()}`,
    subject,
    objective: {
      description: `Allocate ${intent.amount} ${intent.asset} to ${intent.strategyId} on ${intent.network}.`,
      success_predicates: [
        `receipt.decision == allow`,
        `position.${intent.strategyId}.status == open`,
      ],
    },
    action_scope: [
      {
        tool: "tradelayer",
        verb: "allocate_strategy",
        constraints: {
          strategy_id: intent.strategyId,
          network: intent.network,
          max_notional: intent.amount,
        },
      },
    ],
    resources: {
      max_calls: 1,
      max_notional: intent.amount,
      max_fee: intent.maxFee,
      max_compute_seconds: 30,
      network_allowlist: [intent.network],
    },
    invariants: [
      "host retains commit authority",
      "simulation receipt precedes execution",
      "proposal parameters remain identical after approval",
    ],
    evidence: [
      { kind: "receipt", value: "decision_receipt.json" },
      { kind: "state_predicate", value: `position.${intent.strategyId}.status == open` },
    ],
    created_at: now.toISOString(),
    expires_at: expiry.toISOString(),
    metadata: { intent_type: "starter_strategy_allocation" },
  };
}

export function compileActionProposal(
  contract: RunContract,
  grant: AuthorizationGrant | null,
  intent: BitAgentStrategyIntent,
  simulationReceiptHash: string,
): ActionProposal {
  return {
    version: "sch.action-proposal.v1",
    proposal_id: `proposal-${randomUUID()}`,
    contract_id: contract.contract_id,
    grant_id: grant?.grant_id ?? null,
    tool: "tradelayer",
    verb: "allocate_strategy",
    args: {
      strategy_id: intent.strategyId,
      asset: intent.asset,
      amount: intent.amount,
      network: intent.network,
    },
    estimated_effects: {
      notional: intent.amount,
      fee: intent.maxFee,
      compute_seconds: 2,
      network: intent.network,
    },
    justification: "Exact post-simulation proposal for the confirmed strategy intent.",
    model_context_hash: null,
    created_at: new Date().toISOString(),
    metadata: {
      subject: contract.subject,
      simulation_receipt_hash: simulationReceiptHash,
    },
  };
}
