import type {
  ActionProposal,
  AuthorizationGrant,
  DecisionReceipt,
  RunContract,
} from "./types";

export interface GateRequest {
  contract: RunContract;
  grant: AuthorizationGrant | null;
  proposal: ActionProposal;
}

export interface GateResponse {
  receipt: DecisionReceipt;
}

/** Thin boundary around the Small Control Harness gate service. */
export class ContractGateClient {
  constructor(private readonly endpoint: URL) {}

  async evaluate(request: GateRequest): Promise<GateResponse> {
    const response = await fetch(new URL("/v1/contracts/evaluate", this.endpoint), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw new Error(`contract gate failed: ${response.status} ${response.statusText}`);
    }
    return (await response.json()) as GateResponse;
  }
}
