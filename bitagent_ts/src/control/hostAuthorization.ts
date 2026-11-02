import { randomUUID } from "node:crypto";
import { signGrant } from "./canonicalize";
import type { AuthorizationGrant, RunContract, ScopedAction } from "./types";

export interface ConfirmedAuthorization {
  contractId: string;
  subject: string;
  allowedActions: ScopedAction[];
  limits: AuthorizationGrant["limits"];
  expiresAt: string;
  oneShot: boolean;
  confirmationReceiptId: string;
}

/**
 * Host-only authorization issuer.
 * Keep this service outside the model tool registry. The acting model may request
 * a grant and display its canonical scope; only the trusted UI/session layer calls issue().
 */
export class HostAuthorizationService {
  constructor(
    private readonly secret: Buffer,
    private readonly keyId = "local-dev-v1",
    private readonly issuer = "host:user-approval-service",
    private readonly audience = "host:bitagent-executor",
  ) {}

  issue(
    contract: RunContract,
    confirmation: ConfirmedAuthorization,
  ): AuthorizationGrant {
    if (confirmation.contractId !== contract.contract_id) {
      throw new Error("confirmation contract mismatch");
    }
    if (confirmation.subject !== contract.subject) {
      throw new Error("confirmation subject mismatch");
    }
    const grant: AuthorizationGrant = {
      version: "sch.authorization-grant.v1",
      grant_id: `grant-${randomUUID()}`,
      contract_id: contract.contract_id,
      issuer: this.issuer,
      subject: contract.subject,
      audience: this.audience,
      allowed_actions: confirmation.allowedActions,
      limits: confirmation.limits,
      issued_at: new Date().toISOString(),
      expires_at: confirmation.expiresAt,
      nonce: `nonce-${randomUUID()}`,
      one_shot: confirmation.oneShot,
      parent_grant_id: null,
      signature: { alg: "HS256", key_id: this.keyId, value: "" },
      metadata: {
        confirmation_receipt_id: confirmation.confirmationReceiptId,
      },
    };
    grant.signature.value = signGrant(grant, this.secret);
    return grant;
  }
}
