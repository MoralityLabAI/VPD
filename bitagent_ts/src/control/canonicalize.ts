import { createHash, createHmac, timingSafeEqual } from "node:crypto";
import type { AuthorizationGrant } from "./types";

function sortDeep(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortDeep);
  if (value !== null && typeof value === "object") {
    const input = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(input)
        .sort()
        .map((key) => [key, sortDeep(input[key])]),
    );
  }
  return value;
}

export function canonicalJson(value: unknown): string {
  return JSON.stringify(sortDeep(value));
}

export function stableHash(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value), "utf8").digest("hex");
}

export function grantSigningPayload(grant: AuthorizationGrant): AuthorizationGrant {
  return {
    ...grant,
    signature: {
      ...grant.signature,
      value: "",
    },
  };
}

export function signGrant(grant: AuthorizationGrant, secret: Buffer): string {
  return createHmac("sha256", secret)
    .update(canonicalJson(grantSigningPayload(grant)), "utf8")
    .digest("hex");
}

export function verifyGrantSignature(
  grant: AuthorizationGrant,
  secret: Buffer,
): boolean {
  if (grant.signature.alg !== "HS256") return false;
  const expected = Buffer.from(signGrant(grant, secret), "hex");
  const supplied = Buffer.from(grant.signature.value, "hex");
  return supplied.length === expected.length && timingSafeEqual(supplied, expected);
}
