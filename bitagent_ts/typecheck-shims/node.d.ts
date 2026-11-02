declare class Buffer extends Uint8Array {
  static from(value: string, encoding?: string): Buffer;
  static from(value: ArrayBuffer | ArrayLike<number>): Buffer;
}

interface HashLike {
  update(value: string, encoding?: string): HashLike;
  digest(encoding: "hex"): string;
}

declare module "node:crypto" {
  export function randomUUID(): string;
  export function createHash(algorithm: string): HashLike;
  export function createHmac(algorithm: string, key: Buffer): HashLike;
  export function timingSafeEqual(left: Uint8Array, right: Uint8Array): boolean;
}

declare module "node:assert/strict" {
  const assert: {
    equal(actual: unknown, expected: unknown): void;
  };
  export default assert;
}

declare module "node:test" {
  export default function test(name: string, body: () => void | Promise<void>): void;
}
