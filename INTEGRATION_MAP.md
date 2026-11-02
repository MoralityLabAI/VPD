# Integration Map

## Observed BitAgent root

Prior workspace logs showed this local project path:

```text
C:\projects\BitAgent\BitAgent\tradelayer-thorchain-starter\tradelayer-thorchain-starter
```

They also showed an existing adapter at:

```text
src\adapters\tradelayerAdapter.ts
```

Verify both paths before applying the overlay. The public repository location and the current Small Control Harness path were not established by this package.

## BitAgent changes

Copy `bitagent_ts/src/control/` into the BitAgent source tree. Then locate the current flow that calls `tradelayerAdapter.ts`.

Insert the boundary after simulation and before any transaction submission:

```ts
const contract = compileStrategyContract(intent);
const simulation = await simulateStrategy(intent);
const simulationReceiptHash = stableHash(simulation);

const confirmation = await trustedUi.confirm({
  contract,
  exactAction: intent,
  simulation,
});

const grant = hostAuthorizationService.issue(contract, confirmation);
const proposal = compileActionProposal(
  contract,
  grant,
  intent,
  simulationReceiptHash,
);

const { receipt } = await contractGate.evaluate({ contract, grant, proposal });
if (receipt.decision !== "allow") {
  return persistDeniedRun({ contract, grant, proposal, receipt });
}

// Bind execution to the exact proposal digest carried in the receipt.
const result = await tradelayerAdapter.execute(proposal.args);
return verifyAndPersist({ contract, grant, proposal, receipt, result });
```

The confirmation UI may call `HostAuthorizationService.issue`. The model-facing tool registry may not.

## Small Control Harness changes

Place the shared schemas under the harness schema directory. Add the Python package or port its deterministic semantics into the harness language. Register BitAgent as a consequential environment adapter.

Persist these event types:

```text
run_contract_created
action_simulated
authorization_requested
authorization_granted
authorization_revoked
action_proposed
gate_decision
action_executed
action_verified
run_failed
```

Each event should include the contract identifier, proposal identifier where present, stable document hashes, actor identity, monotonic sequence number, and a wall-clock timestamp.

## Receipt binding

Before execution, verify:

```text
receipt.decision == allow
receipt.contract_id == proposal.contract_id
receipt.proposal_id == proposal.proposal_id
receipt.metadata.proposal_hash == sha256(canonical_json(proposal))
```

A production path should sign the decision receipt or transmit it over an authenticated local channel. The local FastAPI service is an integration scaffold.

## Migration sequence

1. Add schemas and type definitions without changing execution behavior.
2. Wrap the existing simulation output in a stable receipt and hash it.
3. Add a shadow gate that logs allow and deny decisions while execution remains regtest-only.
4. Move commit authority behind an allow receipt.
5. Add host confirmation and grant issuance.
6. Add replay, revocation, and budget persistence.
7. Run the paired mechinterp battery on a frozen BitAgent checkpoint.

Every stage should leave the existing test suite green and add one replayable receipt fixture.
