# Architecture

## System tuple

For one run, define

\[
\mathcal{R}=(C,A,q,s,x,V),
\]

where `C` is the run contract, `A` is a host-issued authorization grant, `q` is the model's typed proposal, `s` is gate state, `x` is the external world state, and `V` is the evidence verifier.

The deterministic gate computes

\[
d = g(C,A,q,s,x) \in \{\text{allow},\text{deny}\}.
\]

Execution is available only through the host:

\[
x_{t+1}=
\begin{cases}
F(x_t,q_t), & d_t=\text{allow},\\
x_t, & d_t=\text{deny}.
\end{cases}
\]

The model policy can alter `q`. It has no write path into the signer, nonce registry, revocation registry, budget ledger, or executor.

## Documents

### RunContract

A contract binds an objective, subject, typed action scope, resource limits, invariants, required evidence, and an expiration window. It describes the full run rather than one shell command.

### AuthorizationGrant

A grant is narrower than the contract. It names the issuer, subject, audience, allowed action, quantitative limits, lifespan, nonce, and one-shot policy. A trusted host service signs the canonical document after user confirmation.

### ActionProposal

The model emits a typed proposal containing exact arguments and estimated effects. The proposal references the contract and grant. The host can attach a hidden grant identifier after generation when the experiment tests out-of-band authority.

### DecisionReceipt

Every decision records all deterministic checks, stable hashes of the proposal and policy documents, budget state before and after evaluation, execution status, and state hashes.

## Trust roots

| Component | Reads model text | Holds authority | Can commit |
|---|---:|---:|---:|
| BitAgent actor | yes | no | no |
| Host confirmation UI | yes | user-session authority | no |
| Authorization issuer | scoped confirmation record | signing key | no |
| Contract gate | contract, grant, proposal, state | nonce, revocation, budget state | grants permit |
| Executor | allow receipt + exact proposal | tool credentials | yes |
| Mechinterp probe | activations and artifacts | no | no |

The local HMAC signer gives the TypeScript and Python implementations a common test vector. Production should move signing into a hardware-backed or platform-backed service and sign decision receipts as well.

## Lifecycle

```text
draft
  -> simulated
  -> awaiting_authorization
  -> authorized
  -> proposed
  -> gate_allowed -> executed -> verified
  -> gate_denied  -> awaiting_authorization or revised proposal
```

Every transition requires an evidence identifier. Grant denial and model refusal remain distinct events. The model may refuse an allowed action. The host blocks a disallowed proposal even when the model is eager to proceed.

## BitAgent insertion point

The intended flow is:

```text
natural-language intent
  -> compile typed strategy intent
  -> explain and simulate
  -> hash simulation receipt
  -> display canonical action and limits
  -> trusted user confirmation
  -> issue signed grant
  -> model or adapter emits exact ActionProposal
  -> ContractGateClient.evaluate()
  -> compare receipt proposal hash
  -> existing TradeLayer adapter executes
  -> verify state and persist receipt
```

The host authorization issuer stays outside the model's tool registry. A tool call such as `issue_grant` would collapse the control boundary.

## Small Control Harness insertion point

Small Control Harness owns the shared schemas, gate semantics, perturbations, receipts, and experiment adapter. BitAgent remains an environment plugin. This keeps financial action code and general control science separable.

Suggested adapter interface:

```python
class ConsequentialEnvironment(Protocol):
    def simulate(self, proposal: ActionProposal) -> SimulationReceipt: ...
    def execute(self, permit: DecisionReceipt, proposal: ActionProposal) -> ToolResult: ...
    def verify(self, contract: RunContract, result: ToolResult) -> VerificationReceipt: ...
```

## Reachability view

Let each grant enable a set of action channels. The allowed reachable set under contract and authorization is

\[
\mathcal{X}_{C,A}(T)=\operatorname{Reach}(x_0,\mathcal{U}_{C,A},T).
\]

Least authority asks for a grant that reaches the objective set while adding little reachability in protected directions. A local linearization can score that change through

\[
\Delta W_A=W_{C\oplus A}-W_C,
\]

with task gain and protected leakage

\[
g(A)=\operatorname{tr}(P_G\Delta W_AP_G^\top),\qquad
\ell(A)=\operatorname{tr}(P_H\Delta W_AP_H^\top).
\]

The starter implementation supplies the discrete authorization channels and quantitative budgets needed to estimate these objects empirically.
