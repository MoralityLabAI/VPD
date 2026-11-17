# B4-ONLINE-SVD-01: prospective causal top-k evaluation

Registered before running either pretrained checkpoint, following the numerical
closure audit in PR #156. The 16-board panel is a diagnostic pilot.

## Question

At 60% component density, how much does the zero-fit SVD baseline gain from
selecting masks using the clean target's complete rollout rather than the
perturbed model's current pre-weight activation?

The former is an oracle-informed procedure. The latter is available during
closed-loop inference. Each TiedComponentLinear already receives its current x;
the online selector uses |x @ V| * ||U|| and selects k per token and pass.

## Frozen evaluator

- Checkpoints: SHA-256-verified gaoxin492 and yagizdevre Sudoku TRMs.
- Precision: CUDA BF16; CPU FP32 runs occupy a separate exploratory regime.
- Dataset: first 16 WB-Stall evaluation boards used in B4 v2.
- Readouts: outer-step commits at invocation indices 20, 41 and 62.
- Factors: frozen untrained full-rank SVD for each pretrained linear site.
- Sites: layers.{0,1}.{o_proj,down_proj}, C=512 per module.
- Active components: k=round(0.6*C)=307.
- Ablation: zero-fill inactive component writes, residual off, sphere off.
- Independent same-dtype clean target repeat establishes the zero floor.

## Arms

1. Static first-307 SVD components across all tokens and passes.
2. Oracle top-307 derived from the clean source preweight inputs, then applied
   to a closed-loop perturbed rollout.
3. Online top-307 computed inside each wrapped layer from its current perturbed
   input; future masks are unknown until execution.
4. Same-dtype target repeat.

The online arm receives the clean-informed mask tensor only for post-hoc
Hamming disagreement telemetry, never as an input to its selected writes.
Regression tests substitute contradictory oracle masks and require identical
online output.

## Stored measurements

- Mean and individual board KL(target || arm) at every readout.
- Per-board KL(oracle || online) for selector-induced divergence.
- Paired per-board KL(online) minus KL(oracle), with no floor subtraction.
- Per-board and per-pass mask Hamming disagreement for all four sites.
- Target repeat, model/dtype, and evaluation board hashes.

## Interpretation

Pass-20 paired KL is the first diagnostic comparison. At passes 41 and 62,
large KL is logged as long-horizon numerical sensitivity until the exact
identity-control issue is resolved. Scalar floor subtraction is invalid because
KL is nonlinear. Inspect the per-board distribution.

If clean-informed top-k beats online top-k, the result establishes the value
of clean-state lookahead for this selection rule alone. A trained CI claim
requires CI weights, an explicit causal feature protocol, and a larger
independent evaluation panel.

The current CI architecture concatenates inputs from all four selected modules
at the same pass, although earlier masked modules can change later module
inputs in that pass. Strictly online CI therefore requires a causal feature
design or a preflight step; local activation top-k has no such dependency.

## Execution

    python -m pytest mechinterp/bpd/tests/test_b4_v2_online_svd.py -q
    python mechinterp/bpd/scripts/audit_b4_v2_online_svd.py --mini --device cpu
    python mechinterp/bpd/scripts/audit_b4_v2_online_svd.py --device cuda --out mechinterp/bpd/results/v2/gate_b4_v2_online_svd.json

Mini TRM results are software verification only. The pretrained CUDA
evaluation is prospective and has no reported results at registration.
