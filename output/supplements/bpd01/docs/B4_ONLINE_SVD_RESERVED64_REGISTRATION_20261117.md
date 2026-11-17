# B4-ONLINE-SVD-02: reserved 64-board CPU FP32 comparison

## Source lock, November 17, 2026

This prospective test follows the exploratory first-four-board CPU run in
PR #159 and the implementation/mini-TRM test in PR #158. This file is
committed before any of the 64 reserved evaluations execute.

The registered CPU estimand is a paired comparison at the earliest readout
(pass index 20) between two zero-fit SVD top-k selection procedures:
clean-informed masks from the target model's full trajectory versus online
masks from each masked module's current perturbed input.

The experiment is a numerical/methodological comparison, not a claim
about trained CI or mechanistic necessity. The separate CUDA BF16 identity
control remains open.

## Dataset and identity

Published WB-Stall Sudoku evaluation bundle: 183 inputs. Exclude indices
0 through 15 used for B4 diagnostics, leaving candidates 16..182. Rank
all candidate integer indices by ascending bytes of

    sha256("BPD-ONLINE-SVD-01-RESERVED64-20261117|{idx:03d}")

and take the first 64. Preserve that hash order. The list is:

    31,57,67,154,68,138,83,50,
    92,176,124,151,89,163,76,137,
    23,146,115,132,94,78,21,74,
    135,86,181,112,90,44,113,87,
    131,136,46,97,166,109,60,147,
    110,134,168,174,179,101,150,48,
    62,72,157,116,145,130,107,140,
    171,26,38,165,170,25,34,77

SHA256 of the JSON compact decimal array in that order:
0f474925d83f86839b411c2f1286c5beec5b3630b40275f299e66a267ce97ab0.

They are additional examples from an *existing published dataset*.
They are newly held aside for this comparison. They are neither
independently generated nor guaranteed unseen by other prior analyses.
The loader must verify the exact 183-board source and model hashes.

## Frozen execution

- Models: gaoxin492 and yagizdevre, each loaded from the SHA-256-verified
  pretrained Hugging Face checkpoint.
- Device: GitHub-hosted Linux CPU, torch float32, OMP_NUM_THREADS=2.
- One separate job per checkpoint. Split 64 selected indices into eight
  ordered chunks of eight to limit memory. Use the same source model
  and source-only SVD start for each chunk; no gradients or fitting.
- Run 21 recurrent invocations, expose readout index 20.
- Four shared sites (two o_proj and two down_proj). C=512 components
  per site, k=307 selected at 60% density.
- Delta/residual off, plain zero replacement, sphere rescaling off.
- Same-dtype independent native target repeat must be <=1e-5 KL
  on *each* board or classify the whole checkpoint as invalid.
- Compute static rank307, clean-informed dynamic top307, and true
  current-state online dynamic top307 under identical source weights.
- The online arm must pass the oracle-reference noninterference
  regression test. Clean-input masks are read-only telemetry.

## Registered metrics and hypothesis

For board i define:

    S_i = KL(target || static)_i
    C_i = KL(target || clean_oracle)_i
    O_i = KL(target || online)_i
    D_i = O_i - C_i
    M_i = D_i - 0.10 * S_i

Primary question: is online selection non-inferior to clean-informed
selection within a per-board margin of 10% of the static baseline KL?

Compute the mean M across all 64 boards, bootstrap board pairs with
replacement using 20,000 PCG64 draws and seed 20261117117. Take the
97.5 percentile upper confidence bound for mean M, separately on both
checkpoints, implementing a Bonferroni-protected two-checkpoint gate.
Report *PASS* only if both upper bounds are strictly less than zero.

This relative materiality margin was chosen before the 64-board run
to scale with model baseline accuracy. It is a practical perturbation
tolerance, not a universal mechanistic variation threshold.

Secondary: report mean D and paired 95% bootstrap intervals, individual
signed board effects, win count (D<0), median, and a one-sided paired
sign-flip p-value for online superiority. Use 100,000 fixed-seed
Monte Carlo sign flips with +1 correction, and Holm adjustment across
the two checkpoints. This secondary test cannot upgrade a primary FAIL.

Record 64 per-board KL values per arm and observed mask disagreement
per module/board, with full numeric precision. Also report effect
concentration (sum |D|)^2 / sum D^2 and top-board share of |D|, to
detect a result driven by one unusually sensitive board. A nonpositive
M mean without a passing upper bound is inconclusive.

All comparisons occur at pass 20. There are no final-pass, causal
component-importance, or trained-gate superiority claims from this test.

## Output and fidelity

Each model's JSON must contain the exact board indices, their
SHA256 prefixes, the per-board arms, per-module disagreement, model
loader versions, random seed, and linked workflow run. Preserve
individual-shard results and an independent stdlib or NumPy reducer.
Do not replace a failed or changed run with another subset; source
changes require a new version and explicit registration.

The four-board pilot observed average online minus oracle KL
-0.007747 for gaoxin492 and +0.00001017 for yagizdevre.
Those four pilot boards are excluded here, and no previous 64-board
effects were used to tune this protocol.
