# `papers/shared_weights_dark_fibers`: Unified Looped-Transformer Mechanistic Synthesis

This directory contains the unified manuscript (`v014`):

> **From Shared Weights to Head-Orthogonal Fibers: Virtual Depth, Head-Readable Staircases, and Post-Stall Search in Looped Transformers**  
> Patrick Dugan (November 2026)

It synthesizes seven studies across five experimental arcs while leaving all prior manuscript snapshots (`VPD/output/paper_versions/v001..v013` and `papers/rmp_looped_mechinterp/`) untouched:

1. **Astra / Paper I (`Sept 2026`, `VPD` v003–v008):** Multipass VPD + AdamW vs. Muon Top-32 SAEs, $F_\Delta$ support-switching spikes, and 44-layer embedding-reset vs. state-carry controls on `Nanbeige/Nanbeige4.2-3B` across `ARC-AGI-1`, `ARC-AGI-2`, and `PrimeIntellect/INTELLECT-3-RL` (`logic` and `math` subsets).
2. **Silico (`Nov 2026`, `HUG-CAL-02` & `Stage 1`):** Activation-distribution-matched input-read rotation confirmation refuting late-iteration SVD factor specialization on `tomg-group-umd/huginn-0125` (`±0.0065` nats across 8 frozen factors; trained with random-init warmup and $k=8$ truncated BPTT), plus `Nanbeige4.2-3B` multi-hop pointer-chase Stage 1 and 1-hop VPD heatmap.
3. **Opus (`Oct–Nov 2026`, `RMP` Paper II, `LOOPSCOPE-01`, `ARG-02`):** Closed-loop shared dictionaries & MLP-only transcoders, collision-free feature-mediated variable transfer (`LOOP-T` vs. `FF-U`), Transported Component Attribution (`TCA` across 480 `Huginn` conditions), 24-arm `LoopedChase` monitorability frontier (`Q4–Q9`) + $P_{\text{head}}$ contrast projection (`3/3` seeds retain `1.000` with `0.00` residual loops at $\lambda=0.1$ vs. `+3.06, +5.46` readout-blind-spot loops in `2/3` seeds at $\lambda=1.0$, instantiating Sharma & Vu 2026), Compactification-Theory B1 (`Q11`, `AUC = 0.992`), `SEM-ECHO-01`, and `ARG-02` Extension N1 (`H2a`, pre-run registration commit `65039a5b` on public `Compactification-Theory`) on `Nanbeige4.2-3B`.
4. **`LOOPSCOPE-02` + Two-Checkpoint Manifold & Differential History Probe Controls (`Nov 2026`, [`../../mechinterp/loopscope02/`](../../mechinterp/loopscope02/), public mirror [`VPD/output/supplements/loopscope02`](https://github.com/MoralityLabAI/VPD/tree/main/output/supplements/loopscope02)):** Triangulated evaluation of `Nanbeige4.2-3B`, `Qwen3.5-4B-Base`, `Qwen3-4B-Instruct-2507`, and 7M Attention-`TRM` (`gaoxin492` `step_21700` + `yagizdevre` replication across both `items256` and `bench100`, `16 × 21 = 336` core calls), establishing the observational Detached-ACT vs. BPTT Lyapunov contrast ($F_1/F_6 = 1{,}808.4$ vs. `0.303×` basin contraction), separating off-manifold L2 norm loss from stalled-search fiber sensitivity across both checkpoints (`fiber_regime_controls.json` + `fiber_replication_and_history_probes.json`), documenting primary $k=64$ PCA recovery vs. non-monotone intermediate-$k$ interference on `yagizdevre`, and contrasting differential flip/conflict transition probes (`0.65–0.82` AUC) against visible runner-up logits (`61.8%–82.9%` refuted-digit accuracy) and a shuffled-label null.
5. **`LOOPSCOPE-03` Three-Ledger Decomposition, Data Scaling & Enforced Path Independence (`Nov 2026`, [`../../mechinterp/loopscope03/`](../../mechinterp/loopscope03/), pre-registered commit `b1ec7755`, public mirror [`VPD/output/supplements/loopscope03`](https://github.com/MoralityLabAI/VPD/tree/main/output/supplements/loopscope03)):** Decomposes carried state into $S_t = V_t \oplus C_t \oplus R_t$ across `TRM` (`gaoxin492` + `yagizdevre`), `LoopedChase` (12 `Q4` checkpoints), `Huginn-0125` (`99.94%` cache contraction), and `Nanbeige4.2-3B`, establishing the $z_H \times z_L$ backup channel (`V_only` limit-cycle collapse), `N = 100..8,100` canonical-isomorphism-disjoint refill scaling ($R^2: 0.844 \to 0.915$ primary, $0.689 \to 0.837$ replication), periodic refill ($m=2,4,8$) and covariate-shift-free `Resume` (`0.0` median loop delay across unselected primary mutual solvers vs. `-2` to `-4` median loops and `-10.5` pp `NeverLock` on unresolved `yagizdevre`), replication of Kubo et al. (2026) `C-voting` (`0.836 -> 0.940`; `0.656 -> 0.858`), and unconditional $K=8$ fork scoring confirming Anil et al. (2022) that conditional refill enforces path independence ($\text{PI}_{\text{cell}}(64): 0.603 \to 0.937$) and rescues low-PI boards (`+39.1` pp in `Q1`).

## Contents

- [`main.tex`](main.tex) — LaTeX manuscript source (20 pages)
- [`main.pdf`](main.pdf) — Compiled PDF manuscript
- [`references.bib`](references.bib) — Bibliography (30 entries with verified public `VPD` links for `CC BY 4.0` papers, canonical `arxiv.org/pdf` links for non-CC papers, `PrimeIntellect/INTELLECT-3-RL`, and public `Compactification-Theory` / `VPD/output/supplements` registration commits)
- [`PAPER.md`](PAPER.md) — Self-contained GitHub-readable Markdown edition with embedded PNG figures and full reference list
- [`build_figures.py`](build_figures.py) — Deterministic CPU script regenerating `figures/*.pdf` and `figures/*.png` from [`../../mechinterp/loopscope02/results/`](../../mechinterp/loopscope02/results/) and [`../../mechinterp/loopscope03/results/`](../../mechinterp/loopscope03/results/)
- [`figures/`](figures/) — Vector `.pdf` and 300-DPI `.png` figures (`fig1`–`fig5`)
- [`literature/`](literature/) — License-audited archive + SHA-256 [`manifest.json`](literature/manifest.json) (mirrors the 9 `CC BY 4.0` PDFs in public `MoralityLabAI/VPD`; links + SHA-256 hashes only for the 6 `arXiv non-exclusive 1.0` papers)

## Public `MoralityLabAI/VPD` & Canonical URIs for Cited Looped-MechInterp Papers

- **Sharma & Vu (2026), `arXiv:2606.24898` (`CC BY 4.0`, *Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models*):**
  - Public `VPD` PDF: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2606.24898_sharma2026_readout_blind_spot.pdf
- **Anil et al. (2022), `arXiv:2211.09961` (`CC BY 4.0`, *Path Independent Equilibrium Models Can Better Exploit Test-Time Computation*):**
  - Public `VPD` PDF: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2211.09961_anil2022_path_independent_equilibrium_models.pdf
- **Kubo, Kamiya et al. (2026), `arXiv:2604.13521` (`CC BY 4.0`, *C-voting: Confidence-Based Test-Time Voting without Explicit Energy Functions*):**
  - Public `VPD` PDF: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2604.13521_kubo2026_cvoting_confidence_based_voting.pdf
- **Kaissis et al. (2026), `arXiv:2602.10097` (`CC BY 4.0`, *Step-Resolved Data Attribution for Looped Transformers*):**
  - Public `VPD` PDF: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2602.10097_kaissis2026_step_resolved_data_attribution.pdf
- **Blayney et al. (2026), `arXiv:2604.11791` (`CC BY 4.0`, *A Mechanistic Analysis of Looped Reasoning Language Models*):**
  - Public `VPD` PDF: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2604.11791_blayney2026_mechanistic_analysis_looped_lms.pdf
- **Wu, Hu & Zhang (2026), `arXiv:2609.39892` (`CC BY 4.0`, *Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does*):**
  - Public `VPD` PDF: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2609.39892_wu2026_shared_weights_selected_computations.pdf
- **Sesodia et al. (2026), `arXiv:2610.24135` (`CC BY 4.0`, *Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on Depth-Bound Algorithmic Tasks*):**
  - Public `VPD` PDF: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2610.24135_sesodia2026_inside_the_loop.pdf
- **Olszowiec et al. (2026), `arXiv:2610.18127` (`arXiv non-excl. 1.0`, *Monitoring Latent Reasoning in Looped Language Models*):**
  - Canonical arXiv PDF (non-CC; SHA-256 recorded in `manifest.json`): https://arxiv.org/pdf/2610.18127.pdf
- **Additional `CC BY 4.0` mirrored PDFs (`HRM` `2506.21734`, `Saunshi et al.` `2502.17416`) and canonical `arXiv non-excl. 1.0` links (`DeepLoop` `2607.13491`, `TRM` `2510.04871`, `Huginn-0125` `2502.05171`, `Giannou et al.` `2301.13196`, `Universal Transformers` `1807.03819`):** indexed in [`VPD/output/literature/looped_mechinterp/README.md`](https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/README.md).

## Reproducing Figures and PDF

```powershell
# 1. Regenerate all vector (.pdf) and raster (.png) figures
& "C:\Users\patri\.venvs\bluebeam\Scripts\python.exe" build_figures.py

# 2. Compile main.pdf via MiKTeX
$env:PATH = "C:\Users\patri\AppData\Local\Programs\MiKTeX\miktex\bin\x64;" + $env:PATH
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
