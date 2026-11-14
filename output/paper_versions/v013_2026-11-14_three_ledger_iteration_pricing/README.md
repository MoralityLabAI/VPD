# `papers/shared_weights_dark_fibers`: Unified Looped-Transformer Mechanistic Synthesis

This directory contains the unified manuscript (`v013`):

> **From Shared Weights to Head-Orthogonal Fibers: Virtual Depth, Head-Readable Staircases, and Post-Stall Search in Looped Transformers**  
> Patrick Dugan (November 2026)

It synthesizes seven studies across five experimental arcs while leaving all prior manuscript snapshots (`VPD/output/paper_versions/v001..v012` and `papers/rmp_looped_mechinterp/`) untouched:

1. **Astra / Paper I (`Sept 2026`, `VPD` v003–v008):** Multipass VPD + AdamW vs. Muon Top-32 SAEs, $F_\Delta$ support-switching spikes, and 44-layer embedding-reset vs. state-carry controls on `Nanbeige/Nanbeige4.2-3B` across `ARC-AGI-1`, `ARC-AGI-2`, and `PrimeIntellect/INTELLECT-3-RL` (`logic` and `math` subsets).
2. **Silico (`Nov 2026`, `HUG-CAL-02` & `Stage 1`):** Activation-distribution-matched input-read rotation confirmation refuting late-iteration SVD factor specialization on `tomg-group-umd/huginn-0125` (`±0.0065` nats across 8 frozen factors), plus `Nanbeige4.2-3B` multi-hop pointer-chase Stage 1 and 1-hop VPD heatmap.
3. **Opus (`Oct–Nov 2026`, `RMP` Paper II, `LOOPSCOPE-01`, `ARG-02`):** Closed-loop shared dictionaries & MLP-only transcoders, collision-free feature-mediated variable transfer (`LOOP-T` vs. `FF-U`), Transported Component Attribution (`TCA` across 480 `Huginn` conditions), 24-arm `LoopedChase` monitorability frontier (`Q4–Q9`) + $P_{\text{head}}$ contrast projection (`3/3` seeds retain `1.000` at $\lambda=0.1$; `8/9` across $\lambda \in \{0.01, 0.1\}`, `2/3` at $\lambda=1.0$), Compactification-Theory B1 (`Q11`, `AUC = 0.992`), `SEM-ECHO-01`, and `ARG-02` Extension N1 (`H2a`, pre-run registration commit `65039a5b` on public `Compactification-Theory`) on `Nanbeige4.2-3B`.
4. **`LOOPSCOPE-02` + Two-Checkpoint Manifold & Differential History Probe Controls (`Nov 2026`, [`../../mechinterp/loopscope02/`](../../mechinterp/loopscope02/)):** Triangulated evaluation of `Nanbeige4.2-3B`, `Qwen3.5-4B-Base`, `Qwen3-4B-Instruct-2507`, and 7M Attention-`TRM` (`gaoxin492` `step_21700` + `yagizdevre` replication across both `items256` and `bench100`, `16 × 21 = 336` core calls), establishing the observational Detached-ACT vs. BPTT Lyapunov contrast ($F_1/F_6 = 1{,}808.4$ vs. `0.303×` basin contraction), separating off-manifold L2 norm loss from stalled-search fiber sensitivity across both checkpoints (`fiber_regime_controls.json` + `fiber_replication_and_history_probes.json`), documenting primary $k=64$ PCA recovery vs. non-monotone intermediate-$k$ interference on `yagizdevre`, and contrasting differential flip/conflict transition probes (`0.65–0.82` AUC) against visible runner-up logits (`61.8%–82.9%` refuted-digit accuracy) and a shuffled-label null.
5. **`LOOPSCOPE-03` Three-Ledger Iteration Pricing (`Nov 2026`, [`../../mechinterp/loopscope03/`](../../mechinterp/loopscope03/), pre-registered commit `b1ec7755`):** Decomposes carried state into $S_t = V_t \oplus C_t \oplus R_t$ across `TRM` (`gaoxin492` + `yagizdevre`), `LoopedChase` (12 `Q4` checkpoints), `Huginn-0125`, and `Nanbeige4.2-3B`, establishing the $z_H \times z_L$ backup channel (`V_only` limit-cycle collapse), primary dark-memory deficit (`-7.7` to `-10.4` pp; `+7.65` active search loops at $t=8$) vs. replication cache denoising (`+15.3` to `+16.4` pp above clean at $s=64$), `Huginn-0125` `99.94%` cache contraction, and `100%` solved-fork path independence (Anil et al. 2022) paired with `K=8` `C-voting` (Kubo et al. 2026) matching the `K=8` oracle union `100%`.

## Contents

- [`main.tex`](main.tex) — LaTeX manuscript source
- [`main.pdf`](main.pdf) — Compiled PDF manuscript
- [`references.bib`](references.bib) — Bibliography (29 entries with verified public `VPD` links for `CC BY 4.0` papers, canonical `arxiv.org/pdf` links for non-CC papers, `PrimeIntellect/INTELLECT-3-RL`, and public `Compactification-Theory` pre-run registration commit `65039a5b`)
- [`PAPER.md`](PAPER.md) — Self-contained GitHub-readable Markdown edition with embedded PNG figures and full reference list
- [`build_figures.py`](build_figures.py) — Deterministic CPU script regenerating `figures/*.pdf` and `figures/*.png` from [`../../mechinterp/loopscope02/results/`](../../mechinterp/loopscope02/results/) and [`../../mechinterp/loopscope03/results/`](../../mechinterp/loopscope03/results/)
- [`figures/`](figures/) — Vector `.pdf` and 300-DPI `.png` figures (`fig1`–`fig5`)
- [`literature/`](literature/) — License-audited archive + SHA-256 [`manifest.json`](literature/manifest.json) (mirrors the 8 `CC BY 4.0` PDFs in public `MoralityLabAI/VPD`; links + SHA-256 hashes only for the 6 `arXiv non-exclusive 1.0` papers)

## Public `MoralityLabAI/VPD` & Canonical URIs for Cited Looped-MechInterp Papers

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
