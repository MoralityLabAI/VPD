# `papers/shared_weights_dark_fibers`: Unified Looped-Transformer Mechanistic Synthesis

This directory contains the unified manuscript:

> **From Shared Weights to Head-Orthogonal Fibers: Virtual Depth, Head-Readable Staircases, and Post-Stall Search in Looped Transformers**  
> Patrick Dugan (November 2026)

It synthesizes six studies across four experimental arcs while leaving all prior manuscript snapshots (`VPD/output/paper_versions/v001..v010` and `papers/rmp_looped_mechinterp/`) untouched:

1. **Astra / Paper I (`Sept 2026`, `VPD` v003–v008):** Multipass VPD + AdamW vs. Muon Top-32 SAEs, $F_\Delta$ support-switching spikes, and 44-layer embedding-reset vs. state-carry controls on `Nanbeige/Nanbeige4.2-3B` across `ARC-AGI-1`, `ARC-AGI-2`, and `INTELLECT-3`.
2. **Silico (`Nov 2026`, `HUG-CAL-02` & `Stage 1`):** Activation-distribution-matched input-read rotation confirmation refuting late-iteration SVD factor specialization on `tomg-group-umd/huginn-0125` (`±0.0065` nats across 8 frozen factors), plus `Nanbeige4.2-3B` multi-hop pointer-chase Stage 1 and 1-hop VPD heatmap.
3. **Opus (`Oct–Nov 2026`, `RMP` Paper II, `LOOPSCOPE-01`, `ARG-02`):** Closed-loop shared dictionaries & MLP-only transcoders, collision-free feature-mediated variable transfer (`LOOP-T` vs. `FF-U`), Transported Component Attribution (`TCA` across 480 `Huginn` conditions), 24-arm `LoopedChase` monitorability frontier (`Q4–Q9`) + $P_{\text{head}}$ contrast projection (`1.000` retention at $\lambda=0.1$), Compactification-Theory B1 (`Q11`, `AUC = 0.992`), `SEM-ECHO-01`, and `ARG-02` Extension N1 (`H2a`) on `Nanbeige4.2-3B`.
4. **`LOOPSCOPE-02` + Head-Orthogonal Manifold Controls (`Nov 2026`, [`../../mechinterp/loopscope02/`](../../mechinterp/loopscope02/)):** Triangulated evaluation of `Nanbeige4.2-3B`, `Qwen3.5-4B-Base`, `Qwen3-4B-Instruct-2507`, and 7M Attention-`TRM` (`step_21700` + `yagizdevre` replication, `16 × 21 = 336` core calls) on deductively stratified 9×9 Sudoku (`items256` + `bench100`), establishing the observational Detached-ACT vs. BPTT Lyapunov contrast ($F_1/F_6 = 1{,}808.4$ vs. `0.303×` basin contraction) and dissecting the 501-dimensional `lm_head`-orthogonal subspace via norm-restored, train-mean, donor-swapped, step-1-frozen, top-$k$ PCA ($k=64$ full recovery, $p=1.00$), and linear ridge-probe (`97.1%–97.8%` candidate masks) controls.

## Contents

- [`main.tex`](main.tex) — LaTeX manuscript source
- [`main.pdf`](main.pdf) — Compiled PDF manuscript
- [`references.bib`](references.bib) — Bibliography (27 entries: foundational looped-transformer, SAE/transcoder/DAS/APD/Muon, model/corpus provenance, and all 5 concurrent 2026 looped-interpretability papers)
- [`PAPER.md`](PAPER.md) — Self-contained GitHub-readable Markdown edition with embedded PNG figures and full reference list
- [`build_figures.py`](build_figures.py) — Deterministic CPU script regenerating `figures/*.pdf` and `figures/*.png` from [`../../mechinterp/loopscope02/results/`](../../mechinterp/loopscope02/results/) (including `fiber_regime_controls.json`) and audited cross-program summary tables
- [`figures/`](figures/) — Vector `.pdf` and 300-DPI `.png` figures (`fig1`–`fig4`)
- [`literature/`](literature/) — Fetched PDF archive + SHA-256 [`manifest.json`](literature/manifest.json) for all cited looped-transformer mechanistic interpretability and architecture papers

## Repo URIs for Fetched Cited Looped-MechInterp PDFs

- **Kaissis et al. (2026), `arXiv:2602.10097` (*Step-Resolved Data Attribution for Looped Transformers*):**
  - `LoopedTransformers`: https://github.com/MoralityLabAI/LoopedTransformers/blob/main/papers/shared_weights_dark_fibers/literature/2602.10097_kaissis2026_step_resolved_data_attribution.pdf
  - `VPD`: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2602.10097_kaissis2026_step_resolved_data_attribution.pdf
- **Blayney et al. (2026), `arXiv:2604.11791` (*A Mechanistic Analysis of Looped Reasoning Language Models*):**
  - `LoopedTransformers`: https://github.com/MoralityLabAI/LoopedTransformers/blob/main/papers/shared_weights_dark_fibers/literature/2604.11791_blayney2026_mechanistic_analysis_looped_lms.pdf
  - `VPD`: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2604.11791_blayney2026_mechanistic_analysis_looped_lms.pdf
- **Wu, Hu & Zhang (2026), `arXiv:2609.39892` (*Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does*):**
  - `LoopedTransformers`: https://github.com/MoralityLabAI/LoopedTransformers/blob/main/papers/shared_weights_dark_fibers/literature/2609.39892_wu2026_shared_weights_selected_computations.pdf
  - `VPD`: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2609.39892_wu2026_shared_weights_selected_computations.pdf
- **Olszowiec et al. (2026), `arXiv:2610.18127` (*Monitoring Latent Reasoning in Looped Language Models*):**
  - `LoopedTransformers`: https://github.com/MoralityLabAI/LoopedTransformers/blob/main/papers/shared_weights_dark_fibers/literature/2610.18127_olszowiec2026_monitoring_latent_reasoning.pdf
  - `VPD`: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2610.18127_olszowiec2026_monitoring_latent_reasoning.pdf
- **Sesodia et al. (2026), `arXiv:2610.24135` (*Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on Depth-Bound Algorithmic Tasks*):**
  - `LoopedTransformers`: https://github.com/MoralityLabAI/LoopedTransformers/blob/main/papers/shared_weights_dark_fibers/literature/2610.24135_sesodia2026_inside_the_loop.pdf
  - `VPD`: https://github.com/MoralityLabAI/VPD/blob/main/output/literature/looped_mechinterp/2610.24135_sesodia2026_inside_the_loop.pdf
- **Cited looped-architecture PDFs (`DeepLoop` `2607.13491`, `TRM` `2510.04871`, `HRM` `2506.21734`, `Saunshi et al.` `2502.17416`, `Huginn-0125` `2502.05171`, `Giannou et al.` `2301.13196`, `Universal Transformers` `1807.03819`) + prior MoralityLabAI manuscripts (`VPD v008`, `RMP`):** indexed in [`literature/README.md`](literature/README.md).

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
