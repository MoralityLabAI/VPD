# `papers/shared_weights_dark_fibers`: Unified Looped-Transformer Mechanistic Synthesis

This directory contains the unified manuscript:

> **From Shared Weights to Dark Fibers: Virtual Depth, Head-Readable Staircases, and Unmonitored Search in Looped Transformers**  
> Patrick Dugan (November 2026)

It integrates four sealed experimental threads while leaving all prior manuscript directories (`VPD/output/paper_versions/v001..v008` and `papers/rmp_looped_mechinterp/`) untouched:

1. **Astra / Paper I (`Sept 2026`, `VPD` v003–v008):** Multipass VPD + AdamW vs. Muon Top-32 SAEs, $F_\Delta$ support-switching spikes, and 44-layer embedding-reset vs. state-carry controls on `Nanbeige/Nanbeige4.2-3B` across `ARC-AGI-1`, `ARC-AGI-2`, and `INTELLECT-3`.
2. **Silico (`Nov 2026`, `HUG-CAL-02` & `Stage 1`):** Activation-distribution-matched input-read rotation confirmation refuting late-iteration SVD factor specialization on `tomg-group-umd/huginn-0125` (`±0.0065` nats across 8 frozen factors), plus `Nanbeige4.2-3B` multi-hop pointer-chase Stage 1 and 1-hop VPD heatmap.
3. **Opus (`Oct–Nov 2026`, `RMP` Paper II, `LOOPSCOPE-01`, `ARG-02`):** Closed-loop shared dictionaries & MLP-only transcoders, collision-free feature-mediated variable transfer (`LOOP-T` vs. `FF-U`), Transported Component Attribution (`TCA`), 24-arm `LoopedChase` monitorability frontier (`Q4–Q9`), Compactification-Theory B1 (`Q11`, `AUC = 0.992`), `SEM-ECHO-01`, and `ARG-02` Extension N1 on `Nanbeige4.2-3B`.
4. **`LOOPSCOPE-02` (`Nov 2026`, [`../../mechinterp/loopscope02/`](../../mechinterp/loopscope02/)):** Triangulated evaluation of `Nanbeige4.2-3B`, `Qwen3.5-4B-Base`, `Qwen3-4B-Instruct-2507`, and 7M Attention-`TRM` (`step_21700` + `yagizdevre` replication) on deductively stratified 9×9 Sudoku (`items256` + `bench100`), establishing the Detached-ACT vs. BPTT Lyapunov Dichotomy and causal localization of the 501-dimensional `lm_head`-nullspace dark fiber (`1.000 -> 0.070`, $p = 2.02\times 10^{-28}$).

## Contents

- [`main.tex`](main.tex) — LaTeX manuscript source
- [`main.pdf`](main.pdf) — Compiled PDF manuscript
- [`references.bib`](references.bib) — Bibliography (foundational looped-transformer, SAE/transcoder/DAS/APD, and all 5 concurrent 2026 looped-interpretability papers)
- [`PAPER.md`](PAPER.md) — GitHub-readable Markdown edition with embedded PNG figures
- [`build_figures.py`](build_figures.py) — Deterministic CPU script regenerating `figures/*.pdf` and `figures/*.png` from [`../../mechinterp/loopscope02/results/`](../../mechinterp/loopscope02/results/) and audited cross-program summary tables
- [`figures/`](figures/) — Vector `.pdf` and 300-DPI `.png` figures (`fig1`–`fig4`)

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
