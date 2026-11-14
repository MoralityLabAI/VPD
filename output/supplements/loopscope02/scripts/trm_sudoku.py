"""LOOPSCOPE-02: Pretrained 7M Attention-TRM instrumentation on stratified 9x9 Sudoku.

Loads `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att` (`step_21700`, sha256 verified)
and optional replication `yagizdevre/trm-sudoku-extreme-att` (`model.pt`, sha256 verified)
in standalone pure PyTorch (`TinyRecursiveReasoningModel_ACTV1` parity).

Computes:
  1. Clean trajectory across outer ACT steps s = 1..16 (3 H-cycles x 6 L-cycles per step).
  2. Counterfactual suite (open-loop extrapolation, donor z_H displacement shuffle,
     reset-z_L ablation, donor-z_L swap, step-1 corruption repair, Gaussian kick growth/decay).
  3. SVD rank-one component usage correlations across outer cycles (r_outer) vs. within-cycle
     z_L / z_H role split (r_role) on L_level.layers.{0,1}.{mlp.down_proj, self_attn.o_proj}.
  4. Transported Component Attribution (TCA: fragility F_s and closed-form rotation specificity
     z_read, z_write) unrolled across outer steps.
  5. Q11 Sudoku propagation commutation rho_s (with shuffled-board control) and post-stall
     non-monotone conflict-driven cell revision metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechinterp.loopscope02.scripts import sudoku_prop as SP  # noqa: E402

TRM_CHECKPOINTS = {
    "gaoxin492": {
        "repo_id": "gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att",
        "filename": "step_21700",
        "sha256": "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3",
    },
    "yagizdevre": {
        "repo_id": "yagizdevre/trm-sudoku-extreme-att",
        "filename": "model.pt",
        "sha256": "d67723394a7fe439f1fa498b58196399ee89cd82a794c7da9ce0a8cc3df7fa39",
    },
}
EXIT_POWER_CAP = 75
TOP_SVD = 64


def power_limit_watts() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()[0]
        return float(out.strip())
    except Exception:
        return None


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    orig = x.dtype
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(orig)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig = q.dtype
    qf, kf = q.to(cos.dtype), k.to(cos.dtype)
    c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return ((qf * c) + (rotate_half(qf) * s)).to(orig), ((kf * c) + (rotate_half(kf) * s)).to(orig)


class TRMBlock(nn.Module):
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, expansion: int = 4, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = eps
        raw_inter = int(round(expansion * hidden_size * 2.0 / 3.0))
        inter = ((raw_inter + 255) // 256) * 256 if hidden_size >= 256 else hidden_size * 3
        # Exact shapes from state_dict for hidden_size=512: gate_up=(3072, 512), down=(512, 1536)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, d = h.shape
        q, k, v = [t.view(b, s, self.num_heads, self.head_dim) for t in self.qkv_proj(h).chunk(3, dim=-1)]
        q, k = apply_rope(q, k, cos, sin)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        attn = attn.transpose(1, 2).reshape(b, s, d)
        h = rms_norm(h + self.o_proj(attn), self.eps)
        gate, up = self.gate_up_proj(h).chunk(2, dim=-1)
        h = rms_norm(h + self.down_proj(F.silu(gate) * up), self.eps)
        return h


class PureSudokuTRM(nn.Module):
    """Exact pure-PyTorch implementation of TinyRecursiveReasoningModel_ACTV1 (Sudoku-Extreme)."""

    def __init__(self, hidden_size: int = 512, num_heads: int = 8, l_layers: int = 2,
                 h_cycles: int = 3, l_cycles: int = 6, puzzle_emb_len: int = 16, seq_len: int = 81):
        super().__init__()
        self.hidden_size = hidden_size
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.puzzle_emb_len = puzzle_emb_len
        self.seq_len = seq_len
        self.total_len = puzzle_emb_len + seq_len

        self.register_buffer("H_init", torch.zeros(hidden_size))
        self.register_buffer("L_init", torch.zeros(hidden_size))
        self.embed_tokens = nn.Embedding(11, hidden_size)
        self.register_buffer("puzzle_weight", torch.zeros(1, hidden_size))
        self.layers = nn.ModuleList([TRMBlock(hidden_size, num_heads) for _ in range(l_layers)])
        self.lm_head = nn.Linear(hidden_size, 11, bias=False)
        self.q_head = nn.Linear(hidden_size, 2, bias=True)

        head_dim = hidden_size // num_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(self.total_len, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @classmethod
    def from_state_dict(cls, sd: dict[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16) -> "PureSudokuTRM":
        m = cls()
        p = "_orig_mod.model.inner."
        if p + "H_init" not in sd and "model" in sd:
            sd = sd["model"]
        with torch.no_grad():
            m.H_init.copy_(sd[p + "H_init"].float())
            m.L_init.copy_(sd[p + "L_init"].float())
            m.embed_tokens.weight.copy_(sd[p + "embed_tokens.embedding_weight"].float())
            m.puzzle_weight.copy_(sd[p + "puzzle_emb.weights"].float())
            m.lm_head.weight.copy_(sd[p + "lm_head.weight"].float())
            m.q_head.weight.copy_(sd[p + "q_head.weight"].float())
            m.q_head.bias.copy_(sd[p + "q_head.bias"].float())
            for i, blk in enumerate(m.layers):
                lp = f"{p}L_level.layers.{i}."
                blk.qkv_proj.weight.copy_(sd[lp + "self_attn.qkv_proj.weight"].float())
                blk.o_proj.weight.copy_(sd[lp + "self_attn.o_proj.weight"].float())
                blk.gate_up_proj.weight.copy_(sd[lp + "mlp.gate_up_proj.weight"].float())
                blk.down_proj.weight.copy_(sd[lp + "mlp.down_proj.weight"].float())
        m.to(dtype=dtype)
        m.cos = m.cos.float()
        m.sin = m.sin.float()
        return m.eval()

    def encode_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        b = inputs.shape[0]
        dt = self.embed_tokens.weight.dtype
        emb = self.embed_tokens(inputs)
        puz = F.pad(self.puzzle_weight.to(dt), (0, (self.puzzle_emb_len - 1) * self.hidden_size))
        puz = puz.view(1, self.puzzle_emb_len, self.hidden_size).expand(b, -1, -1)
        return math.sqrt(self.hidden_size) * torch.cat([puz, emb], dim=1)

    def init_states(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.embed_tokens.weight.dtype
        dev = self.embed_tokens.weight.device
        zh = self.H_init.to(device=dev, dtype=dt).view(1, 1, -1).expand(batch_size, self.total_len, -1).clone()
        zl = self.L_init.to(device=dev, dtype=dt).view(1, 1, -1).expand(batch_size, self.total_len, -1).clone()
        return zh, zl

    def run_l_level(self, state: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        h = state + injection
        for blk in self.layers:
            h = blk(h, self.cos, self.sin)
        return h

    def act_step(self, z_h: torch.Tensor, z_l: torch.Tensor, x_emb: torch.Tensor,
                 grad_last_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        ambient_grad = torch.is_grad_enabled()
        for hc in range(self.h_cycles):
            use_grad = ambient_grad and (not grad_last_only or hc == self.h_cycles - 1)
            with torch.set_grad_enabled(use_grad):
                for _ in range(self.l_cycles):
                    z_l = self.run_l_level(z_l, z_h + x_emb)
                z_h = self.run_l_level(z_h, z_l)
        return z_h, z_l

    def digit_logits(self, z_h: torch.Tensor) -> torch.Tensor:
        """Return logits over digits 1..9 (vocab indices 2..10) for the 81 board cells: [B, 81, 9]."""
        raw = self.lm_head(z_h)[:, self.puzzle_emb_len:, 2:11]
        return raw.float()


def encode_questions(questions: list[str], device: torch.device | str) -> torch.Tensor:
    arr = [[1 if ch == "." else int(ch) + 1 for ch in q] for q in questions]
    return torch.tensor(arr, dtype=torch.long, device=device)


def encode_answers(answers: list[str], device: torch.device | str) -> torch.Tensor:
    """Zero-indexed digits 0..8 matching digit_logits [:, :, 0..8] for digits '1'..'9'."""
    arr = [[int(ch) - 1 for ch in a] for a in answers]
    return torch.tensor(arr, dtype=torch.long, device=device)


def load_trm_checkpoint(tag: str, device: str, dtype: torch.dtype = torch.bfloat16) -> tuple[PureSudokuTRM, str]:
    from huggingface_hub import hf_hub_download

    spec = TRM_CHECKPOINTS[tag]
    path = Path(hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"]))
    digest = SP.sha256_file(path)
    if digest != spec["sha256"]:
        raise RuntimeError(f"TRM checkpoint SHA-256 mismatch for {tag}: {digest} != {spec['sha256']}")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    model = PureSudokuTRM.from_state_dict(sd, dtype=dtype).to(device)
    return model, digest


def trm_matrices(model: PureSudokuTRM):
    out = []
    for li, blk in enumerate(model.layers):
        out.append((f"L{li}.o_proj", blk.o_proj))
        out.append((f"L{li}.down_proj", blk.down_proj))
    return out


def svd_factors(model: PureSudokuTRM, top_k: int = TOP_SVD):
    facs = {}
    for name, mod in trm_matrices(model):
        w = mod.weight.detach().double()
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        r = min(top_k, s.numel())
        facs[name] = (
            u[:, :r].float().to(w.device),
            s[:r].float().to(w.device),
            vh[:r, :].T.contiguous().float().to(w.device),
        )
    return facs


@torch.no_grad()
def evaluate_trajectories_and_usage(
    model: PureSudokuTRM,
    questions: list[str],
    answers: list[str],
    donor_indices: list[int],
    n_steps: int = 16,
    batch_size: int = 64,
    record_usage: bool = True,
) -> dict:
    """Run clean s=1..n_steps + counterfactuals + component usage across z_L vs z_H calls."""
    dev = next(model.parameters()).device
    n = len(questions)
    inp_all = encode_questions(questions, dev)
    ans_all = encode_answers(answers, dev)

    total_hc = n_steps * model.h_cycles
    probs_steps = np.zeros((n_steps, n, 81, 9), dtype=np.float32)
    probs_hcycles = np.zeros((total_hc, n, 81, 9), dtype=np.float32)
    z_h_step1 = []
    z_h_stepN = []

    facs = svd_factors(model) if record_usage else {}
    # Usage accumulators: per matrix, per ACT step (0..n_steps-1), role in {'zL', 'zH'} -> [TOP_SVD]
    usage_sum = {
        name: {"zL": np.zeros((n_steps, facs[name][1].numel()), dtype=np.float64),
               "zH": np.zeros((n_steps, facs[name][1].numel()), dtype=np.float64)}
        for name in facs
    }

    for b0 in range(0, n, batch_size):
        b1 = min(n, b0 + batch_size)
        bsz = b1 - b0
        x = model.encode_inputs(inp_all[b0:b1])
        zh, zl = model.init_states(bsz)

        call_state = {"step": 0, "role": "zL"}
        handles = []
        if record_usage:
            for name, mod in trm_matrices(model):
                _, s_vec, v_mat = facs[name]

                def make_hook(nm=name, sv=s_vec, vm=v_mat):
                    def _hk(_m, inputs, _out):
                        xin = inputs[0][:, model.puzzle_emb_len:, :].float()  # board cells [b, 81, d_in]
                        mag = (xin @ vm).abs().mean(dim=1) * sv  # [b, r]
                        usage_sum[nm][call_state["role"]][call_state["step"]] += mag.sum(0).cpu().numpy()
                    return _hk

                handles.append(mod.register_forward_hook(make_hook()))

        try:
            for s in range(n_steps):
                call_state["step"] = s
                for hc in range(model.h_cycles):
                    call_state["role"] = "zL"
                    for _lc in range(model.l_cycles):
                        zl = model.run_l_level(zl, zh + x)
                    call_state["role"] = "zH"
                    zh = model.run_l_level(zh, zl)
                    hc_logits = model.digit_logits(zh)
                    hc_probs = torch.softmax(hc_logits, dim=-1).cpu().numpy()
                    probs_hcycles[s * model.h_cycles + hc, b0:b1] = hc_probs
                probs_steps[s, b0:b1] = hc_probs
                if s == 0:
                    z_h_step1.append(zh.cpu())
                if s == n_steps - 1:
                    z_h_stepN.append(zh.cpu())
        finally:
            for h in handles:
                h.remove()

    zh1_all = torch.cat(z_h_step1, dim=0)  # [N, 97, 512]
    zhN_all = torch.cat(z_h_stepN, dim=0)

    # Counterfactual 1: Open-loop extrapolation at step N: z_H_open = z_H(1) + (N - 1)*(z_H(1) - H_init)
    h_init_cpu = model.H_init.detach().cpu().view(1, 1, -1).to(zh1_all.dtype)
    zh_open = zh1_all + (n_steps - 1) * (zh1_all - h_init_cpu)
    probs_open = []
    for b0 in range(0, n, batch_size):
        logits = model.digit_logits(zh_open[b0:b0 + batch_size].to(dev))
        probs_open.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs_open_arr = np.concatenate(probs_open, axis=0)

    # Counterfactual 2: Donor z_H displacement shuffle: z_H(1)_i + (z_H(N)_donor - z_H(1)_donor)
    donor_idx_t = torch.tensor(donor_indices, dtype=torch.long)
    zh_shuf = zh1_all + (zhN_all[donor_idx_t] - zh1_all[donor_idx_t])
    probs_shuf = []
    for b0 in range(0, n, batch_size):
        logits = model.digit_logits(zh_shuf[b0:b0 + batch_size].to(dev))
        probs_shuf.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs_shuf_arr = np.concatenate(probs_shuf, axis=0)

    # Counterfactual 3: Reset z_L -> L_init between every ACT step (tests persistent latent carry necessity)
    # Counterfactual 3b: Project z_H onto the 11-dim lm_head readout subspace + reset z_L between ACT steps
    #   (preserves step-s output logits identically while erasing the 501-dim orthogonal fiber of z_H!)
    w_head = model.lm_head.weight.detach().float()  # [11, D]
    _, _, vh_head = torch.linalg.svd(w_head, full_matrices=False)  # vh_head: [11, D] orthonormal rows
    p_head = vh_head.T @ vh_head  # [D, D] orthogonal projector onto rowspace(lm_head)
    h_init_perp = (model.H_init.detach().float() @ (torch.eye(model.hidden_size, device=dev) - p_head)).view(1, 1, -1)

    probs_reset_zl = np.zeros((n, 81, 9), dtype=np.float32)
    probs_fiber_erased = np.zeros((n, 81, 9), dtype=np.float32)
    for b0 in range(0, n, batch_size):
        b1 = min(n, b0 + batch_size)
        x = model.encode_inputs(inp_all[b0:b1])
        zh, _ = model.init_states(b1 - b0)
        zh_fib, _ = model.init_states(b1 - b0)
        for s_idx in range(n_steps):
            _, zl_fresh = model.init_states(b1 - b0)
            zh, _ = model.act_step(zh, zl_fresh, x)
            _, zl_fresh2 = model.init_states(b1 - b0)
            zh_fib, _ = model.act_step(zh_fib, zl_fresh2, x)
            if s_idx < n_steps - 1:
                # Preserve puzzle prefix embeddings + project 81 board cells onto rowspace(lm_head)
                board_zh = zh_fib[:, model.puzzle_emb_len:, :].float()
                board_proj = (board_zh @ p_head) + h_init_perp
                zh_fib = torch.cat([zh_fib[:, :model.puzzle_emb_len, :], board_proj.to(zh_fib.dtype)], dim=1)
        probs_reset_zl[b0:b1] = torch.softmax(model.digit_logits(zh), dim=-1).cpu().numpy()
        probs_fiber_erased[b0:b1] = torch.softmax(model.digit_logits(zh_fib), dim=-1).cpu().numpy()

    # Counterfactual 4: Step-1 corruption repair + Gaussian kick perturbation ratio
    # Corrupt 10 empty cells' z_H vectors after step 1 with donor states + measure kick norm growth/decay
    probs_corrupt_step1 = np.zeros((n, 81, 9), dtype=np.float32)
    probs_corrupt_repaired = np.zeros((n, 81, 9), dtype=np.float32)
    kick_ratios = np.zeros((n_steps, n), dtype=np.float32)
    g_rng = torch.Generator(device="cpu").manual_seed(20261113)

    for b0 in range(0, n, batch_size):
        b1 = min(n, b0 + batch_size)
        bsz = b1 - b0
        x = model.encode_inputs(inp_all[b0:b1])
        zh, zl = model.init_states(bsz)
        zh, zl = model.act_step(zh, zl, x)

        # Clean trajectory from step 1 onward for kick comparison
        zh_clean, zl_clean = zh.clone(), zl.clone()
        # Gaussian kick 0.25 * RMS(zh) after step 1
        rms = zh.float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        noise = torch.randn(zh.shape, generator=g_rng, dtype=torch.float32).to(dev) * (0.25 * rms)
        kick_norm0 = noise.flatten(1).norm(dim=1).clamp_min(1e-8)
        zh_kick = (zh.float() + noise).to(zh.dtype)
        zl_kick = zl.clone()
        kick_ratios[0, b0:b1] = 1.0

        # Donor 10-empty-cell patch on z_H after step 1
        d_sub = donor_idx_t[b0:b1]
        zh_donor_b = zh1_all[d_sub].to(dev)
        zh_corr = zh.clone()
        for bi in range(bsz):
            q_str = questions[b0 + bi]
            empties = [idx for idx, ch in enumerate(q_str) if ch == "."]
            perm = torch.randperm(len(empties), generator=g_rng)[: min(10, len(empties))]
            for pi in perm:
                pos = model.puzzle_emb_len + empties[int(pi)]
                zh_corr[bi, pos] = zh_donor_b[bi, pos]
        probs_corrupt_step1[b0:b1] = torch.softmax(model.digit_logits(zh_corr), dim=-1).cpu().numpy()
        zl_corr = zl.clone()

        for s in range(1, n_steps):
            zh_clean, zl_clean = model.act_step(zh_clean, zl_clean, x)
            zh_kick, zl_kick = model.act_step(zh_kick, zl_kick, x)
            zh_corr, zl_corr = model.act_step(zh_corr, zl_corr, x)
            diff = (zh_kick.float() - zh_clean.float()).flatten(1).norm(dim=1)
            kick_ratios[s, b0:b1] = (diff / kick_norm0).cpu().numpy()
        probs_corrupt_repaired[b0:b1] = torch.softmax(model.digit_logits(zh_corr), dim=-1).cpu().numpy()

    # Summarize usage correlations
    usage_stats = {}
    if record_usage:
        for name in usage_sum:
            zl_u = usage_sum[name]["zL"] / (n * model.h_cycles * model.l_cycles)
            zh_u = usage_sum[name]["zH"] / (n * model.h_cycles)
            # Pairwise outer correlation across steps s=1..n_steps-1 (1-based steps 2..16)
            zh_corr_mat = np.corrcoef(zh_u[1:])
            zl_corr_mat = np.corrcoef(zl_u[1:])
            iu = np.triu_indices(zh_corr_mat.shape[0], k=1)
            r_outer_zh = float(np.mean(zh_corr_mat[iu]))
            r_outer_zl = float(np.mean(zl_corr_mat[iu]))
            r_step2_step8 = float(np.corrcoef(zh_u[1], zh_u[min(7, n_steps - 1)])[0, 1])
            # Within-step role correlation between z_L and z_H usage
            role_corrs = [float(np.corrcoef(zl_u[s], zh_u[s])[0, 1]) for s in range(1, n_steps)]
            usage_stats[name] = {
                "r_outer_zH_mean": r_outer_zh,
                "r_outer_zL_mean": r_outer_zl,
                "r_zH_step2_vs_step8": r_step2_step8,
                "r_role_zL_vs_zH_mean": float(np.mean(role_corrs)),
                "r_step1_vs_step2_zH": float(np.corrcoef(zh_u[0], zh_u[1])[0, 1]),
            }

    return {
        "probs_steps": probs_steps,
        "probs_hcycles": probs_hcycles,
        "probs_open": probs_open_arr,
        "probs_donor_zH": probs_shuf_arr,
        "probs_reset_zL": probs_reset_zl,
        "probs_fiber_erased": probs_fiber_erased,
        "probs_corrupt_step1": probs_corrupt_step1,
        "probs_corrupt_repaired": probs_corrupt_repaired,
        "kick_ratios": kick_ratios,
        "usage_stats": usage_stats,
        "targets": ans_all.cpu().numpy(),
    }


def run_trm_tca(
    model_bf16: PureSudokuTRM,
    questions: list[str],
    answers: list[str],
    target_indices: list[int],
    unroll_steps: int = 6,
    chunk_size: int = 8,
) -> dict:
    """Differentiable unrolled TCA across outer steps 1..unroll_steps (1 H-cycle grad chain per step
    or full differentiable unroll in float32) on target empty cells."""
    dev = next(model_bf16.parameters()).device
    # Clone a float32 model for exact unrolled gradients without bf16 underflow
    import copy
    model = copy.deepcopy(model_bf16).float().eval()
    model.requires_grad_(False)
    facs = svd_factors(model, top_k=TOP_SVD)
    mats = trm_matrices(model)
    m_cnt = len(mats)
    n_items = len(questions)
    r_cnt = TOP_SVD

    frag = np.zeros((m_cnt, unroll_steps, n_items), dtype=np.float32)
    zr_abs_gt3 = np.zeros((m_cnt, unroll_steps), dtype=np.float32)
    zw_abs_gt3 = np.zeros((m_cnt, unroll_steps), dtype=np.float32)

    inp_all = encode_questions(questions, dev)
    tgt_digits = torch.tensor([int(answers[i][target_indices[i]]) - 1 for i in range(n_items)],
                              dtype=torch.long, device=dev)

    zr_counts = np.zeros((m_cnt, unroll_steps), dtype=np.int64)
    zw_counts = np.zeros((m_cnt, unroll_steps), dtype=np.int64)

    for b0 in range(0, n_items, chunk_size):
        b1 = min(n_items, b0 + chunk_size)
        bsz = b1 - b0
        model.zero_grad(set_to_none=True)

        # Capture z_H commit call (last call of each ACT step) per matrix
        step_x: dict[int, list[torch.Tensor]] = {mi: [] for mi in range(m_cnt)}
        step_g: dict[int, dict[int, torch.Tensor]] = {mi: {} for mi in range(m_cnt)}
        active = {"capture": False, "s": 0}
        handles = []

        for mi, (_name, mod) in enumerate(mats):
            def make_hk(m_idx=mi):
                def _hk(_mod, inp, out):
                    if not active["capture"]:
                        return
                    s_idx = active["s"]
                    step_x[m_idx].append(inp[0].detach())

                    def _save(grad, m_i=m_idx, s_i=s_idx):
                        step_g[m_i][s_i] = grad.detach()

                    out.register_hook(_save)
                return _hk

            handles.append(mod.register_forward_hook(make_hk()))

        try:
            x = model.encode_inputs(inp_all[b0:b1]).detach().requires_grad_(True)
            zh, zl = model.init_states(bsz)
            for s in range(unroll_steps):
                active["s"] = s
                for hc in range(model.h_cycles):
                    active["capture"] = False
                    for _lc in range(model.l_cycles):
                        zl = model.run_l_level(zl, zh + x)
                    # Capture the final z_H commit call of each outer ACT step
                    active["capture"] = (hc == model.h_cycles - 1)
                    zh = model.run_l_level(zh, zl)
                active["capture"] = False

            logits = model.digit_logits(zh)  # [bsz, 81, 9]
            t_pos = torch.tensor(target_indices[b0:b1], dtype=torch.long, device=dev)
            cell_logits = logits[torch.arange(bsz, device=dev), t_pos]  # [bsz, 9]
            td = tgt_digits[b0:b1]
            zc = cell_logits.gather(1, td[:, None])[:, 0]
            other = cell_logits.scatter(1, td[:, None], float("-inf"))
            margin = zc - torch.logsumexp(other, dim=1)
            margin.sum().backward()

            for mi, (name, _mod) in enumerate(mats):
                u_mat, s_vec, v_mat = facs[name]
                din, dout = v_mat.shape[0], u_mat.shape[0]
                for s in range(unroll_steps):
                    xb = step_x[mi][s]
                    ab = step_g[mi][s]
                    xv = xb @ v_mat
                    au = ab @ u_mat
                    delta = -s_vec * (xv * au).sum(dim=1)  # [bsz, r]
                    xtau = torch.einsum("bpi,bpr->bir", xb, au)
                    atxv = torch.einsum("bpo,bpr->bor", ab, xv)
                    sdr = s_vec * xtau.norm(dim=1) / math.sqrt(din)
                    sdw = s_vec * atxv.norm(dim=1) / math.sqrt(dout)
                    zr = (delta / sdr.clamp_min(1e-12)).abs()
                    zw = (delta / sdw.clamp_min(1e-12)).abs()
                    zr_counts[mi, s] += int((zr > 3.0).sum().item())
                    zw_counts[mi, s] += int((zw > 3.0).sum().item())
                    g_gram = xb @ xb.transpose(1, 2)
                    h_gram = ab @ ab.transpose(1, 2)
                    frag[mi, s, b0:b1] = (g_gram * h_gram).sum(dim=(1, 2)).clamp_min(0).sqrt().cpu().numpy()
        finally:
            for h in handles:
                h.remove()
            del step_x, step_g

    denom = float(n_items * r_cnt)
    zr_abs_gt3 = zr_counts / denom
    zw_abs_gt3 = zw_counts / denom
    f_mean = frag.mean(axis=2)  # [M, S]
    ratios = (f_mean[:, 0] / np.maximum(f_mean[:, -1], 1e-12)).tolist()
    return {
        "unroll_steps": unroll_steps,
        "n_items": n_items,
        "matrices": [m[0] for m in mats],
        "frag_mean_by_step": f_mean.tolist(),
        "f1_over_flate_by_matrix": ratios,
        "f1_over_flate_mean": float(np.mean(ratios)),
        "zr_gt3_share_by_step": zr_abs_gt3.tolist(),
        "zw_gt3_share_by_step": zw_abs_gt3.tolist(),
        "zr_gt3_overall": float(zr_abs_gt3.mean()),
        "zw_gt3_overall": float(zw_abs_gt3.mean()),
    }


def compute_compactification_and_revisions(
    probs_steps: np.ndarray,
    questions: list[str],
    answers: list[str],
    prop_solved_flags: list[bool],
    probs_hcycles: np.ndarray | None = None,
    seed: int = 20261113,
) -> dict:
    """Compute Q11 G_prop commutation rho_s, within-board deductive staircase, and non-monotone revisions."""
    n_steps, n_boards, _, _ = probs_steps.shape
    fine_probs = probs_hcycles if probs_hcycles is not None else probs_steps
    n_fine = fine_probs.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_boards)
    shuf = np.empty(n_boards, dtype=np.int64)
    shuf[perm] = np.roll(perm, -1)

    rho_clean = np.zeros((n_steps - 1, n_boards), dtype=np.float32)
    rho_shuf = np.zeros((n_steps - 1, n_boards), dtype=np.float32)
    conflicts_by_step = np.zeros((n_steps, n_boards), dtype=np.int32)
    exact_by_step = np.zeros((n_steps, n_boards), dtype=bool)
    non_monotone_flips = np.zeros(n_boards, dtype=np.int32)
    had_conflict_before_solve = np.zeros(n_boards, dtype=bool)

    givens_list = [SP.parse_grid(q) for q in questions]
    traces_list = [SP.trace_propagation(q) for q in questions]
    ans_grids = [[int(ch) for ch in a] for a in answers]

    tau_by_group: dict[str, list[float]] = {"r1": [], "r2": [], "r3plus": [], "stalled": []}
    unc_by_group: dict[str, list[float]] = {"r1": [], "r2": [], "r3plus": [], "stalled": []}
    concord_wins = 0.0
    concord_shuf_wins = 0.0
    concord_pairs = 0

    for b in range(n_boards):
        giv = givens_list[b]
        tr = traces_list[b]
        giv_shuf = givens_list[int(shuf[b])]
        empties = [i for i in range(81) if giv[i] == 0]
        preds_steps = np.argmax(probs_steps[:, b, :, :], axis=-1) + 1  # [S, 81] in 1..9
        preds_fine = np.argmax(fine_probs[:, b, :, :], axis=-1) + 1    # [K, 81] in 1..9

        for s in range(n_steps):
            grid_s = [giv[i] if giv[i] != 0 else int(preds_steps[s, i]) for i in range(81)]
            conflicts_by_step[s, b] = SP.count_unit_conflicts(grid_s)
            exact_by_step[s, b] = (grid_s == ans_grids[b])
            if s < n_steps - 1 and empties:
                gp = SP.g_prop_belief(probs_steps[s, b], giv)
                gp_sh = SP.g_prop_belief(probs_steps[s, b], giv_shuf)
                diff = probs_steps[s + 1, b, empties] - gp[empties]
                diff_sh = probs_steps[s + 1, b, empties] - gp_sh[empties]
                rho_clean[s, b] = float(np.linalg.norm(diff, axis=-1).mean())
                rho_shuf[s, b] = float(np.linalg.norm(diff_sh, axis=-1).mean())

        # Count empty cells whose argmax changes across outer steps and fine H-cycles
        flips = 0
        cell_tau: dict[int, int] = {}
        cell_unc: dict[int, float] = {}
        for i in empties:
            seq = preds_fine[:, i]
            flips += int(np.sum(seq[1:] != seq[:-1]))
            true_d = ans_grids[b][i]
            # Lock-in cycle (1..n_fine, or n_fine+1 if not solved at final cycle)
            if seq[-1] != true_d:
                tau_i = n_fine + 1
            else:
                tau_i = n_fine
                for k in range(n_fine - 2, -1, -1):
                    if seq[k] == true_d:
                        tau_i = k + 1
                    else:
                        break
            unc_i = float(np.sum(1.0 - fine_probs[:, b, i, true_d - 1]))
            cell_tau[i] = tau_i
            cell_unc[i] = unc_i
            cr = tr.cell_round[i]
            grp_key = "stalled" if cr < 0 else ("r1" if cr == 1 else ("r2" if cr == 2 else "r3plus"))
            tau_by_group[grp_key].append(float(tau_i))
            unc_by_group[grp_key].append(unc_i)

        # Within-board deductive-depth concordance across empty-cell pairs (i, j) with 1 <= r(i) < r(j)
        prop_empties = [i for i in empties if tr.cell_round[i] >= 1]
        if len(prop_empties) >= 2:
            rounds_vec = [tr.cell_round[i] for i in prop_empties]
            shuf_rounds = list(rng.permutation(rounds_vec))
            uncs_vec = [cell_unc[i] for i in prop_empties]
            for a_i in range(len(prop_empties)):
                for b_j in range(a_i + 1, len(prop_empties)):
                    ri, rj = rounds_vec[a_i], rounds_vec[b_j]
                    ui, uj = uncs_vec[a_i], uncs_vec[b_j]
                    if ri != rj:
                        concord_pairs += 1
                        if (ri < rj and ui < uj) or (ri > rj and ui > uj):
                            concord_wins += 1.0
                        elif ui == uj:
                            concord_wins += 0.5
                    sri, srj = shuf_rounds[a_i], shuf_rounds[b_j]
                    if sri != srj:
                        if (sri < srj and ui < uj) or (sri > srj and ui > uj):
                            concord_shuf_wins += 1.0
                        elif ui == uj:
                            concord_shuf_wins += 0.5

        non_monotone_flips[b] = flips
        if exact_by_step[-1, b] and np.any(conflicts_by_step[:-1, b] > 0):
            had_conflict_before_solve[b] = True

    prop_idx = [i for i, f in enumerate(prop_solved_flags) if f]
    stall_idx = [i for i, f in enumerate(prop_solved_flags) if not f]

    # Commutation separation Mann-Whitney U AUC + paired win-rate on propagation-solvable boards (steps 1..4)
    auc_comm = None
    paired_comm = None
    if prop_idx:
        c_scores = rho_clean[: min(4, n_steps - 1), prop_idx].mean(axis=0)
        s_scores = rho_shuf[: min(4, n_steps - 1), prop_idx].mean(axis=0)
        mw_wins = sum(float(sv > cv) + 0.5 * float(sv == cv) for cv in c_scores for sv in s_scores)
        auc_comm = mw_wins / (len(c_scores) * len(s_scores))
        pw_wins = sum(float(sv > cv) + 0.5 * float(sv == cv) for cv, sv in zip(c_scores, s_scores))
        paired_comm = pw_wins / len(c_scores)

    stalled_solved = [i for i in stall_idx if exact_by_step[-1, i]]
    prop_solved_by_trm = [i for i in prop_idx if exact_by_step[-1, i]]

    return {
        "n_boards": n_boards,
        "n_fine_cycles": n_fine,
        "n_prop_solvable": len(prop_idx),
        "n_stalled": len(stall_idx),
        "exact_solve_curve_all": exact_by_step.mean(axis=1).tolist(),
        "exact_solve_curve_prop": exact_by_step[:, prop_idx].mean(axis=1).tolist() if prop_idx else [],
        "exact_solve_curve_stalled": exact_by_step[:, stall_idx].mean(axis=1).tolist() if stall_idx else [],
        "conflicts_curve_stalled": conflicts_by_step[:, stall_idx].mean(axis=1).tolist() if stall_idx else [],
        "rho_clean_curve_prop": rho_clean[:, prop_idx].mean(axis=1).tolist() if prop_idx else [],
        "rho_shuf_curve_prop": rho_shuf[:, prop_idx].mean(axis=1).tolist() if prop_idx else [],
        "commutation_vs_shuffled_auc_prop": auc_comm,
        "commutation_vs_shuffled_paired_win_prop": paired_comm,
        "staircase_lockin_cycle_mean": {
            k: (float(np.mean(v)) if v else None) for k, v in tau_by_group.items()
        },
        "staircase_cum_uncertainty_mean": {
            k: (float(np.mean(v)) if v else None) for k, v in unc_by_group.items()
        },
        "within_board_round_concordance": float(concord_wins / concord_pairs) if concord_pairs > 0 else None,
        "within_board_round_concordance_shuf_null": (
            float(concord_shuf_wins / concord_pairs) if concord_pairs > 0 else None
        ),
        "within_board_concordance_pairs": concord_pairs,
        "stalled_solve_rate_step16": float(len(stalled_solved) / max(len(stall_idx), 1)),
        "prop_solve_rate_step16": float(len(prop_solved_by_trm) / max(len(prop_idx), 1)),
        "stalled_solved_with_nonmonotone_flips_share": float(
            np.mean([non_monotone_flips[i] > 0 for i in stalled_solved])
        ) if stalled_solved else 0.0,
        "stalled_solved_with_prior_conflict_share": float(
            np.mean([had_conflict_before_solve[i] for i in stalled_solved])
        ) if stalled_solved else 0.0,
        "mean_flips_stalled_solved": float(
            np.mean([non_monotone_flips[i] for i in stalled_solved])
        ) if stalled_solved else 0.0,
        "mean_flips_prop_solved": float(
            np.mean([non_monotone_flips[i] for i in prop_solved_by_trm])
        ) if prop_solved_by_trm else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("mechinterp/loopscope02/results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mem-fraction", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--min-watts", type=float, default=40.0)
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        watts = power_limit_watts()
        if watts is not None and watts < args.min_watts:
            print(f"[power-guard] GPU enforced limit {watts}W < {args.min_watts}W; pausing (75).")
            return EXIT_POWER_CAP
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        torch.backends.cuda.matmul.allow_tf32 = False

    args.out.mkdir(parents=True, exist_ok=True)
    bundle_path = args.out / "benchmark_bundle.json"
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    else:
        bundle = SP.build_benchmark_bundle()
        bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    items = bundle["items"]
    id_to_idx = {it["item_id"]: idx for idx, it in enumerate(items)}
    donor_indices = [id_to_idx[it["donor_item_id"]] for it in items]
    questions = [it["question"] for it in items]
    answers = [it["answer"] for it in items]
    target_indices = [int(it["target_idx"]) for it in items]
    prop_flags = [bool(it["prop_solved"]) for it in items]

    t0 = time.time()
    model, sha = load_trm_checkpoint("gaoxin492", args.device)
    traj = evaluate_trajectories_and_usage(
        model, questions, answers, donor_indices, n_steps=args.steps, batch_size=64, record_usage=True
    )

    # Extract target-cell predictions & logprobs across all 256 stratified items
    n = len(items)
    tgt_d = np.array([int(it["target_digit"]) - 1 for it in items], dtype=np.int64)
    t_idx = np.array(target_indices, dtype=np.int64)
    ar = np.arange(n)

    step_cell_probs = traj["probs_steps"][:, ar, t_idx, :]  # [S, 256, 9]
    step_preds = step_cell_probs.argmax(axis=-1)            # [S, 256]

    # Full-board exact match on the 256 items + counterfactuals
    ans_all = traj["targets"]  # [256, 81]
    board_exact_steps = (traj["probs_steps"].argmax(axis=-1) == ans_all[None, :, :]).all(axis=-1)  # [S, 256]
    board_exact_open = (traj["probs_open"].argmax(axis=-1) == ans_all).all(axis=-1)
    board_exact_donor = (traj["probs_donor_zH"].argmax(axis=-1) == ans_all).all(axis=-1)
    board_exact_reset_zl = (traj["probs_reset_zL"].argmax(axis=-1) == ans_all).all(axis=-1)
    board_exact_fiber_erased = (traj["probs_fiber_erased"].argmax(axis=-1) == ans_all).all(axis=-1)
    board_exact_corr1 = (traj["probs_corrupt_step1"].argmax(axis=-1) == ans_all).all(axis=-1)
    board_exact_corr_rep = (traj["probs_corrupt_repaired"].argmax(axis=-1) == ans_all).all(axis=-1)

    open_preds = traj["probs_open"][ar, t_idx, :].argmax(axis=-1)
    donor_preds = traj["probs_donor_zH"][ar, t_idx, :].argmax(axis=-1)
    reset_zl_preds = traj["probs_reset_zL"][ar, t_idx, :].argmax(axis=-1)
    fiber_erased_preds = traj["probs_fiber_erased"][ar, t_idx, :].argmax(axis=-1)
    corr1_preds = traj["probs_corrupt_step1"][ar, t_idx, :].argmax(axis=-1)
    corr_rep_preds = traj["probs_corrupt_repaired"][ar, t_idx, :].argmax(axis=-1)

    # Run TCA on S1_naked1 (64 items) + S3_prop3plus (64 items) = 128 items
    tca_sel = [i for i, it in enumerate(items) if it["stratum"] in ("S1_naked1", "S3_prop3plus")]
    tca_res = run_trm_tca(
        model,
        [questions[i] for i in tca_sel],
        [answers[i] for i in tca_sel],
        [target_indices[i] for i in tca_sel],
        unroll_steps=6,
        chunk_size=2,
    )

    # Also evaluate the 100 canonical test_sudoku_bench boards
    bench = bundle["bench_boards"]
    b_donors = [(i + 1) % len(bench) for i in range(len(bench))]
    b_traj = evaluate_trajectories_and_usage(
        model,
        [b["question"] for b in bench],
        [b["answer"] for b in bench],
        b_donors,
        n_steps=args.steps,
        batch_size=50,
        record_usage=False,
    )
    b_comp = compute_compactification_and_revisions(
        b_traj["probs_steps"],
        [b["question"] for b in bench],
        [b["answer"] for b in bench],
        [bool(b["prop_solved"]) for b in bench],
        probs_hcycles=b_traj["probs_hcycles"],
    )
    b_ans = b_traj["targets"]
    b_reset_zl_exact = (b_traj["probs_reset_zL"].argmax(axis=-1) == b_ans).all(axis=-1)
    b_fiber_erased_exact = (b_traj["probs_fiber_erased"].argmax(axis=-1) == b_ans).all(axis=-1)
    b_clean16_exact = (b_traj["probs_steps"][-1].argmax(axis=-1) == b_ans).all(axis=-1)
    b_stalled_mask = np.array([not bool(b["prop_solved"]) for b in bench], dtype=bool)

    # Replication check on yagizdevre checkpoint (100 sudoku-bench boards)
    model_rep, sha_rep = load_trm_checkpoint("yagizdevre", args.device)
    rep_traj = evaluate_trajectories_and_usage(
        model_rep,
        [b["question"] for b in bench],
        [b["answer"] for b in bench],
        b_donors,
        n_steps=args.steps,
        batch_size=50,
        record_usage=False,
    )
    rep_exact16 = float(((rep_traj["probs_steps"][-1].argmax(axis=-1) == b_ans).all(axis=-1)).mean())
    rep_exact1 = float(((rep_traj["probs_steps"][0].argmax(axis=-1) == b_ans).all(axis=-1)).mean())

    items_comp = compute_compactification_and_revisions(
        traj["probs_steps"], questions, answers, prop_flags, probs_hcycles=traj["probs_hcycles"]
    )

    elapsed = time.time() - t0
    out_json = {
        "model": "TinyRecursiveModels-Sudoku-Extreme-att (gaoxin492/step_21700)",
        "checkpoint_sha256": sha,
        "replication_model": "yagizdevre/trm-sudoku-extreme-att (model.pt)",
        "replication_sha256": sha_rep,
        "replication_bench100_exact_step1": rep_exact1,
        "replication_bench100_exact_step16": rep_exact16,
        "bundle_sha256": bundle["bundle_sha256"],
        "steps": args.steps,
        "elapsed_sec": elapsed,
        "items_target_correct_by_step": (step_preds == tgt_d[None, :]).astype(int).tolist(),
        "items_target_open_correct": (open_preds == tgt_d).astype(int).tolist(),
        "items_target_donor_correct": (donor_preds == tgt_d).astype(int).tolist(),
        "items_target_reset_zL_correct": (reset_zl_preds == tgt_d).astype(int).tolist(),
        "items_target_fiber_erased_correct": (fiber_erased_preds == tgt_d).astype(int).tolist(),
        "items_target_corrupt_step1_correct": (corr1_preds == tgt_d).astype(int).tolist(),
        "items_target_corrupt_repaired_correct": (corr_rep_preds == tgt_d).astype(int).tolist(),
        "items_board_exact_by_step": board_exact_steps.astype(int).tolist(),
        "items_board_exact_open": board_exact_open.astype(int).tolist(),
        "items_board_exact_donor": board_exact_donor.astype(int).tolist(),
        "items_board_exact_reset_zL": board_exact_reset_zl.astype(int).tolist(),
        "items_board_exact_fiber_erased": board_exact_fiber_erased.astype(int).tolist(),
        "items_board_exact_corrupt_step1": board_exact_corr1.astype(int).tolist(),
        "items_board_exact_corrupt_repaired": board_exact_corr_rep.astype(int).tolist(),
        "kick_ratio_curve_mean": traj["kick_ratios"].mean(axis=1).tolist(),
        "usage_stats": traj["usage_stats"],
        "tca": tca_res,
        "items_compactification": items_comp,
        "bench100_compactification": b_comp,
        "bench100_clean16_exact": b_clean16_exact.astype(int).tolist(),
        "bench100_reset_zL_exact": b_reset_zl_exact.astype(int).tolist(),
        "bench100_fiber_erased_exact": b_fiber_erased_exact.astype(int).tolist(),
        "bench100_stalled_mask": b_stalled_mask.astype(int).tolist(),
    }
    (args.out / "trm_results.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(
        f"[TRM DONE] {elapsed:.1f}s | bench100 exact s1={b_comp['exact_solve_curve_all'][0]:.3f} -> "
        f"s16={b_comp['exact_solve_curve_all'][-1]:.3f} (rep={rep_exact16:.3f}) | "
        f"stalled s16={b_comp['stalled_solve_rate_step16']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
