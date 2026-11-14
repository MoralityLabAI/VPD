#!/usr/bin/env python3
"""LOOPSCOPE-03 Stage 1: Conditional Cache Refill, Capacity Sweep, and the z_L x z_H Cross.

Splits TRM's carried state S_t = (z_H(t), z_L(t)) at outer step boundary t -> t+1 into:
  - V_t (Visible / Head-Readable Ledger): z_H_board(t) @ P_head (rank-11 lm_head row space)
  - C_t (Monitorable Conditional Cache): H_hat(x, V_t) predicted strictly from (clues x, V_t)
  - R_t (Path-Dependent Memory Residual): H_t - H_hat(x, V_t)

Sweeps refill capacity across 7 nested model classes trained strictly on `train.csv[0:900]`
(selected on held-out `train.csv[900:1000]`, zero overlap with `items256` or `bench100`):
  - C0_init (0 params: sphere-calibrated H_init / L_init)
  - C0_mean (global train_900 mean, zero V_t dependence)
  - C1_linear_local (cell-local ridge on (x_i, V_{t,i}) + cell position, no peer visibility)
  - C2_linear_peer (peer-context ridge adding Sudoku candidate mask + row/col/box probs aggregates)
  - C3_tf_2L_128 (2-layer d=128 RoPE+SwiGLU 97-token Transformer)
  - C4_tf_2L_256 (2-layer d=256 RoPE+SwiGLU 97-token Transformer)
  - C5_tf_2L_512 (2-layer d=512 RoPE+SwiGLU 97-token Transformer — exact TRM core match)
  - C6_tf_4L_512 (4-layer d=512 RoPE+SwiGLU 97-token Transformer — 2x TRM depth/params)
  - C5_tf_2L_512_stepclock (auxiliary control receiving step index min(s, 15))

Evaluates:
  1. Out-of-sample R^2 and directional cosine on `val_100` (WB-Prop vs WB-Stall, across steps).
  2. Full z_H x z_L factorial cross + capacity ladder on `items256` and `bench100` across horizons
     s in {16, 32, 64}, plus periodic visible-board cycle / revisit rates across s = 1..64.
  3. Both primary (`gaoxin492`) and independent replication (`yagizdevre`) TRM checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mechinterp.loopscope02.scripts import fiber_regime_controls as FRC  # noqa: E402
from mechinterp.loopscope02.scripts import score_loopscope02 as SC       # noqa: E402
from mechinterp.loopscope02.scripts import sudoku_prop as SP             # noqa: E402
from mechinterp.loopscope02.scripts import trm_sudoku as TS              # noqa: E402

RESULTS_DIR = REPO_ROOT / "mechinterp" / "loopscope03" / "results"
DEFAULT_CACHE_DIR = Path(
    r"C:\Users\patri\.gemini\antigravity\brain\683081f5-5cee-4346-8423-d0c90341cef0\scratch\loopscope03_models"
)

# Precompute row, col, and box peer index tables [81, 8] for fast vectorized peer features
ROW_PEERS = np.zeros((81, 8), dtype=np.int64)
COL_PEERS = np.zeros((81, 8), dtype=np.int64)
BOX_PEERS = np.zeros((81, 8), dtype=np.int64)
for _i in range(81):
    _r, _c = divmod(_i, 9)
    _br, _bc = (_r // 3) * 3, (_c // 3) * 3
    ROW_PEERS[_i] = [_r * 9 + _j for _j in range(9) if _j != _c]
    COL_PEERS[_i] = [_j * 9 + _c for _j in range(9) if _j != _r]
    BOX_PEERS[_i] = [
        (_br + _dr) * 9 + (_bc + _dc)
        for _dr in range(3)
        for _dc in range(3)
        if (_br + _dr) * 9 + (_bc + _dc) != _i
    ]


def compute_head_projector_f64(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return float32 (V_head [512, 11], P_head [512, 512], P_perp [512, 512]) from float64 SVD."""
    w = weight.detach().cpu().double()
    _u, s, vh = torch.linalg.svd(w, full_matrices=False)
    rank = int((s > 1e-8 * s[0]).sum().item())
    v64 = vh[:rank].T.contiguous()
    p64 = v64 @ v64.T
    p_perp64 = torch.eye(w.shape[1], dtype=torch.float64) - p64
    return v64.float(), p64.float(), p_perp64.float()


def sphere_calibrate_orth(
    raw_orth: torch.Tensor,
    v_proj: torch.Tensor,
    p_perp: torch.Tensor,
    v_head: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project raw_orth [..., 512] onto P_perp and scale to sqrt(max(512 - ||v_proj||^2, 1e-6))."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        rf = raw_orth.float()
        vf = v_proj.float()
        if v_head is not None:
            vh = v_head.float()
            u = rf - (rf @ vh) @ vh.T
        else:
            u = rf @ p_perp.float()
        v_norm_sq = vf.pow(2).sum(dim=-1, keepdim=True)
        target_r = (512.0 - v_norm_sq).clamp_min(1e-6).sqrt()
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return u * (target_r / u_norm)


def rms_calibrate_vec(raw_vec: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Project raw_vec [..., 512] onto the Post-RMSNorm sphere (L2 norm = sqrt(512))."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        xf = raw_vec.float()
        return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)


def build_cand_masks(qs: list[str]) -> torch.Tensor:
    out = np.zeros((len(qs), 81, 9), dtype=np.float32)
    for i, q in enumerate(qs):
        out[i] = SP.candidate_mask(SP.parse_grid(q)).astype(np.float32)
    return torch.from_numpy(out)


@torch.no_grad()
def collect_trm_states(
    model: TS.PureSudokuTRM,
    qs: list[str],
    n_steps: int = 16,
    bsz: int = 100,
) -> dict[str, torch.Tensor]:
    """Collect clean (zh, zl) trajectories [N, S, 97, 512] in bfloat16 on CPU."""
    dev = next(model.parameters()).device
    n = len(qs)
    inp = TS.encode_questions(qs, dev)
    x_all = model.encode_inputs(inp)
    zh_out = torch.empty((n, n_steps, 97, 512), dtype=torch.bfloat16)
    zl_out = torch.empty((n, n_steps, 97, 512), dtype=torch.bfloat16)

    for b0 in range(0, n, bsz):
        b1 = min(n, b0 + bsz)
        zh, zl = model.init_states(b1 - b0)
        xb = x_all[b0:b1]
        for s in range(n_steps):
            zh, zl = model.act_step(zh, zl, xb)
            zh_out[b0:b1, s] = zh.to(torch.bfloat16).cpu()
            zl_out[b0:b1, s] = zl.to(torch.bfloat16).cpu()

    return {
        "inp_tokens": inp.cpu(),               # [N, 81] in 1..10
        "cand_masks": build_cand_masks(qs),    # [N, 81, 9] float32
        "zh": zh_out,                          # [N, S, 97, 512] bf16
        "zl": zl_out,                          # [N, S, 97, 512] bf16
    }


def extract_visible_cell_features(
    inp_tokens: torch.Tensor,
    cand_masks: torch.Tensor,
    v_coords: torch.Tensor,
    w_digit: torch.Tensor,
    include_peers: bool,
) -> torch.Tensor:
    """Build strictly visible features from (inp_tokens [M, 81], cand_masks [M, 81, 9], v_coords [M, 81, 11]).

    w_digit: [9, 11] mapping v_coords -> 9 digit logits identically to model.digit_logits.
    """
    m = inp_tokens.shape[0]
    dev = v_coords.device
    clue_oh = F.one_hot((inp_tokens - 1).clamp(0, 9), num_classes=10).float()  # [M, 81, 10]
    logits = v_coords @ w_digit.T                                              # [M, 81, 9]
    probs = torch.softmax(logits, dim=-1)                                      # [M, 81, 9]
    ent = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)            # [M, 81, 1]
    pos_oh = torch.eye(81, device=dev, dtype=torch.float32).unsqueeze(0).expand(m, -1, -1)

    if not include_peers:
        return torch.cat([clue_oh, v_coords, probs, ent, pos_oh], dim=-1)      # [M, 81, 112]

    # Exact peer aggregates strictly from visible probs & argmax + given clues
    given_dig = inp_tokens - 2  # -1 for empty, 0..8 for clue '1'..'9'
    pred_dig = logits.argmax(dim=-1)
    board_dig = torch.where(given_dig >= 0, given_dig, pred_dig)               # [M, 81]
    board_oh = F.one_hot(board_dig, num_classes=9).float()                     # [M, 81, 9]
    eff_probs = torch.where((given_dig >= 0).unsqueeze(-1), board_oh, probs)   # [M, 81, 9]

    rp = torch.as_tensor(ROW_PEERS, device=dev)
    cp = torch.as_tensor(COL_PEERS, device=dev)
    bp = torch.as_tensor(BOX_PEERS, device=dev)

    r_sum = eff_probs[:, rp, :].sum(dim=2)
    c_sum = eff_probs[:, cp, :].sum(dim=2)
    b_sum = eff_probs[:, bp, :].sum(dim=2)
    r_max = eff_probs[:, rp, :].amax(dim=2)
    c_max = eff_probs[:, cp, :].amax(dim=2)
    b_max = eff_probs[:, bp, :].amax(dim=2)
    peer_conf = (board_oh[:, rp, :].sum(dim=2) + board_oh[:, cp, :].sum(dim=2) + board_oh[:, bp, :].sum(dim=2))

    return torch.cat(
        [clue_oh, v_coords, probs, ent, cand_masks, r_sum, c_sum, b_sum, r_max, c_max, b_max, peer_conf, pos_oh],
        dim=-1,
    )  # [M, 81, 184]


class LinearRefillModel:
    """Closed-form Ridge Conditional Refill mapping (x, V_t) -> (zh_orth, zh_pref, zl)."""

    def __init__(self, include_peers: bool, reg: float = 50.0):
        self.include_peers = include_peers
        self.reg = reg
        self.mu_cell: torch.Tensor | None = None
        self.sd_cell: torch.Tensor | None = None
        self.w_cell: torch.Tensor | None = None   # [D_feat + 1, 1024] -> (zh_orth 512, zl_board 512)
        self.mu_pref: torch.Tensor | None = None
        self.sd_pref: torch.Tensor | None = None
        self.w_pref: torch.Tensor | None = None   # [D_pool + 1, 32 * 512] -> (zh_pref 16x512, zl_pref 16x512)

    @torch.no_grad()
    def fit(
        self,
        train_data: dict[str, torch.Tensor],
        v_head: torch.Tensor,
        w_digit: torch.Tensor,
        p_perp: torch.Tensor,
        dev: str = "cuda",
    ) -> None:
        n_b, n_s, _, _ = train_data["zh"].shape
        m_tot = n_b * n_s
        inp_rep = train_data["inp_tokens"].unsqueeze(1).expand(-1, n_s, -1).reshape(m_tot, 81)
        cand_rep = train_data["cand_masks"].unsqueeze(1).expand(-1, n_s, -1, -1).reshape(m_tot, 81, 9)
        zh_flat = train_data["zh"].reshape(m_tot, 97, 512)
        zl_flat = train_data["zl"].reshape(m_tot, 97, 512)

        v_head_d = v_head.to(dev)
        w_dig_d = w_digit.to(dev)
        p_perp_d = p_perp.to(dev)

        # Streaming mean/std and Gram accumulation on GPU in float64 to avoid huge RAM allocation
        sample_v = zh_flat[:2, 16:, :].float().to(dev) @ v_head_d
        sample_f = extract_visible_cell_features(
            inp_rep[:2].to(dev), cand_rep[:2].to(dev), sample_v, w_dig_d, self.include_peers
        )
        d_feat = sample_f.shape[-1]

        sum_f = torch.zeros(d_feat, dtype=torch.float64, device=dev)
        sum_f2 = torch.zeros(d_feat, dtype=torch.float64, device=dev)
        count_rows = 0
        chunk = 512
        for i0 in range(0, m_tot, chunk):
            i1 = min(m_tot, i0 + chunk)
            zb = zh_flat[i0:i1, 16:, :].float().to(dev)
            vc = zb @ v_head_d
            feat = extract_visible_cell_features(
                inp_rep[i0:i1].to(dev), cand_rep[i0:i1].to(dev), vc, w_dig_d, self.include_peers
            ).reshape(-1, d_feat).double()
            sum_f += feat.sum(dim=0)
            sum_f2 += (feat * feat).sum(dim=0)
            count_rows += feat.shape[0]

        mu_c = sum_f / count_rows
        var_c = (sum_f2 / count_rows - mu_c * mu_c).clamp_min(1e-8)
        sd_c = var_c.sqrt()

        gram = torch.zeros((d_feat + 1, d_feat + 1), dtype=torch.float64, device=dev)
        xty = torch.zeros((d_feat + 1, 1024), dtype=torch.float64, device=dev)

        # Also accumulate pooled board features -> prefix (16*512 zh_pref + 16*512 zl_pref = 16384)
        pool_list = []
        pref_y_list = []

        for i0 in range(0, m_tot, chunk):
            i1 = min(m_tot, i0 + chunk)
            zb = zh_flat[i0:i1, 16:, :].float().to(dev)
            vc = zb @ v_head_d
            feat3 = extract_visible_cell_features(
                inp_rep[i0:i1].to(dev), cand_rep[i0:i1].to(dev), vc, w_dig_d, self.include_peers
            )
            fn = ((feat3.reshape(-1, d_feat).double() - mu_c) / sd_c)
            ones = torch.ones((fn.shape[0], 1), dtype=torch.float64, device=dev)
            xa = torch.cat([fn, ones], dim=1)

            zh_orth = (zb @ p_perp_d).reshape(-1, 512).double()
            zl_b = zl_flat[i0:i1, 16:, :].float().to(dev).reshape(-1, 512).double()
            yb = torch.cat([zh_orth, zl_b], dim=1)

            gram += xa.T @ xa
            xty += xa.T @ yb

            # Board-pooled visible summary (exclude 81-d pos one-hot at tail)
            pool_f = feat3[:, :, :-81].mean(dim=1).double()  # [bs, d_feat - 81]
            pool_list.append(pool_f.cpu())
            yp = torch.cat(
                [zh_flat[i0:i1, :16, :].float().reshape(i1 - i0, -1),
                 zl_flat[i0:i1, :16, :].float().reshape(i1 - i0, -1)],
                dim=1,
            ).double()
            pref_y_list.append(yp.cpu())

        reg_eye = self.reg * torch.eye(d_feat + 1, dtype=torch.float64, device=dev)
        reg_eye[-1, -1] = 0.0
        w_c = torch.linalg.solve(gram + reg_eye, xty).float().cpu()

        pool_all = torch.cat(pool_list, dim=0).to(dev)
        pref_y_all = torch.cat(pref_y_list, dim=0).to(dev)
        mu_p = pool_all.mean(dim=0)
        sd_p = pool_all.std(dim=0).clamp_min(1e-5)
        xp = torch.cat([(pool_all - mu_p) / sd_p, torch.ones((pool_all.shape[0], 1), dtype=torch.float64, device=dev)], dim=1)
        reg_p = self.reg * torch.eye(xp.shape[1], dtype=torch.float64, device=dev)
        reg_p[-1, -1] = 0.0
        w_p = torch.linalg.solve(xp.T @ xp + reg_p, xp.T @ pref_y_all).float().cpu()

        self.mu_cell = mu_c.float().cpu()
        self.sd_cell = sd_c.float().cpu()
        self.w_cell = w_c
        self.mu_pref = mu_p.float().cpu()
        self.sd_pref = sd_p.float().cpu()
        self.w_pref = w_p

    @torch.no_grad()
    def predict(
        self,
        inp_tokens: torch.Tensor,
        cand_masks: torch.Tensor,
        v_coords: torch.Tensor,
        v_proj: torch.Tensor,
        w_digit: torch.Tensor,
        p_perp: torch.Tensor,
        calibrate_sphere: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = v_coords.device
        feat3 = extract_visible_cell_features(inp_tokens, cand_masks, v_coords, w_digit, self.include_peers)
        bsz, _, d_feat = feat3.shape
        mu_c = self.mu_cell.to(dev)
        sd_c = self.sd_cell.to(dev)
        w_c = self.w_cell.to(dev)
        fn = (feat3.reshape(-1, d_feat) - mu_c) / sd_c
        xa = torch.cat([fn, torch.ones((fn.shape[0], 1), dtype=torch.float32, device=dev)], dim=1)
        out_c = (xa @ w_c).view(bsz, 81, 1024)
        zh_orth_raw = out_c[:, :, :512]
        zl_board_raw = out_c[:, :, 512:]

        pool_f = feat3[:, :, :-81].mean(dim=1)
        xp = torch.cat(
            [(pool_f - self.mu_pref.to(dev)) / self.sd_pref.to(dev),
             torch.ones((bsz, 1), dtype=torch.float32, device=dev)],
            dim=1,
        )
        out_p = (xp @ self.w_pref.to(dev)).view(bsz, 32, 512)
        zh_pref_raw = out_p[:, :16, :]
        zl_pref_raw = out_p[:, 16:, :]
        zl_raw = torch.cat([zl_pref_raw, zl_board_raw], dim=1)

        if calibrate_sphere:
            zh_orth = sphere_calibrate_orth(zh_orth_raw, v_proj, p_perp)
            zh_pref = rms_calibrate_vec(zh_pref_raw)
            zl_hat = rms_calibrate_vec(zl_raw)
        else:
            zh_orth = zh_orth_raw @ p_perp
            zh_pref = zh_pref_raw
            zl_hat = zl_raw
        return zh_orth, zh_pref, zl_hat


class RefillTransformer(nn.Module):
    """97-token RoPE+SwiGLU Conditional Refill Transformer mapping (x, V_t) -> (zh_orth, zh_pref, zl)."""

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        use_step_clock: bool = False,
        puz_len: int = 16,
        seq_len: int = 81,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_step_clock = use_step_clock
        self.puz_len = puz_len
        self.seq_len = seq_len
        self.total_len = puz_len + seq_len

        # Visible per-cell input: [clue_oh(10), v_coords(11), probs(9), ent(1), cand_mask(9)] = 40-d
        self.in_board = nn.Linear(40, d_model)
        self.pref_emb = nn.Parameter(torch.randn(1, puz_len, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, self.total_len, d_model) * 0.02)
        if use_step_clock:
            self.step_emb = nn.Embedding(16, d_model)

        self.blocks = nn.ModuleList([TS.TRMBlock(hidden_size=d_model, num_heads=n_heads) for _ in range(n_layers)])
        self.head_zh_pref = nn.Linear(d_model, 512)
        self.head_zh_orth = nn.Linear(d_model, 512)
        self.head_zl = nn.Linear(d_model, 512)

        head_dim = d_model // n_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(self.total_len, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        inp_tokens: torch.Tensor,
        cand_masks: torch.Tensor,
        v_coords: torch.Tensor,
        v_proj: torch.Tensor,
        w_digit: torch.Tensor,
        p_perp: torch.Tensor,
        step_idx: torch.Tensor | None = None,
        calibrate_sphere: bool = True,
        v_head: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = inp_tokens.shape[0]
        clue_oh = F.one_hot((inp_tokens - 1).clamp(0, 9), num_classes=10).float()
        logits = v_coords.float() @ w_digit.float().T
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)
        vis_in = torch.cat([clue_oh, v_coords.float() * 0.2, probs, ent * 0.5, cand_masks.float()], dim=-1)

        dt = self.in_board.weight.dtype
        h_board = self.in_board(vis_in.to(dt))
        h_pref = self.pref_emb.to(dt).expand(bsz, -1, -1)
        h = torch.cat([h_pref, h_board], dim=1) + self.pos_emb.to(dt)
        if self.use_step_clock:
            assert step_idx is not None
            h = h + self.step_emb(step_idx.clamp(0, 15)).to(dt).unsqueeze(1)

        h = TS.rms_norm(h)
        for blk in self.blocks:
            h = blk(h, self.cos, self.sin)

        zh_pref_raw = self.head_zh_pref(h[:, : self.puz_len, :]).float()
        zh_orth_raw = self.head_zh_orth(h[:, self.puz_len :, :]).float()
        zl_raw = self.head_zl(h).float()

        if calibrate_sphere:
            zh_orth = sphere_calibrate_orth(zh_orth_raw, v_proj, p_perp, v_head=v_head)
            zh_pref = rms_calibrate_vec(zh_pref_raw)
            zl_hat = rms_calibrate_vec(zl_raw)
        else:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                ro_f = zh_orth_raw.float()
                zh_orth = (
                    ro_f - (ro_f @ v_head.float()) @ v_head.float().T
                    if v_head is not None
                    else ro_f @ p_perp.float()
                )
                zh_pref = zh_pref_raw.float()
                zl_hat = zl_raw.float()
        return zh_orth, zh_pref, zl_hat


def train_refill_transformer(
    net: RefillTransformer,
    train_data: dict[str, torch.Tensor],
    val_data: dict[str, torch.Tensor],
    v_head: torch.Tensor,
    w_digit: torch.Tensor,
    p_perp: torch.Tensor,
    epochs: int = 36,
    batch_size: int = 256,
    lr: float = 1.2e-3,
    dev: str = "cuda",
) -> tuple[RefillTransformer, dict]:
    """Train RefillTransformer on train_900 and select best epoch on val_100 spherical MSE."""
    net = net.to(dev)
    v_head_d = v_head.to(dev)
    w_dig_d = w_digit.to(dev)
    p_perp_d = p_perp.to(dev)

    n_tr, n_s, _, _ = train_data["zh"].shape
    m_tr = n_tr * n_s
    # Keep bf16 training & validation buffers directly on GPU and precompute rank-11 visible coords
    tr_inp = train_data["inp_tokens"].unsqueeze(1).expand(-1, n_s, -1).reshape(m_tr, 81).to(dev)
    tr_cand = train_data["cand_masks"].unsqueeze(1).expand(-1, n_s, -1, -1).reshape(m_tr, 81, 9).to(dev)
    tr_step = torch.arange(n_s, dtype=torch.long, device=dev).unsqueeze(0).expand(n_tr, -1).reshape(m_tr)
    tr_zh = train_data["zh"].reshape(m_tr, 97, 512).to(dev)
    tr_zl = train_data["zl"].reshape(m_tr, 97, 512).to(dev)
    with torch.no_grad():
        tr_vc = torch.empty((m_tr, 81, v_head_d.shape[1]), dtype=torch.float32, device=dev)
        for i0 in range(0, m_tr, 1024):
            i1 = min(m_tr, i0 + 1024)
            tr_vc[i0:i1] = tr_zh[i0:i1, 16:, :].float() @ v_head_d

    n_va = val_data["zh"].shape[0]
    m_va = n_va * n_s
    va_inp = val_data["inp_tokens"].unsqueeze(1).expand(-1, n_s, -1).reshape(m_va, 81).to(dev)
    va_cand = val_data["cand_masks"].unsqueeze(1).expand(-1, n_s, -1, -1).reshape(m_va, 81, 9).to(dev)
    va_step = torch.arange(n_s, dtype=torch.long, device=dev).unsqueeze(0).expand(n_va, -1).reshape(m_va)
    va_zh = val_data["zh"].reshape(m_va, 97, 512).to(dev)
    va_zl = val_data["zl"].reshape(m_va, 97, 512).to(dev)
    with torch.no_grad():
        va_vc = va_zh[:, 16:, :].float() @ v_head_d

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    total_steps = epochs * ((m_tr + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=3e-5)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    rng = np.random.default_rng(20261114)
    t0 = time.time()

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if dev == "cuda" else False
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for ep in range(epochs):
            net.train()
            perm = torch.from_numpy(rng.permutation(m_tr)).to(dev)
            for b0 in range(0, m_tr, batch_size):
                idx = perm[b0 : b0 + batch_size]
                inp_b = tr_inp[idx]
                cand_b = tr_cand[idx]
                step_b = tr_step[idx]
                zh_b = tr_zh[idx].float()
                zl_b = tr_zl[idx].float()
                vc = tr_vc[idx]
                vp = vc @ v_head_d.T
                tgt_orth = zh_b[:, 16:, :] - vp
                tgt_pref = zh_b[:, :16, :]

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    pred_orth, pred_pref, pred_zl = net(
                        inp_b, cand_b, vc, vp, w_dig_d, p_perp_d,
                        step_idx=step_b, calibrate_sphere=True, v_head=v_head_d,
                    )
                loss_orth = (pred_orth - tgt_orth).pow(2).mean()
                loss_pref = (pred_pref - tgt_pref).pow(2).mean()
                loss_zl = (pred_zl - zl_b).pow(2).mean()
                loss = loss_orth + 0.25 * loss_pref + loss_zl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()

            # Evaluate on held-out val_100 every 3 epochs (and final epoch)
            if (ep + 1) % 3 == 0 or ep == epochs - 1:
                net.eval()
                val_sum = 0.0
                val_cnt = 0
                with torch.no_grad():
                    for b0 in range(0, m_va, 512):
                        b1 = min(m_va, b0 + 512)
                        inp_b = va_inp[b0:b1]
                        cand_b = va_cand[b0:b1]
                        step_b = va_step[b0:b1]
                        zh_b = va_zh[b0:b1].float()
                        zl_b = va_zl[b0:b1].float()
                        vc = va_vc[b0:b1]
                        vp = vc @ v_head_d.T
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                            po, pp, pz = net(
                                inp_b, cand_b, vc, vp, w_dig_d, p_perp_d,
                                step_idx=step_b, calibrate_sphere=True, v_head=v_head_d,
                            )
                        v_loss = (
                            (po - (zh_b[:, 16:, :] - vp)).pow(2).mean()
                            + 0.25 * (pp - zh_b[:, :16, :]).pow(2).mean()
                            + (pz - zl_b).pow(2).mean()
                        )
                        val_sum += float(v_loss.item()) * (b1 - b0)
                        val_cnt += (b1 - b0)
                val_mse = val_sum / max(val_cnt, 1)
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    best_epoch = ep + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    finally:
        if dev == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    return net, {
        "n_params": net.num_parameters(),
        "best_epoch": best_epoch,
        "best_val_objective": best_val_loss,
        "train_sec": float(time.time() - t0),
    }


@torch.no_grad()
def evaluate_val_r2(
    predictor_fn,
    val_data: dict[str, torch.Tensor],
    val_prop_flags: np.ndarray,
    mean_baseline: dict[str, torch.Tensor],
    v_head: torch.Tensor,
    p_perp: torch.Tensor,
    dev: str = "cuda",
) -> dict:
    """Compute held-out val_100 R^2 (1 - SSE_model / SSE_global_mean) and cosine similarity."""
    v_head_d = v_head.to(dev)
    p_perp_d = p_perp.to(dev)
    mu_orth = mean_baseline["mu_orth"].to(dev)   # [1, 81, 512]
    mu_pref = mean_baseline["mu_pref"].to(dev)   # [1, 16, 512]
    mu_zl = mean_baseline["mu_zl"].to(dev)       # [1, 97, 512]

    n_va, n_s, _, _ = val_data["zh"].shape
    sse_orth = np.zeros((n_va, n_s), dtype=np.float64)
    sst_orth = np.zeros((n_va, n_s), dtype=np.float64)
    sse_zl = np.zeros((n_va, n_s), dtype=np.float64)
    sst_zl = np.zeros((n_va, n_s), dtype=np.float64)
    sse_pref = np.zeros((n_va, n_s), dtype=np.float64)
    sst_pref = np.zeros((n_va, n_s), dtype=np.float64)
    cos_orth = np.zeros((n_va, n_s), dtype=np.float64)
    cos_zl = np.zeros((n_va, n_s), dtype=np.float64)

    inp_d = val_data["inp_tokens"].to(dev)
    cand_d = val_data["cand_masks"].to(dev)

    for s in range(n_s):
        zh_s = val_data["zh"][:, s].to(dev).float()
        zl_s = val_data["zl"][:, s].to(dev).float()
        zb = zh_s[:, 16:, :]
        vc = zb @ v_head_d
        vp = vc @ v_head_d.T
        tgt_orth = zb - vp
        tgt_pref = zh_s[:, :16, :]

        po, pp, pz = predictor_fn(inp_d, cand_d, vc, vp, s)
        # Baseline sphere-calibrated global mean
        bo = sphere_calibrate_orth(mu_orth.expand(n_va, -1, -1), vp, p_perp_d)
        bp = mu_pref.expand(n_va, -1, -1)
        bz = mu_zl.expand(n_va, -1, -1)

        sse_orth[:, s] = (po - tgt_orth).pow(2).sum(dim=(1, 2)).cpu().numpy()
        sst_orth[:, s] = (bo - tgt_orth).pow(2).sum(dim=(1, 2)).cpu().numpy()
        sse_zl[:, s] = (pz - zl_s).pow(2).sum(dim=(1, 2)).cpu().numpy()
        sst_zl[:, s] = (bz - zl_s).pow(2).sum(dim=(1, 2)).cpu().numpy()
        sse_pref[:, s] = (pp - tgt_pref).pow(2).sum(dim=(1, 2)).cpu().numpy()
        sst_pref[:, s] = (bp - tgt_pref).pow(2).sum(dim=(1, 2)).cpu().numpy()

        cos_orth[:, s] = F.cosine_similarity(po, tgt_orth, dim=-1).mean(dim=-1).cpu().numpy()
        cos_zl[:, s] = F.cosine_similarity(pz, zl_s, dim=-1).mean(dim=-1).cpu().numpy()

    def summarize_mask(mask: np.ndarray) -> dict:
        r2_o = 1.0 - float(sse_orth[mask].sum() / max(sst_orth[mask].sum(), 1e-12))
        r2_z = 1.0 - float(sse_zl[mask].sum() / max(sst_zl[mask].sum(), 1e-12))
        r2_p = 1.0 - float(sse_pref[mask].sum() / max(sst_pref[mask].sum(), 1e-12))
        r2_joint = 1.0 - float(
            (sse_orth[mask].sum() + sse_pref[mask].sum() + sse_zl[mask].sum())
            / max(sst_orth[mask].sum() + sst_pref[mask].sum() + sst_zl[mask].sum(), 1e-12)
        )
        by_step_o = {
            str(st + 1): 1.0 - float(sse_orth[mask, st].sum() / max(sst_orth[mask, st].sum(), 1e-12))
            for st in (0, 1, 3, 7, 11, 15)
        }
        by_step_z = {
            str(st + 1): 1.0 - float(sse_zl[mask, st].sum() / max(sst_zl[mask, st].sum(), 1e-12))
            for st in (0, 1, 3, 7, 11, 15)
        }
        return {
            "r2_zh_orth": r2_o,
            "r2_zl": r2_z,
            "r2_zh_pref": r2_p,
            "r2_joint_hidden": r2_joint,
            "cos_zh_orth": float(cos_orth[mask].mean()),
            "cos_zl": float(cos_zl[mask].mean()),
            "r2_zh_orth_by_step": by_step_o,
            "r2_zl_by_step": by_step_z,
        }

    return {
        "all": summarize_mask(np.ones(n_va, dtype=bool)),
        "prop": summarize_mask(val_prop_flags),
        "stalled": summarize_mask(~val_prop_flags),
    }


@torch.no_grad()
def verify_zero_leak_and_logit_parity(
    model: TS.PureSudokuTRM,
    refill_fn,
    inp_tokens: torch.Tensor,
    cand_masks: torch.Tensor,
    v_head: torch.Tensor,
    p_perp: torch.Tensor,
    dev: str = "cuda",
) -> dict[str, float]:
    """Programmatically verify that refill_fn depends ONLY on (x, V_t) and preserves logits < 1e-5."""
    bsz = min(8, inp_tokens.shape[0])
    inp_b = inp_tokens[:bsz].to(dev)
    cand_b = cand_masks[:bsz].to(dev)
    xb = model.encode_inputs(inp_b)
    zh, zl = model.init_states(bsz)
    zh, zl = model.act_step(zh, zl, xb)

    v_head_d = v_head.to(dev)
    p_perp_d = p_perp.to(dev)

    zb1 = zh[:, 16:, :].float()
    vc1 = zb1 @ v_head_d
    vp1 = vc1 @ v_head_d.T
    po1, pp1, pz1 = refill_fn(inp_b, cand_b, vc1, vp1, 0)

    # Construct counterfactual zh2 with IDENTICAL V_t and arbitrary random orthogonal/prefix state
    g = torch.Generator(device=dev).manual_seed(999)
    rand_orth = torch.randn((bsz, 81, 512), generator=g, device=dev) @ p_perp_d
    zb2 = vp1 + rand_orth
    vc2 = zb2 @ v_head_d
    vp2 = vc2 @ v_head_d.T
    po2, pp2, pz2 = refill_fn(inp_b, cand_b, vc2, vp2, 0)

    zh_rebuilt = torch.cat([pp1, vp1 + po1], dim=1).to(zh.dtype)
    orig_logits = model.digit_logits(zh)
    reb_logits = model.digit_logits(zh_rebuilt)
    # Also check float32 exact logit diff prior to bf16 cast
    w_head = model.lm_head.weight.detach().float()
    f32_logit_diff = float(((zb1 @ w_head.T)[:, :, 2:11] - ((vp1 + po1) @ w_head.T)[:, :, 2:11]).abs().max().item())
    bf16_logit_diff = float((orig_logits - reb_logits).abs().max().item())
    leak_diff = float(max((po1 - po2).abs().max().item(), (pp1 - pp2).abs().max().item(), (pz1 - pz2).abs().max().item()))

    assert f32_logit_diff < 1e-4, f"Float32 logit parity violated: {f32_logit_diff}"
    assert leak_diff < 1e-4, f"Hidden-state leak detected in refill_fn: {leak_diff}"
    return {
        "max_f32_logit_diff": f32_logit_diff,
        "max_bf16_logit_diff": bf16_logit_diff,
        "max_hidden_perturbation_leak": leak_diff,
    }


@torch.no_grad()
def run_closed_loop_arm(
    model: TS.PureSudokuTRM,
    qs: list[str],
    ans: list[str],
    prop_flags: list[bool],
    strata: list[str] | None,
    v_head: torch.Tensor,
    p_perp: torch.Tensor,
    zh_orth_mode: str,
    zh_pref_mode: str,
    zl_mode: str,
    refill_fn,
    mean_base: dict[str, torch.Tensor],
    max_steps: int = 64,
    bsz: int = 64,
    calibrate_sphere: bool = True,
) -> dict:
    """Run closed-loop TRM for s = 1..max_steps applying specified inter-step ledger rules."""
    dev = next(model.parameters()).device
    dt = model.embed_tokens.weight.dtype
    n = len(qs)
    inp_all = TS.encode_questions(qs, dev)
    ans_np = np.array([[int(ch) - 1 for ch in a] for a in ans], dtype=np.int64)
    empty_mask = np.array([[ch == "." for ch in q] for q in qs], dtype=bool)
    given_np = np.array([[-1 if ch == "." else int(ch) - 1 for ch in q] for q in qs], dtype=np.int64)
    cand_all = build_cand_masks(qs).to(dev)
    x_all = model.encode_inputs(inp_all)

    v_head_d = v_head.to(dev)
    p_perp_d = p_perp.to(dev)
    h_init_vec = model.H_init.detach().float().to(dev).view(1, 1, 512)
    l_init_vec = model.L_init.detach().float().to(dev).view(1, 1, 512)
    mu_orth = mean_base["mu_orth"].to(dev)
    mu_pref = mean_base["mu_pref"].to(dev)
    mu_zl = mean_base["mu_zl"].to(dev)

    preds_all = np.empty((max_steps, n, 81), dtype=np.int8)

    for b0 in range(0, n, bsz):
        b1 = min(n, b0 + bsz)
        m = b1 - b0
        zh, zl = model.init_states(m)
        xb = x_all[b0:b1]
        inp_b = inp_all[b0:b1]
        cand_b = cand_all[b0:b1]

        for s in range(max_steps):
            zh, zl = model.act_step(zh, zl, xb)
            preds_all[s, b0:b1] = model.digit_logits(zh).argmax(dim=-1).cpu().numpy().astype(np.int8)

            if s < max_steps - 1 and not (zh_orth_mode == "live" and zh_pref_mode == "live" and zl_mode == "live"):
                zb = zh[:, 16:, :].float()
                vc = zb @ v_head_d
                vp = vc @ v_head_d.T

                need_refill = ("refill" in (zh_orth_mode, zh_pref_mode, zl_mode))
                if need_refill:
                    ro, rp, rz = refill_fn(inp_b, cand_b, vc, vp, s, calibrate_sphere=calibrate_sphere)
                else:
                    ro = rp = rz = None

                # 1. z_H board orthogonal fiber
                if zh_orth_mode == "live":
                    new_zb = zb
                elif zh_orth_mode == "norm_only":
                    orig_n = zb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    new_zb = vp * (orig_n / vp.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                elif zh_orth_mode == "init_sphere":
                    new_zb = vp + sphere_calibrate_orth(h_init_vec.expand(m, 81, -1), vp, p_perp_d)
                elif zh_orth_mode == "mean_sphere":
                    new_zb = vp + sphere_calibrate_orth(mu_orth.expand(m, -1, -1), vp, p_perp_d)
                elif zh_orth_mode == "refill":
                    new_zb = vp + ro
                else:
                    raise ValueError(zh_orth_mode)

                # 2. z_H prefix [:, :16, :]
                if zh_pref_mode == "live":
                    new_zp = zh[:, :16, :].float()
                elif zh_pref_mode == "init_sphere":
                    new_zp = rms_calibrate_vec(h_init_vec.expand(m, 16, -1))
                elif zh_pref_mode == "mean_sphere":
                    new_zp = mu_pref.expand(m, -1, -1)
                elif zh_pref_mode == "refill":
                    new_zp = rp
                else:
                    raise ValueError(zh_pref_mode)

                # 3. z_L [:, :97, :]
                if zl_mode == "live":
                    new_zl = zl.float()
                elif zl_mode == "reset_init":
                    new_zl = l_init_vec.expand(m, 97, -1)
                elif zl_mode == "mean_sphere":
                    new_zl = mu_zl.expand(m, -1, -1)
                elif zl_mode == "refill":
                    new_zl = rz
                else:
                    raise ValueError(zl_mode)

                zh = torch.cat([new_zp, new_zb], dim=1).to(dt)
                zl = new_zl.to(dt)

    # Full-board merged grids across s = 1..max_steps
    merged = np.where(empty_mask[None, :, :], preds_all, given_np[None, :, :])  # [S, N, 81]
    exact_steps = (merged == ans_np[None, :, :]).all(axis=-1)                   # [S, N]

    # Measure unsolved board revisits / periodic limit cycles across steps 1..max_steps:
    # A step s >= 2 is a "refuted-board revisit" if board[s] != ans and board[s] != board[s-1]
    # AND board[s] appeared at some earlier step u <= s - 2!
    # Also count exact repeat/periodic cycle rate among boards unsolved at s = max_steps.
    board_revisit_counts = np.zeros(n, dtype=np.int32)
    distinct_boards_visited = np.zeros(n, dtype=np.int32)
    trapped_in_exact_cycle = np.zeros(n, dtype=bool)
    first_solve_step = np.full(n, max_steps + 1, dtype=np.int32)

    for bi in range(n):
        sol_steps = np.where(exact_steps[:, bi])[0]
        if len(sol_steps) > 0:
            first_solve_step[bi] = int(sol_steps[0]) + 1

        seen_first: dict[bytes, int] = {}
        revisits = 0
        for s in range(max_steps):
            b_bytes = merged[s, bi].tobytes()
            if not exact_steps[s, bi]:
                if s >= 1 and b_bytes != merged[s - 1, bi].tobytes() and b_bytes in seen_first:
                    revisits += 1
            if b_bytes not in seen_first:
                seen_first[b_bytes] = s
        board_revisit_counts[bi] = revisits
        distinct_boards_visited[bi] = len(seen_first)
        if not exact_steps[-1, bi]:
            # Check if any board in steps [max_steps//2 .. max_steps-1] repeats
            tail_bytes = [merged[s, bi].tobytes() for s in range(max_steps // 2, max_steps)]
            trapped_in_exact_cycle[bi] = (len(set(tail_bytes)) < len(tail_bytes))

    prop_arr = np.array(prop_flags, dtype=bool)
    stall_arr = ~prop_arr

    def summarize_sub(mask: np.ndarray) -> dict:
        cnt = int(mask.sum())
        if cnt == 0:
            return {}
        return {
            "n": cnt,
            "solve_s16": float(exact_steps[15, mask].mean()),
            "solve_s32": float(exact_steps[min(31, max_steps - 1), mask].mean()),
            "solve_s64": float(exact_steps[max_steps - 1, mask].mean()),
            "ever_solved_by_s64": float((first_solve_step[mask] <= max_steps).mean()),
            "mean_first_solve_step_capped65": float(first_solve_step[mask].mean()),
            "mean_refuted_board_revisits_s64": float(board_revisit_counts[mask].mean()),
            "any_refuted_board_revisit_rate": float((board_revisit_counts[mask] > 0).mean()),
            "unsolved_tail_exact_cycle_rate": float(trapped_in_exact_cycle[mask].mean()),
            "mean_distinct_boards_s64": float(distinct_boards_visited[mask].mean()),
        }

    out = {
        "all": summarize_sub(np.ones(n, dtype=bool)),
        "wb_prop": summarize_sub(prop_arr),
        "wb_stall": summarize_sub(stall_arr),
        "exact_s16_flags": exact_steps[15].tolist(),
        "exact_s64_flags": exact_steps[max_steps - 1].tolist(),
    }
    if strata is not None:
        st_arr = np.array(strata)
        out["by_stratum_s16"] = {
            st: float(exact_steps[15, st_arr == st].mean())
            for st in ("S1_naked1", "S2_prop2", "S3_prop3plus", "S4_stalled")
        }
        out["by_stratum_s64"] = {
            st: float(exact_steps[max_steps - 1, st_arr == st].mean())
            for st in ("S1_naked1", "S2_prop2", "S3_prop3plus", "S4_stalled")
        }
    return out


def attach_mcnemar_vs_clean(arms: dict[str, dict], prop_flags: list[bool]) -> None:
    clean_s16 = np.array(arms["clean"]["exact_s16_flags"], dtype=bool)
    clean_s64 = np.array(arms["clean"]["exact_s64_flags"], dtype=bool)
    prop_arr = np.array(prop_flags, dtype=bool)
    stall_arr = ~prop_arr

    for name, res in arms.items():
        s16 = np.array(res["exact_s16_flags"], dtype=bool)
        s64 = np.array(res["exact_s64_flags"], dtype=bool)
        res["mcnemar_s16_wb_stall_vs_clean"] = SC.exact_mcnemar_p(clean_s16[stall_arr], s16[stall_arr])
        res["mcnemar_s64_wb_stall_vs_clean"] = SC.exact_mcnemar_p(clean_s64[stall_arr], s64[stall_arr])
        res["mcnemar_s16_wb_prop_vs_clean"] = SC.exact_mcnemar_p(clean_s16[prop_arr], s16[prop_arr])
        del res["exact_s16_flags"]
        del res["exact_s64_flags"]


def run_checkpoint_stage1(
    ckpt_tag: str,
    train_qs: list[str],
    val_qs: list[str],
    val_prop_flags: np.ndarray,
    items_qs: list[str],
    items_ans: list[str],
    items_prop: list[bool],
    items_strata: list[str],
    bench_qs: list[str],
    bench_ans: list[str],
    bench_prop: list[bool],
    cache_dir: Path,
    dev: str = "cuda",
) -> dict:
    done_json_path = cache_dir / f"{ckpt_tag}_stage1_done.json"
    if done_json_path.exists() and (cache_dir / f"{ckpt_tag}_refill_bundle.pt").exists():
        print(f"\n=== Stage 1 Conditional Refill on checkpoint: {ckpt_tag} (loaded cached receipt) ===", flush=True)
        return json.loads(done_json_path.read_text(encoding="utf-8"))

    t_ck = time.time()
    print(f"\n=== Stage 1 Conditional Refill on checkpoint: {ckpt_tag} ===", flush=True)
    model, sha = TS.load_trm_checkpoint(ckpt_tag, device=dev, dtype=torch.bfloat16)
    v_head, _p_head, p_perp = compute_head_projector_f64(model.lm_head.weight)
    w_digit = (model.lm_head.weight.detach().cpu().float()[2:11, :] @ v_head).contiguous()  # [9, 11]

    print(f"[{ckpt_tag}] Collecting clean train_900 + val_100 trajectories...", flush=True)
    tr_data = collect_trm_states(model, train_qs, n_steps=16, bsz=100)
    va_data = collect_trm_states(model, val_qs, n_steps=16, bsz=100)

    # Compute global train_900 mean baseline
    zh_tr_b = tr_data["zh"][:, :, 16:, :].float()
    vc_tr = zh_tr_b @ v_head
    vp_tr = vc_tr @ v_head.T
    orth_tr = zh_tr_b - vp_tr
    mu_orth = (orth_tr.mean(dim=(0, 1), keepdim=False).unsqueeze(0) @ p_perp).contiguous()  # [1, 81, 512]
    mu_pref = rms_calibrate_vec(tr_data["zh"][:, :, :16, :].float().mean(dim=(0, 1)).unsqueeze(0)).contiguous()
    mu_zl = rms_calibrate_vec(tr_data["zl"].float().mean(dim=(0, 1)).unsqueeze(0)).contiguous()
    del zh_tr_b, vc_tr, vp_tr, orth_tr
    mean_base = {"mu_orth": mu_orth, "mu_pref": mu_pref, "mu_zl": mu_zl}

    v_head_d = v_head.to(dev)
    w_dig_d = w_digit.to(dev)
    p_perp_d = p_perp.to(dev)

    # Fit C1 (linear local) and C2 (linear peer)
    print(f"[{ckpt_tag}] Fitting C1_linear_local and C2_linear_peer...", flush=True)
    c1 = LinearRefillModel(include_peers=False, reg=50.0)
    c1.fit(tr_data, v_head, w_digit, p_perp, dev=dev)
    c2 = LinearRefillModel(include_peers=True, reg=50.0)
    c2.fit(tr_data, v_head, w_digit, p_perp, dev=dev)

    # Fit Transformer ladder C3..C6 (+ stepclock check on primary) with matched head_dim=64
    tf_specs = [
        ("C3_tf_2L_128", dict(d_model=128, n_layers=2, n_heads=2, use_step_clock=False), 1.5e-3),
        ("C4_tf_2L_256", dict(d_model=256, n_layers=2, n_heads=4, use_step_clock=False), 1.3e-3),
        ("C5_tf_2L_512", dict(d_model=512, n_layers=2, n_heads=8, use_step_clock=False), 1.0e-3),
        ("C6_tf_4L_512", dict(d_model=512, n_layers=4, n_heads=8, use_step_clock=False), 8.0e-4),
    ]
    if ckpt_tag == "gaoxin492":
        tf_specs.append(
            ("C5_tf_2L_512_stepclock", dict(d_model=512, n_layers=2, n_heads=8, use_step_clock=True), 1.0e-3)
        )

    tf_models: dict[str, RefillTransformer] = {}
    tf_train_meta: dict[str, dict] = {}
    ladder_cache_path = cache_dir / f"{ckpt_tag}_all_tf_ladder_v2.pt"
    if ladder_cache_path.exists():
        print(f"[{ckpt_tag}] Loading cached Transformer ladder from {ladder_cache_path}...", flush=True)
        saved_l = torch.load(ladder_cache_path, map_location="cpu", weights_only=False)
        tf_train_meta = saved_l["meta"]
        for name, kw, _lr_val in tf_specs:
            net = RefillTransformer(**kw)
            net.load_state_dict(saved_l["states"][name])
            tf_models[name] = net.to(dev).eval()
    else:
        for name, kw, lr_val in tf_specs:
            print(f"[{ckpt_tag}] Training {name}...", flush=True)
            torch.manual_seed(20261114)
            net = RefillTransformer(**kw)
            net, meta = train_refill_transformer(
                net, tr_data, va_data, v_head, w_digit, p_perp,
                epochs=30, batch_size=512, lr=lr_val * 1.25, dev=dev,
            )
            tf_models[name] = net
            tf_train_meta[name] = meta
            print(
                f"  -> {name}: params={meta['n_params']:,}, best_ep={meta['best_epoch']}, "
                f"val_obj={meta['best_val_objective']:.5f}, {meta['train_sec']:.1f}s",
                flush=True,
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "meta": tf_train_meta,
                "states": {k: { p_k: p_v.cpu() for p_k, p_v in m.state_dict().items() } for k, m in tf_models.items()},
            },
            ladder_cache_path,
        )

    # Save C5 and C6 weights + mean_base for Stage 2 & Stage 3 reuse
    cache_dir.mkdir(parents=True, exist_ok=True)
    best_tf_name = "C6_tf_4L_512" if (
        tf_train_meta["C6_tf_4L_512"]["best_val_objective"] <= tf_train_meta["C5_tf_2L_512"]["best_val_objective"]
    ) else "C5_tf_2L_512"
    torch.save(
        {
            "ckpt_tag": ckpt_tag,
            "trm_sha256": sha,
            "best_tf_name": best_tf_name,
            "v_head": v_head,
            "p_perp": p_perp,
            "w_digit": w_digit,
            "mean_base": mean_base,
            "C5_tf_2L_512_state": {k: v.cpu() for k, v in tf_models["C5_tf_2L_512"].state_dict().items()},
            "C6_tf_4L_512_state": {k: v.cpu() for k, v in tf_models["C6_tf_4L_512"].state_dict().items()},
        },
        cache_dir / f"{ckpt_tag}_refill_bundle.pt",
    )

    # Build uniform predictor wrappers
    h_init_d = model.H_init.detach().float().to(dev).view(1, 1, 512)
    l_init_d = model.L_init.detach().float().to(dev).view(1, 1, 512)

    def make_wrapper(kind: str):
        if kind == "C0_init":
            return lambda inp, cand, vc, vp, s, calibrate_sphere=True: (
                sphere_calibrate_orth(h_init_d.expand(inp.shape[0], 81, -1), vp, p_perp_d, v_head=v_head_d),
                rms_calibrate_vec(h_init_d.expand(inp.shape[0], 16, -1)),
                l_init_d.expand(inp.shape[0], 97, -1),
            )
        if kind == "C0_mean":
            return lambda inp, cand, vc, vp, s, calibrate_sphere=True: (
                sphere_calibrate_orth(mu_orth.to(dev).expand(inp.shape[0], -1, -1), vp, p_perp_d, v_head=v_head_d),
                mu_pref.to(dev).expand(inp.shape[0], -1, -1),
                mu_zl.to(dev).expand(inp.shape[0], -1, -1),
            )
        if kind == "C1_linear_local":
            return lambda inp, cand, vc, vp, s, calibrate_sphere=True: c1.predict(
                inp, cand, vc, vp, w_dig_d, p_perp_d, calibrate_sphere=calibrate_sphere
            )
        if kind == "C2_linear_peer":
            return lambda inp, cand, vc, vp, s, calibrate_sphere=True: c2.predict(
                inp, cand, vc, vp, w_dig_d, p_perp_d, calibrate_sphere=calibrate_sphere
            )
        net = tf_models[kind]

        def _tf_wrap(inp, cand, vc, vp, s, calibrate_sphere=True, _net=net):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                return _net(
                    inp,
                    cand,
                    vc,
                    vp,
                    w_dig_d,
                    p_perp_d,
                    step_idx=torch.full((inp.shape[0],), min(int(s), 15), dtype=torch.long, device=inp.device),
                    calibrate_sphere=calibrate_sphere,
                    v_head=v_head_d,
                )

        return _tf_wrap

    # Verify continuous float32 zero hidden-state leak and logit parity on best Transformer refill
    best_net_f32 = tf_models[best_tf_name]
    parity_check = verify_zero_leak_and_logit_parity(
        model,
        lambda inp, cand, vc, vp, s, calibrate_sphere=True: best_net_f32(
            inp,
            cand,
            vc,
            vp,
            w_dig_d,
            p_perp_d,
            step_idx=torch.full((inp.shape[0],), min(int(s), 15), dtype=torch.long, device=inp.device),
            calibrate_sphere=calibrate_sphere,
            v_head=v_head_d,
        ),
        va_data["inp_tokens"],
        va_data["cand_masks"],
        v_head,
        p_perp,
        dev=dev,
    )
    print(f"[{ckpt_tag}] Zero-leak & logit parity verified: {parity_check}", flush=True)

    # Evaluate held-out val_100 R^2 across all capacity tiers
    val_r2 = {}
    all_tiers = ["C0_init", "C0_mean", "C1_linear_local", "C2_linear_peer"] + list(tf_models.keys())
    for tier in all_tiers:
        val_r2[tier] = evaluate_val_r2(
            make_wrapper(tier), va_data, val_prop_flags, mean_base, v_head, p_perp, dev=dev
        )
        print(
            f"  [val R^2] {tier:24s}: joint={val_r2[tier]['all']['r2_joint_hidden']:.4f} "
            f"(prop={val_r2[tier]['prop']['r2_joint_hidden']:.4f}, "
            f"stall={val_r2[tier]['stalled']['r2_joint_hidden']:.4f})",
            flush=True,
        )

    del tr_data, va_data
    if dev == "cuda":
        torch.cuda.empty_cache()

    # Define closed-loop suite: z_H x z_L factorial cross + capacity ladder
    best_fn = make_wrapper(best_tf_name)
    arm_specs = [
        # (arm_name, zh_orth_mode, zh_pref_mode, zl_mode, wrapper, calibrate_sphere)
        ("clean", "live", "live", "live", None, True),
        # Single-channel ablations (from v009-v012)
        ("live_zH__reset_zL", "live", "live", "reset_init", None, True),
        ("live_zH__mean_zL", "live", "live", "mean_sphere", None, True),
        ("norm_board_zH__live_zL", "norm_only", "live", "live", None, True),
        ("mean_board_zH__live_zL", "mean_sphere", "live", "live", None, True),
        ("mean_all_zH__live_zL", "mean_sphere", "mean_sphere", "live", None, True),
        # THE z_L x z_H CROSS (True V-only unconditional baselines: stripping BOTH z_H and z_L!)
        ("V_only__mean_board_zH__reset_zL", "mean_sphere", "live", "reset_init", None, True),
        ("V_only__init_all", "init_sphere", "init_sphere", "reset_init", None, True),
        ("V_only__mean_all", "mean_sphere", "mean_sphere", "mean_sphere", None, True),
        # Hybrid single-channel refill crosses using best Transformer refill
        ("refill_all_zH__live_zL", "refill", "refill", "live", best_fn, True),
        ("live_zH__refill_zL", "live", "live", "refill", best_fn, True),
        ("refill_all_zH__reset_zL", "refill", "refill", "reset_init", best_fn, True),
        ("refill_board_zH__live_pref__refill_zL", "refill", "live", "refill", best_fn, True),
        # Full Capacity Ladder on TRUE V + C (ALL hidden state z_H_orth + z_H_pref + z_L refilled!)
        ("VC_C1_linear_local", "refill", "refill", "refill", make_wrapper("C1_linear_local"), True),
        ("VC_C2_linear_peer", "refill", "refill", "refill", make_wrapper("C2_linear_peer"), True),
        ("VC_C3_tf_2L_128", "refill", "refill", "refill", make_wrapper("C3_tf_2L_128"), True),
        ("VC_C4_tf_2L_256", "refill", "refill", "refill", make_wrapper("C4_tf_2L_256"), True),
        ("VC_C5_tf_2L_512", "refill", "refill", "refill", make_wrapper("C5_tf_2L_512"), True),
        ("VC_C6_tf_4L_512", "refill", "refill", "refill", make_wrapper("C6_tf_4L_512"), True),
        # Uncalibrated MSE shrinkage check on C5
        ("VC_C5_raw_uncalibrated", "refill", "refill", "refill", make_wrapper("C5_tf_2L_512"), False),
    ]
    if "C5_tf_2L_512_stepclock" in tf_models:
        arm_specs.append(
            ("VC_C5_tf_2L_512_stepclock", "refill", "refill", "refill", make_wrapper("C5_tf_2L_512_stepclock"), True)
        )

    items_arms: dict[str, dict] = {}
    bench_arms: dict[str, dict] = {}
    for arm_name, zho_m, zhp_m, zl_m, fn_w, cal_s in arm_specs:
        items_arms[arm_name] = run_closed_loop_arm(
            model, items_qs, items_ans, items_prop, items_strata, v_head, p_perp,
            zho_m, zhp_m, zl_m, fn_w, mean_base, max_steps=64, bsz=64, calibrate_sphere=cal_s,
        )
        bench_arms[arm_name] = run_closed_loop_arm(
            model, bench_qs, bench_ans, bench_prop, None, v_head, p_perp,
            zho_m, zhp_m, zl_m, fn_w, mean_base, max_steps=64, bsz=64, calibrate_sphere=cal_s,
        )
        it_p = items_arms[arm_name]["wb_prop"]["solve_s16"]
        it_st16 = items_arms[arm_name]["wb_stall"]["solve_s16"]
        it_st64 = items_arms[arm_name]["wb_stall"]["solve_s64"]
        it_cyc = items_arms[arm_name]["wb_stall"]["unsolved_tail_exact_cycle_rate"]
        b_st16 = bench_arms[arm_name]["wb_stall"]["solve_s16"]
        print(
            f"  [closed-loop] {arm_name:36s} | items256 Prop@16={it_p:.3f} Stall@16={it_st16:.3f} "
            f"Stall@64={it_st64:.3f} (cyc={it_cyc:.3f}) | bench100 Stall@16={b_st16:.3f}",
            flush=True,
        )

    attach_mcnemar_vs_clean(items_arms, items_prop)
    attach_mcnemar_vs_clean(bench_arms, bench_prop)

    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    res_dict = {
        "checkpoint_tag": ckpt_tag,
        "checkpoint_sha256": sha,
        "elapsed_sec": float(time.time() - t_ck),
        "best_refill_model": best_tf_name,
        "zero_leak_and_logit_parity": parity_check,
        "transformer_training_meta": tf_train_meta,
        "val_100_r2_by_capacity": val_r2,
        "items256_closed_loop": items_arms,
        "bench100_closed_loop": bench_arms,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    done_json_path.write_bytes((json.dumps(res_dict, indent=2) + "\n").encode("utf-8"))
    return res_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "stage1_conditional_refill.json")
    parser.add_argument("--mem-fraction", type=float, default=0.65)
    parser.add_argument("--min-watts", type=float, default=40.0)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        watts = TS.power_limit_watts()
        if watts is not None and watts < args.min_watts:
            print(f"[power-guard] GPU limit {watts}W < {args.min_watts}W; exiting 75.")
            return TS.EXIT_POWER_CAP
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.time()
    all_train_qs, _all_train_ans = FRC.load_train_boards(1000)
    assert len(all_train_qs) == 1000, f"Expected 1000 train.csv boards, got {len(all_train_qs)}"
    train_qs = all_train_qs[:900]
    val_qs = all_train_qs[900:1000]
    val_prop_flags = np.array([bool(SP.trace_propagation(q).solved) for q in val_qs], dtype=bool)

    bundle = json.loads((REPO_ROOT / "mechinterp" / "loopscope02" / "results" / "benchmark_bundle.json").read_text("utf-8"))
    items = bundle["items"]
    items_qs = [it["question"] for it in items]
    items_ans = [it["answer"] for it in items]
    items_prop = [bool(it["prop_solved"]) for it in items]
    items_strata = [it["stratum"] for it in items]
    bench_qs, bench_ans, bench_prop = FRC.load_bench100()

    assert set(all_train_qs).isdisjoint(set(items_qs)), "Overlap detected between train.csv and items256!"
    assert set(all_train_qs).isdisjoint(set(bench_qs)), "Overlap detected between train.csv and bench100!"
    print(
        f"Verified zero puzzle overlap: train_900={len(train_qs)}, val_100={len(val_qs)} "
        f"(prop={int(val_prop_flags.sum())}, stall={int((~val_prop_flags).sum())}), "
        f"items256={len(items_qs)}, bench100={len(bench_qs)}",
        flush=True,
    )

    primary_res = run_checkpoint_stage1(
        "gaoxin492", train_qs, val_qs, val_prop_flags,
        items_qs, items_ans, items_prop, items_strata,
        bench_qs, bench_ans, bench_prop, args.cache_dir, dev=dev,
    )
    repl_res = run_checkpoint_stage1(
        "yagizdevre", train_qs, val_qs, val_prop_flags,
        items_qs, items_ans, items_prop, items_strata,
        bench_qs, bench_ans, bench_prop, args.cache_dir, dev=dev,
    )

    payload = {
        "preregistration_commit": "b1ec77550997cbc5ae2cd84618af74e8f34e38e7",
        "elapsed_sec": float(time.time() - t0),
        "dataset_splits": {
            "train_n": len(train_qs),
            "val_n": len(val_qs),
            "val_n_prop": int(val_prop_flags.sum()),
            "val_n_stalled": int((~val_prop_flags).sum()),
            "items256_n": len(items_qs),
            "bench100_n": len(bench_qs),
            "overlap_with_test": 0,
        },
        "primary_gaoxin492": primary_res,
        "replication_yagizdevre": repl_res,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(f"\nWrote Stage 1 receipt to {args.out} in {payload['elapsed_sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
