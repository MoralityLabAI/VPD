"""Pass-tied Causal Importance (CI) networks and Clock / Visible Gap diagnostics.

Implements the three pass-tied CI parameterizations specified for BPD:
  1. ``state_only``: ``g_{b,p,tau} = f_phi(X_{b,p,tau})``
  2. ``state_plus_clock``: ``g_{b,p,tau} = f_phi(X_{b,p,tau}, e(tau))``
  3. ``visible_only``: ``g_{b,p,tau} = f_phi(x_{b,p}, V_{b,p,tau})``

where ``phi`` is shared across all passes ``tau = 0, ..., T - 1`` so the CI
network has ``O(1)`` parameters in ``T`` and generalizes zero-shot to unseen pass
counts ``T' > T``.

The **Clock Gap** (``state_only`` vs ``state_plus_clock``) tests whether mechanisms
are addressed by pass clock or by recurrent state. The **Visible Gap**
(``visible_only`` vs ``state_only``) is the parameter-space counterpart of
LOOPSCOPE-03's dark-fiber trajectory residual ``R_t``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.components import (
    RolloutTrajectory,
    lower_leaky_sigmoid,
    upper_leaky_sigmoid,
)
from mechinterp.bpd.config import CIConfig


def _rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _RotaryPositionEmbedding(nn.Module):
    """Lightweight RoPE over sequence positions ``0..P-1``."""

    inv_freq: Tensor

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        # q, k: [N, H, P, D_h]
        seq_len = q.shape[2]
        positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [P, D_h]
        cos = emb.cos().view(1, 1, seq_len, -1)
        sin = emb.sin().view(1, 1, seq_len, -1)
        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot


class _CITransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer block operating across sequence positions within one pass."""

    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, rope_base: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = _RotaryPositionEmbedding(self.head_dim, base=rope_base)
        self.fc1 = nn.Linear(d_model, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, d_model)

    def forward(self, h: Tensor, *, is_causal: bool = False) -> Tensor:
        n_bt, seq_len, d = h.shape
        normed = _rms_norm(h)
        q, k, v = (
            self.qkv(normed)
            .view(n_bt, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = self.rope(q, k)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal and seq_len > 1)
        attn = attn.transpose(1, 2).reshape(n_bt, seq_len, d)
        h = h + self.o_proj(attn)
        h = h + self.fc2(F.gelu(self.fc1(_rms_norm(h))))
        return h


class _SinusoidalLearnedClock(nn.Module):
    """Hybrid sinusoidal + clamped learned pass-clock embedding generalizing to OOD passes."""

    inv_freq: Tensor

    def __init__(self, clock_dim: int, max_passes: int) -> None:
        super().__init__()
        if clock_dim < 2 or clock_dim % 2 != 0:
            raise ValueError(f"clock_dim must be even and >= 2, got {clock_dim}")
        self.clock_dim = clock_dim
        self.max_passes = max(max_passes, 1)
        half = clock_dim // 2
        inv_freq = 1.0 / (1000.0 ** (torch.arange(0, half, dtype=torch.float32) / max(half, 1)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.learned = nn.Embedding(self.max_passes, clock_dim)
        nn.init.normal_(self.learned.weight, mean=0.0, std=0.02)

    def forward(self, pass_indices: Tensor) -> Tensor:
        # pass_indices: [T] long
        pf = pass_indices.float()
        angles = torch.outer(pf, self.inv_freq)  # [T, clock_dim // 2]
        sin_emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [T, clock_dim]
        clamped = pass_indices.clamp(min=0, max=self.max_passes - 1)
        return sin_emb + self.learned(clamped)


@dataclass
class CIOutputs:
    """Pass-indexed causal importance outputs per shared module.

    Each dictionary maps ``module_name`` -> tensor of shape ``[B, T, P, C]``:
      - ``lower_leaky``: clamped to ``[0, 1]`` with resurrection leak below 0 (used in forward masks)
      - ``upper_leaky``: clamped below 1, slope ``alpha`` above 1 (used in minimality + switch loss)
      - ``pre_sigmoid``: unconstrained pre-activation logits
    """

    lower_leaky: dict[str, Tensor]
    upper_leaky: dict[str, Tensor]
    pre_sigmoid: dict[str, Tensor]


def _stack_pass_dict(
    states_by_pass: Mapping[int, Tensor],
    n_passes: int,
    pass_offset: int,
    seq_len: int,
) -> Tensor:
    """Stack ``tau -> [B, P_s, D]`` into ``[B, T, P, D]``, broadcasting ``P_s == 1`` if needed."""
    slices: list[Tensor] = []
    for local_t in range(n_passes):
        tau = pass_offset + local_t
        if tau not in states_by_pass:
            raise KeyError(
                f"visible_only CI requires visible_states at pass tau={tau}; available={sorted(states_by_pass)}"
            )
        st = states_by_pass[tau].float()
        if st.ndim == 2:
            st = st.unsqueeze(1)
        if st.shape[1] == 1 and seq_len > 1:
            st = st.expand(st.shape[0], seq_len, st.shape[2])
        elif st.shape[1] != seq_len:
            raise ValueError(
                f"visible_states[{tau}] has seq_len={st.shape[1]}, expected 1 or {seq_len}"
            )
        slices.append(st)
    return torch.stack(slices, dim=1)


class TiedCINetwork(nn.Module):
    """Pass-tied Causal Importance network shared identically across all loop passes."""

    def __init__(
        self,
        module_dims: Mapping[str, int],
        c_per_module: Mapping[str, int],
        cfg: CIConfig | None = None,
        *,
        visible_dim: int | None = None,
        prompt_dim: int | None = None,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg or CIConfig()
        self.module_order = sorted(c_per_module.keys())
        self.c_per_module = {k: int(c_per_module[k]) for k in self.module_order}
        self.module_dims = {k: int(module_dims[k]) for k in self.module_order}
        self.is_causal = is_causal

        self.slices: dict[str, slice] = {}
        offset = 0
        for name in self.module_order:
            c = self.c_per_module[name]
            self.slices[name] = slice(offset, offset + c)
            offset += c
        self.total_c = offset

        if self.cfg.variant == "visible_only":
            if visible_dim is None or prompt_dim is None:
                raise ValueError(
                    "visible_only TiedCINetwork requires explicit visible_dim and prompt_dim"
                )
            in_dim = int(visible_dim) + int(prompt_dim)
            self.clock: _SinusoidalLearnedClock | None = None
        elif self.cfg.variant == "state_plus_clock":
            in_dim = sum(self.module_dims.values()) + self.cfg.clock_dim
            self.clock = _SinusoidalLearnedClock(self.cfg.clock_dim, self.cfg.max_passes)
        elif self.cfg.variant == "state_only":
            in_dim = sum(self.module_dims.values())
            self.clock = None
        else:
            raise ValueError(f"Unknown CI variant: {self.cfg.variant!r}")

        self.in_dim = in_dim
        self.proj_in = nn.Linear(in_dim, self.cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                _CITransformerBlock(
                    d_model=self.cfg.d_model,
                    n_heads=self.cfg.n_heads,
                    mlp_hidden=self.cfg.mlp_hidden,
                    rope_base=self.cfg.rope_base,
                )
                for _ in range(self.cfg.n_blocks)
            ]
        )
        self.out_proj = nn.Linear(self.cfg.d_model, self.total_c)
        # Initialize bias near +0.75 so initial masks start mostly open (~0.75 in [0, 1]), matching VPD practice
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.75)

    def _assemble_features(
        self,
        module_inputs: Mapping[str, Tensor],
        *,
        visible_states: Mapping[int, Tensor] | Tensor | None = None,
        prompt_features: Tensor | None = None,
        pass_offset: int = 0,
    ) -> tuple[Tensor, int, int, int]:
        first = module_inputs[self.module_order[0]]
        bsz, n_passes, seq_len, _ = first.shape
        dev = first.device

        if self.cfg.variant == "visible_only":
            if visible_states is None or prompt_features is None:
                raise ValueError(
                    "visible_only CI forward requires both visible_states and prompt_features"
                )
            if isinstance(visible_states, Tensor):
                vis = visible_states.float()
            else:
                vis = _stack_pass_dict(visible_states, n_passes, pass_offset, seq_len)
            pf = prompt_features.float()
            if pf.ndim == 2:
                pf = pf.unsqueeze(1)
            if pf.shape[1] == 1 and seq_len > 1:
                pf = pf.expand(bsz, seq_len, pf.shape[2])
            pf_t = _rms_norm(pf).unsqueeze(1).expand(bsz, n_passes, seq_len, pf.shape[2])
            vis_t = _rms_norm(vis)
            feats = torch.cat([pf_t, vis_t], dim=-1)
            return feats, bsz, n_passes, seq_len

        normed_parts = [_rms_norm(module_inputs[name]) for name in self.module_order]
        if self.cfg.variant == "state_plus_clock":
            assert self.clock is not None
            pass_idx = torch.arange(
                pass_offset, pass_offset + n_passes, device=dev, dtype=torch.long
            )
            clk = self.clock(pass_idx).view(1, n_passes, 1, self.cfg.clock_dim)
            clk = clk.expand(bsz, n_passes, seq_len, self.cfg.clock_dim)
            normed_parts.append(clk)

        feats = torch.cat(normed_parts, dim=-1)
        return feats, bsz, n_passes, seq_len

    def forward(
        self,
        module_inputs: Mapping[str, Tensor],
        *,
        visible_states: Mapping[int, Tensor] | Tensor | None = None,
        prompt_features: Tensor | None = None,
        pass_offset: int = 0,
    ) -> CIOutputs:
        feats, bsz, n_passes, seq_len = self._assemble_features(
            module_inputs,
            visible_states=visible_states,
            prompt_features=prompt_features,
            pass_offset=pass_offset,
        )
        flat = feats.reshape(bsz * n_passes, seq_len, self.in_dim)
        h = self.proj_in(flat)
        for blk in self.blocks:
            h = blk(h, is_causal=self.is_causal)
        raw = self.out_proj(_rms_norm(h)).view(bsz, n_passes, seq_len, self.total_c)

        alpha = self.cfg.leaky_alpha
        lower: dict[str, Tensor] = {}
        upper: dict[str, Tensor] = {}
        pre: dict[str, Tensor] = {}
        for name in self.module_order:
            sl = self.slices[name]
            sub = raw[..., sl]
            pre[name] = sub
            lower[name] = lower_leaky_sigmoid(sub, alpha=alpha)
            upper[name] = upper_leaky_sigmoid(sub, alpha=alpha)
        return CIOutputs(lower_leaky=lower, upper_leaky=upper, pre_sigmoid=pre)

    def forward_trajectory(self, traj: RolloutTrajectory, *, pass_offset: int = 0) -> CIOutputs:
        """Convenience wrapper extracting CI inputs directly from a ``RolloutTrajectory``."""
        return self.forward(
            traj.module_inputs,
            visible_states=traj.visible_states,
            prompt_features=traj.prompt_features,
            pass_offset=pass_offset,
        )


class UnrolledCINetwork(nn.Module):
    """Untied per-pass CI baseline for Gate B1 (standard VPD unrolled across ``T`` passes)."""

    def __init__(
        self,
        module_dims: Mapping[str, int],
        c_per_module: Mapping[str, int],
        n_passes: int,
        cfg: CIConfig | None = None,
        *,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        if n_passes < 1:
            raise ValueError("n_passes must be >= 1")
        base_cfg = cfg or CIConfig()
        state_cfg = CIConfig(
            variant="state_only",
            d_model=base_cfg.d_model,
            n_blocks=base_cfg.n_blocks,
            n_heads=base_cfg.n_heads,
            mlp_hidden=base_cfg.mlp_hidden,
            max_passes=1,
            max_seq_len=base_cfg.max_seq_len,
            clock_dim=base_cfg.clock_dim,
            rope_base=base_cfg.rope_base,
            leaky_alpha=base_cfg.leaky_alpha,
        )
        self.n_passes = n_passes
        self.module_order = sorted(c_per_module.keys())
        self.per_pass_nets = nn.ModuleList(
            [
                TiedCINetwork(module_dims, c_per_module, state_cfg, is_causal=is_causal)
                for _ in range(n_passes)
            ]
        )

    def forward(self, module_inputs: Mapping[str, Tensor]) -> CIOutputs:
        first = module_inputs[self.module_order[0]]
        n_passes = first.shape[1]
        if n_passes != self.n_passes:
            raise ValueError(f"Expected n_passes={self.n_passes}, got {n_passes}")

        per_pass_out: list[CIOutputs] = []
        for tau in range(self.n_passes):
            single_in = {k: v[:, tau : tau + 1] for k, v in module_inputs.items()}
            per_pass_out.append(self.per_pass_nets[tau](single_in))

        lower = {
            k: torch.cat([o.lower_leaky[k] for o in per_pass_out], dim=1) for k in self.module_order
        }
        upper = {
            k: torch.cat([o.upper_leaky[k] for o in per_pass_out], dim=1) for k in self.module_order
        }
        pre = {
            k: torch.cat([o.pre_sigmoid[k] for o in per_pass_out], dim=1) for k in self.module_order
        }
        return CIOutputs(lower_leaky=lower, upper_leaky=upper, pre_sigmoid=pre)


def active_component_budget(ci_out: CIOutputs, threshold: float = 0.5) -> dict[str, float]:
    """Compute active component statistics across batch, positions, and passes."""
    per_pass_sums: list[Tensor] = []
    any_pass_sums: list[Tensor] = []
    switch_sums: list[Tensor] = []
    for name in sorted(ci_out.lower_leaky.keys()):
        g = ci_out.lower_leaky[name].detach().float()  # [B, T, P, C]
        active = (g > threshold).float()
        per_pass_sums.append(active.sum(dim=-1).mean())  # active components per (b, tau, p)
        any_pass_sums.append(active.amax(dim=1).sum(dim=-1).mean())  # distinct recruited across tau
        if g.shape[1] > 1:
            switch_sums.append((g[:, 1:] - g[:, :-1]).abs().sum(dim=-1).mean())
        else:
            switch_sums.append(torch.zeros((), device=g.device))

    mean_active_per_pass = float(torch.stack(per_pass_sums).sum().item())
    recruited_any_pass = float(torch.stack(any_pass_sums).sum().item())
    mean_tv_switches = float(torch.stack(switch_sums).sum().item())
    return {
        "mean_active_per_pass": mean_active_per_pass,
        "recruited_any_pass": recruited_any_pass,
        "mean_tv_switches": mean_tv_switches,
    }


def compute_ci_variant_gaps(
    variant_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Compute Clock Gap and Visible Gap from evaluated CI variant metrics.

    Expects ``variant_metrics`` to contain keys ``"state_only"``, ``"state_plus_clock"``,
    and ``"visible_only"``, each with ``"recon_loss"`` and ``"mean_active_per_pass"``.

    Conventions:
      - ``clock_gap_recon = recon(state_only) - recon(state_plus_clock)``
        (positive when explicit pass clock improves reconstruction beyond state alone).
      - ``visible_gap_recon = recon(visible_only) - recon(state_only)``
        (positive when dark-fiber / off-head state carries routing information
        inaccessible from ``(x, V_tau)`` alone — the parameter-space counterpart of ``R_t``).
    """
    st = variant_metrics["state_only"]
    clk = variant_metrics["state_plus_clock"]
    vis = variant_metrics["visible_only"]
    return {
        "clock_gap_recon": float(st["recon_loss"] - clk["recon_loss"]),
        "visible_gap_recon": float(vis["recon_loss"] - st["recon_loss"]),
        "clock_gap_active": float(st["mean_active_per_pass"] - clk["mean_active_per_pass"]),
        "visible_gap_active": float(vis["mean_active_per_pass"] - st["mean_active_per_pass"]),
    }
