"""BPD adapter for ``Huginn-0125`` and the ``HUG-CAL-02`` 8-factor negative control.

Supports:
  1. ``HuginnBPDAdapter``: wraps a loaded ``tomg-group-umd/huginn-0125`` recurrent
     transformer (prelude -> ``T`` shared ``core_block`` iterations -> coda + ``lm_head``
     readout at every pass), discovering the 4 recurrent core blocks' shared linear
     projections automatically.
  2. ``SettlingCoreLooper`` & ``evaluate_hug_cal02_eight_factors``: self-contained
     4-core-block random-warmup settling model (plus real-model entrypoint) for
     Gate B2, auditing 8 top singular factors across ``core.{0..3}.{o_proj, down_proj}``
     (and optional planted task-aligned positive control) to verify that isotropic
     settling factors yield ~zero net-of-rotation-null importance while reporting
     raw and net importance side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.adapters.trm import compute_head_projector_f64
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.nulls import ComponentNullEntry, evaluate_component_null_report


def discover_huginn_core_linears(model: nn.Module, c: int = 32) -> dict[str, int]:
    """Discover ``nn.Linear`` layers inside ``core_block`` (or ``core``) of a Huginn model."""
    found: dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ("core_block" in name or name.startswith("core.")):
            found[name] = c
    if not found:
        raise ValueError("No recurrent core_block nn.Linear modules found on model")
    return found


class HuginnBPDAdapter:
    """Adapter for HuggingFace ``tomg-group-umd/huginn-0125`` recurrent depth models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        lm_head = getattr(model, "lm_head", None)
        if not isinstance(lm_head, nn.Linear):
            raise TypeError("Expected model.lm_head to be an nn.Linear")
        v_head, p_head, _ = compute_head_projector_f64(lm_head.weight[: min(64, lm_head.out_features)])
        self.v_head = v_head
        self.p_head = p_head
        self.visible_dim = int(v_head.shape[1])
        self.prompt_dim = int(lm_head.in_features)

    def rollout(
        self,
        input_ids: Tensor,
        n_passes: int,
        pass_offset: int = 0,
        init_state: Tensor | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], Tensor]:
        """Run prelude -> ``n_passes`` recurrent steps -> per-pass coda/lm_head readout."""
        transformer = getattr(self.model, "transformer")
        lm_head = getattr(self.model, "lm_head")
        dev = lm_head.weight.device
        toks = input_ids.to(dev)

        x_emb = transformer.wte(toks)
        for blk in transformer.prelude:
            x_emb = blk(x_emb)

        state = init_state.to(dev) if init_state is not None else torch.zeros_like(x_emb)
        adapter = getattr(transformer, "adapter", None)
        ln_f = getattr(transformer, "ln_f", nn.Identity())
        coda = getattr(transformer, "coda", ())
        vh = self.v_head.to(device=dev, dtype=torch.float32)

        readouts: dict[int, Tensor] = {}
        boundary_states: dict[int, Tensor] = {}
        visible_states: dict[int, Tensor] = {}

        for local_k in range(n_passes):
            tau = pass_offset + local_k
            visible_states[tau] = state.float() @ vh
            if adapter is not None:
                step_in = adapter(torch.cat([state, x_emb], dim=-1))
            else:
                step_in = state + x_emb
            h = step_in
            for blk in transformer.core_block:
                h = blk(h)
            state = h
            coda_h = state
            for blk in coda:
                coda_h = blk(coda_h)
            logits = lm_head(ln_f(coda_h))
            readouts[tau] = logits
            boundary_states[tau] = state

        return readouts, boundary_states, visible_states, x_emb.float()


class _SettlingBlock(nn.Module):
    """Recurrent Pre-RMSNorm block with ``o_proj`` and ``down_proj`` matching Huginn's 4-block core."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_up_proj = nn.Linear(d_model, 4 * d_model, bias=False)
        self.down_proj = nn.Linear(2 * d_model, d_model, bias=False)

    @staticmethod
    def _rms(x: Tensor, eps: float = 1e-6) -> Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)

    def forward(self, h: Tensor) -> Tensor:
        b, p, d = h.shape
        xn = self._rms(h)
        q, k, v = (
            self.qkv_proj(xn)
            .view(b, p, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, p, d)
        h = h + self.o_proj(attn)
        g, u = self.gate_up_proj(self._rms(h)).chunk(2, dim=-1)
        h = h + self.down_proj(F.silu(g) * u)
        return h


class SettlingCoreLooper(nn.Module):
    """4-block recurrent settling loop reproducing ``HUG-CAL-02`` fragility physics.

    Each recurrent pass injects prompt embeddings ``e`` into latent state ``s`` via
    ``adapter([s, e])`` and runs ``core.{0..3}``. Early passes experience high
    non-normal amplification while later passes contract toward an equilibrium,
    creating steep cross-pass fragility without requiring factor-specific semantics
    on the isotropic bulk weights.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_core_blocks: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.wte = nn.Embedding(vocab_size, d_model)
        self.adapter = nn.Linear(2 * d_model, d_model, bias=False)
        self.core = nn.ModuleList([_SettlingBlock(d_model, n_heads) for _ in range(n_core_blocks)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        with torch.no_grad():
            self.wte.weight.normal_(0.0, 0.5, generator=gen)
            # Contractive state carry + prompt re-injection -> classic settling dynamics
            eye = torch.eye(d_model)
            self.adapter.weight[:, :d_model].copy_(0.55 * eye)
            self.adapter.weight[:, d_model:].copy_(0.45 * eye)
            for blk in self.core:
                for lin in (blk.qkv_proj, blk.o_proj, blk.gate_up_proj, blk.down_proj):
                    # Isotropic random weights -> raw sensitivity without privileged read alignment
                    lin.weight.normal_(0.0, 0.85 / (lin.weight.shape[1] ** 0.5), generator=gen)
            self.lm_head.weight.normal_(0.0, 1.5 / (d_model ** 0.5), generator=gen)

    def rollout(
        self,
        tokens: Tensor,
        n_passes: int,
        pass_offset: int = 0,
        init_state: Tensor | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], Tensor]:
        dev = self.lm_head.weight.device
        tok = tokens.to(dev)
        x_emb = self.wte(tok)
        state = init_state.to(dev) if init_state is not None else torch.zeros_like(x_emb)

        readouts: dict[int, Tensor] = {}
        boundary_states: dict[int, Tensor] = {}
        visible_states: dict[int, Tensor] = {}

        for local_k in range(n_passes):
            tau = pass_offset + local_k
            visible_states[tau] = self.lm_head( _SettlingBlock._rms(state) ).float()
            h = self.adapter(torch.cat([state, x_emb], dim=-1))
            for blk in self.core:
                h = blk(h)
            state = h
            logits = self.lm_head(_SettlingBlock._rms(state))
            readouts[tau] = logits
            boundary_states[tau] = state

        return readouts, boundary_states, visible_states, x_emb.float()


@dataclass
class HugCal02AuditResult:
    """Gate B2 negative-control summary over the 8 ``HUG-CAL-02`` settling factors."""

    factor_entries: list[ComponentNullEntry]
    max_raw_importance: float
    mean_raw_importance: float
    mean_abs_net_importance: float
    max_abs_z_rot: float
    passed_negative_control: bool


def evaluate_hug_cal02_eight_factors(
    model: SettlingCoreLooper,
    tokens: Tensor,
    *,
    n_passes: int = 6,
    n_rotations: int = 16,
    z_threshold: float = 2.5,
    seed: int = 0,
) -> HugCal02AuditResult:
    """Audit the 8 leading ``mlp.down_proj`` SVD factors (top-2 per block across ``core.{0..3}``).

    Matches ``HUG-CAL-02`` (8 frozen ``mlp.proj`` factors ``f0, f1`` across ``core0..core3``,
    evaluated against orthogonal read rotations matched to exact per-iteration projection RMS
    while fixing write vector ``u``). Reports both ``raw_importance`` and ``net_importance``.
    """
    four_mlp_modules = {f"core.{i}.down_proj": 4 for i in range(4)}
    bpd_model = BPDModel(model, four_mlp_modules, max_passes=n_passes, init_svd=True)

    with torch.no_grad():
        with bpd_model.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
            ro, bs, vis, prm = model.rollout(tokens, n_passes, 0, None)
        mod_in = bpd_model.collect_recorded_inputs(expected_passes=n_passes)
    clean_traj = RolloutTrajectory(
        readouts=ro,
        boundary_states=bs,
        module_inputs=mod_in,
        visible_states=vis,
        prompt_features=prm,
    )

    def _run() -> RolloutTrajectory:
        p_ro, p_bs, p_vis, p_prm = model.rollout(tokens, n_passes, 0, None)
        return RolloutTrajectory(
            readouts=p_ro,
            boundary_states=p_bs,
            module_inputs=mod_in,
            visible_states=p_vis,
            prompt_features=p_prm,
        )

    report = evaluate_component_null_report(
        bpd_model,
        clean_traj,
        _run,
        components_by_module={m: [0, 1] for m in four_mlp_modules},
        passes=[],  # evaluate multi-pass tied factor ablation (pass_idx = -1)
        include_all_pass=True,
        n_rotations=n_rotations,
        seed=seed,
    )
    entries = report.entries
    raws = [e.raw_importance for e in entries]
    abs_nets = [abs(e.net_importance) for e in entries]
    abs_zs = [abs(e.z_rot) for e in entries]

    max_raw = max(raws) if raws else 0.0
    mean_raw = sum(raws) / max(len(raws), 1)
    mean_abs_net = sum(abs_nets) / max(len(abs_nets), 1)
    max_z = max(abs_zs) if abs_zs else 0.0

    return HugCal02AuditResult(
        factor_entries=entries,
        max_raw_importance=max_raw,
        mean_raw_importance=mean_raw,
        mean_abs_net_importance=mean_abs_net,
        max_abs_z_rot=max_z,
        passed_negative_control=(max_raw > 0.0 and max_z <= z_threshold),
    )
