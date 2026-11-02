"""VPD math from Goodfire nano_param_decomp; see VPD_BACKEND_LICENSE and provenance.

Sections A-F retained. Installation is supplied by recurrent.py; the original
single-invocation installer is unused. No upstream training or logging entry point.
"""
import math
from dataclasses import dataclass
from typing import Any, Literal, cast
from typing_extensions import override
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# --- Section A: Config ---


@dataclass
class Config:
    """Configuration for the Parameter Decomposition method.

    Defaults are set to the values used by the pile-4L decomposition in the VPD paper.
    """

    # Maps each `nn.Linear` submodule path to its component count. Required; tied to the
    # specific target model.
    C_per_module: dict[str, int]

    # Training schedule
    n_steps: int = 400_000
    batch_size: int = 64  # global batch (divided across ranks)
    seq_len: int = 512
    seed: int = 0

    # Main optimizer (AdamW over components + CI transformer)
    main_lr: float = 5e-5
    main_lr_final_frac: float = 0.1

    # Faithfulness warmup (AdamW over components only; runs before main loop)
    faithfulness_warmup_steps: int = 400
    faithfulness_warmup_lr: float = 1e-3

    # Loss coefficients
    coeff_faith: float = 1e7
    coeff_imp: float = 2e-4
    coeff_stoch: float = 0.5
    coeff_ppgd: float = 0.5

    # Importance minimality (L_p with linear p-anneal, plus log description term)
    p_start: float = 2.0
    p_end: float = 0.4
    imp_eps: float = 1e-12
    imp_beta: float = 0.5

    # Leaky-hard sigmoid slope outside [0, 1]
    leaky_alpha: float = 0.01

    # CI transformer (global_shared_transformer)
    ci_d_model: int = 2048
    ci_n_blocks: int = 8
    ci_n_heads: int = 16
    ci_mlp_hidden: int = 8192
    ci_rope_base: float = 10000.0

    # Persistent PGD (per_batch_per_position scope, Adam)
    ppgd_lr: float = 0.01
    ppgd_lr_final_frac: float = 0.1
    ppgd_warmup_pct: float = 0.025
    ppgd_beta1: float = 0.5
    ppgd_beta2: float = 0.99
    ppgd_eps: float = 1e-8
    ppgd_inner_steps: int = 2

    # Gradient clipping
    grad_clip_components: float = 0.01

    # Evaluation
    eval_freq: int = 1000
    slow_eval_freq: int = 10000
    slow_eval_on_first_step: bool = True
    eval_batch_size: int = 128
    ci_alive_threshold: float = 0.0
    rounding_threshold: float = 0.0
    pgd_eval_step_size: float = 0.1
    pgd_eval_n_steps: int = 20

    # Logging
    log_every: int = 200
    use_wandb: bool = False
    wandb_project: str = "param-decomp"
    wandb_run_name: str | None = None


# --- Section B: Leaky-hard sigmoids ---


class _LowerLeakyHardSigmoid(torch.autograd.Function):
    """Forward: `clamp(x, 0, 1)`. Backward: pass-through on `(0, 1)`; in the `x <= 0` region,
    only return `alpha * grad_output` when `grad_output < 0` (i.e. gradient wants `y` to
    increase — this can 'resurrect' a dead component). When `grad_output >= 0` in that
    region, return 0 — no point pushing `x` further negative when `y` is already saturated.
    Above 1, gradient is blocked.
    """

    @staticmethod
    @override
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.clamp(0.0, 1.0)

    @staticmethod
    @override
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        grad_output = grad_outputs[0]
        (x,) = ctx.saved_tensors
        alpha: float = ctx.alpha
        zero = torch.zeros_like(grad_output)
        grad = torch.where(
            x <= 0,
            torch.where(grad_output < 0, alpha * grad_output, zero),
            torch.where(x <= 1, grad_output, zero),
        )
        return grad, None


def lower_leaky(x: Tensor, alpha: float) -> Tensor:
    return cast(Tensor, _LowerLeakyHardSigmoid.apply(x, alpha))


def upper_leaky(x: Tensor, alpha: float) -> Tensor:
    """For `x > 1` return `1 + alpha*(x-1)` (linear continuation, grad alpha); otherwise `clamp(x, 0, 1)`.
    Native autograd gives the correct profile (0 below 0, 1 in-range, alpha above 1), so no
    custom Function is needed.
    """
    return torch.where(x > 1, 1 + alpha * (x - 1), x.clamp(0.0, 1.0))


# --- Section C: ComponentLinear wrapper + install helper ---


class ComponentLinear(nn.Module):
    """Replaces one `nn.Linear` in the target model.

    Parameters:
        V: [d_in, C], the component basis in input space.
        U: [C, d_out], the component output transforms.
    The original weight and bias are kept as frozen buffers (`W_target`, `bias`).

    Two forward modes:
        "target":     returns F.linear(x, W_target, bias) and caches x for the CI function.
        "component":  returns the masked component output, optionally routed against the
                      target output per (batch, pos) by `routing_mask`.

    In "component" mode the caller sets `self.mask` ([B, S, C]), `self.delta_mask` ([B, S]),
    and optionally `self.routing_mask` ([B, S], bool). `None` routing_mask means "component
    everywhere".

    The weight delta is `W_target - (V @ U).T`; it is masked per-position by the scalar
    `delta_mask`. The delta acts as an extra "spillover" component that absorbs whatever
    `V @ U` doesn't yet explain.
    """

    def __init__(self, linear: nn.Linear, C: int) -> None:
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.C = C
        self.register_buffer("W_target", linear.weight.detach().clone())
        # Stubs type `nn.Linear.bias` as `Parameter`, but at runtime it is None when `bias=False`.
        linear_bias = cast(Tensor | None, linear.bias)
        self.register_buffer(
            "bias", linear_bias.detach().clone() if linear_bias is not None else None
        )
        self.V = nn.Parameter(torch.empty(d_in, C).normal_(0.0, 1.0 / math.sqrt(d_in)))
        self.U = nn.Parameter(torch.empty(C, d_out).normal_(0.0, 1.0 / math.sqrt(C)))
        # Transient state set per forward from the training loop
        self.mode: Literal["target", "component"] = "target"
        self.mask: Tensor | None = None
        self.delta_mask: Tensor | None = None
        self.routing_mask: Tensor | None = None
        self.last_input: Tensor | None = None
        self.cache_output: bool = False
        self.last_output: Tensor | None = None

    def weight_delta(self) -> Tensor:
        return self.W_target - (self.V @ self.U).T

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "target":
            self.last_input = x.detach()
            out = F.linear(x, self.W_target, self.bias)
            if self.cache_output:
                self.last_output = out.detach()
            return out
        assert self.mask is not None and self.delta_mask is not None
        comp_acts = x @ self.V  # [B, S, C]
        comp_out = (comp_acts * self.mask) @ self.U
        if self.bias is not None:
            comp_out = comp_out + self.bias
        delta_out = F.linear(x, self.weight_delta())  # [B, S, d_out]
        comp_out = comp_out + self.delta_mask.unsqueeze(-1) * delta_out
        if self.routing_mask is not None:
            target_out = F.linear(x, self.W_target, self.bias)
            comp_out = torch.where(self.routing_mask.unsqueeze(-1), comp_out, target_out)
        if self.cache_output:
            self.last_output = comp_out.detach()
        return comp_out


def install_components(model: nn.Module, module_to_c: dict[str, int]) -> dict[str, ComponentLinear]:
    """Freeze every target parameter and replace each listed `nn.Linear` in place."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers: dict[str, ComponentLinear] = {}
    for path, C in module_to_c.items():
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        linear = model.get_submodule(path)
        assert isinstance(linear, nn.Linear), f"{path} is not nn.Linear: {type(linear)}"
        wrapper = ComponentLinear(linear, C)
        setattr(parent, attr, wrapper)
        wrappers[path] = wrapper
    return wrappers


# --- Section D: CI transformer (global_shared_transformer) ---


def precompute_rope(
    seq_len: int, head_dim: int, base: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    assert head_dim % 2 == 0
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [seq_len, half]
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Split-in-half RoPE. x: [B, H, S, head_dim]; cos/sin: [S, half]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CIAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE on Q, K."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    @override
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class CIBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attn = CIAttention(cfg.ci_d_model, cfg.ci_n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.ci_d_model, cfg.ci_mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.ci_mlp_hidden, cfg.ci_d_model),
        )

    @override
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(F.rms_norm(x, (x.shape[-1],)), cos, sin)
        x = x + self.mlp(F.rms_norm(x, (x.shape[-1],)))
        return x


class CITransformer(nn.Module):
    """The causal-importance function: a shared transformer that sees all layers at once.

    Inputs: dict of pre-weight activations (one entry per target module, shape [B, S, d_in_m]).
    Each is RMS-normed (no learned scale) and concatenated along the feature dim; a linear
    projection maps the concatenation to d_model, `n_blocks` transformer blocks run over the
    sequence, and a second linear projection maps back to the total component dimension.
    The output is split per module and passed through the two leaky-hard sigmoids.

    Modules are concatenated in alphabetical-path order (see `module_order`), so the layout
    of `proj_in` columns / `proj_out` rows is independent of the input dict's iteration order.
    """

    def __init__(
        self, d_in_per_module: dict[str, int], c_per_module: dict[str, int], cfg: Config
    ) -> None:
        super().__init__()
        self.module_order = sorted(d_in_per_module.keys())
        self.cfg = cfg
        total_in = sum(d_in_per_module.values())
        total_C = sum(c_per_module[name] for name in self.module_order)
        self.proj_in = nn.Linear(total_in, cfg.ci_d_model)
        self.blocks = nn.ModuleList([CIBlock(cfg) for _ in range(cfg.ci_n_blocks)])
        self.proj_out = nn.Linear(cfg.ci_d_model, total_C)
        self.c_splits: list[int] = [c_per_module[n] for n in self.module_order]
        head_dim = cfg.ci_d_model // cfg.ci_n_heads
        cos, sin = precompute_rope(cfg.seq_len, head_dim, cfg.ci_rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @override
    def forward(
        self, acts: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        normed = [F.rms_norm(acts[n], (acts[n].shape[-1],)) for n in self.module_order]
        x = torch.cat(normed, dim=-1)
        x = self.proj_in(x)
        S = x.shape[1]
        cos, sin = self.rope_cos[:S], self.rope_sin[:S]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.proj_out(x)  # [B, S, total_C]
        per_module = dict(zip(self.module_order, logits.split(self.c_splits, dim=-1), strict=True))
        alpha = self.cfg.leaky_alpha
        ci_lower = {n: lower_leaky(v, alpha) for n, v in per_module.items()}
        ci_upper = {n: upper_leaky(v, alpha) for n, v in per_module.items()}
        return ci_lower, ci_upper, per_module


# --- Section E: Losses + mask/routing sampling ---


def faithfulness_loss(wrappers: dict[str, ComponentLinear]) -> Tensor:
    """Sum of squared weight-delta errors divided by total delta element count."""
    sum_sq = torch.zeros((), device=next(iter(wrappers.values())).V.device)
    numel = 0
    for w in wrappers.values():
        delta = w.weight_delta()
        sum_sq = sum_sq + delta.pow(2).sum()
        numel += delta.numel()
    return sum_sq / numel


def anneal_p(step: int, total_steps: int, p_start: float, p_end: float) -> float:
    t = min(max(step / total_steps, 0.0), 1.0)
    return p_start + (p_end - p_start) * t


def importance_minimality_loss(
    ci_upper: dict[str, Tensor], p: float, eps: float, beta: float, world_size: int
) -> Tensor:
    """Per-module: sum_c [ mean[c] + beta * mean[c] * log2(1 + sum[c] * world_size) ].

    `mean` / `sum` are over batch and sequence dims (local to this rank). The `* world_size`
    term rescales the local sum into an estimate of the global per-component total.
    """
    total = torch.zeros((), device=next(iter(ci_upper.values())).device)
    for v in ci_upper.values():
        vals = (v + eps).pow(p)  # [B, S, C]
        batch_seq_dims = tuple(range(vals.ndim - 1))
        sum_c = vals.sum(dim=batch_seq_dims)  # [C]
        n = math.prod(vals.shape[:-1])
        mean_c = sum_c / n
        total = total + (mean_c + beta * mean_c * torch.log2(1 + sum_c * world_size)).sum()
    return total


def kl_logits(pred: Tensor, target: Tensor) -> Tensor:
    """KL(softmax(target) || softmax(pred)) averaged over all non-vocab positions.

    `target` is treated as the reference distribution (equivalent to the original model's
    output); we don't backprop through it.
    """
    log_q = F.log_softmax(pred, dim=-1)
    p = F.softmax(target.detach(), dim=-1)
    kl_per_pos = F.kl_div(log_q, p, reduction="none").sum(dim=-1)
    return kl_per_pos.mean()


def sample_continuous_masks(
    ci_lower: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """mask = ci + (1 - ci) * U(0, 1); delta_mask ~ U(0, 1) per (batch, pos)."""
    masks: dict[str, Tensor] = {}
    delta_masks: dict[str, Tensor] = {}
    for name, ci in ci_lower.items():
        u = torch.rand_like(ci)
        masks[name] = ci + (1 - ci) * u
        delta_masks[name] = torch.rand(*ci.shape[:-1], device=ci.device, dtype=ci.dtype)
    return masks, delta_masks


def sample_uniform_k_subset_routing(
    module_names: list[str], batch_dims: tuple[int, ...], device: torch.device
) -> dict[str, Tensor]:
    """For each (batch, pos), sample k ~ Uniform{1..M} and route to a random k-subset."""
    M = len(module_names)
    k = torch.randint(1, M + 1, batch_dims, device=device)  # [*batch_dims]
    noise = torch.rand(M, *batch_dims, device=device)
    ranks = noise.argsort(dim=0).argsort(dim=0)  # [M, *batch_dims]
    return {name: ranks[i] < k for i, name in enumerate(module_names)}


def set_wrapper_masks(
    wrappers: dict[str, ComponentLinear],
    masks: dict[str, Tensor],
    delta_masks: dict[str, Tensor],
    routing: dict[str, Tensor] | None,
) -> None:
    for name, w in wrappers.items():
        w.mode = "component"
        w.mask = masks[name]
        w.delta_mask = delta_masks[name]
        w.routing_mask = None if routing is None else routing[name]


def clear_wrapper_masks(wrappers: dict[str, ComponentLinear]) -> None:
    for w in wrappers.values():
        w.mode = "target"
        w.mask = None
        w.delta_mask = None
        w.routing_mask = None


def stochastic_recon_loss(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    input_ids: Tensor,
    target_logits: Tensor,
    ci_lower: dict[str, Tensor],
) -> Tensor:
    """One-sample stochastic-mask reconstruction with uniform-k-subset per-position routing."""
    B, S = input_ids.shape
    masks, delta_masks = sample_continuous_masks(ci_lower)
    routing = sample_uniform_k_subset_routing(list(wrappers), (B, S), input_ids.device)
    set_wrapper_masks(wrappers, masks, delta_masks, routing)
    try:
        pred = target_model(input_ids)
    finally:
        clear_wrapper_masks(wrappers)
    return kl_logits(pred, target_logits)


# --- Section F: Persistent PGD ---


class PersistentPGD:
    """Per-module adversarial sources that persist across training steps.

    Scope is `per_batch_per_position`: sources have shape `[local_B, S, C + 1]` on each
    rank (+1 for the delta component's scalar mask), with no cross-rank sync. We maintain
    Adam state (m, v) alongside each source and a global step counter t.

    Each training step:
      1) `warmup`: run `inner_steps` (=2) PGD updates on the current batch, all-layers routed.
      2) `recon_loss`: forward once with current sources (no mutation) to produce the loss
         term that enters the main total; the caller extracts source grads from this via
         `torch.autograd.grad(loss, sources, retain_graph=True)` before `total.backward()`.
      3) `external_step`: Adam-update sources with the extracted grads after `total.backward()`.
    """

    def __init__(
        self,
        wrappers: dict[str, ComponentLinear],
        local_B: int,
        seq_len: int,
        device: torch.device,
        cfg: Config,
    ) -> None:
        self.cfg = cfg
        self.sources: dict[str, Tensor] = {}
        self.m: dict[str, Tensor] = {}
        self.v: dict[str, Tensor] = {}
        for name, w in wrappers.items():
            shape = (local_B, seq_len, w.C + 1)
            src = torch.rand(shape, device=device).requires_grad_(True)
            self.sources[name] = src
            self.m[name] = torch.zeros(shape, device=device)
            self.v[name] = torch.zeros(shape, device=device)
        self.t = 0

    def _masks_from_sources(
        self, ci_lower: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        masks: dict[str, Tensor] = {}
        delta_masks: dict[str, Tensor] = {}
        for name, ci in ci_lower.items():
            s = self.sources[name]
            masks[name] = ci + (1 - ci) * s[..., : ci.shape[-1]]
            delta_masks[name] = s[..., -1]
        return masks, delta_masks

    def recon_loss(
        self,
        target_model: nn.Module,
        wrappers: dict[str, ComponentLinear],
        input_ids: Tensor,
        target_logits: Tensor,
        ci_lower: dict[str, Tensor],
    ) -> Tensor:
        masks, delta_masks = self._masks_from_sources(ci_lower)
        set_wrapper_masks(wrappers, masks, delta_masks, routing=None)  # all-layers
        try:
            pred = target_model(input_ids)
        finally:
            clear_wrapper_masks(wrappers)
        return kl_logits(pred, target_logits)

    def warmup(
        self,
        target_model: nn.Module,
        wrappers: dict[str, ComponentLinear],
        input_ids: Tensor,
        target_logits: Tensor,
        ci_lower: dict[str, Tensor],
        lr: float,
    ) -> None:
        for _ in range(self.cfg.ppgd_inner_steps):
            loss = self.recon_loss(target_model, wrappers, input_ids, target_logits, ci_lower)
            grads = torch.autograd.grad(loss, list(self.sources.values()), retain_graph=False)
            self._adam_step(dict(zip(self.sources, grads, strict=True)), lr)

    def external_step(self, grads: dict[str, Tensor], lr: float) -> None:
        self._adam_step(grads, lr)

    def _adam_step(self, grads: dict[str, Tensor], lr: float) -> None:
        self.t += 1
        bc1 = 1 - self.cfg.ppgd_beta1**self.t
        bc2 = 1 - self.cfg.ppgd_beta2**self.t
        with torch.no_grad():
            for name, src in self.sources.items():
                g = grads[name]
                m, v = self.m[name], self.v[name]
                m.mul_(self.cfg.ppgd_beta1).add_(g, alpha=1 - self.cfg.ppgd_beta1)
                v.mul_(self.cfg.ppgd_beta2).addcmul_(g, g, value=1 - self.cfg.ppgd_beta2)
                src.add_(lr * (m / bc1) / ((v / bc2).sqrt() + self.cfg.ppgd_eps))
                src.clamp_(0.0, 1.0)



