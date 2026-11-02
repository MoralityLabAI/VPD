"""Invocation-aware adapter for the original VPD losses and CI transformer.

Physical matrices own V/U once. Invocation views own caches and masks, but no
parameters. This deliberately supports single-process, uncheckpointed forwards.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from torch import nn

from experiments.nanbeige_looped import vpd_math as vpd


class InvocationView:
    def __init__(self, core):
        self.core = core
        self.mode = "target"
        self.mask = self.delta_mask = self.routing_mask = None
        self.last_input = self.last_output = None
        self.cache_output = False

    @property
    def C(self):
        return self.core.C

    @property
    def W_target(self):
        return self.core.W_target


class RepeatedLinear(nn.Module):
    def __init__(self, linear, components, passes):
        super().__init__()
        self.core = vpd.ComponentLinear(linear, components).to(linear.weight.device)
        # V/U remain FP32; target buffers retain the source dtype. CUDA forwards
        # use BF16 autocast so component math and the original model interoperate.
        self.views = [InvocationView(self.core) for _ in range(passes)]
        self.calls = 0
        self.intervention = None

    def reset(self):
        self.calls = 0
        for view in self.views:
            view.last_input = view.last_output = None

    def forward(self, x):
        if self.calls >= len(self.views):
            raise RuntimeError("Unexpected invocation; generation/checkpoint recomputation unsupported")
        invocation = self.calls
        self.calls += 1
        view = self.views[invocation]
        for attr in ("mode", "mask", "delta_mask", "routing_mask", "cache_output"):
            setattr(self.core, attr, getattr(view, attr))
        out = self.core(x)
        view.last_input = x.detach()
        if self.intervention is not None:
            component, alpha, selected = self.intervention
            if view.mode != "target":
                raise RuntimeError("Parameter-effect measurement requires target mode")
            if invocation in selected:
                # Exactly the linear contribution of alpha * outer(U_c, V_c),
                # applied at this invocation without mutating the shared weights.
                out = out + alpha * ((x @ self.core.V[:, component:component + 1])
                                     @ self.core.U[component:component + 1]).to(out.dtype)
        if view.cache_output:
            view.last_output = out.detach()
        return out


class RecurrentTarget(nn.Module):
    def __init__(self, model, component_counts, *, passes=2):
        super().__init__()
        if passes < 1 or not component_counts:
            raise ValueError("Positive passes and at least one physical matrix required")
        if getattr(model, "is_gradient_checkpointing", False):
            raise ValueError("Disable gradient checkpointing before invocation-aware VPD")
        paths = list(component_counts)
        linears = [model.get_submodule(path) for path in paths]
        if any(not isinstance(module, nn.Linear) for module in linears):
            raise ValueError("Only nn.Linear matrices are supported")
        if len({id(module.weight) for module in linears}) != len(linears):
            raise ValueError("List each shared physical weight only once")
        if any(count < 1 for count in component_counts.values()):
            raise ValueError("Component counts must be positive")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model.eval()
        self.physical = {}
        self.views = {}
        self.passes = passes
        for path, linear in zip(paths, linears, strict=True):
            wrapper = RepeatedLinear(linear, component_counts[path], passes)
            parent_path, _, attr = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, attr, wrapper)
            self.physical[path] = wrapper
            for invocation, view in enumerate(wrapper.views):
                self.views[f"{path}::pass_{invocation}"] = view

    def forward(self, input_ids):
        for wrapper in self.physical.values():
            wrapper.reset()
        output = self.model(input_ids)
        for path, wrapper in self.physical.items():
            if wrapper.calls != self.passes:
                raise RuntimeError(f"Incomplete forward at {path}: {wrapper.calls}/{self.passes}")
        return output.logits if hasattr(output, "logits") else output

    def autocast(self):
        parameter = next(self.model.parameters())
        device = parameter.device
        if device.type == "cuda" or (device.type == "cpu" and parameter.dtype == torch.bfloat16):
            return torch.autocast(device.type, dtype=torch.bfloat16)
        return nullcontext()

    @contextmanager
    def intervene(self, path, component, alpha, passes):
        wrapper = self.physical[path]
        selected = frozenset(passes)
        if (not selected or not selected.issubset(range(self.passes))
                or not 0 <= component < wrapper.core.C
                or not torch.isfinite(torch.tensor(alpha))):
            raise ValueError("Invalid component, coefficient, or intervention passes")
        if wrapper.intervention is not None:
            raise RuntimeError("Nested intervention on the same matrix")
        vpd.clear_wrapper_masks(self.views)
        wrapper.intervention = (component, alpha, selected)
        try:
            yield
        finally:
            wrapper.intervention = None


def train_calibration(target, train_batches, cfg, *, variable_lengths=False,
                      saved_tensors_on_cpu=False, progress_callback=None):
    """Bounded VPD optimization with all four upstream objectives.

CI and PPGD operate on invocation keys. Faithfulness and optimizer parameters
are counted once per physical matrix. Returns a fitted *calibration*, not a
validated mechanistic decomposition. Batches are unpadded. Variable lengths use
separate persistent PGD state for each sequence length and shared model factors.
"""
    if cfg.n_steps < 1 or not train_batches:
        raise ValueError("A positive training budget and nonempty training split are required")
    shape = train_batches[0].shape
    if variable_lengths:
        if any(batch.ndim!=2 or batch.shape[0]!=cfg.batch_size or not 1<=batch.shape[1]<=cfg.seq_len
               for batch in train_batches):
            raise ValueError("Variable batches must retain the configured batch size and sequence bound")
    elif len(shape) != 2 or any(batch.shape != shape for batch in train_batches):
        raise ValueError("VPD calibration needs fixed [batch, sequence] shapes, without padding")
    if not variable_lengths and tuple(shape) != (cfg.batch_size, cfg.seq_len):
        raise ValueError("Batch shape differs from VPD configuration")
    device = train_batches[0].device
    if saved_tensors_on_cpu:
        # Copy tensors needed for backward to host memory. Forward arithmetic,
        # objective terms and full-prefix conditioning are unchanged; no forward
        # recomputation is introduced into the invocation-counting adapter.
        with torch.autograd.graph.save_on_cpu(pin_memory=device.type == 'cuda'):
            return train_calibration(target, train_batches, cfg,
                variable_lengths=variable_lengths, progress_callback=progress_callback)
    cores = {path: wrapper.core for path, wrapper in target.physical.items()}
    component_params = [p for core in cores.values() for p in (core.V, core.U)]
    if len({id(p) for p in component_params}) != len(component_params):
        raise RuntimeError("Duplicated physical parameters in optimizer")
    ci = vpd.CITransformer(
        {name: int(view.W_target.shape[1]) for name, view in target.views.items()},
        {name: view.C for name, view in target.views.items()}, cfg,
    ).to(device)
    warmup = torch.optim.AdamW(component_params, lr=cfg.faithfulness_warmup_lr,
                              weight_decay=0.0)
    for _ in range(cfg.faithfulness_warmup_steps):
        warmup.zero_grad(set_to_none=True)
        loss = vpd.faithfulness_loss(cores)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite faithfulness loss")
        loss.backward()
        warmup.step()
    optimizer = torch.optim.AdamW(component_params + list(ci.parameters()),
                                 lr=cfg.main_lr, weight_decay=0.0)
    pgd_by_length = {}
    history = []
    try:
        for step in range(cfg.n_steps):
            tokens = train_batches[step % len(train_batches)]
            length = tokens.shape[1]
            if length not in pgd_by_length:
                pgd_by_length[length] = vpd.PersistentPGD(target.views, cfg.batch_size, length, device, cfg)
            pgd = pgd_by_length[length]
            vpd.clear_wrapper_masks(target.views)
            with target.autocast():
                reference = target(tokens)
                acts = {name: view.last_input for name, view in target.views.items()}
                lower, upper, _ = ci(acts)
                pgd.warmup(target, target.views, tokens, reference, lower, cfg.ppgd_lr)
                losses = {
                    "faith": vpd.faithfulness_loss(cores),
                    "importance": vpd.importance_minimality_loss(
                        upper, vpd.anneal_p(step, cfg.n_steps, cfg.p_start, cfg.p_end),
                        cfg.imp_eps, cfg.imp_beta, 1),
                    "stochastic": vpd.stochastic_recon_loss(
                        target, target.views, tokens, reference, lower),
                    "adversarial": pgd.recon_loss(target, target.views, tokens, reference, lower),
                }
            total = (cfg.coeff_faith * losses["faith"] + cfg.coeff_imp * losses["importance"]
                     + cfg.coeff_stoch * losses["stochastic"]
                     + cfg.coeff_ppgd * losses["adversarial"])
            if not all(torch.isfinite(value) for value in [total, *losses.values()]):
                raise RuntimeError("Nonfinite VPD objective")
            source_grads = torch.autograd.grad(losses["adversarial"], list(pgd.sources.values()),
                                              retain_graph=True)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            parameters = component_params + list(ci.parameters())
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError("Nonfinite VPD gradient")
            torch.nn.utils.clip_grad_norm_(component_params, cfg.grad_clip_components)
            optimizer.step()
            pgd.external_step(dict(zip(pgd.sources, source_grads, strict=True)), cfg.ppgd_lr)
            history.append({"step": step + 1, **{name: float(value.detach())
                                                 for name, value in losses.items()}})
            if variable_lengths:
                history[-1]['sequence_length'] = length
            if progress_callback is not None:
                progress_callback(dict(history[-1]))
            if step == 0 or (step + 1) % 5 == 0 or step + 1 == cfg.n_steps:
                print(f"VPD step {step + 1}/{cfg.n_steps}: {history[-1]}", flush=True)
    finally:
        vpd.clear_wrapper_masks(target.views)
    return ci, history
