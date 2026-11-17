"""Strictly causal CI gating for closed-loop TRM rollouts (BPD-01 Experiment C).

``BPDTrainer.evaluate()`` computes ``ci_net.forward_trajectory(clean_traj)`` before the
masked rollout, so its gates are read from the clean target trajectory: a
clean-informed oracle. A closed-loop evaluation must instead derive the gate at
invocation ``tau`` from information the perturbed rollout has at ``tau``.

Within-pass dependency. The ``state_only`` and ``state_plus_clock`` networks
concatenate the pre-weight inputs of every wrapped module at the same invocation.
Inside one ``run_l_level`` call the wrapped modules execute in the order
``layers.0.o_proj -> layers.0.down_proj -> layers.1.o_proj -> layers.1.down_proj``
(``module_execution_order`` measures it), so the gate for ``layers.0.o_proj`` reads
inputs of three modules that run after it and depend on its masked output. Those
features do not exist when the first gated module executes.

Causal observation schedules implemented here:

``preflight_target``
    Before invocation ``tau`` runs, execute that invocation once with target
    weights from the current perturbed state and record the four module inputs
    ``X~_tau``; gate with ``f(X~_tau)``; discard the preflight output and run the
    masked invocation. Uses only ``(z_main, z_inj)`` at ``tau``. At ``tau = 0`` the
    preflight inputs equal the clean inputs.
``lagged``
    Gate invocation ``tau`` with ``f(X_{tau-1})``, the inputs the masked modules
    actually received at ``tau - 1`` (the preceding invocation, whatever its z_L/z_H
    role). Invocation 0 uses ``preflight_target``. Causal and preflight-free, one
    invocation stale.

Gate selection: ``topk`` keeps the ``k`` highest CI logits per module, token and
invocation (budget-matched to the SVD selectors); ``threshold`` keeps CI > t;
``continuous`` uses the lower-leaky CI value itself (``BPDTrainer.evaluate`` semantics).
"""
from __future__ import annotations

from typing import Any, Literal, Mapping

import torch
from torch import Tensor

from mechinterp.bpd.ci import TiedCINetwork
from mechinterp.bpd.components import BPDModel, PassMaskSchedule

Observation = Literal["preflight_target", "lagged"]
Selection = Literal["topk", "threshold", "continuous"]


def module_execution_order(bpd: BPDModel, run_once: Any) -> list[str]:
    """Order in which wrapped modules first execute during ``run_once()``."""
    order: list[str] = []
    handles = []
    for name, w in bpd.wrappers.items():
        def _pre(_m: Any, _inp: Any, _name: str = name) -> None:
            if _name not in order:
                order.append(_name)
        handles.append(w.register_forward_pre_hook(_pre))
    try:
        with torch.no_grad():
            run_once()
    finally:
        for h in handles:
            h.remove()
    return order


def select_gates(
    pre_sigmoid: Tensor,
    lower_leaky: Tensor,
    *,
    selection: Selection,
    k: int | None = None,
    threshold: float = 0.5,
) -> Tensor:
    """Turn CI outputs ``[..., C]`` into a component mask under the chosen rule."""
    if selection == "continuous":
        return lower_leaky.float()
    if selection == "threshold":
        return (lower_leaky > threshold).float()
    if selection == "topk":
        if k is None or not 1 <= k <= pre_sigmoid.shape[-1]:
            raise ValueError(f"topk selection needs 1 <= k <= C, got {k}")
        idx = pre_sigmoid.float().topk(k, dim=-1).indices
        return torch.zeros_like(pre_sigmoid, dtype=torch.float32).scatter_(-1, idx, 1.0)
    raise ValueError(f"unknown selection {selection!r}")


def _ci_on_single_pass(ci_net: TiedCINetwork, inputs: Mapping[str, Tensor], tau: int) -> Any:
    """CI outputs for one invocation; ``inputs[m]`` is ``[B, 1, P, d_in]``."""
    return ci_net(inputs, pass_offset=tau)


@torch.no_grad()
def clean_informed_ci_masks(
    bpd: BPDModel,
    adapter: Any,
    ci_net: TiedCINetwork,
    inputs: Tensor,
    n_passes: int,
    *,
    selection: Selection,
    k: int | None = None,
    threshold: float = 0.5,
    pass_chunk: int = 7,
) -> dict[str, Tensor]:
    """Masks ``[B, T, P, C]`` computed from the clean target trajectory (oracle gates)."""
    with bpd.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        adapter.rollout(inputs, n_passes, 0, None)
    clean_in = bpd.collect_recorded_inputs(expected_passes=n_passes)
    for w in bpd.wrappers.values():
        w.recorded_inputs.clear()
    parts: dict[str, list[Tensor]] = {m: [] for m in bpd.module_order}
    for t0 in range(0, n_passes, pass_chunk):
        t1 = min(t0 + pass_chunk, n_passes)
        out = ci_net({m: clean_in[m][:, t0:t1] for m in bpd.module_order}, pass_offset=t0)
        for m in bpd.module_order:
            parts[m].append(select_gates(out.pre_sigmoid[m], out.lower_leaky[m],
                                         selection=selection, k=k, threshold=threshold))
    return {m: torch.cat(v, dim=1) for m, v in parts.items()}


@torch.no_grad()
def causal_ci_rollout(
    bpd: BPDModel,
    adapter: Any,
    ci_net: TiedCINetwork,
    inputs: Tensor,
    n_passes: int,
    *,
    observation: Observation = "preflight_target",
    selection: Selection = "topk",
    k: int | None = None,
    threshold: float = 0.5,
    delta_value: float = 0.0,
    reference_masks: Mapping[str, Tensor] | None = None,
    replay_masks: Mapping[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Closed-loop TRM rollout whose gates are computed online from the perturbed state.

    Mirrors ``TRMBPDAdapter.rollout`` invocation by invocation. ``reference_masks``
    (e.g. clean-informed masks ``[B, T, P, C]``) are read only for disagreement
    telemetry and never enter the gate computation.

    ``replay_masks`` ([B, T, P, C]) is a verification mode: the loop runs every step
    (preflight, CI, telemetry) but applies these masks instead of the CI gates. Replaying
    clean-informed masks must reproduce the open-loop masked rollout exactly.
    Returns readouts ``{tau: [B, 81, 9]}``, boundary states, per-invocation mask
    density per module, and per-invocation disagreement with the reference.
    """
    model = adapter.model
    dev = model.embed_tokens.weight.device
    inp = inputs.to(dev)
    x_emb = model.encode_inputs(inp)
    z_h, z_l = model.init_states(inp.shape[0])
    cyc = adapter.calls_per_h_cycle

    readouts: dict[int, Tensor] = {}
    states: dict[int, Tensor] = {}
    density: dict[str, list[float]] = {m: [] for m in bpd.module_order}
    disagreement: dict[str, list[list[float]]] = {m: [] for m in bpd.module_order}
    last_masked_inputs: dict[str, Tensor] | None = None
    bsz = int(inp.shape[0])

    for tau in range(n_passes):
        is_l = (tau % cyc) < model.l_cycles
        z_main, z_inj = (z_l, z_h + x_emb) if is_l else (z_h, z_l)

        if observation == "lagged" and last_masked_inputs is not None:
            obs = last_masked_inputs
        elif observation in ("preflight_target", "lagged"):
            with bpd.target_capture(pass_offset=tau, update_means=False, detach_inputs=True):
                model.run_l_level(z_main, z_inj)
            obs = bpd.collect_recorded_inputs(expected_passes=1)
            for w in bpd.wrappers.values():
                w.recorded_inputs.clear()
        else:
            raise ValueError(f"unknown observation {observation!r}")

        ci_out = _ci_on_single_pass(ci_net, obs, tau)
        sched: dict[str, PassMaskSchedule] = {}
        for m in bpd.module_order:
            mask = select_gates(ci_out.pre_sigmoid[m], ci_out.lower_leaky[m],
                                selection=selection, k=k, threshold=threshold)
            if replay_masks is not None:
                mask = replay_masks[m][:, tau : tau + 1].to(device=mask.device, dtype=mask.dtype)
            density[m].append(float(mask.mean()))
            if reference_masks is not None:
                ref = reference_masks[m][:, tau : tau + 1].to(mask.device)
                disagreement[m].append(
                    [float(v) for v in (mask - ref).abs().mean(dim=(1, 2, 3)).tolist()]
                )
            sched[m] = PassMaskSchedule(
                comp_mask=mask,
                delta_mask=torch.full((bsz, 1, mask.shape[2]), float(delta_value), device=mask.device),
            )

        with bpd.masked_components(sched, pass_offset=tau, record_inputs=(observation == "lagged")):
            new = model.run_l_level(z_main, z_inj)
            if observation == "lagged":
                last_masked_inputs = {
                    m: t.detach() for m, t in bpd.collect_recorded_inputs(expected_passes=1).items()
                }
                for w in bpd.wrappers.values():
                    w.recorded_inputs.clear()

        if is_l:
            z_l = new
        else:
            z_h = new
            if adapter.is_readout_pass(tau):
                readouts[tau] = model.digit_logits(z_h)
        states[tau] = new

    return {
        "readouts": readouts,
        "boundary_states": states,
        "density_by_module_by_pass": density,
        "disagreement_by_module_by_pass_by_board": disagreement if reference_masks is not None else None,
    }


def within_pass_dependency(
    ci_net: TiedCINetwork,
    module_inputs: Mapping[str, Tensor],
    *,
    tau: int = 0,
) -> dict[str, dict[str, float]]:
    """``|d sum(gate logits of A) / d X_B|_F`` for every module pair at one invocation.

    A nonzero entry for module ``B`` executing after ``A`` means A's gate reads a
    feature that is computed downstream of A's own masked write.
    """
    leaves = {m: t.detach().clone().float().requires_grad_(True) for m, t in module_inputs.items()}
    out = ci_net(leaves, pass_offset=tau)
    result: dict[str, dict[str, float]] = {}
    for a in ci_net.module_order:
        grads = torch.autograd.grad(out.pre_sigmoid[a].sum(), list(leaves.values()),
                                    retain_graph=True, allow_unused=True)
        result[a] = {
            b: (0.0 if g is None else float(g.norm())) for b, g in zip(leaves.keys(), grads)
        }
    return result
