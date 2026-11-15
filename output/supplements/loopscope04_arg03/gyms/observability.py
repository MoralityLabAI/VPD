"""Sampled readout-kernel observability Gramian for a stationary TRM outer step.

Finite angular pairs are *exactly matched at t* in the full lm_head readout
and in per-cell L2 norm. Their subsequent readout differences approximate
columns of D(q o F^j) restricted to head-orthogonal, sphere-tangent directions.
The returned spectrum is a sampled local diagnostic, not a global proof of
dark-fiber dimensionality or path-dependent reasoning.
"""
from __future__ import annotations

import math
import torch

from .core import project_visible, row_basis


@torch.no_grad()
def _sample_orthonormal_tangent_basis(
    zh: torch.Tensor,
    head: torch.Tensor,
    *,
    board_start: int,
    probes: int = 8,
    seed: int = 20261115,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build `probes` sphere-tangent fields [probes, n_cells, D], each of per-cell norm `radius`.

    When per-cell tangent dimension (D - rank(head) - 1) >= probes (500 >= probes on TRM),
    performs exact per-cell Gram-Schmidt so any unit vector c in R^probes produces
    sum_p c_p U_p with exact per-cell radius without rescaling distortion.
    """
    if zh.shape[0] != 1 or zh.ndim != 3 or zh.dtype != torch.float32:
        raise ValueError("requires one float32 [1,T,D] source state")
    basis = row_basis(head).to(zh.device)
    board = zh[:, board_start:, :]  # [1, C, D]
    vis = project_visible(board, basis)
    orth = board - vis
    radius = orth.norm(dim=-1, keepdim=True)  # [1, C, 1]
    if torch.any(radius < 1e-5):
        raise ValueError("orthogonal carrier radius too small")
    kernel_dim = int(zh.shape[-1] - basis.shape[1])
    cell_tangent_dim = kernel_dim - 1
    gen = torch.Generator(device="cpu").manual_seed(seed)
    fields: list[torch.Tensor] = []
    per_cell_gs = cell_tangent_dim >= probes
    for _ in range(probes):
        rand = torch.randn(tuple(board.shape), generator=gen, dtype=torch.float32).to(zh.device)
        tangent = rand - project_visible(rand, basis)
        tangent = tangent - ((tangent * orth).sum(-1, keepdim=True) / radius.square()) * orth
        if per_cell_gs and fields:
            for prev in fields:
                proj_coeff = (tangent * prev).sum(-1, keepdim=True) / radius.square()
                tangent = tangent - proj_coeff * prev
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        if torch.any(tangent_norm < 1e-6):
            raise ValueError("tangent collapses; insufficient kernel")
        tangent = tangent / tangent_norm * radius
        fields.append(tangent)
    u_stack = torch.cat(fields, dim=0)  # [probes, C, D]
    return u_stack, vis, orth, radius, kernel_dim


@torch.no_grad()
def sphere_tangent_pairs(
    zh: torch.Tensor, head: torch.Tensor, *,
    board_start: int, probes: int = 8, angle: float = 0.04, seed: int = 20261115,
    atol: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    """Return [2*probes,T,D] paired states (+angle,-angle) on matched readout fibers."""
    if zh.shape[0]!=1 or zh.ndim!=3 or zh.dtype!=torch.float32:
        raise ValueError("requires one float32 [1,T,D] source state")
    if probes<1 or not (0<angle<0.25):
        raise ValueError("invalid local tangent sampling")
    u_stack, vis, orth, _radius, kernel_dim = _sample_orthonormal_tangent_basis(
        zh, head, board_start=board_start, probes=probes, seed=seed
    )
    board = zh[:, board_start:, :]
    all_states = []
    for p_idx in range(probes):
        tangent = u_stack[p_idx : p_idx + 1]
        for sign in (1, -1):
            out = zh.clone()
            out[:, board_start:, :] = vis + math.cos(angle) * orth + sign * math.sin(angle) * tangent
            all_states.append(out)
    states = torch.cat(all_states, dim=0)
    w = head.float().to(zh.device)
    full_read = states[:, board_start:, :] @ w.T
    ref = board @ w.T
    leak = float((full_read - ref).abs().max())
    norm_error = float((states[:, board_start:, :].norm(dim=-1) - board.norm(dim=-1)).abs().max())
    if max(leak, norm_error) > atol:
        raise AssertionError(f"gramian tangent pre-readout parity failed: {leak}, {norm_error}")
    return states, {
        "max_readout_error": leak,
        "max_l2_error": norm_error,
        "probes": probes,
        "angle": angle,
        "kernel_dim": kernel_dim,
        "per_cell_orthonormal": bool((kernel_dim - 1) >= probes),
    }


@torch.no_grad()
def sampled_trm_gramian(model, zh: torch.Tensor, zl: torch.Tensor, x_emb: torch.Tensor,
                       *, board_start: int, probes: int = 8, angle: float = 0.04,
                       horizon: int = 4, seed: int = 20261115) -> dict:
    """Gramian over fixed orthonormal angular tangent directions at one clean state.

    The matrix is probes x probes and its units are squared readout response
    per radian, since all directions rotate each board-cell orthogonal radius.
    """
    if horizon<1:
        raise ValueError("horizon must be >=1")
    states,certificate=sphere_tangent_pairs(
        zh,model.lm_head.weight,board_start=board_start,probes=probes,
        angle=angle,seed=seed)
    batch=2*probes
    zl_batch=zl.expand(batch,-1,-1).clone()
    x_batch=x_emb.expand(batch,-1,-1)
    gram=torch.zeros((probes,probes),dtype=torch.float64,device=zh.device)
    responses=[]
    per_lag_traces=[]
    for _j in range(1,horizon+1):
        states,zl_batch=model.act_step(states,zl_batch,x_batch)
        readout=model.lm_head(states)[:,board_start:,:].float()
        # Central differences in the angular fiber chart.
        delta=(readout[0::2]-readout[1::2])/(2*angle)
        flat=delta.reshape(probes,-1).double()
        step_gram = flat @ flat.T
        gram+=step_gram
        responses.append(float(delta.norm(dim=(-1,-2)).mean()))
        per_lag_traces.append(float(step_gram.trace()))
    gram=(gram+gram.T)/2
    evals, _evecs = torch.linalg.eigh(gram)
    values=evals.clamp_min(0).cpu().tolist()[::-1]
    top=values[0] if values else 0.0
    bot=values[-1] if values else 0.0
    return {
        "schema":"darkfibers.sampled_gramian.v1",
        "certificate":certificate,"horizon":horizon,
        "eigenvalues_descending":[float(v) for v in values],
        "condition_ratio_top_to_bottom": float(top / max(bot, 1e-12)) if top > 0 else 1.0,
        "sampled_rank_relative_1e-3":sum(v>top*1e-3 for v in values) if top>0 else 0,
        "per_lag_mean_response":[float(v) for v in responses],
        "per_lag_gramian_trace":[float(v) for v in per_lag_traces],
        "frobenius_trace":float(gram.trace()),
    }


@torch.no_grad()
def causal_gramian_validation(
    model,
    zh: torch.Tensor,
    zl: torch.Tensor,
    x_emb: torch.Tensor,
    *,
    answer_digits: list[int] | None = None,
    board_start: int,
    probes: int = 8,
    fd_angle: float = 0.04,
    gramian_horizon: int = 4,
    eval_horizon: int = 12,
    test_angles: tuple[float, ...] = (0.04, 0.15, 0.35, 0.75),
    seed: int = 20261115,
    atol: float = 1e-3,
) -> dict:
    """Estimate sampled G_k and causally test high-eig vs low-eig vs fresh-random directions.

    All three directions (`high_eig`, `low_eig`, `random_dir`) have identical per-cell sphere
    radius, zero pre-intervention full-head logit error, and identical perturbation L2 norm at
    each tested finite angular amplitude `beta` in `test_angles`.
    """
    if gramian_horizon < 1 or eval_horizon < 1:
        raise ValueError("gramian_horizon and eval_horizon must be >= 1")
    u_stack, vis, orth, radius, kernel_dim = _sample_orthonormal_tangent_basis(
        zh, model.lm_head.weight, board_start=board_start, probes=probes, seed=seed
    )
    board = zh[:, board_start:, :]
    w_head = model.lm_head.weight.float().to(zh.device)

    # 1. Estimate G_k on the orthonormal basis u_stack
    pair_states = []
    for p_idx in range(probes):
        t_p = u_stack[p_idx : p_idx + 1]
        for sgn in (1.0, -1.0):
            st = zh.clone()
            st[:, board_start:, :] = vis + math.cos(fd_angle) * orth + sgn * math.sin(fd_angle) * t_p
            pair_states.append(st)
    states_fd = torch.cat(pair_states, dim=0)
    pre_leak = float(((states_fd[:, board_start:, :] @ w_head.T) - (board @ w_head.T)).abs().max())
    pre_norm = float((states_fd[:, board_start:, :].norm(dim=-1) - board.norm(dim=-1)).abs().max())
    if max(pre_leak, pre_norm) > atol:
        raise AssertionError(f"gramian basis parity failed: {pre_leak}, {pre_norm}")

    zl_fd = zl.expand(2 * probes, -1, -1).clone()
    x_fd = x_emb.expand(2 * probes, -1, -1)
    gram = torch.zeros((probes, probes), dtype=torch.float64, device=zh.device)
    responses = []
    per_lag_traces = []
    for _j in range(1, gramian_horizon + 1):
        states_fd, zl_fd = model.act_step(states_fd, zl_fd, x_fd)
        ro = model.lm_head(states_fd)[:, board_start:, :].float()
        delta = (ro[0::2] - ro[1::2]) / (2.0 * fd_angle)
        flat = delta.reshape(probes, -1).double()
        sg = flat @ flat.T
        gram += sg
        responses.append(float(delta.norm(dim=(-1, -2)).mean()))
        per_lag_traces.append(float(sg.trace()))
    gram = 0.5 * (gram + gram.T)
    evals, evecs = torch.linalg.eigh(gram)  # ascending order
    c_low = evecs[:, 0].float()
    c_high = evecs[:, -1].float()

    # Synthesize unit-radius tangent fields for high_eig, low_eig, and fresh random_dir
    def combine_unit(coeff: torch.Tensor) -> torch.Tensor:
        comb = (coeff.view(probes, 1, 1) * u_stack).sum(dim=0, keepdim=True)  # [1, C, D]
        comb = comb - project_visible(comb, row_basis(model.lm_head.weight).to(zh.device))
        comb = comb - ((comb * orth).sum(-1, keepdim=True) / radius.square()) * orth
        return comb / comb.norm(dim=-1, keepdim=True).clamp_min(1e-8) * radius

    t_high = combine_unit(c_high)
    t_low = combine_unit(c_low)
    t_rand, _, _, _, _ = _sample_orthonormal_tangent_basis(
        zh, model.lm_head.weight, board_start=board_start, probes=1, seed=seed + 77777
    )

    # Clean reference rollout from (zh, zl) for eval_horizon steps
    zh_clean, zl_clean = zh.clone(), zl.clone()
    clean_readouts = []
    clean_preds = []
    digit_slice = slice(2, 11) if w_head.shape[0] >= 11 else slice(None)
    for _step in range(1, eval_horizon + 1):
        zh_clean, zl_clean = model.act_step(zh_clean, zl_clean, x_emb)
        ro_c = model.lm_head(zh_clean)[:, board_start:, :].float()
        clean_readouts.append(ro_c)
        clean_preds.append(ro_c[:, :, digit_slice].argmax(dim=-1)[0])

    # Evaluate high_eig, low_eig, random_dir across all test_angles in one batched rollout
    arm_specs = []
    init_list = []
    for beta in test_angles:
        for d_name, t_field in (("high_eig", t_high), ("low_eig", t_low), ("random_dir", t_rand)):
            st = zh.clone()
            st[:, board_start:, :] = vis + math.cos(beta) * orth + math.sin(beta) * t_field
            l_err = float(((st[:, board_start:, :] @ w_head.T) - (board @ w_head.T)).abs().max())
            n_err = float((st[:, board_start:, :].norm(dim=-1) - board.norm(dim=-1)).abs().max())
            if max(l_err, n_err) > atol:
                raise AssertionError(f"causal gramian arm {d_name}@{beta} parity failed: {l_err}, {n_err}")
            pert_l2 = float((st[:, board_start:, :] - board).norm().item())
            arm_specs.append((beta, d_name, l_err, n_err, pert_l2))
            init_list.append(st)

    zh_b = torch.cat(init_list, dim=0)
    zl_b = zl.expand(len(init_list), -1, -1).clone()
    x_b = x_emb.expand(len(init_list), -1, -1)

    lag_metrics: dict[int, torch.Tensor] = {}
    pred_at_h = None
    k_check = min(gramian_horizon, eval_horizon)
    cum_sq = torch.zeros(len(init_list), dtype=torch.float32)
    for step in range(1, eval_horizon + 1):
        zh_b, zl_b = model.act_step(zh_b, zl_b, x_b)
        ro_b = model.lm_head(zh_b)[:, board_start:, :].float()
        diff_l2 = (ro_b - clean_readouts[step - 1]).reshape(len(init_list), -1).norm(dim=-1)
        if step in (1, k_check, eval_horizon):
            lag_metrics[step] = diff_l2.cpu()
        if step <= k_check:
            cum_sq = cum_sq + diff_l2.square().cpu()
        if step == eval_horizon:
            pred_at_h = ro_b[:, :, digit_slice].argmax(dim=-1)  # [B, C]

    by_angle = {}
    ans_t = (
        torch.tensor([d - 1 for d in answer_digits], device=zh.device, dtype=torch.long)
        if answer_digits is not None
        else None
    )
    clean_solved_end = bool(torch.equal(clean_preds[-1], ans_t)) if ans_t is not None else None

    for idx, (beta, d_name, l_err, n_err, pert_l2) in enumerate(arm_specs):
        key = f"angle_{beta:.2f}"
        by_angle.setdefault(key, {"angle_rad": float(beta), "directions": {}})
        flips_end = int((pred_at_h[idx] != clean_preds[-1]).sum().item())
        rec = {
            "pre_readout_max_error": l_err,
            "pre_cell_l2_max_error": n_err,
            "initial_perturbation_l2": pert_l2,
            "lag1_readout_l2": float(lag_metrics[1][idx].item()),
            f"lag{k_check}_readout_l2": float(lag_metrics[k_check][idx].item()),
            f"cum_h{k_check}_rms_readout_l2": float(cum_sq[idx].sqrt().item()),
            f"lag{eval_horizon}_readout_l2": float(lag_metrics[eval_horizon][idx].item()),
            f"board_flips_vs_clean_at_h{eval_horizon}": flips_end,
        }
        if ans_t is not None:
            rec[f"exact_solve_at_h{eval_horizon}"] = bool(torch.equal(pred_at_h[idx], ans_t))
        by_angle[key]["directions"][d_name] = rec

    values_desc = evals.clamp_min(0).cpu().tolist()[::-1]
    top = values_desc[0] if values_desc else 0.0
    bot = values_desc[-1] if values_desc else 0.0
    return {
        "schema": "darkfibers.sampled_gramian_causal.v1",
        "certificate": {
            "max_readout_error": pre_leak,
            "max_l2_error": pre_norm,
            "probes": probes,
            "angle": fd_angle,
            "kernel_dim": kernel_dim,
            "per_cell_orthonormal": bool((kernel_dim - 1) >= probes),
        },
        "horizon": gramian_horizon,
        "eval_horizon": eval_horizon,
        "clean_exact_solve_at_eval_horizon": clean_solved_end,
        "eigenvalues_descending": [float(v) for v in values_desc],
        "condition_ratio_top_to_bottom": float(top / max(bot, 1e-12)) if top > 0 else 1.0,
        "sampled_rank_relative_1e-3": sum(v > top * 1e-3 for v in values_desc) if top > 0 else 0,
        "per_lag_mean_response": [float(v) for v in responses],
        "per_lag_gramian_trace": [float(v) for v in per_lag_traces],
        "frobenius_trace": float(gram.trace()),
        "causal_direction_tests": by_angle,
    }


@torch.no_grad()
def causal_gramian_validation_batch(
    model,
    zh_batch: torch.Tensor,
    zl_batch: torch.Tensor,
    x_batch: torch.Tensor,
    *,
    answers_digits: list[list[int]] | None = None,
    board_start: int,
    probes: int = 8,
    fd_angle: float = 0.04,
    gramian_horizon: int = 4,
    eval_horizon: int = 12,
    test_angles: tuple[float, ...] = (0.04, 0.15, 0.35, 0.75),
    seeds: list[int],
    atol: float = 1e-3,
) -> list[dict]:
    """Batched GPU version of causal_gramian_validation preserving exact per-board seeds."""
    m = zh_batch.shape[0]
    if len(seeds) != m:
        raise ValueError("seeds length must equal batch size")
    dev = zh_batch.device
    w_head = model.lm_head.weight.float().to(dev)
    basis = row_basis(model.lm_head.weight).to(dev)
    digit_slice = slice(2, 11) if w_head.shape[0] >= 11 else slice(None)

    # 1. Sample per-board orthonormal tangent bases and stack all [m * 2 * probes] FD trajectories
    bases_meta = []
    fd_zh_list, fd_zl_list, fd_x_list = [], [], []
    for bi in range(m):
        zh_i = zh_batch[bi : bi + 1]
        zl_i = zl_batch[bi : bi + 1]
        x_i = x_batch[bi : bi + 1]
        u_stack, vis, orth, radius, kernel_dim = _sample_orthonormal_tangent_basis(
            zh_i, model.lm_head.weight, board_start=board_start, probes=probes, seed=seeds[bi]
        )
        board_i = zh_i[:, board_start:, :]
        pair_states = []
        for p_idx in range(probes):
            t_p = u_stack[p_idx : p_idx + 1]
            for sgn in (1.0, -1.0):
                st = zh_i.clone()
                st[:, board_start:, :] = vis + math.cos(fd_angle) * orth + sgn * math.sin(fd_angle) * t_p
                pair_states.append(st)
        st_fd_i = torch.cat(pair_states, dim=0)
        pre_leak = float(((st_fd_i[:, board_start:, :] @ w_head.T) - (board_i @ w_head.T)).abs().max())
        pre_norm = float((st_fd_i[:, board_start:, :].norm(dim=-1) - board_i.norm(dim=-1)).abs().max())
        if max(pre_leak, pre_norm) > atol:
            raise AssertionError(f"gramian basis parity failed: {pre_leak}, {pre_norm}")
        bases_meta.append((u_stack, vis, orth, radius, kernel_dim, pre_leak, pre_norm))
        fd_zh_list.append(st_fd_i)
        fd_zl_list.append(zl_i.expand(2 * probes, -1, -1).clone())
        fd_x_list.append(x_i.expand(2 * probes, -1, -1))

    zh_fd = torch.cat(fd_zh_list, dim=0)
    zl_fd = torch.cat(fd_zl_list, dim=0)
    x_fd = torch.cat(fd_x_list, dim=0)

    grams = [torch.zeros((probes, probes), dtype=torch.float64, device=dev) for _ in range(m)]
    responses = [[] for _ in range(m)]
    per_lag_traces = [[] for _ in range(m)]

    for _j in range(1, gramian_horizon + 1):
        zh_fd, zl_fd = model.act_step(zh_fd, zl_fd, x_fd)
        ro_all = model.lm_head(zh_fd)[:, board_start:, :].float().view(m, 2 * probes, -1, w_head.shape[0])
        for bi in range(m):
            ro = ro_all[bi]
            delta = (ro[0::2] - ro[1::2]) / (2.0 * fd_angle)
            flat = delta.reshape(probes, -1).double()
            sg = flat @ flat.T
            grams[bi] += sg
            responses[bi].append(float(delta.norm(dim=(-1, -2)).mean()))
            per_lag_traces[bi].append(float(sg.trace()))

    # 2. Construct clean + 3 * len(test_angles) continuation arms per board and roll out together
    n_arms_per_board = 1 + 3 * len(test_angles)  # index 0 = clean, 1.. = directional arms
    ev_zh_list, ev_zl_list, ev_x_list = [], [], []
    per_board_arm_specs = []
    per_board_eig_meta = []

    for bi in range(m):
        zh_i = zh_batch[bi : bi + 1]
        zl_i = zl_batch[bi : bi + 1]
        x_i = x_batch[bi : bi + 1]
        board_i = zh_i[:, board_start:, :]
        u_stack, vis, orth, radius, _kdim, _pl, _pn = bases_meta[bi]
        g_mat = 0.5 * (grams[bi] + grams[bi].T)
        evals, evecs = torch.linalg.eigh(g_mat)
        c_low = evecs[:, 0].float()
        c_high = evecs[:, -1].float()
        per_board_eig_meta.append((evals, g_mat))

        def _comb_unit(coeff: torch.Tensor) -> torch.Tensor:
            comb = (coeff.view(probes, 1, 1) * u_stack).sum(dim=0, keepdim=True)
            comb = comb - project_visible(comb, basis)
            comb = comb - ((comb * orth).sum(-1, keepdim=True) / radius.square()) * orth
            return comb / comb.norm(dim=-1, keepdim=True).clamp_min(1e-8) * radius

        t_high = _comb_unit(c_high)
        t_low = _comb_unit(c_low)
        t_rand, _, _, _, _ = _sample_orthonormal_tangent_basis(
            zh_i, model.lm_head.weight, board_start=board_start, probes=1, seed=seeds[bi] + 77777
        )

        b_inits = [zh_i.clone()]
        b_specs = []
        for beta in test_angles:
            for d_name, t_field in (("high_eig", t_high), ("low_eig", t_low), ("random_dir", t_rand)):
                st = zh_i.clone()
                st[:, board_start:, :] = vis + math.cos(beta) * orth + math.sin(beta) * t_field
                l_err = float(((st[:, board_start:, :] @ w_head.T) - (board_i @ w_head.T)).abs().max())
                n_err = float((st[:, board_start:, :].norm(dim=-1) - board_i.norm(dim=-1)).abs().max())
                if max(l_err, n_err) > atol:
                    raise AssertionError(f"causal gramian arm {d_name}@{beta} parity failed: {l_err}, {n_err}")
                pert_l2 = float((st[:, board_start:, :] - board_i).norm().item())
                b_specs.append((beta, d_name, l_err, n_err, pert_l2))
                b_inits.append(st)

        per_board_arm_specs.append(b_specs)
        ev_zh_list.append(torch.cat(b_inits, dim=0))
        ev_zl_list.append(zl_i.expand(n_arms_per_board, -1, -1).clone())
        ev_x_list.append(x_i.expand(n_arms_per_board, -1, -1))

    zh_ev = torch.cat(ev_zh_list, dim=0)
    zl_ev = torch.cat(ev_zl_list, dim=0)
    x_ev = torch.cat(ev_x_list, dim=0)

    k_check = min(gramian_horizon, eval_horizon)
    n_dir_arms = n_arms_per_board - 1
    lag_metrics: dict[int, torch.Tensor] = {}
    cum_sq = torch.zeros((m, n_dir_arms), dtype=torch.float32)
    clean_pred_end = None
    dir_pred_end = None

    for step in range(1, eval_horizon + 1):
        zh_ev, zl_ev = model.act_step(zh_ev, zl_ev, x_ev)
        ro_ev = model.lm_head(zh_ev)[:, board_start:, :].float().view(m, n_arms_per_board, -1, w_head.shape[0])
        ro_clean = ro_ev[:, 0:1]  # [m, 1, C, V]
        ro_dirs = ro_ev[:, 1:]    # [m, n_dir_arms, C, V]
        diff_l2 = (ro_dirs - ro_clean).reshape(m, n_dir_arms, -1).norm(dim=-1).cpu()
        if step in (1, k_check, eval_horizon):
            lag_metrics[step] = diff_l2
        if step <= k_check:
            cum_sq = cum_sq + diff_l2.square()
        if step == eval_horizon:
            clean_pred_end = ro_clean[:, 0, :, digit_slice].argmax(dim=-1)  # [m, C]
            dir_pred_end = ro_dirs[:, :, :, digit_slice].argmax(dim=-1)     # [m, n_dir_arms, C]

    results = []
    for bi in range(m):
        _u, _v, _o, _r, kernel_dim, pre_leak, pre_norm = bases_meta[bi]
        evals, g_mat = per_board_eig_meta[bi]
        b_specs = per_board_arm_specs[bi]
        ans_t = (
            torch.tensor([d - 1 for d in answers_digits[bi]], device=dev, dtype=torch.long)
            if answers_digits is not None
            else None
        )
        clean_solved_end = bool(torch.equal(clean_pred_end[bi], ans_t)) if ans_t is not None else None
        by_angle = {}
        for idx, (beta, d_name, l_err, n_err, pert_l2) in enumerate(b_specs):
            key = f"angle_{beta:.2f}"
            by_angle.setdefault(key, {"angle_rad": float(beta), "directions": {}})
            flips_end = int((dir_pred_end[bi, idx] != clean_pred_end[bi]).sum().item())
            rec = {
                "pre_readout_max_error": l_err,
                "pre_cell_l2_max_error": n_err,
                "initial_perturbation_l2": pert_l2,
                "lag1_readout_l2": float(lag_metrics[1][bi, idx].item()),
                f"lag{k_check}_readout_l2": float(lag_metrics[k_check][bi, idx].item()),
                f"cum_h{k_check}_rms_readout_l2": float(cum_sq[bi, idx].sqrt().item()),
                f"lag{eval_horizon}_readout_l2": float(lag_metrics[eval_horizon][bi, idx].item()),
                f"board_flips_vs_clean_at_h{eval_horizon}": flips_end,
            }
            if ans_t is not None:
                rec[f"exact_solve_at_h{eval_horizon}"] = bool(torch.equal(dir_pred_end[bi, idx], ans_t))
            by_angle[key]["directions"][d_name] = rec

        values_desc = evals.clamp_min(0).cpu().tolist()[::-1]
        top = values_desc[0] if values_desc else 0.0
        bot = values_desc[-1] if values_desc else 0.0
        results.append({
            "schema": "darkfibers.sampled_gramian_causal.v1",
            "certificate": {
                "max_readout_error": pre_leak,
                "max_l2_error": pre_norm,
                "probes": probes,
                "angle": fd_angle,
                "kernel_dim": kernel_dim,
                "per_cell_orthonormal": bool((kernel_dim - 1) >= probes),
            },
            "horizon": gramian_horizon,
            "eval_horizon": eval_horizon,
            "clean_exact_solve_at_eval_horizon": clean_solved_end,
            "eigenvalues_descending": [float(v) for v in values_desc],
            "condition_ratio_top_to_bottom": float(top / max(bot, 1e-12)) if top > 0 else 1.0,
            "sampled_rank_relative_1e-3": sum(v > top * 1e-3 for v in values_desc) if top > 0 else 0,
            "per_lag_mean_response": [float(v) for v in responses[bi]],
            "per_lag_gramian_trace": [float(v) for v in per_lag_traces[bi]],
            "frobenius_trace": float(g_mat.trace()),
            "causal_direction_tests": by_angle,
        })
    return results
