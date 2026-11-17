"""BPD-01 regressions: float64 divergence helpers, closure-mechanism pieces, online-SVD extras."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.adapters.trm import TRMBPDAdapter, build_mini_trm, default_trm_c_per_module
from mechinterp.bpd.components import AblationConfig, BPDModel
from mechinterp.bpd.scripts import b4_numerics as N
from mechinterp.bpd.scripts.audit_b4_v2_closure_mechanism import (
    _wrapper_identity_output,
    decide,
)
from mechinterp.bpd.scripts.audit_b4_v2_online_svd import audit_one


def test_kl64_zero_on_identity_and_positive_otherwise() -> None:
    a = torch.randn(3, 81, 9)
    assert N.kl64_per_board(a, a) == [0.0, 0.0, 0.0]
    b = a.clone()
    b[1, 0, 0] += 5.0
    kl = N.kl64_per_board(b, a)
    assert kl[0] == 0.0 and kl[2] == 0.0 and kl[1] > 0.0


def test_argmax_flips_counts_changed_cells() -> None:
    t = torch.zeros(2, 81, 9)
    t[..., 0] = 1.0
    p = t.clone()
    p[0, :5, 3] = 2.0
    assert N.argmax_flips_per_board(p, t) == [5, 0]


def test_bf16_ulp_matches_grid_spacing() -> None:
    vals = torch.tensor([1.0, 1.5, 2.0, 0.75, -3.0])
    ulp = N.bf16_ulp(vals)
    expected = torch.tensor([2.0 ** -7, 2.0 ** -7, 2.0 ** -6, 2.0 ** -8, 2.0 ** -6])
    torch.testing.assert_close(ulp, expected, rtol=0, atol=0)
    # One ulp step really is the next representable bfloat16 value.
    one = torch.tensor(1.0, dtype=torch.bfloat16)
    assert float(one + torch.tensor(2.0 ** -7, dtype=torch.bfloat16)) == 1.0 + 2.0 ** -7


def test_relative_state_divergence_is_per_board() -> None:
    z = torch.ones(2, 4, 8)
    z2 = z.clone()
    z2[1] *= 1.5
    d = N.relative_state_divergence(z2, z)
    assert d[0] == 0.0 and math.isclose(d[1], 0.5, rel_tol=1e-12)


def test_output_perturbation_single_and_every_call() -> None:
    lin = nn.Linear(4, 4, bias=False).double()
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    clean = [lin(x) for _ in range(3)]

    hook = N.OutputPerturbation(1e-3, at_call=1, seed=7)
    h = lin.register_forward_hook(hook)
    outs = [lin(x) for _ in range(3)]
    h.remove()
    assert torch.equal(outs[0], clean[0]) and torch.equal(outs[2], clean[2])
    rel = ((outs[1] - clean[1]).norm(dim=-1) / clean[1].norm(dim=-1))
    torch.testing.assert_close(rel, torch.full_like(rel, 1e-3), rtol=1e-9, atol=0)

    every = N.OutputPerturbation(1e-3, at_call=-1, seed=7)
    h = lin.register_forward_hook(every)
    outs = [lin(x) for _ in range(3)]
    h.remove()
    assert every.calls == 3
    assert all(not torch.equal(o, c) for o, c in zip(outs, clean))
    # Independent draws per invocation.
    assert not torch.allclose(outs[0] - clean[0], outs[1] - clean[1])


def test_exact_fp64_rmsnorm_patch_scoped_and_dtype_preserving() -> None:
    from mechinterp.loopscope02.scripts import trm_sudoku as trm_mod

    original = trm_mod.rms_norm
    x64 = torch.randn(2, 5, 16, dtype=torch.float64) * 3
    xb = torch.randn(2, 5, 16).to(torch.bfloat16)
    ref_b = original(xb)
    with N.exact_fp64_rmsnorm():
        assert trm_mod.rms_norm is not original
        y64 = trm_mod.rms_norm(x64)
        assert y64.dtype == torch.float64
        exact = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + 1e-5)
        torch.testing.assert_close(y64, exact, rtol=0, atol=0)
        assert torch.equal(trm_mod.rms_norm(xb), ref_b)
    assert trm_mod.rms_norm is original


def test_paired_bootstrap_reports_sign_counts() -> None:
    res = N.paired_bootstrap([-1.0, -2.0, -3.0, 0.0], n_boot=500, n_signflip=2000, seed=1)
    assert res["n_negative"] == 3 and res["n_zero"] == 1
    assert res["bootstrap_ci95"][1] <= 0.0
    assert 0.0 < res["signflip_two_sided_p"] <= 1.0


def test_wrapper_identity_output_reproduces_linear() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(16, 8, bias=False)
    model = nn.Sequential(lin)
    bpd = BPDModel(model, {"0": 8}, max_passes=4,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False), init_svd=True)
    w = bpd.wrappers["0"]
    x = torch.randn(2, 3, 16)
    ref = F.linear(x, w.W_target)
    for delta in (0.0, 1.0):
        y = _wrapper_identity_output(w, x, 2, delta)
        torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)
        assert w.mode == "target" and w.schedule is None


def _arm(kl62: list[float]) -> dict:
    def block(vals: list[float]) -> dict:
        return {"kl64": N.summarize(vals), "argmax_flips_per_board": [0] * len(vals)}
    return {"pass_20": block([0.0, 0.0]), "pass_41": block([0.0, 0.0]), "pass_62": block(kl62)}


def test_decide_closure_and_implementation_rules() -> None:
    arm_ok = {"mismatch_frac_mean": 1e-4, "max_eff_ulps": 1.0, "rel_err_vs_fp64_mean": 1.0e-3}
    local = {"m": {"native_bf16_rel_err_vs_fp64_mean": 1.0e-3,
                   "arms": {"fp32_linear": arm_ok, "svd_all_on_delta_off": arm_ok,
                            "svd_all_on_delta_on": arm_ok, "trained_all_on_delta_on": arm_ok}}}
    res = {
        "local_identity_error_bf16": local,
        "bf16_regime": {"fp32_linear_at_4_sites": _arm([0.5, 0.0]),
                        "svd_all_on_delta_off": _arm([0.5, 0.0]),
                        "svd_all_on_delta_on": _arm([0.0, 0.2]),
                        "trained_all_on_delta_on": _arm([0.0, 0.0])},
        "fp32_regimes": {"x": {"svd_all_on_delta_off": _arm([1e-9, 0.0]),
                               "svd_all_on_delta_on": _arm([0.0, 0.0]),
                               "trained_all_on_delta_on": _arm([2e-6, 0.0])}},
    }
    d = decide(res)
    assert d["implementation_mismatch_detected"] is False
    assert d["identity_closure_by_regime"]["native_bf16"]["pass_20"]["closed"] is True
    assert d["identity_closure_by_regime"]["native_bf16"]["pass_62"]["closed"] is False
    assert d["identity_closure_by_regime"]["fp32_x"]["pass_62"]["closed"] is True
    assert d["bf16_identity_divergent_boards_pass62"] == [0, 1]
    bad = copy.deepcopy(res)
    bad["local_identity_error_bf16"]["m"]["arms"]["svd_all_on_delta_on"]["rel_err_vs_fp64_mean"] = 2e-3
    assert decide(bad)["implementation_mismatch_detected"] is True


def test_online_svd_mini_records_floor_flips_and_density() -> None:
    model = build_mini_trm(hidden_size=32, num_heads=4, l_layers=2, h_cycles=1,
                           l_cycles=2, puzzle_emb_len=4, seq_len=16, seed=113).float().eval()
    inputs = torch.randint(1, 11, (2, 16), dtype=torch.long)
    res = audit_one("mini", model, inputs, n_passes=6, c=32, keep_frac=0.6, device="cpu")  # full rank
    floor = res["float64_kl_and_argmax_flips"]["svd_all_on_identity_floor"]
    for tau in res["readouts"]:
        assert floor[f"pass_{tau}"]["kl64"]["max"] < 1e-8
    assert res["component_density"]["static_rank_k"] == res["k_keep"] / 32
    for v in res["component_density"]["clean_oracle_dynamic_topk_measured"].values():
        assert abs(v - res["k_keep"] / 32) < 1e-6
    online = res["float64_kl_and_argmax_flips"]["online_dynamic_topk"]
    for tau in res["readouts"]:
        assert len(online[f"pass_{tau}"]["argmax_flips_per_board"]) == 2
