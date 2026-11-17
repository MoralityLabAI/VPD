"""Causal CI gating regressions on the mini TRM (BPD-01 Experiment C)."""
from __future__ import annotations

import torch

from mechinterp.bpd.adapters.trm import TRMBPDAdapter, build_mini_trm, default_trm_c_per_module
from mechinterp.bpd.causal_ci import (
    causal_ci_rollout,
    clean_informed_ci_masks,
    module_execution_order,
    within_pass_dependency,
)
from mechinterp.bpd.ci import TiedCINetwork
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import AblationConfig, CIConfig

N_PASSES = 6


def _setup(c: int = 32, seed: int = 3):
    torch.manual_seed(seed)
    model = build_mini_trm(hidden_size=32, num_heads=4, l_layers=2, h_cycles=1,
                           l_cycles=2, puzzle_emb_len=4, seq_len=16, seed=113).float().eval()
    c_map = default_trm_c_per_module(model, c=c)
    bpd = BPDModel(model, c_map, max_passes=N_PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False), init_svd=True)
    adapter = TRMBPDAdapter(bpd.target_model, readout_at="outer_step")
    dims = {m: bpd.wrappers[m].d_in for m in bpd.module_order}
    ci = TiedCINetwork(dims, bpd.c_per_module,
                       CIConfig(variant="state_only", d_model=32, n_blocks=1, n_heads=4,
                                mlp_hidden=64, max_passes=16, max_seq_len=32)).eval()
    with torch.no_grad():
        ci.out_proj.weight.normal_(0.0, 0.5)
    inputs = torch.randint(1, 11, (2, 16), dtype=torch.long)
    return bpd, adapter, ci, inputs


def test_module_execution_order_is_layer_then_attention_before_mlp() -> None:
    bpd, adapter, _ci, inputs = _setup()
    order = module_execution_order(bpd, lambda: adapter.rollout(inputs, 1, 0, None))
    assert order == ["layers.0.o_proj", "layers.0.down_proj", "layers.1.o_proj", "layers.1.down_proj"]
    # Sorted module_order used by the CI network differs from execution order.
    assert bpd.module_order != order


def test_state_only_gate_reads_downstream_module_inputs() -> None:
    bpd, adapter, ci, inputs = _setup()
    with bpd.target_capture(pass_offset=0):
        adapter.rollout(inputs, 1, 0, None)
    x = bpd.collect_recorded_inputs(expected_passes=1)
    dep = within_pass_dependency(ci, x)
    # First-executed module's gate depends on the last-executed module's input.
    assert dep["layers.0.o_proj"]["layers.1.down_proj"] > 0.0


def test_all_open_causal_gates_reproduce_target() -> None:
    bpd, adapter, ci, inputs = _setup()
    with torch.no_grad():
        ci.out_proj.weight.zero_()
        ci.out_proj.bias.fill_(5.0)  # lower-leaky clamps to exactly 1
    ref, _, _, _ = adapter.rollout(inputs, N_PASSES, 0, None)
    res = causal_ci_rollout(bpd, adapter, ci, inputs, N_PASSES,
                            observation="preflight_target", selection="continuous")
    assert sorted(res["readouts"]) == sorted(ref)
    for tau in ref:
        torch.testing.assert_close(res["readouts"][tau], ref[tau], rtol=1e-5, atol=1e-5)


def test_no_lookahead_truncated_rollout_matches_prefix() -> None:
    bpd, adapter, ci, inputs = _setup()
    long = causal_ci_rollout(bpd, adapter, ci, inputs, N_PASSES, selection="topk", k=19)
    short = causal_ci_rollout(bpd, adapter, ci, inputs, 4, selection="topk", k=19)
    for tau in range(4):
        assert torch.equal(long["boundary_states"][tau], short["boundary_states"][tau])
    assert torch.equal(long["readouts"][2], short["readouts"][2])


def test_reference_masks_are_telemetry_only_and_tau0_matches_clean() -> None:
    bpd, adapter, ci, inputs = _setup()
    oracle = clean_informed_ci_masks(bpd, adapter, ci, inputs, N_PASSES, selection="topk", k=19)
    rand = {m: (torch.rand_like(v) > 0.5).float() for m, v in oracle.items()}
    a = causal_ci_rollout(bpd, adapter, ci, inputs, N_PASSES, selection="topk", k=19,
                          reference_masks=oracle)
    b = causal_ci_rollout(bpd, adapter, ci, inputs, N_PASSES, selection="topk", k=19,
                          reference_masks=rand)
    for tau in a["readouts"]:
        assert torch.equal(a["readouts"][tau], b["readouts"][tau])
    # Invocation 0 sees the clean state, so preflight gates equal the clean-informed gates.
    for m in bpd.module_order:
        assert a["disagreement_by_module_by_pass_by_board"][m][0] == [0.0, 0.0]
        assert all(abs(d - 19 / 32) < 1e-6 for d in a["density_by_module_by_pass"][m])


def test_lagged_observation_runs_and_differs_from_preflight() -> None:
    bpd, adapter, ci, inputs = _setup()
    pre = causal_ci_rollout(bpd, adapter, ci, inputs, N_PASSES, selection="topk", k=19)
    lag = causal_ci_rollout(bpd, adapter, ci, inputs, N_PASSES, observation="lagged",
                            selection="topk", k=19)
    assert torch.equal(pre["boundary_states"][0], lag["boundary_states"][0])
    assert any(not torch.equal(pre["readouts"][t], lag["readouts"][t]) for t in pre["readouts"])


def test_replaying_clean_informed_masks_reproduces_open_loop_rollout() -> None:
    """The causal loop (preflight, CI, telemetry) with T2's masks replayed equals the T2 rollout."""
    from mechinterp.bpd.components import PassMaskSchedule

    bpd, adapter, ci, inputs = _setup(c=16)  # c < full rank so dropped components change the state
    k = 9
    t2 = clean_informed_ci_masks(bpd, adapter, ci, inputs, N_PASSES, selection="topk", k=k)
    sched = {m: PassMaskSchedule(comp_mask=v, delta_mask=torch.zeros(inputs.shape[0], N_PASSES, 1))
             for m, v in t2.items()}
    with torch.no_grad(), bpd.masked_components(sched, pass_offset=0):
        open_loop, _, _, _ = adapter.rollout(inputs, N_PASSES, 0, None)
    replay = causal_ci_rollout(bpd, adapter, ci, inputs, N_PASSES, observation="preflight_target",
                               selection="topk", k=k, reference_masks=t2, replay_masks=t2)
    for t, v in open_loop.items():
        assert torch.equal(replay["readouts"][t], v), t
    # Telemetry still reports the CI's own causal gates, which differ from T2 after tau = 0.
    causal = causal_ci_rollout(bpd, adapter, ci, inputs, N_PASSES, observation="preflight_target",
                               selection="topk", k=k, reference_masks=t2)
    assert all(v == 0.0 for rows in causal["disagreement_by_module_by_pass_by_board"].values() for v in rows[0])
