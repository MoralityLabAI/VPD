"""Fast, checkpoint-free CPU regression tests for local causal fiber gyms."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from mechinterp.dark_fiber_gyms import arg03_roster as R
from mechinterp.dark_fiber_gyms import core as C
from mechinterp.dark_fiber_gyms import loopscope04_trm as TR
from mechinterp.dark_fiber_gyms import sudoku_roster as SU
from mechinterp.dark_fiber_gyms import observability as OBS


def test_sphere_swap_preserves_all_head_logits_and_radius():
    torch.manual_seed(4)
    w = torch.randn(3,8)
    src = torch.randn(2,19,8)
    donor = torch.randn_like(src)
    out, cert = C.trm_manifold_swap(src,donor,w,board_start=2,atol=1e-4)
    assert cert["orthogonal_dim"] == 5
    assert torch.equal(out[:,:2],src[:,:2])
    assert (out[:,2:]@w.T-src[:,2:]@w.T).abs().max() < 1e-4
    assert torch.allclose(out[:,2:].norm(dim=-1),src[:,2:].norm(dim=-1),atol=1e-5)
    assert (out[:,2:]-src[:,2:]).norm().item()>1e-4


def test_float16_swap_refused():
    x=torch.randn(1,5,8,dtype=torch.float16)
    with pytest.raises(ValueError,match="float32"):
        C.trm_manifold_swap(x,x,torch.randn(3,8),board_start=1)


def test_nanbeige_context_preserves_last_three_exactly():
    a=torch.randn(16,18,6)
    b=torch.randn(16,18,6)
    out,meta=C.nanbeige_context_swap(a,b,width=6)
    assert meta["positions"] == list(range(9,15))
    assert torch.equal(a[:,-3:],out[:,-3:])
    assert torch.equal(out[:,9:15],b[:,9:15])
    assert torch.equal(out[:,:9],a[:,:9])


class ToyLoop:
    n=1
    def prepare(self,ids):
        ids=ids.float()
        emb=torch.stack([ids,ids.square()/100,ids.sin(),ids.cos()],dim=-1)
        return emb,{}
    def run(self,h,lo,hi,ctx):
        return h+0.17*h.roll(1,dims=1)
    def readout(self,h,ctx):
        w=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
        return torch.log_softmax(h[:,-3:,:]@w.T,dim=-1)


class ToyEngine:
    op_ids=dict(zip("INSR",range(4)))
    loop=ToyLoop()
    def rows(self,row):
        ids=torch.stack([torch.tensor(list(range(10))+[i,j],dtype=torch.long)
                         for i in range(4) for j in range(4)])
        return ids,10


def test_nanbeige_mock_full_readout_equality_and_later_causality():
    r={"case_id":"toy","public_sha256":"0"*64}
    out=NB.evaluate_case(ToyEngine(),r,sigma=0.1,width=5,seed=27,depths=(1,2,4))
    assert out["max_precontinuation_logprob_error"]==0.0
    assert out["baseline"]["1"]==out["context_swap"]["1"]
    assert out["postloop_readout_distance"]["2"]>0.0
    assert len(out["baseline"]["4"])==64
    assert set(out["temperature_controls"]) == {"T0.5","T0.75","T1.25","T2"}
    assert "4" in out["open_loop"]


def test_program_score_layout():
    logp=torch.log_softmax(torch.randn(16,3,10),dim=-1)
    values=C.arg02_program_scores(logp,dict(zip("INSR",range(4))))
    assert len(values)==64
    assert abs(values[0]-float(logp[0,0,0]+logp[0,1,0]+logp[0,2,0]))<1e-6


def test_ridge_diagnostic_and_mcnemar():
    rng=np.random.default_rng(4)
    x=rng.standard_normal((80,4))
    w=rng.standard_normal((5,7))
    y=np.column_stack([np.ones(80),x])@w
    fitted=TR.ridge_fit(x,y,alpha=1e-5)
    assert np.abs(TR.ridge_predict(x,fitted)-y).max()<1e-4
    res=R.exact_mcnemar([True,True,True],[False,False,False])
    assert res["gain"]==3 and res["loss"]==0
    assert res["exact_two_sided_p"]==0.25


def test_sudoku_generator_and_uniqueness():
    import random
    rng=random.Random(9)
    solved=SU.generate_solution(rng)
    assert len(solved)==81 and SU.count_solutions(solved)==1
    puzzle=SU.delete_clues(solved,78,rng)
    assert puzzle is not None and puzzle.count(".")==3
    assert SU.count_solutions(puzzle)==1
    relabel="".join("2" if c=="1" else "1" if c=="2" else c for c in puzzle)
    assert TR.canonical_digit_relabel(puzzle)==TR.canonical_digit_relabel(relabel)


def _write_boards(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=("question","answer"))
        w.writeheader()
        for row in rows:w.writerow(row)


def test_dataset_overlap_rejected(tmp_path):
    old=tmp_path/"old.csv"
    q="."*81
    a=SU.generate_solution(__import__("random").Random(5))
    _write_boards(old,[{"question":q,"answer":a}])
    test=[{"question":q,"answer":a,"id":"test"}]
    with pytest.raises(ValueError,match="overlaps excluded"):
        TR.audit_boards([],test,[old])


def test_nanbeige_dry_run_rejects_hidden_fields(tmp_path,capsys):
    public=tmp_path/"public.jsonl"
    base={"case_id":"toy","public_sha256":"a"*64,"topology":"fork","observed_degree":12,
          "query":{},"demos":[{},{}]}
    public.write_text(json.dumps(base)+"\n")
    assert NB.main(["--public-jsonl",str(public),"--compactification-root",str(tmp_path),
                    "--out-dir",str(tmp_path/"out"),"--dry-run"])==0
    base["future_secret"]="blocked"
    public.write_text(json.dumps(base)+"\n")
    with pytest.raises(ValueError,match="hidden"):
        NB.main(["--public-jsonl",str(public),"--compactification-root",str(tmp_path),
                 "--out-dir",str(tmp_path/"out"),"--dry-run"])


class TinyRecurrentMap:
    def __init__(self):
        self.lm_head=torch.nn.Linear(6,2,bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[0,0]=1.0
            self.lm_head.weight[1,1]=1.0

    def act_step(self, zh,zl,x):
        y=zh.clone()
        y[:,:,0]=zh[:,:,0]+0.2*zh[:,:,2]
        y[:,:,1]=zh[:,:,1]+0.1*zh[:,:,3]
        return y,zl


def test_sampled_gramian_local_tangents_exact_and_delayed():
    model=TinyRecurrentMap()
    torch.manual_seed(5)
    zh=torch.randn(1,7,6)
    zl=torch.zeros_like(zh)
    x=torch.zeros_like(zh)
    pairs,certificate=OBS.sphere_tangent_pairs(
        zh,model.lm_head.weight,board_start=2,probes=5,angle=0.03)
    assert pairs.shape==(10,7,6)
    assert certificate["max_readout_error"] < 1e-5
    assert certificate["max_l2_error"] < 1e-5
    result=OBS.sampled_trm_gramian(
        model,zh,zl,x,board_start=2,probes=5,angle=0.03,horizon=3)
    assert result["sampled_rank_relative_1e-3"] >= 1
    assert result["eigenvalues_descending"][0]>0
    assert len(result["per_lag_mean_response"])==3


def test_sudoku_full_automorphism_isomorphism_and_json_exclusion(tmp_path):
    import random
    rng = random.Random(20261115)
    solved = SU.generate_diverse_solution(rng)
    assert len(solved) == 81 and SU.count_solutions(solved) == 1
    puzzle = SU.delete_clues(solved, 46, rng)
    assert puzzle is not None and SU.count_solutions(puzzle) == 1

    # Apply transpose + band swap (band 0 <-> band 2) + row perm + stack swap + digit relabel
    t_grid = [puzzle[c * 9 + r] for r in range(9) for c in range(9)]
    row_perm = [7, 6, 8, 4, 3, 5, 1, 2, 0]
    col_perm = [3, 5, 4, 8, 6, 7, 2, 0, 1]
    dig_map = dict(zip("123456789", "947281635"))
    iso_chars = []
    for r in range(9):
        for c in range(9):
            ch = t_grid[row_perm[r] * 9 + col_perm[c]]
            iso_chars.append("." if ch == "." else dig_map[ch])
    iso_puzzle = "".join(iso_chars)

    assert TR.canonical_digit_relabel(puzzle) != TR.canonical_digit_relabel(iso_puzzle)
    assert TR.sudoku_wl_signature(puzzle) == TR.sudoku_wl_signature(iso_puzzle)
    assert TR.are_sudoku_isomorphic(puzzle, iso_puzzle)

    # Non-isomorphic puzzle with identical clue count must not match
    other_sol = SU.generate_diverse_solution(random.Random(99991))
    other_puz = SU.delete_clues(other_sol, 46, random.Random(99992))
    assert other_puz is not None
    assert not TR.are_sudoku_isomorphic(puzzle, other_puz)

    # JSON exclusion file containing original puzzle must block isomorphic copy in audit_boards
    excl_json = tmp_path / "prior_items.json"
    excl_json.write_text(json.dumps({"items": [{"question": puzzle, "answer": solved}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Sudoku automorphism"):
        TR.audit_boards([], [{"question": iso_puzzle, "answer": solved, "id": "iso"}], [excl_json])


def test_nanbeige_norm_matched_random_context_and_ranking():
    torch.manual_seed(19)
    clean = torch.randn(16, 18, 6)
    donor = torch.randn(16, 18, 6)
    patched, _ = C.nanbeige_context_swap(clean, donor, width=6, readout_tail=3)
    rand_p, rmeta = C.nanbeige_norm_matched_random_context_swap(clean, patched, width=6, readout_tail=3, seed=7)
    assert torch.equal(rand_p[:, -3:, :], clean[:, -3:, :])
    assert torch.equal(rand_p[:, :9, :], clean[:, :9, :])
    assert abs(rmeta["patch_l2"] - rmeta["reference_patch_l2"]) < 1e-4
    d_donor = (patched[:, 9:15, :] - clean[:, 9:15, :]).norm(dim=-1)
    d_rand = (rand_p[:, 9:15, :] - clean[:, 9:15, :]).norm(dim=-1)
    assert torch.allclose(d_donor, d_rand, atol=1e-5)

    comp = C.ranking_comparison(list(range(64)), list(reversed(range(64))))
    assert comp["top1_changed"] is True
    assert comp["top12_overlap"] == 0
    assert comp["spearman_rho"] < -0.99


def test_detect_board_limit_cycle():
    a = [1] * 81
    b = [2] * 81
    c_fixed = C.detect_board_limit_cycle([a, b, b, b, b])
    assert c_fixed["terminal_period"] == 1 and c_fixed["oscillatory_cycle"] is False
    c_osc = C.detect_board_limit_cycle([a, b, a, b, a, b])
    assert c_osc["terminal_period"] == 2 and c_osc["oscillatory_cycle"] is True


def test_causal_gramian_validation_high_beats_low():
    class AsymmetricDelayedMap:
        def __init__(self):
            self.lm_head = torch.nn.Linear(8, 2, bias=False)
            with torch.no_grad():
                self.lm_head.weight.zero_()
                self.lm_head.weight[0, 0] = 1.0
                self.lm_head.weight[1, 1] = 1.0

        def act_step(self, zh, zl, x):
            y = zh.clone()
            # Strong delayed readout coupling along hidden coordinate 2, zero along coordinates 4..7
            y[:, :, 0] = zh[:, :, 0] + 1.5 * zh[:, :, 2]
            y[:, :, 1] = zh[:, :, 1] + 0.02 * zh[:, :, 3]
            return y, zl

    model = AsymmetricDelayedMap()
    torch.manual_seed(12)
    zh = torch.randn(1, 6, 8)
    zl = torch.zeros_like(zh)
    x = torch.zeros_like(zh)
    res = OBS.causal_gramian_validation(
        model, zh, zl, x, board_start=2, probes=4, fd_angle=0.03,
        gramian_horizon=2, eval_horizon=3, test_angles=(0.05, 0.25), seed=41
    )
    assert res["eigenvalues_descending"][0] > res["eigenvalues_descending"][-1]
    for k in ("angle_0.05", "angle_0.25"):
        b_arm = res["causal_direction_tests"][k]["directions"]
        assert abs(b_arm["high_eig"]["initial_perturbation_l2"] - b_arm["low_eig"]["initial_perturbation_l2"]) < 1e-4
        assert abs(b_arm["high_eig"]["initial_perturbation_l2"] - b_arm["random_dir"]["initial_perturbation_l2"]) < 1e-4
        assert b_arm["high_eig"]["cum_h2_rms_readout_l2"] > b_arm["low_eig"]["cum_h2_rms_readout_l2"]


def test_oracle_isolation_and_test_calibration_guard(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "spiral_ln.algebraic_reasoning_oracle", object())
    with pytest.raises(RuntimeError, match="oracle isolation"):
        NB.assert_oracle_isolated()
    monkeypatch.delitem(sys.modules, "spiral_ln.algebraic_reasoning_oracle", raising=False)
    NB.assert_oracle_isolated()

    with pytest.raises(RuntimeError, match="development"):
        R.main([
            "grade",
            "--evalmath-root", str(tmp_path),
            "--compactification-root", str(tmp_path),
            "--split-dir", str(tmp_path),
            "--split", "test",
            "--tape", str(tmp_path / "none.jsonl"),
            "--out", str(tmp_path / "out.json"),
            "--fit-dev-calibration", str(tmp_path / "cal.json"),
        ])


def test_causal_gramian_batch_matches_single():
    model = TinyRecurrentMap()
    torch.manual_seed(77)
    zh = torch.randn(2, 7, 6)
    zl = torch.zeros_like(zh)
    x = torch.zeros_like(zh)
    seeds = [101, 202]
    b_res = OBS.causal_gramian_validation_batch(
        model, zh, zl, x, board_start=2, probes=3, fd_angle=0.03,
        gramian_horizon=2, eval_horizon=3, test_angles=(0.05, 0.25), seeds=seeds
    )
    for i in range(2):
        s_res = OBS.causal_gramian_validation(
            model, zh[i:i+1], zl[i:i+1], x[i:i+1], board_start=2, probes=3, fd_angle=0.03,
            gramian_horizon=2, eval_horizon=3, test_angles=(0.05, 0.25), seed=seeds[i]
        )
        assert np.allclose(b_res[i]["eigenvalues_descending"], s_res["eigenvalues_descending"], atol=1e-4)
        for ak in ("angle_0.05", "angle_0.25"):
            for dn in ("high_eig", "low_eig", "random_dir"):
                assert abs(
                    b_res[i]["causal_direction_tests"][ak]["directions"][dn]["cum_h2_rms_readout_l2"]
                    - s_res["causal_direction_tests"][ak]["directions"][dn]["cum_h2_rms_readout_l2"]
                ) < 1e-4


def test_campaign_summary_statistical_primitives(tmp_path):
    from mechinterp.dark_fiber_gyms import campaign_summary as CS

    x = [10.0 + i for i in range(12)]
    y = [1.0 + i for i in range(12)]
    st = CS.exact_sign_test(x, y)
    assert st["positive"] == 12 and st["negative"] == 0
    assert abs(st["exact_sign_p_two_sided"] - 2.0 / (2**12)) < 1e-12

    wk = CS.wilcoxon_signed_rank_normal(x, y)
    assert wk["n_nonzero"] == 12 and wk["p_two_sided"] < 0.005

    ci = CS.paired_bootstrap_ci(x, y, n_boot=500, seed=7)
    assert abs(ci["estimate"] - 9.0) < 1e-12
    assert abs(ci["ci95_low"] - 9.0) < 1e-12

    sp = CS.spearman_corr([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0])
    assert abs(sp["rho"] - 1.0) < 1e-12
    sp_ties = CS.spearman_corr([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 20.0, 40.0], n_boot=200)
    assert abs(sp_ties["rho"] - 1.0) < 1e-12

    # Verify _summarize_arg03_tape matches evaluate_case schema end-to-end
    rec = NB.evaluate_case(ToyEngine(), {"case_id": "toy1", "public_sha256": "0" * 64}, sigma=0.1, width=5, seed=27)
    tape_p = tmp_path / "toy_tape.jsonl"
    tape_p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    ts = CS._summarize_arg03_tape(tape_p)
    assert ts["n_rows"] == 1
    assert ts["max_precontinuation_donor_logprob_error"] == 0.0
    assert "spearman_rho_mean_ci" in ts["ranking_changes_by_depth"]["2"]["context_swap"]

    # Verify manifest path sanitization strips local filesystem roots while preserving digests
    dirty = {
        "source_csv": r"C:\Users\alice\.cache\hf\test_hard.csv",
        "source_runner": r"C:\Users\alice\Documents\algernon-arg02",
        "excluded_sha256": {
            r"C:\Users\alice\secret_dir\sudoku_items256.json": "abc123",
            "/home/alice/data/eval.csv": "def456",
        },
    }
    clean = CS._sanitize_manifest_paths(dirty)
    assert clean["source_csv"] == "test_hard.csv"
    assert clean["source_runner"] == "algernon-arg02"
    assert clean["excluded_sha256"] == {"sudoku_items256.json": "abc123", "eval.csv": "def456"}


def test_streamed_nanbeige_loop_bit_exact():
    """Verify StreamedNanbeigeLoop with layer staging produces bit-exact outputs vs full resident."""
    import copy

    class _MockLayer(torch.nn.Module):
        def __init__(self, d_model: int, layer_idx: int):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.layer_idx = layer_idx
            self.proj = torch.nn.Linear(d_model, d_model, bias=False)

        def forward(self, h, attention_mask=None, position_ids=None, past_key_value=None,
                    output_attentions=False, use_cache=False, cache_position=None, loop_idx=0):
            # Causal cumulative mix + linear projection so context affects later steps
            c = torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
            return (h + 0.25 * torch.tanh(self.proj(c)),)

    class _MockBase(torch.nn.Module):
        def __init__(self, vocab: int = 32, d_model: int = 12, n_layers: int = 3):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(vocab, d_model)
            self.layers = torch.nn.ModuleList([_MockLayer(d_model, i) for i in range(n_layers)])
            self.norm = torch.nn.LayerNorm(d_model)

        def _update_causal_mask(self, *args, **kwargs):
            return None

    class _MockCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"num_hidden_layers": 3})()
            self.model = _MockBase()
            self.lm_head = torch.nn.Linear(12, 32, bias=False)

    torch.manual_seed(19)
    full_m = _MockCausalLM().eval()
    split_m = copy.deepcopy(full_m).eval()
    stage = _MockLayer(12, 0).eval()

    loop_full = NB.StreamedNanbeigeLoop(full_m, torch.device("cpu"), n_gpu_layers=3, stage_layer=None)
    loop_split = NB.StreamedNanbeigeLoop(split_m, torch.device("cpu"), n_gpu_layers=1, stage_layer=stage)

    class _Eng:
        def __init__(self, lp):
            self.loop = lp
            self.op_ids = {"I": 3, "N": 4, "S": 5, "R": 6}

        def rows(self, row, *, variant="canonical"):
            shift = {"canonical": 0, "swap": 1, "space": 2, "swap_space": 3}[variant]
            base = [1, 2 + shift, 7, 8, 9, 10, 11, 12, 13, 14]
            grid = [base + [a, b] for a in (3, 4, 5, 6) for b in (3, 4, 5, 6)]
            return torch.tensor(grid, dtype=torch.long), len(base)

    row = {"case_id": "s1", "public_sha256": "a" * 64}
    r_full = NB.evaluate_case(_Eng(loop_full), row, sigma=0.05, width=4, seed=99)
    r_split = NB.evaluate_case(_Eng(loop_split), row, sigma=0.05, width=4, seed=99)

    for d_key in ("1", "2", "4", "8"):
        assert r_full["baseline"][d_key] == r_split["baseline"][d_key]
        assert r_full["context_swap"][d_key] == r_split["context_swap"][d_key]
        assert r_full["random_context_norm_matched"][d_key] == r_split["random_context_norm_matched"][d_key]
        assert r_full["equal_compute"][d_key] == r_split["equal_compute"][d_key]





