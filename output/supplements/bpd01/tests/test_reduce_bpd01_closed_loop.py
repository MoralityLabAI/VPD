"""Registered H4 reduction (B4-CAUSAL-CI-02) on the committed B4-CAUSAL-CI-01 receipt."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from mechinterp.bpd.scripts.reduce_bpd01 import closed_loop_fix_effect

RECEIPT = Path(__file__).resolve().parents[1] / "results" / "v2" / "gate_b4_causal_ci.json"


def test_h4_identity_is_zero_and_inconclusive() -> None:
    old = json.loads(RECEIPT.read_text(encoding="utf-8"))
    fx, md = closed_loop_fix_effect(old, old)
    for ck, rows in fx["by_checkpoint"].items():
        assert rows["pass_20"]["mean"] == 0.0
        assert rows["pass_20"]["n_negative"] == rows["pass_20"]["n_positive"] == 0
    assert fx["H4_registered_pass20_decision"] == "inconclusive"
    assert any("H4 registered pass-20 decision" in line for line in md)


def test_h4_detects_uniform_improvement() -> None:
    old = json.loads(RECEIPT.read_text(encoding="utf-8"))
    new = copy.deepcopy(old)
    for ck in new["checkpoints"]:
        for t in (20, 41, 62):
            arm = new["checkpoints"][ck]["arms"]["T3"][f"pass_{t}"]
            arm["kl64"] = [v - 1.0 for v in arm["kl64"]]
    fx, _ = closed_loop_fix_effect(new, old)
    assert fx["H4_registered_pass20_decision"].startswith("closed-loop CI better")
    assert fx["by_checkpoint"]["gaoxin492"]["pass_20"]["n_negative"] == 64
