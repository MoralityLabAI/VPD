"""Export live Stage 1a-4 variables and (V, U) factors from a running/suspended run_trm_bpd.py process via stock debugpy.

Provenance utility used to extract `gate_b4_v2_stage4_receipt.json` and
`bpd_b4_v2_stage4_weights.pt` from `PID 275016` at the Stage 4 -> Stage 5a boundary.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from debugpy._vendored.pydevd.pydevd_attach_to_process import add_code_to_python_process

sys.path.insert(0, os.path.dirname(add_code_to_python_process.__file__))
run_python_code = add_code_to_python_process.run_python_code

PROCESS_ALL_ACCESS = 0x1F0FFF


def suspend_pid(pid: int) -> int:
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not h:
        raise RuntimeError(f"OpenProcess failed for {pid}")
    try:
        return int(ctypes.windll.ntdll.NtSuspendProcess(h))
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def resume_pid(pid: int) -> int:
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not h:
        raise RuntimeError(f"OpenProcess failed for {pid}")
    try:
        rc = 0
        for _ in range(4):
            rc = int(ctypes.windll.ntdll.NtResumeProcess(h))
        return rc
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def write_exporter_script(
    exporter_py: Path, out_json: Path, out_status: Path, out_weights: Path | None
) -> str:
    j_str = out_json.as_posix()
    s_str = out_status.as_posix()
    w_str = out_weights.as_posix() if out_weights is not None else ""
    script_body = f'''import sys, json, traceback
status_path = "{s_str}"
json_path = "{j_str}"
weights_path = "{w_str}"
try:
    target_locals = None
    for tid, frame in sys._current_frames().items():
        f = frame
        while f is not None:
            if f.f_code.co_name == "run_trm_bpd_pipeline":
                target_locals = dict(f.f_locals)
                break
            f = f.f_back
        if target_locals is not None:
            break

    if target_locals is None:
        with open(status_path, "w", encoding="utf-8") as fp:
            fp.write("ERR: run_trm_bpd_pipeline frame not found")
    else:
        def _clean(obj):
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, dict):
                return {{str(k): _clean(v) for k, v in obj.items()}}
            if isinstance(obj, (list, tuple)):
                return [_clean(x) for x in obj]
            if hasattr(obj, "item") and callable(obj.item):
                try:
                    return obj.item()
                except Exception:
                    pass
            return str(obj)

        pgd_sh = target_locals.get("pgd_shared")
        pgd_pp = target_locals.get("pgd_per_pass")
        sched_rep = target_locals.get("sched_rep")
        ap_ref = target_locals.get("active_pool_ref") or {{}}
        ap_tgt = target_locals.get("active_pool_tgt") or {{}}
        train_in = target_locals.get("train_inputs")

        out = {{
            "gate": "B4_v2_stages_1a_through_4_receipt",
            "status": "completed_stages_1a_1b_2_3a_3b_4_stopped_after_stage_4",
            "gate_b4_verdict": "incomplete",
            "gate_b4_passed": None,
            "registration_commit": "180f1bcbbc633a16bfd887a9593252cefd7a6a65",
            "c_per_module": _clean(target_locals.get("c_per_mod")),
            "num_train_boards": int(train_in.shape[0]) if train_in is not None and hasattr(train_in, "shape") else None,
            "train_batch_size": _clean(target_locals.get("eff_bsz")),
            "total_invocations_trained": _clean(target_locals.get("total_invocations")),
            "hinge_kappa": _clean(target_locals.get("hinge_kappa")),
            "rollout_mode_counts": _clean(target_locals.get("mode_counts")),
            "trajectory_state_matching_modes": {{
                "outer_readout_only_default": {{
                    "match_inner_boundary_states": False,
                    "eval_metrics": _clean(target_locals.get("ref_eval_metrics")),
                    "ood_eval_metrics": _clean(target_locals.get("ref_ood_metrics")),
                    "pass_null_losses": _clean(target_locals.get("ref_null_losses")),
                    "last_train_pass_loss_shares": _clean(target_locals.get("ref_last_shares")),
                }},
                "with_inner_boundary_states": {{
                    "match_inner_boundary_states": True,
                    "rollout_mode_counts": _clean(target_locals.get("inner_mode_counts")),
                    "eval_metrics": _clean(target_locals.get("ref_inner_eval_metrics")),
                    "ood_eval_metrics": _clean(target_locals.get("ref_inner_ood_metrics")),
                    "pass_null_losses": _clean(target_locals.get("inner_null_losses")),
                    "last_train_pass_loss_shares": _clean(target_locals.get("inner_last_shares")),
                }},
            }},
            "target_decomposition_summary": {{
                "eval_metrics": _clean(target_locals.get("tgt_eval_metrics")),
                "ood_eval_metrics": _clean(target_locals.get("tgt_ood_metrics")),
                "pass_null_losses": _clean(target_locals.get("tgt_null_losses")),
                "last_train_pass_loss_shares": _clean(target_locals.get("tgt_last_shares")),
            }},
            "stationary_fraction": float(sched_rep.is_stationary.float().mean().item()) if sched_rep is not None and hasattr(sched_rep, "is_stationary") else None,
            "active_pool_sizes_ref": {{str(m): len(v) for m, v in ap_ref.items()}},
            "active_pool_sizes_tgt": {{str(m): len(v) for m, v in ap_tgt.items()}},
            "pgd_stability": {{
                "shared_batch_and_pass": _clean(getattr(pgd_sh, "loss_by_steps", None)),
                "shared_amplification": _clean(getattr(pgd_sh, "amplification_ratio", None)),
                "per_pass_resonant": _clean(getattr(pgd_pp, "loss_by_steps", None)),
                "per_pass_amplification": _clean(getattr(pgd_pp, "amplification_ratio", None)),
            }},
            "switch_penalty_sweep": _clean(target_locals.get("switch_sweep_results")),
        }}

        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(out, fp, indent=2)
            fp.write("\\n")

        if weights_path:
            import torch
            bpd_ref = target_locals.get("bpd_ref")
            bpd_tgt = target_locals.get("bpd_tgt")
            ckpt = {{}}
            if bpd_ref is not None and hasattr(bpd_ref, "wrappers"):
                ckpt["bpd_ref"] = {{
                    m: {{"V": w.V.detach().cpu(), "U": w.U.detach().cpu()}}
                    for m, w in bpd_ref.wrappers.items()
                }}
            if bpd_tgt is not None and hasattr(bpd_tgt, "wrappers"):
                ckpt["bpd_tgt"] = {{
                    m: {{"V": w.V.detach().cpu(), "U": w.U.detach().cpu()}}
                    for m, w in bpd_tgt.wrappers.items()
                }}
            if ckpt:
                torch.save(ckpt, weights_path)

        with open(status_path, "w", encoding="utf-8") as fp:
            fp.write("OK")
except Exception:
    with open(status_path, "w", encoding="utf-8") as fp:
        fp.write("EXC:\\n" + traceback.format_exc())
'''
    exporter_py.write_text(script_body, encoding="utf-8")
    path_bytes = list(exporter_py.as_posix().encode("utf-8"))
    cmd = f"import codecs;p=codecs.utf_8_decode(bytearray({path_bytes}))[0];exec(open(p).read())"
    assert not ({'"', "'", "\r", "\n"} & set(cmd))
    return cmd


def run_dummy_test(scratch_dir: Path) -> None:
    pid_file = scratch_dir / "_dummy_real_pid.txt"
    pid_file.unlink(missing_ok=True)
    dummy_script = scratch_dir / "_dummy_proc.py"
    dummy_script.write_text(
        f"import os, time\n"
        f"open({pid_file.as_posix()!r}, 'w').write(str(os.getpid()))\n"
        "class _P:\n"
        "    loss_by_steps = {'0': 0.5, '160': 12.3}\n"
        "    amplification_ratio = 24.6\n"
        "def run_trm_bpd_pipeline():\n"
        "    c_per_mod = 512\n"
        "    eff_bsz = 32\n"
        "    total_invocations = 63\n"
        "    hinge_kappa = 0.1\n"
        "    mode_counts = {'clean_window': 120, 'closed_loop': 120}\n"
        "    ref_eval_metrics = {'recon_loss': 0.53, 'final_kl': 0.11}\n"
        "    ref_ood_metrics = {'recon_loss': 0.88, 'final_kl': 0.19}\n"
        "    ref_null_losses = {0: 1e-4}\n"
        "    ref_last_shares = {62: 0.4}\n"
        "    inner_mode_counts = {'closed_loop': 120}\n"
        "    ref_inner_eval_metrics = {'recon_loss': 0.059}\n"
        "    ref_inner_ood_metrics = {'recon_loss': 0.09}\n"
        "    inner_null_losses = {0: 1e-4}\n"
        "    inner_last_shares = {62: 0.3}\n"
        "    tgt_eval_metrics = {'recon_loss': 0.13, 'final_kl': 0.08}\n"
        "    tgt_ood_metrics = {'recon_loss': 0.21, 'final_kl': 0.14}\n"
        "    tgt_null_losses = {0: 1e-4}\n"
        "    tgt_last_shares = {62: 0.35}\n"
        "    pgd_shared = _P()\n"
        "    pgd_per_pass = _P()\n"
        "    switch_sweep_results = [{'coeff_switch': 0.0, 'recon_loss': 0.51}]\n"
        "    while True:\n"
        "        time.sleep(0.05)\n"
        "run_trm_bpd_pipeline()\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(dummy_script)])
    try:
        for _ in range(30):
            if pid_file.exists():
                break
            time.sleep(0.1)
        real_pid = int(pid_file.read_text().strip())
        suspend_pid(real_pid)
        time.sleep(0.2)
        resume_pid(real_pid)
        time.sleep(0.2)
        out_json = scratch_dir / "dummy_out.json"
        out_status = scratch_dir / "dummy_status.txt"
        exporter_py = scratch_dir / "_dummy_exporter.py"
        out_json.unlink(missing_ok=True)
        out_status.unlink(missing_ok=True)
        cmd = write_exporter_script(exporter_py, out_json, out_status, None)
        run_python_code(real_pid, cmd, connect_debugger_tracing=False, show_debug_info=0)
        for _ in range(30):
            if out_status.exists():
                break
            time.sleep(0.2)
        print("DUMMY_STATUS:", out_status.read_text(encoding="utf-8") if out_status.exists() else "MISSING")
        assert out_json.exists(), "dummy_out.json was not written"
        parsed = json.loads(out_json.read_text(encoding="utf-8"))
        assert parsed["switch_penalty_sweep"][0]["recon_loss"] == 0.51
        assert parsed["pgd_stability"]["shared_amplification"] == 24.6
        print("DUMMY_TEST_OK")
    finally:
        proc.kill()


def extract_live_pid(
    scratch_dir: Path, pid: int, out_json: Path, out_status: Path, out_weights: Path
) -> None:
    out_status.unlink(missing_ok=True)
    exporter_py = scratch_dir / "_live_exporter.py"
    cmd = write_exporter_script(exporter_py, out_json, out_status, out_weights)
    print(f"Resuming PID {pid} briefly for debugpy export...")
    resume_pid(pid)
    time.sleep(1.0)
    run_python_code(pid, cmd, connect_debugger_tracing=False, show_debug_info=0)
    for _ in range(300):
        if out_status.exists():
            break
        time.sleep(0.2)
    if not out_status.exists():
        suspend_pid(pid)
        raise RuntimeError("Timed out waiting for status file; re-suspended PID")
    st = out_status.read_text(encoding="utf-8")
    print("LIVE_STATUS:", st)
    if st != "OK":
        suspend_pid(pid)
        raise RuntimeError(f"Live export failed: {st}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dummy-test", action="store_true")
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-weights", type=Path, default=None)
    ap.add_argument("--scratch-dir", type=Path, default=None)
    args = ap.parse_args()

    scratch = args.scratch_dir or Path(tempfile.mkdtemp(prefix="bpd_stage4_extract_"))
    scratch.mkdir(parents=True, exist_ok=True)

    if args.dummy_test:
        run_dummy_test(scratch)
        return

    assert args.pid is not None and args.out_json is not None
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    weights_out = args.out_weights or (args.out_json.parent / "bpd_b4_v2_stage4_weights.pt")
    extract_live_pid(
        scratch,
        args.pid,
        args.out_json,
        scratch / "live_extract_status.txt",
        weights_out,
    )


if __name__ == "__main__":
    main()
