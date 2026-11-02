"""Bounded, filename-only recovery search; never reads credentials or executes recovery code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

ROOTS = [
    "E:/Recovered_C_projects/Spiral_target/projects/SmallControlHarness",
    "E:/Recovered_C_projects/Spiral_target/projects/HRM-re",
    "E:/Recovered_C_projects/Spiral_target/projects/trm_observability_harness",
    "E:/Recovered_C_projects/Git_reconstructed/HRM-re",
    "E:/Recovered_C_projects/Git_reconstructed_loose/HRM-re_union",
]
PATTERNS = ["captures.jsonl", "calibration_tokens.jsonl", "sae_quality.json", "manifest.json",
            "*capture*manifest*.json", "*activation*manifest*.json", "baseline_*.npz", "activation_*.npz"]
EXCLUSIONS = ["!**/.git/**", "!**/.venv/**", "!**/node_modules/**", "!**/__pycache__/**"]
SECONDS_PER_ROOT = 25


def decoded(output):
    return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output or ""


def search_root(root, *, runner=subprocess.run):
    command = ["rg", "--files", "--hidden", "--no-ignore"]
    for pattern in PATTERNS+EXCLUSIONS:
        command.extend(["-g", pattern])
    command.append(root)
    try:
        result = runner(command, capture_output=True, text=True, timeout=SECONDS_PER_ROOT, check=False)
        status = "complete" if result.returncode in (0, 1) else "failed"
        output, code = decoded(result.stdout), result.returncode
    except subprocess.TimeoutExpired as error:
        status, output, code = "timed_out_partial", decoded(error.stdout), None
    paths = sorted(set(line for line in output.splitlines() if line))
    return {"root": root, "status": status, "returncode": code, "timeout_seconds": SECONDS_PER_ROOT,
            "matching_filenames": paths, "match_count": len(paths),
            "scope": "filename patterns only; no array integrity, model identity, precision, freshness or Q2 eligibility certified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a search receipt")
    results = []
    for root in ROOTS:
        result = search_root(root)
        results.append(result)
        print(json.dumps({"root": root, "status": result["status"], "match_count": result["match_count"]}), flush=True)
    body = {"schema": "sch.vpd-q2-recovery-capture-filename-search.v1", "observed_utc": datetime.now(timezone.utc).isoformat(),
            "roots": results, "filename_patterns": PATTERNS, "excluded_directories": EXCLUSIONS,
            "ignore_rules_disabled": True,
            "limitations": ["No full-disk search, Git-object inspection, archive extraction or deleted-file recovery was performed.",
                            "Timeout/failed roots are incomplete searches; zero output is not evidence of absence.",
                            "Filename matches are leads, not eligible captures. Nonstandard filenames and excluded trees may contain other data.",
                            "An earlier broad rg scan of Spiral_target/projects and Remote_clones was interrupted without results and is not counted as a complete search."],
            "boundary": {"recovery_files_modified": False, "recovery_code_executed": False, "capture_or_training_performed": False,
                         "paid_compute_launched": False, "additional_eligible_packs": 0},
            "runner_sha256": sha256_file(Path(__file__))}
    report = {**body, "report_id": content_sha256(body)}
    write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "complete_roots": sum(r["status"] == "complete" for r in results),
                      "matching_paths": sum(r["match_count"] for r in results)}), flush=True)


if __name__ == "__main__":
    main()
