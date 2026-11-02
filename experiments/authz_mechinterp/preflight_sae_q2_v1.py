"""Bounded local Q2 evidence replay. Fixture success never authorizes a research run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_SECONDS = 45
NOT_READY_EXIT = 3
TESTS = ["tests/test_vpd_sae_paired_initialization.py", "tests/test_vpd_sae_paired_activation_binding.py",
         "tests/test_vpd_sae_paired_schedule.py", "tests/test_vpd_sae_trajectory_breadth_plan.py"]
RECEIPTS = {
    "initialization_chain": "artifacts/vpd_sae_paired_initialization_v1/report.json",
    "source_inventory": "artifacts/vpd_sae_source_family_inventory_v1/report.json",
    "phase_quality": "artifacts/vpd_q4_sae_phase_quality_v1/report.json",
}
MODULES = {
    "initialization_chain": "experiments.authz_mechinterp.check_sae_paired_initialization_v1",
    "source_inventory": "experiments.authz_mechinterp.audit_sae_source_family_inventory_v1",
    "phase_quality": "experiments.authz_mechinterp.audit_q4_sae_phase_quality_v1",
}
REQUIRED_RESEARCH_EVIDENCE = {
    "fresh_role_disjoint_pack": "Reviewed actual Q2 pack, semantic-family/alias audit, exposure review and frozen role/ID manifest.",
    "capture_producer_and_real_arrays": "Pinned model/capture stack, actual full-width captures, producer hook/pass verification and real array binding.",
    "research_initialization": "Equal full-size initial parameters and common complete-pool bias verified on the intended runtime.",
    "paired_optimizer_integration": "Reviewed trainer consuming the bound schedule for exactly 1000 matched updates, with numerical/failure controls.",
    "scientific_decision_rule": "Frozen useful-effect margin, prompt non-degradation rule, uncertainty/power analysis and joint stopping decision.",
    "explicit_training_and_spending_authority": "Explicit approval covering weight fitting, exact run conditions and any spending cap.",
    "independent_provider_shutdown": "Verified termination mechanism independent of this task, finite deadline and provider receipt/status checks.",
}


def checks():
    return [{"id": "q2_unit_tests", "argv": [sys.executable, "-B", "-m", "pytest", "-q", *TESTS]}] + [
        {"id": key, "argv": [sys.executable, "-B", "-m", module, "--verify", RECEIPTS[key]]}
        for key, module in MODULES.items()]


def execute_check(check, *, runner=subprocess.run):
    try:
        result = runner(check["argv"], cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        return {"id": check["id"], "status": "timed_out", "timeout_seconds": TIMEOUT_SECONDS}
    if result.returncode != 0:
        return {"id": check["id"], "status": "failed", "returncode": result.returncode, "reason": "nonzero_verifier_exit"}
    if check["id"] == "q2_unit_tests":
        match = re.search(r"\b(\d+) passed\b", result.stdout)
        if not match or int(match.group(1)) != 64:
            return {"id": check["id"], "status": "failed", "reason": "expected_64_fixture_tests_not_reported"}
        return {"id": check["id"], "status": "pass", "passed_tests": 64}
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"id": check["id"], "status": "failed", "reason": "verifier_did_not_return_json"}
    receipt_path = ROOT / RECEIPTS[check["id"]]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (output.get("report_id") != receipt.get("report_id")
            or receipt.get("report_id") != content_sha256({k: v for k, v in receipt.items() if k != "report_id"})):
        return {"id": check["id"], "status": "failed", "reason": "verifier_receipt_identity_differs"}
    return {"id": check["id"], "status": "pass", "verified_report_id": receipt["report_id"]}


def readiness(check_results):
    expected = [c["id"] for c in checks()]
    if [r["id"] for r in check_results] != expected:
        raise ValueError("all fixed local checks must be accounted for in order")
    if any(r["status"] not in ("pass", "failed", "timed_out", "not_run") for r in check_results):
        raise ValueError("invalid check status")
    local_pass = all(r["status"] == "pass" for r in check_results)
    return {"local_checks_status": "pass" if local_pass else "failed_or_incomplete",
            "research_readiness": "not_ready", "launch_authorized": False,
            "reason": "This V1 command accepts no research or authorization proofs; it verifies fixtures and archived evidence only.",
            "research_requirements": [{"id": key, "status": "not_verified_by_this_preflight", "required_evidence": value}
                                      for key, value in REQUIRED_RESEARCH_EVIDENCE.items()]}


def build_report(*, executor=execute_check):
    watched = [Path(__file__).relative_to(ROOT).as_posix(), *TESTS, *RECEIPTS.values(),
               "artifacts/vpd_sae_trajectory_breadth_plan_v1/report.json",
               "artifacts/vpd_sae_paired_schedule_v1/report.json",
               "artifacts/vpd_sae_paired_schedule_v1/slot_indices.json",
               "artifacts/vpd_sae_paired_activation_binding_v1/report.json"]
    before = {name: sha256_file(ROOT/name) for name in watched}
    results, stopped = [], False
    for check in checks():
        result = {"id": check["id"], "status": "not_run", "reason": "earlier_local_check_failed"} if stopped else executor(check)
        results.append(result)
        stopped |= result["status"] != "pass"
    after = {name: sha256_file(ROOT/name) for name in watched}
    if before != after:
        raise ValueError("watched preflight inputs changed during replay")
    decision = readiness(results)
    evidence = None
    if decision["local_checks_status"] == "pass":
        inventory = json.loads((ROOT/RECEIPTS["source_inventory"]).read_text(encoding="utf-8"))
        quality = json.loads((ROOT/RECEIPTS["phase_quality"]).read_text(encoding="utf-8"))
        generated = [r for r in quality["summary"] if r["phase"] == "generated"]
        if len(generated) != 1:
            raise ValueError("exactly one generated-phase summary required")
        evidence = {"source_family_count_check": inventory["inventory"]["F1_necessary_family_count_check"],
                    "archived_generated_phase_quality": generated[0],
                    "fixture_scope": "synthetic rows and width-5, ten-feature initialization only",
                    "current_provider_pod_status": "not_queried_unknown"}
    body = {"schema": "sch.vpd-sae-q2-local-preflight.v1", "checks": results, "decision": decision,
            "verified_evidence_summary": evidence, "watched_input_sha256": before,
            "execution_contract": {"commands": checks(), "per_check_timeout_seconds": TIMEOUT_SECONDS,
                                   "stop_after_first_failure": True, "not_ready_exit_code": NOT_READY_EXIT,
                                   "local_failure_exit_code": 1, "launch_option_exists": False},
            "boundary": {"research_weights_modified": False, "research_training_performed": False,
                         "paid_compute_launched": False, "provider_status_verified": False, "operative_admissions": 0},
            "limitations": ["Only the four fixed local checks run; this is not the entire project suite or the full canonical release audit.",
                            "Missing research proofs are not inferred absent everywhere on disk; they are not accepted or certified by this fixture-only V1 interface.",
                            "A timeout is a failed local check, not evidence that any remote pod has stopped.",
                            "All local checks passing still returns not_ready and exit 3; it never grants training or spending permission."]}
    return {**body, "report_id": content_sha256(body)}


def exit_code(report):
    if report["decision"]["launch_authorized"] or report["decision"]["research_readiness"] != "not_ready":
        raise ValueError("fixture-only preflight cannot authorize a launch")
    return NOT_READY_EXIT if report["decision"]["local_checks_status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("refusing to overwrite an existing report")
    report = build_report()
    if args.verify:
        if report != json.loads(args.verify.read_text(encoding="utf-8")):
            raise ValueError("preflight report does not reproduce")
    else:
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "checks": report["checks"], "decision": report["decision"],
                      "verified_evidence_summary": report["verified_evidence_summary"]}, indent=2))
    raise SystemExit(exit_code(report))


if __name__ == "__main__":
    main()
