import json
import subprocess
from types import SimpleNamespace

import pytest

from experiments.authz_mechinterp import preflight_sae_q2_v1 as preflight


def passed_checks():
    return [{"id": c["id"], "status": "pass"} for c in preflight.checks()]


def test_all_local_passes_never_authorize_research():
    decision = preflight.readiness(passed_checks())
    assert decision["local_checks_status"] == "pass"
    assert decision["research_readiness"] == "not_ready"
    assert decision["launch_authorized"] is False
    assert len(decision["research_requirements"]) == 7
    assert preflight.exit_code({"decision": decision}) == 3


@pytest.mark.parametrize("status", ["failed", "timed_out", "not_run"])
def test_incomplete_local_checks_are_non_success(status):
    checks = passed_checks()
    checks[1]["status"] = status
    assert preflight.exit_code({"decision": preflight.readiness(checks)}) == 1


def test_missing_reordered_or_invalid_status_checks_rejected():
    for checks in (passed_checks()[:-1], passed_checks()[::-1]):
        with pytest.raises(ValueError):
            preflight.readiness(checks)
    checks = passed_checks()
    checks[0]["status"] = "skipped_but_ok"
    with pytest.raises(ValueError):
        preflight.readiness(checks)


def test_timeout_is_explicit_not_permission_to_proceed():
    def runner(argv, **kwargs):
        assert kwargs["timeout"] == 45
        raise subprocess.TimeoutExpired(argv, 45)
    result = preflight.execute_check(preflight.checks()[0], runner=runner)
    assert result["status"] == "timed_out"


@pytest.mark.parametrize("code,stdout,expected", [(0, "64 passed in 0.1s", "pass"), (0, "63 passed, 1 skipped", "failed"), (1, "64 passed", "failed")])
def test_process_exit_and_exact_fixture_count_both_checked(code, stdout, expected):
    result = preflight.execute_check(preflight.checks()[0], runner=lambda *a, **kw: SimpleNamespace(returncode=code, stdout=stdout))
    assert result["status"] == expected


def test_success_exit_with_wrong_receipt_is_rejected():
    result = preflight.execute_check(preflight.checks()[1], runner=lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps({"report_id": "wrong"})))
    assert result["status"] == "failed"


def test_json_success_cannot_be_replaced_with_arbitrary_output():
    result = preflight.execute_check(preflight.checks()[1], runner=lambda *a, **kw: SimpleNamespace(returncode=0, stdout="ok"))
    assert result["reason"] == "verifier_did_not_return_json"


def test_failed_check_stops_remaining_subprocesses():
    called = []
    def executor(check):
        called.append(check["id"])
        return {"id": check["id"], "status": "failed"}
    report = preflight.build_report(executor=executor)
    assert called == ["q2_unit_tests"]
    assert all(r["status"] == "not_run" for r in report["checks"][1:])
    assert report["verified_evidence_summary"] is None
    assert preflight.exit_code(report) == 1


def test_authorization_bit_cannot_turn_fixture_preflight_into_launcher():
    decision = preflight.readiness(passed_checks())
    decision["launch_authorized"] = True
    with pytest.raises(ValueError):
        preflight.exit_code({"decision": decision})
