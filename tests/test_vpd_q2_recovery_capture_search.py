import subprocess
from types import SimpleNamespace

from experiments.authz_mechinterp.search_q2_recovery_captures_v1 import search_root


def test_ignored_artifacts_are_searched_and_results_deduplicated():
    def runner(argv, **kwargs):
        assert "--no-ignore" in argv and "--hidden" in argv
        assert kwargs["timeout"] == 25
        return SimpleNamespace(returncode=0, stdout="b\na\na\n")
    result = search_root("fixture-root", runner=runner)
    assert result["status"] == "complete"
    assert result["matching_filenames"] == ["a", "b"]


def test_timeout_preserves_partial_paths_without_absence_claim():
    def runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 25, output=b"partial/path\n")
    result = search_root("fixture-root", runner=runner)
    assert result["status"] == "timed_out_partial"
    assert result["matching_filenames"] == ["partial/path"]


def test_no_matches_and_scan_error_remain_distinct():
    empty = search_root("fixture-root", runner=lambda *a, **kw: SimpleNamespace(returncode=1, stdout=""))
    failed = search_root("fixture-root", runner=lambda *a, **kw: SimpleNamespace(returncode=2, stdout=""))
    assert empty["status"] == "complete" and empty["match_count"] == 0
    assert failed["status"] == "failed"
