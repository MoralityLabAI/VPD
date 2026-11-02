from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_binding_readiness_v1 import (
    _bindings,
    build_report,
    verify_benchmark,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    OUTPUT_PARENT,
    audit_completed_bindings,
    build_readiness,
    verify_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "artifacts/vpd_cluster_to_sae_binding_readiness_v1/readiness.json"
BENCHMARK = (
    ROOT / "artifacts/vpd_cluster_to_sae_binding_readiness_benchmark_v1/report.json"
)


def test_binding_readiness_template_reproduces() -> None:
    observed = json.loads(READINESS.read_text(encoding="utf-8"))
    assert observed == build_readiness(project_root=ROOT)
    verification = verify_readiness(READINESS, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["required_bindings"] == 10
    assert verification["missing_bindings"] == 10
    assert verification["output_isolation_verified"] is False
    assert verification["execution_authorized"] is False


def test_binding_readiness_benchmark_reproduces() -> None:
    observed = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["controls"] == 17
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_completed_binding_audit_is_non_authorizing() -> None:
    audit = audit_completed_bindings(
        project_root=ROOT,
        bindings=_bindings(ROOT),
        output_root_exists=False,
    )
    assert audit["bindings_complete"] is True
    assert audit["output_root_nonexistent"] is True
    assert audit["model_content_opened"] is False
    assert audit["adapter_content_opened"] is False
    assert audit["execution_authorized"] is False


def test_binding_audit_rejects_existing_and_escaped_output_roots() -> None:
    bindings = _bindings(ROOT)
    with pytest.raises(ValueError, match="already exists"):
        audit_completed_bindings(
            project_root=ROOT, bindings=bindings, output_root_exists=True
        )
    with pytest.raises(ValueError, match="direct child"):
        audit_completed_bindings(
            project_root=ROOT,
            bindings={
                **bindings,
                "output_root": str((ROOT / "artifacts/escaped-run").resolve()),
            },
            output_root_exists=False,
        )
    assert Path(bindings["output_root"]).parent == (ROOT / OUTPUT_PARENT).resolve()
