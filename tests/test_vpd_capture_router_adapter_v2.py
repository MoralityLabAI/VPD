from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_capture_router_adapter_v2 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_capture_router_adapter_benchmark_v2/report.json"


def test_v3_capture_router_adapter_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "10486291b569bd2eeea91f8429702b6c668463d727013cad98baf2f676628688"
    )
    assert verification["controls"] == 18
    assert verification["v3_guarded_lineage_preserved"] is True
    assert verification["production_path_empirically_exercised"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False


def test_v3_capture_router_adapter_preserves_guarded_identity_without_authority() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["v3_execution_identity_is_router_identity"] is True
    assert controls["router_receipt_preserves_v3_execution_identity"] is True
    assert controls["predecessor_v2_identity_preserved"] is True
    assert controls["guarded_job_identity_preserved"] is True
    assert controls["capture_artifacts_are_read_only"] is True
    assert controls["production_context_rejected_for_synthetic"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["persistent_empirical_router_receipt_produced"] is False
    assert boundary["production_path_empirically_exercised"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["sae_training_authorized"] is False
