from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v21 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v13 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v21.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v13/runbook.json"


def test_protocol_v21_reproduces_router_and_planner_lineage() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["router_benchmark_id"] == (
        "0d5668cdbec6629f1632b3cf0f26a69695c95ec137fcaa2cae0342cde1056ac1"
    )
    assert verification["planner_benchmark_id"] == (
        "d79ff71b50d0bfc0bddb649609234b6bf9fbf9be78ef985daa773ebd5687ca10"
    )
    assert verification["empirical_cluster_routing_ready"] is False
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert observed["unified_lineage"]["duplicated_category_archives_prohibited"]
    assert observed["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v13_reproduces_and_uses_safe_wrapper_commands() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "79f1c87f1cfa84fb05d575be0265cdf7e41405e1e791306e1b36c9b41b0ea757"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["sae_training_role_capture_planner_ready"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert "verify_partition_identity_evidence_wrapper_v1" in command_ids
    assert "verify_cluster_episode_routing_benchmark_v1" in command_ids
    assert "verify_cluster_aware_sae_plan_benchmark_v1" in command_ids
    assert "verify_cluster_partition_identity_benchmark_v4" not in command_ids
    assert "verify_cluster_partition_identity_admission_benchmark_v4" not in command_ids
    assert observed["scientific_boundary"]["execution_authorized"] is False
