from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v6 import (
    canonical_inputs as protocol_inputs,
)
from experiments.authz_mechinterp.derive_vpd_partition_audit_gate_plan_v2 import (
    verify_gate_plan_v2,
)
from experiments.authz_mechinterp.evaluate_vpd_partition_audit import (
    MEASUREMENT_IDENTITY_KEYS,
    _require_production_screens,
    validate_cross_archive_measurement_identity,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline_v4 import (
    canonical_inputs as receipt_inputs,
    verify_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "artifacts" / "vpd_partition_audit_gate_v2" / "plan.json"
DISCOVERY_CAMPAIGN = (
    ROOT / "artifacts" / "vpd_effect_capture_campaign_v4" / "plan.json"
)
PROTOCOL = ROOT / "configs" / "evals" / "vpd_activation_category_compartments_v6.json"
RECEIPT = (
    ROOT / "artifacts" / "vpd_unified_compartment_pipeline_v4" / "receipt.json"
)


def _gate_inputs() -> dict[str, Path]:
    values = protocol_inputs(ROOT)
    return {
        "predecessor_gate_plan_path": values["predecessor_gate_plan_path"],
        "predecessor_protocol_path": values["v4_protocol_path"],
        "pack_dir": values["pack_dir"],
        "benchmark_path": values["benchmark_path"],
        "metadata_bundle_dir": values["bundle_dir"],
        "discovery_campaign_plan_path": values["discovery_campaign_plan_path"],
    }


def test_gate_v2_binds_non_authorizing_exact_job_graph() -> None:
    verification = verify_gate_plan_v2(GATE, **_gate_inputs())
    assert verification["status"] == "pass"
    assert verification["work_units"] == verification["job_groups"] == 16
    assert verification["execution_authorized"] is False
    plan = json.loads(GATE.read_text(encoding="utf-8"))
    roles = plan["execution_plan"]["bundle_roles"]
    assert roles["capture_packet_bundle"]["activation_category_assignment_claimed"] is False
    assert roles["frozen_category_aware_bundle"]["same_packet_ids_as_capture_packet_bundle_required"] is False
    merge_command = plan["execution_plan"]["merge_step"]["command"]
    assert merge_command[merge_command.index("--bundle") + 1] == "{CAPTURE_PACKET_BUNDLE}"
    evaluate_command = plan["execution_plan"]["evaluation_step"]["command"]
    assert evaluate_command[evaluate_command.index("--bundle-dir") + 1] == "{CATEGORY_AWARE_BUNDLE}"
    work_unit_ids = {row["work_unit_id"] for row in plan["capture_work_units"]}
    jobs = plan["execution_plan"]["job_groups"]
    assert {row["work_unit_id"] for row in jobs} == work_unit_ids
    assert all(row["execution_authorized"] is False for row in jobs)
    capture_bundle = ROOT / plan["source"]["current_bundle_evidence"]["path"]
    capture_manifest = json.loads(
        (capture_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    packet_workers = {}
    for entry in capture_manifest["artifacts"]["packets"]:
        packet = json.loads(
            (capture_bundle / entry["packet_path"]).read_text(encoding="utf-8")
        )
        packet_workers[packet["packet_id"]] = packet["distribution"]["assigned_worker"]
    expected_private_keys = {
        f"WORKER_PRIVATE_KEY_{worker.upper().replace('-', '_')}"
        for worker in packet_workers.values()
    }
    required = set(plan["execution_plan"]["required_placeholders"])
    assert expected_private_keys <= required
    assert "WORKER_ID" not in required
    assert "WORKER_PRIVATE_KEY" not in required
    assert "DTYPE" not in required
    campaign = json.loads(DISCOVERY_CAMPAIGN.read_text(encoding="utf-8"))
    bound_campaign = plan["source"]["discovery_effect_capture_campaign"]
    assert bound_campaign["campaign_plan_id"] == campaign["campaign_plan_id"]
    assert bound_campaign["capture_parameters"] == campaign["parameters"]
    expected_capture_arguments = {
        "--maximum-episodes": str(campaign["parameters"]["maximum_episodes"]),
        "--selection-seed": str(campaign["parameters"]["selection_seed"]),
        "--epsilon": str(campaign["parameters"]["epsilon"]),
        "--batch-size": str(campaign["parameters"]["batch_size"]),
        "--uncertainty-blocks": str(campaign["parameters"]["uncertainty_blocks"]),
        "--uncertainty-sketch-dimension": str(
            campaign["parameters"]["uncertainty_sketch_dimension"]
        ),
        "--dtype": campaign["parameters"]["dtype"],
    }
    for job in jobs:
        commands = [step["command"] for step in job["steps"]]
        assert commands[0][2] == (
            "experiments.authz_mechinterp.compute_vpd_effect_shard_v2"
        )
        flat = [token for command in commands for token in command]
        assert "--activation-category-split-role" not in flat
        assert "--activation-category-split-manifest" not in flat
        assert "experiments.authz_mechinterp.attest_partition_audit_effect_receipt" in flat
        assert "--private-key" in flat
        shard = commands[0][commands[0].index("--output-shard") + 1]
        assert shard.startswith("{SHARDS_DIR}/")
        assert shard.endswith(f"{job['work_unit_id']}.npz")
        assigned_worker = packet_workers[job["packet_id"]]
        assert job["assigned_worker"] == assigned_worker
        compute_worker = commands[0][commands[0].index("--worker-id") + 1]
        assert compute_worker == assigned_worker
        private_key_name = f"WORKER_PRIVATE_KEY_{assigned_worker.upper().replace('-', '_')}"
        signing_key = commands[2][commands[2].index("--private-key") + 1]
        assert job["private_key_placeholder"] == private_key_name
        assert signing_key == "{" + private_key_name + "}"
        worker_plan = plan["execution_plan"]["worker_signing"]["workers"][
            assigned_worker
        ]
        assert job["work_unit_id"] in worker_plan["work_unit_ids"]
        for argument, expected_value in expected_capture_arguments.items():
            assert commands[0][commands[0].index(argument) + 1] == expected_value
        assert "{DTYPE}" not in flat


def test_gate_v2_rejects_readdressed_policy_tampering(tmp_path: Path) -> None:
    plan = json.loads(GATE.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(plan)
    tampered["execution_plan"]["execution_authorized"] = True
    body = copy.deepcopy(tampered)
    body.pop("gate_plan_id")
    tampered["gate_plan_id"] = content_sha256(body)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="not reproducible"):
        verify_gate_plan_v2(path, **_gate_inputs())


def test_gate_v2_rejects_readdressed_discovery_capture_parameter_drift(
    tmp_path: Path,
) -> None:
    campaign = json.loads(DISCOVERY_CAMPAIGN.read_text(encoding="utf-8"))
    campaign["parameters"]["dtype"] = "bfloat16"
    body = copy.deepcopy(campaign)
    body.pop("campaign_plan_id")
    campaign["campaign_plan_id"] = content_sha256(body)
    path = tmp_path / "readdressed-campaign.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    inputs = _gate_inputs()
    inputs["discovery_campaign_plan_path"] = path
    with pytest.raises(ValueError, match="not reproducible"):
        verify_gate_plan_v2(GATE, **inputs)


def test_production_screen_contract_is_fail_closed() -> None:
    screens = {
        "pair_block_direction_robustness_screen": {
            "mode": "pair_block_mean_countsketch_direction_and_relative_norm_coverage",
            "required": True,
        },
        "condition_robustness_screen": {
            "mode": "condition_mean_countsketch_direction_and_relative_norm_coverage",
            "required": True,
        },
        "template_robustness_screen": {
            "mode": "template_group_mean_countsketch_direction_and_relative_norm_coverage",
            "required": True,
        },
    }
    assert set(_require_production_screens(screens)) == {"pair", "condition", "template"}
    invalid = copy.deepcopy(screens)
    invalid["condition_robustness_screen"]["required"] = False
    with pytest.raises(ValueError, match="condition screen"):
        _require_production_screens(invalid)


def test_cross_archive_measurement_identity_is_fail_closed() -> None:
    identity = {
        key: (
            ["replayed", "valid_authority"]
            if key == "activation_categories"
            else {"schema": f"fixture-{key}"}
            if key.startswith("activation_category_")
            else 0.1
            if key == "perturbation_epsilon"
            else 1707
            if key == "selection_seed"
            else 0
            if key == "generated_tokens"
            else "a" * 64
            if key in {"adapter_sha256", "runner_sha256", "activation_category_contract_sha256"}
            else f"fixture-{key}"
        )
        for key in MEASUREMENT_IDENTITY_KEYS
    }
    discovery = {"identity": copy.deepcopy(identity)}
    audit = {"identity": copy.deepcopy(identity)}
    verification = validate_cross_archive_measurement_identity(
        discovery_manifest=discovery, audit_manifest=audit
    )
    assert verification["status"] == "pass"
    for key in ("model_id", "revision", "adapter_sha256", "effect_space_id"):
        changed = copy.deepcopy(audit)
        changed["identity"][key] = "b" * 64
        with pytest.raises(ValueError, match=key):
            validate_cross_archive_measurement_identity(
                discovery_manifest=discovery, audit_manifest=changed
            )


def test_protocol_v6_and_unified_lineage_v4_reproduce() -> None:
    receipt = verify_receipt(RECEIPT, **receipt_inputs(ROOT))
    assert receipt["status"] == "pass"
    assert receipt["partition_audit_work_units"] == 16
    assert receipt["execution_authorized"] is False
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["schema"] == "sch.vpd-activation-category-protocol.v6"
    benchmark = protocol["effect_capture"]["partition_audit_campaign"][
        "model_free_capture_pipeline_benchmark"
    ]
    assert benchmark["status"] == "pass"
    assert benchmark["scenarios"] == 7
    capture_contract = protocol["effect_capture"]["partition_audit_campaign"][
        "verified_discovery_capture_contract"
    ]
    assert capture_contract["capture_parameters"]["dtype"] == "float16"
    assert protocol["effect_capture"]["partition_audit_campaign"][
        "model_load_dtype_is_not_runtime_configurable"
    ] is True
    assert protocol["effect_capture"]["partition_audit_campaign"][
        "receipt_level_campaign_measurement_conformance_required"
    ] is True
    assert protocol["effect_capture"]["partition_audit_campaign"][
        "model_load_dtype_evidence_boundary"
    ].endswith("not_worker_observed_field")
