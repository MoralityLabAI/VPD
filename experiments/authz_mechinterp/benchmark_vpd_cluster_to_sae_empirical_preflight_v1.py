from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v1 import (
    REQUIRED_BINDINGS,
    _validate_bindings,
    _verify_campaign_address,
    build_preflight,
    verify_preflight,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


MODEL_FRAMEWORKS = {"torch", "transformers", "peft", "accelerate"}


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def _complete_bindings() -> dict[str, Any]:
    digest = "a" * 64
    return {
        "model_id": "synthetic/model-id-not-loaded",
        "model_revision": "synthetic-revision-not-loaded",
        "model_files_manifest_sha256": digest,
        "adapter_path": "synthetic/adapter-path-not-opened",
        "adapter_weights_sha256": digest,
        "adapter_config_sha256": digest,
        "device": "synthetic-device-not-used",
        "wall_time_ceiling_seconds": 1,
        "cost_ceiling_usd": 1.0,
        "output_root": "artifacts/synthetic-preflight-output-not-created",
        "authorization_receipt_id": digest,
        "authorization_receipt_sha256": digest,
        "explicit_user_authorization_statement": (
            "synthetic fixture only; this is not real execution authorization"
        ),
    }


def build_report(project_root: Path) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    template = build_preflight(project_root=project_root)
    after_template = MODEL_FRAMEWORKS & set(sys.modules)
    complete = build_preflight(
        project_root=project_root,
        bindings=_complete_bindings(),
    )
    after_complete = MODEL_FRAMEWORKS & set(sys.modules)
    campaign = json.loads(
        (
            project_root / "artifacts/vpd_effect_capture_campaign_v4/plan.json"
        ).read_text(encoding="utf-8")
    )
    tampered_campaign = copy.deepcopy(campaign)
    tampered_campaign["resources"]["total"]["compute_jobs"] -= 1
    invalid_digest = _complete_bindings()
    invalid_digest["adapter_weights_sha256"] = "not-a-digest"
    invalid_cost = _complete_bindings()
    invalid_cost["cost_ceiling_usd"] = 0
    unknown_binding = _complete_bindings()
    unknown_binding["load_model_now"] = True
    roles = template["capture_roles"]
    controls = {
        "template_reports_all_thirteen_missing_bindings": template[
            "missing_required_bindings"
        ]
        == list(REQUIRED_BINDINGS),
        "template_is_blocked_not_execution_ready": template["status"]
        == "blocked_missing_immutable_identities_and_explicit_authorization",
        "synthetic_complete_fixture_still_does_not_authorize_execution": complete[
            "status"
        ]
        == "identity_complete_requires_separate_authenticated_runtime_authorization"
        and complete["scientific_boundary"]["execution_authorized"] is False,
        "exact_four_role_dataset_hashes_and_role_ids_bound": set(roles)
        == {"discovery", "calibration", "cluster_validation", "sae_training"}
        and all(len(row["dataset_sha256"]) == 64 for row in roles.values())
        and len({row["split_role_id"] for row in roles.values()}) == 4,
        "selection_and_routing_extension_uses_are_separate": all(
            roles[role]["preflight_use"] == "cluster_selection_effect_capture"
            for role in ("discovery", "calibration", "cluster_validation")
        )
        and roles["sae_training"]["preflight_use"]
        == "routing_assignment_effect_capture_only_before_dictionary_learning",
        "full_ceiling_is_41352192_batch_forwards": template["resource_ceiling"][
            "maximum_batch_forwards_at_batch_size_4"
        ]
        == 41_352_192,
        "full_ceiling_is_164900736_episode_forward_equivalents": template[
            "resource_ceiling"
        ]["maximum_episode_forward_equivalents"]
        == 164_900_736,
        "payload_ceiling_is_4046192640_bytes": template["resource_ceiling"][
            "estimated_float32_effect_payload_bytes"
        ]
        == 4_046_192_640,
        "final_evaluation_absent_from_capture_and_stage_order": "final_evaluation"
        not in roles
        and all(
            "final_evaluation" not in stage for stage in template["stage_order"]
        ),
        "stage_order_gates_before_later_compute": template["stage_order"].index(
            "discovery_gate"
        )
        < template["stage_order"].index("calibration_capture")
        < template["stage_order"].index("calibration_gate")
        < template["stage_order"].index("cluster_validation_capture")
        < template["stage_order"].index("partition_identity_admission")
        < template["stage_order"].index("sae_training_routing_capture"),
        "campaign_content_tamper_rejected": _rejects(
            lambda: _verify_campaign_address(tampered_campaign)
        ),
        "invalid_digest_cost_and_unknown_binding_rejected": _rejects(
            lambda: _validate_bindings(invalid_digest)
        )
        and _rejects(lambda: _validate_bindings(invalid_cost))
        and _rejects(lambda: _validate_bindings(unknown_binding)),
        "preflight_does_not_import_model_frameworks": before
        == after_template
        == after_complete,
        "preflight_opens_no_model_or_adapter_weights": template[
            "scientific_boundary"
        ]["model_or_adapter_weights_opened"]
        is False
        and complete["scientific_boundary"]["model_or_adapter_weights_opened"]
        is False,
        "preflight_performs_no_inference_routing_or_training": all(
            template["scientific_boundary"][field] is False
            for field in (
                "model_inference_performed",
                "route_allocations_created",
                "sae_dictionary_trained",
                "weights_loaded_or_mutated",
            )
        ),
    }
    if not all(controls.values()):
        raise ValueError(
            "cluster-to-SAE empirical preflight benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-to-sae-empirical-preflight-benchmark.v1",
        "status": "pass",
        "template_preflight_id": template["preflight_id"],
        "complete_synthetic_fixture_preflight_id": complete["preflight_id"],
        "results": {
            "missing_required_bindings": template["missing_required_bindings"],
            "resource_ceiling": template["resource_ceiling"],
            "capture_role_ids": {
                role: row["split_role_id"] for role, row in sorted(roles.items())
            },
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "source": {
            "preflight_builder_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/build_vpd_cluster_to_sae_empirical_preflight_v1.py"
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "synthetic_complete_bindings_are_not_real_identities": True,
            "execution_authorized": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "sae_training_authorized": False,
            "release_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = observed.get("benchmark_id")
    body = {key: value for key, value in observed.items() if key != "benchmark_id"}
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("cluster-to-SAE empirical preflight benchmark is not reproducible")
    preflight_verification = verify_preflight(
        project_root / "artifacts/vpd_cluster_to_sae_empirical_preflight_v1/preflight.json",
        project_root=project_root,
    )
    return {
        "schema": "sch.vpd-cluster-to-sae-empirical-preflight-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "template_preflight_id": preflight_verification["preflight_id"],
        "controls": observed["summary"]["controls"],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the no-model cluster-to-SAE empirical preflight."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
