from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_cluster_aware_sae_empirical_budget_binding_readiness_v1 import (
    ALL_BINDINGS,
    ARTIFACT_BINDING_PHASES,
    build_binding_readiness,
    canonical_inputs,
    verify_binding_readiness,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
    REQUIRED_EXECUTION_BINDINGS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


def _digest(label: str) -> str:
    return hashlib.sha256(f"vpd-synthetic-binding:{label}".encode()).hexdigest()


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def _synthetic_complete_bindings(project_root: Path) -> dict[str, Any]:
    bindings: dict[str, Any] = {
        "model_id": "synthetic-model-id-not-opened",
        "model_revision": "synthetic-immutable-revision-not-opened",
        "model_files_manifest_sha256": _digest("model-files-manifest"),
        "adapter_path": str(
            (project_root / "artifacts/synthetic-adapter-fixture-not-opened").resolve()
        ),
        "adapter_weights_sha256": _digest("adapter-weights"),
        "adapter_config_sha256": _digest("adapter-config"),
        "device": "cpu",
        "wall_time_ceiling_seconds": 1,
        "cost_ceiling_usd": 0.01,
        "output_root": str(
            (
                project_root
                / "artifacts/vpd_cluster_to_sae_empirical_capture_runs/synthetic-binding-fixture-never-created"
            ).resolve()
        ),
    }
    for field_name in ALL_BINDINGS:
        bindings.setdefault(field_name, _digest(field_name))
    return bindings


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    inputs = canonical_inputs(project_root)
    readiness_path = (
        project_root
        / "artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1/readiness.json"
    )
    default_verification = verify_binding_readiness(readiness_path, **inputs)
    default = json.loads(readiness_path.read_text(encoding="utf-8"))
    complete_bindings = _synthetic_complete_bindings(project_root)
    complete = build_binding_readiness(**inputs, bindings=complete_bindings)

    def rejects(bindings: dict[str, Any]) -> bool:
        return _rejects(lambda: build_binding_readiness(**inputs, bindings=bindings))

    unknown = dict(complete_bindings)
    unknown["unknown_binding"] = _digest("unknown")
    placeholder_text = dict(complete_bindings)
    placeholder_text["model_id"] = "TODO"
    uniform_digest = dict(complete_bindings)
    uniform_digest["execution_blueprint_id"] = "0" * 64
    uppercase_digest = dict(complete_bindings)
    uppercase_digest["execution_blueprint_sha256"] = _digest(
        "execution_blueprint_sha256"
    ).upper()
    short_digest = dict(complete_bindings)
    short_digest["budget_admission_id"] = "abc"
    invalid_device = dict(complete_bindings)
    invalid_device["device"] = "gpu"
    invalid_wall_time = dict(complete_bindings)
    invalid_wall_time["wall_time_ceiling_seconds"] = 0
    invalid_cost = dict(complete_bindings)
    invalid_cost["cost_ceiling_usd"] = float("inf")
    outside_output = dict(complete_bindings)
    outside_output["output_root"] = str(
        (project_root / "artifacts/outside-dedicated-capture-parent").resolve()
    )
    one_missing = dict(complete_bindings)
    one_missing.pop("runtime_attachment_claim_sha256")
    partial = build_binding_readiness(**inputs, bindings=one_missing)
    tampered_readiness = copy.deepcopy(default)
    tampered_readiness["summary"]["missing_bindings"].pop()

    phase_counts = {
        row["phase"]: row["required_bindings"] for row in default["phase_summaries"]
    }
    controls = {
        "default_worksheet_reproduces": default_verification["status"] == "pass",
        "default_requires_exactly_50_bindings": default["summary"][
            "required_bindings"
        ]
        == 50
        and len(default["binding_rows"]) == 50,
        "default_reports_all_50_missing": default["summary"]["supplied_bindings"]
        == 0
        and len(default["summary"]["missing_bindings"]) == 50,
        "phase_partition_is_exact_and_exhaustive": phase_counts
        == {
            "capture_execution_identity": 10,
            "empirical_capture": 8,
            "routing_and_partition": 10,
            "execution_blueprint_and_budget": 6,
            "signed_release_and_compute_authority": 8,
            "durable_runtime_transition": 8,
        },
        "all_default_phases_are_incomplete_and_nonauthorizing": all(
            row["syntax_complete"] is False
            and row["content_verified"] is False
            and row["authority_granted"] is False
            for row in default["phase_summaries"]
        ),
        "pinned_budget_policy_is_bound": default["source"]["budget_policy"][
            "budget_policy_id"
        ]
        == "5e9f4fc0841d0e71a2eef6f9ac91f5e27d0702c0e0aadfa5a6f511dad50719f2",
        "capture_binding_template_is_bound": default["source"][
            "capture_binding_readiness_template"
        ]["readiness_id"]
        == "77e2c766499f22d401ffa60166b31dbc8fe4d0ecaefa349ab3b636202fef1472",
        "synthetic_complete_fixture_reaches_syntax_only": complete["status"]
        == "syntax_complete_unopened_awaiting_real_artifact_graph_and_signed_authority"
        and complete["summary"]["syntax_complete"] is True
        and complete["summary"]["missing_bindings"] == [],
        "synthetic_complete_fixture_preserves_output_isolation": complete["summary"][
            "capture_output_isolation_verified"
        ]
        is True,
        "synthetic_complete_fixture_opens_no_content": complete[
            "completed_syntax_audit"
        ]["model_or_adapter_content_opened"]
        is False
        and complete["completed_syntax_audit"]["artifact_graph_content_opened"]
        is False,
        "synthetic_completion_does_not_verify_cross_artifact_graph": complete["summary"]
        ["cross_artifact_content_verified"]
        is False
        and complete["completed_syntax_audit"]["cross_artifact_binding_verified"]
        is False,
        "synthetic_completion_does_not_verify_budget_binding": complete["summary"][
            "budget_binding_verified"
        ]
        is False,
        "synthetic_completion_does_not_verify_signed_authority": complete["summary"][
            "signed_authority_verified"
        ]
        is False,
        "synthetic_completion_does_not_verify_runtime_transition": complete["summary"]
        ["durable_runtime_transition_verified"]
        is False,
        "unknown_binding_rejected": rejects(unknown),
        "placeholder_text_rejected": rejects(placeholder_text),
        "uniform_placeholder_digest_rejected": rejects(uniform_digest),
        "uppercase_digest_rejected": rejects(uppercase_digest),
        "short_digest_rejected": rejects(short_digest),
        "invalid_device_rejected": rejects(invalid_device),
        "nonpositive_wall_time_rejected": rejects(invalid_wall_time),
        "nonfinite_cost_rejected": rejects(invalid_cost),
        "output_outside_dedicated_parent_rejected": rejects(outside_output),
        "one_missing_binding_remains_blocked": partial["status"]
        == "blocked_missing_empirical_blueprint_budget_and_authority_bindings"
        and partial["summary"]["missing_bindings"]
        == ["runtime_attachment_claim_sha256"],
        "worksheet_content_tamper_changes_address": tampered_readiness[
            "binding_readiness_id"
        ]
        != content_sha256(
            {
                key: value
                for key, value in tampered_readiness.items()
                if key != "binding_readiness_id"
            }
        ),
        "default_persists_no_synthetic_complete_fixture": default[
            "scientific_boundary"
        ]["synthetic_complete_fixture_persisted"]
        is False,
        "worksheet_remains_nonexecuting": all(
            default["scientific_boundary"][field_name] is False
            for field_name in (
                "model_loaded",
                "model_inference_performed",
                "weights_loaded_or_mutated",
                "empirical_capture_present",
                "empirical_routing_bundle_present",
                "empirical_execution_blueprint_present",
                "real_signed_receipts_present",
                "durable_runtime_transition_present",
                "budget_binding_verified",
                "execution_authorized",
                "sae_training_authorized",
                "paid_compute_authorized",
                "real_actions_authorized",
            )
        ),
        "benchmark_imports_no_model_frameworks": before_frameworks
        == (MODEL_FRAMEWORKS & set(sys.modules)),
        "synthetic_output_root_not_created": not Path(
            complete_bindings["output_root"]
        ).exists(),
    }
    if not all(controls.values()):
        raise ValueError(
            "empirical budget binding readiness benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-empirical-budget-binding-readiness-benchmark.v1",
        "status": "pass",
        "source": {
            "binding_readiness_path": readiness_path.relative_to(project_root).as_posix(),
            "binding_readiness_sha256": sha256_file(readiness_path),
            "binding_readiness_id": default["binding_readiness_id"],
            "builder_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/build_cluster_aware_sae_empirical_budget_binding_readiness_v1.py"
            ),
        },
        "controls": controls,
        "summary": {
            "passed": sum(controls.values()),
            "controls": len(controls),
            "required_bindings": 50,
            "missing_bindings": 50,
            "capture_execution_identity_bindings": len(REQUIRED_EXECUTION_BINDINGS),
            "artifact_graph_bindings": sum(
                len(fields) for fields in ARTIFACT_BINDING_PHASES.values()
            ),
            "phases": 6,
            "synthetic_complete_fixture_persisted": False,
        },
        "scientific_boundary": {
            "synthetic_complete_fixture_used_in_memory_only": True,
            "empirical_artifact_graph_present": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "weights_mutated": False,
            "persistent_output_created": False,
            "budget_binding_verified": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("empirical budget binding readiness benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-empirical-budget-binding-readiness-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "binding_readiness_id": observed["source"]["binding_readiness_id"],
        "controls": observed["summary"]["controls"],
        "required_bindings": observed["summary"]["required_bindings"],
        "missing_bindings": observed["summary"]["missing_bindings"],
        "phases": observed["summary"]["phases"],
        "budget_binding_verified": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify empirical-blueprint budget binding readiness benchmark V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=root)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result["benchmark_id"],
                "binding_readiness_id": result.get(
                    "binding_readiness_id",
                    result.get("source", {}).get("binding_readiness_id"),
                ),
                "controls": result.get(
                    "controls", result.get("summary", {}).get("controls")
                ),
                "execution_authorized": result.get(
                    "execution_authorized",
                    result.get("scientific_boundary", {}).get("execution_authorized"),
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
