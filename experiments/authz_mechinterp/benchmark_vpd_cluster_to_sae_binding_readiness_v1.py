from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
    OUTPUT_PARENT,
    audit_completed_bindings,
    build_readiness,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def _bindings(project_root: Path) -> dict[str, Any]:
    return {
        "model_id": "synthetic/model-not-loaded",
        "model_revision": "synthetic-revision-not-loaded",
        "model_files_manifest_sha256": "1" * 64,
        "adapter_path": str(Path("C:/synthetic/adapter-not-opened").resolve()),
        "adapter_weights_sha256": "2" * 64,
        "adapter_config_sha256": "3" * 64,
        "device": "cpu",
        "wall_time_ceiling_seconds": 3600,
        "cost_ceiling_usd": 1.0,
        "output_root": str(
            (
                project_root
                / OUTPUT_PARENT
                / "synthetic-readiness-fixture-do-not-create"
            ).resolve()
        ),
    }


def build_report(project_root: Path) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    template = build_readiness(project_root=project_root)
    bindings = _bindings(project_root)
    completed = build_readiness(project_root=project_root, bindings=bindings)
    audit = audit_completed_bindings(
        project_root=project_root, bindings=bindings, output_root_exists=False
    )

    def mutated(**changes: Any) -> dict[str, Any]:
        return {**bindings, **changes}

    allowed_parent = (project_root / OUTPUT_PARENT).resolve()
    outside = str((project_root / "artifacts/outside-capture-root").resolve())
    nested = str((allowed_parent / "parent" / "nested-run-name").resolve())
    noncanonical = str(allowed_parent / "segment" / ".." / "noncanonical-run")
    after = MODEL_FRAMEWORKS & set(sys.modules)
    controls = {
        "template_lists_exactly_ten_missing_bindings": (
            template["summary"]["required_bindings"] == 10
            and len(template["summary"]["missing_bindings"]) == 10
            and template["summary"]["supplied_bindings"] == 0
        ),
        "template_join_has_no_fabricated_authority_inputs": all(
            template["authorization_join"][field] is None
            for field in (
                "preflight_path",
                "authority_registry_path",
                "signed_receipt_path",
                "verification_as_of_utc",
            )
        ),
        "synthetic_complete_bindings_pass_syntax_and_isolation": (
            audit["bindings_complete"]
            and audit["output_root_nonexistent"]
            and audit["output_root_direct_child"]
            and audit["execution_authorized"] is False
        ),
        "completed_worksheet_still_awaits_signed_authorization": (
            completed["status"]
            == "identity_complete_output_isolated_awaiting_signed_authorization"
            and completed["summary"]["missing_bindings"] == []
            and completed["scientific_boundary"]["execution_authorized"] is False
        ),
        "existing_output_root_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings=bindings,
                output_root_exists=True,
            )
        ),
        "outside_output_root_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings=mutated(output_root=outside),
                output_root_exists=False,
            )
        ),
        "output_parent_itself_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings=mutated(output_root=str(allowed_parent)),
                output_root_exists=False,
            )
        ),
        "nested_output_root_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings=mutated(output_root=nested),
                output_root_exists=False,
            )
        ),
        "noncanonical_output_root_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings=mutated(output_root=noncanonical),
                output_root_exists=False,
            )
        ),
        "relative_adapter_path_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings=mutated(adapter_path="relative/adapter"),
                output_root_exists=False,
            )
        ),
        "unsupported_device_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings=mutated(device="synthetic-device"),
                output_root_exists=False,
            )
        ),
        "missing_binding_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings={key: value for key, value in bindings.items() if key != "device"},
                output_root_exists=False,
            )
        ),
        "unknown_binding_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings={**bindings, "authorization_receipt_id": "4" * 64},
                output_root_exists=False,
            )
        ),
        "nonpositive_wall_time_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings=mutated(wall_time_ceiling_seconds=0),
                output_root_exists=False,
            )
        ),
        "malformed_digest_rejected": _rejects(
            lambda: audit_completed_bindings(
                project_root=project_root,
                bindings=mutated(adapter_weights_sha256="not-a-digest"),
                output_root_exists=False,
            )
        ),
        "readiness_imports_no_model_frameworks": before == after,
        "all_readiness_paths_remain_non_authorizing": all(
            audit[field] is False
            for field in ("execution_authorized", "sae_training_authorized")
        )
        and all(
            completed["scientific_boundary"][field] is False
            for field in (
                "execution_authorized",
                "sae_training_authorized",
                "final_evaluation_authorized",
                "real_actions_authorized",
            )
        ),
    }
    if not all(controls.values()):
        raise ValueError(
            "cluster-to-SAE binding readiness benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-to-sae-binding-readiness-benchmark.v1",
        "status": "pass",
        "results": {
            "template_readiness_id": template["readiness_id"],
            "synthetic_completed_readiness_id": completed["readiness_id"],
            "synthetic_completed_audit": audit,
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "source": {
            "readiness_builder_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/build_vpd_cluster_to_sae_binding_readiness_v1.py"
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "synthetic_bindings_only": True,
            "model_or_adapter_content_opened": False,
            "model_framework_imported": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "authorization_receipt_present": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "release_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = str(observed.get("benchmark_id", ""))
    body = {key: value for key, value in observed.items() if key != "benchmark_id"}
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("cluster-to-SAE binding readiness benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-to-sae-binding-readiness-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "missing_bindings": 10,
        "model_or_adapter_content_opened": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the non-authorizing cluster-to-SAE binding worksheet."
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
