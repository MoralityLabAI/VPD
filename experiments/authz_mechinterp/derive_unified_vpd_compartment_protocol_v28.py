from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_bound_content_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v27 import (
    canonical_inputs as protocol_v27_inputs,
    verify_protocol as verify_protocol_v27,
)


SCHEMA = "sch.vpd-activation-category-protocol.v28"
EXPECTED_V27_SHA256 = (
    "36c66bfb8591961342ea717bde6289f2511e32fbe2577959dad683ab51b94eca"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V28 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V27_SHA256:
        raise ValueError("protocol V27 differs from its pinned SHA-256")
    verification = verify_protocol_v27(path, **protocol_v27_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V27 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    content_verifier_path: Path,
    content_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_benchmark(
        content_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(content_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_to_sae_bound_content_v1"] = {
        "schema": "sch.vpd-cluster-to-sae-bound-content-contract.v1",
        "status": "streaming_verifier_ready_real_model_manifest_and_bindings_absent",
        "verifier": {
            "path": _relative(content_verifier_path, project_root),
            "sha256": sha256_file(content_verifier_path),
            "model_manifest_schema": (
                "sch.vpd-cluster-to-sae-model-files-manifest.v1"
            ),
            "bound_verification_schema": (
                "sch.vpd-cluster-to-sae-bound-content-verification.v1"
            ),
        },
        "benchmark": {
            "path": _relative(content_benchmark_path, project_root),
            "sha256": sha256_file(content_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "one_way_order": [
            "stream_model_files_and_write_manifest_outside_model_root",
            "bind_manifest_file_sha256_and_adapter_digests_in_preflight_v2",
            "verify_exact_model_tree_and_adapter_bytes_against_preflight",
            "verify_real_signed_scope_exact_authorization_receipt",
            "atomically_create_new_isolated_output_root",
            "begin_zero_action_capture",
        ],
        "model_manifest_preflight_back_reference_forbidden": True,
        "model_tree_exact_coverage_required": True,
        "model_tree_reparse_points_forbidden": True,
        "adapter_weight_file_must_be_unique": True,
        "adapter_configuration_digest_required": True,
        "real_model_manifest_present": False,
        "real_bound_content_verification_present": False,
        "model_or_adapter_loaded": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["streaming_bound_content_verifier_ready"] = True
    result["unified_lineage"]["bound_content_controls"] = benchmark["summary"][
        "controls"
    ]
    result["unified_lineage"]["content_manifest_preflight_cycle_free"] = True
    result["unified_lineage"]["real_model_manifest_present"] = False
    result["unified_lineage"]["real_bound_content_verification_present"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v27_verification": predecessor_verification,
        "bound_content_benchmark_verification": benchmark_verification,
        "bound_content_verifier_sha256": sha256_file(content_verifier_path),
        "bound_content_derivation_sha256": sha256_file(Path(__file__)),
        "manifest_then_preflight_one_way_order_required": True,
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_content_fixture_only"] = True
    result["scientific_boundary"]["real_model_manifest_present"] = False
    result["scientific_boundary"]["real_model_or_adapter_content_opened"] = False
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["sae_dictionary_training_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V28 is not reproducible")
    contract = observed["component_compartments"]["cluster_to_sae_bound_content_v1"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v28",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "bound_content_benchmark_id": contract["benchmark"]["benchmark_id"],
        "bound_content_controls": contract["benchmark"]["controls"]["controls"],
        "real_model_manifest_present": False,
        "real_bound_content_verification_present": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v27.json"
        ),
        "content_verifier_path": (
            project_root
            / "experiments/authz_mechinterp/verify_vpd_cluster_to_sae_bound_content_v1.py"
        ),
        "content_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_bound_content_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V28.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_protocol(**inputs)
        write_json(args.output, result)
        protocol_sha256 = sha256_file(args.output)
    else:
        result = verify_protocol(args.verify, **inputs)
        protocol_sha256 = result["protocol_sha256"]
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result.get("status", "derived"),
                "protocol_sha256": protocol_sha256,
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
