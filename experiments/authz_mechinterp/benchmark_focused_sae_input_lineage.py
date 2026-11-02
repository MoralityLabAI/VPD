from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_focused_vpd_sae import (
    validate_packet_effect_lineage,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import (
    validate_packet_effect_input_lineage,
)


SCHEMA = "sch.vpd-focused-sae-input-lineage-benchmark.v1"


def _scenario(
    scenario_id: str,
    expected_status: str,
    call: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        result = call()
        observed = str(result["status"])
        detail = result
    except ValueError as exc:
        observed = "reject"
        detail = {"error": str(exc)}
    return {
        "scenario_id": scenario_id,
        "expected_status": expected_status,
        "observed_status": observed,
        "result": detail,
        "status": "pass" if observed == expected_status else "fail",
    }


def build_benchmark() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "activation_effects.npz"
        manifest = root / "effect_merge_manifest.json"
        other_archive = root / "other_effects.npz"
        other_manifest = root / "other_merge_manifest.json"
        archive.write_bytes(b"exact-discovery-effect-archive")
        manifest.write_bytes(b"exact-discovery-effect-merge-manifest")
        other_archive.write_bytes(b"substituted-discovery-effect-archive")
        other_manifest.write_bytes(b"substituted-discovery-effect-merge-manifest")
        packet = {
            "source": {
                "effects_sha256": sha256_file(archive),
                "effects_merge_manifest_sha256": sha256_file(manifest),
            },
            "sae_study": {
                "activation_compartment": {"category": "replayed"},
            },
        }
        missing = copy.deepcopy(packet)
        missing["source"] = {
            "effects_sha256": None,
            "effects_merge_manifest_sha256": None,
        }
        legacy = copy.deepcopy(missing)
        legacy["sae_study"]["activation_compartment"] = None
        exact_inputs = {
            "effect_archive_sha256": sha256_file(archive),
            "effect_merge_manifest_sha256": sha256_file(manifest),
            "packet_effect_lineage": {
                "schema": "sch.vpd-focused-sae-packet-effect-lineage.v1",
                "status": "pass",
                "effect_archive_sha256": sha256_file(archive),
                "effect_merge_manifest_sha256": sha256_file(manifest),
                "empirical_category_claim_allowed": True,
            },
        }
        substituted_reducer_inputs = copy.deepcopy(exact_inputs)
        substituted_reducer_inputs["effect_archive_sha256"] = sha256_file(
            other_archive
        )
        misstated_reducer_inputs = copy.deepcopy(exact_inputs)
        misstated_reducer_inputs["packet_effect_lineage"]["status"] = "pass_unverified"
        scenarios = [
            _scenario(
                "exact_packet_bound_archive_and_manifest_pass",
                "pass",
                lambda: validate_packet_effect_lineage(
                    packet=packet,
                    effect_archive_path=archive,
                    effect_manifest_path=manifest,
                ),
            ),
            _scenario(
                "substituted_effect_archive_rejects",
                "reject",
                lambda: validate_packet_effect_lineage(
                    packet=packet,
                    effect_archive_path=other_archive,
                    effect_manifest_path=manifest,
                ),
            ),
            _scenario(
                "substituted_effect_manifest_rejects",
                "reject",
                lambda: validate_packet_effect_lineage(
                    packet=packet,
                    effect_archive_path=archive,
                    effect_manifest_path=other_manifest,
                ),
            ),
            _scenario(
                "category_packet_without_effect_lineage_rejects",
                "reject",
                lambda: validate_packet_effect_lineage(
                    packet=missing,
                    effect_archive_path=archive,
                    effect_manifest_path=manifest,
                ),
            ),
            _scenario(
                "legacy_metadata_packet_is_explicitly_unattested",
                "unattested_legacy_metadata_pilot",
                lambda: validate_packet_effect_lineage(
                    packet=legacy,
                    effect_archive_path=archive,
                    effect_manifest_path=manifest,
                ),
            ),
            _scenario(
                "exact_signed_input_lineage_passes_reducer",
                "pass",
                lambda: validate_packet_effect_input_lineage(
                    packet=packet,
                    inputs=exact_inputs,
                ),
            ),
            _scenario(
                "substituted_signed_effect_archive_rejects_at_reducer",
                "reject",
                lambda: validate_packet_effect_input_lineage(
                    packet=packet,
                    inputs=substituted_reducer_inputs,
                ),
            ),
            _scenario(
                "misstated_signed_lineage_receipt_rejects_at_reducer",
                "reject",
                lambda: validate_packet_effect_input_lineage(
                    packet=packet,
                    inputs=misstated_reducer_inputs,
                ),
            ),
        ]
    project_root = Path(__file__).resolve().parents[2]
    body = {
        "schema": SCHEMA,
        "status": "pass" if all(row["status"] == "pass" for row in scenarios) else "fail",
        "scenarios": scenarios,
        "source": {
            "benchmark_runner_sha256": sha256_file(Path(__file__)),
            "focused_study_runner_sha256": sha256_file(
                project_root / "experiments/authz_mechinterp/run_focused_vpd_sae.py"
            ),
            "result_reducer_sha256": sha256_file(
                project_root / "experiments/authz_mechinterp/merge_vpd_sae_results.py"
            ),
        },
        "scientific_boundary": {
            "synthetic_file_lineage_only": True,
            "model_inference_performed": False,
            "sae_training_performed": False,
            "does_not_establish_bonsai_cluster_or_feature_assignments": True,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(report)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("focused SAE input-lineage benchmark content address differs")
    if report.get("status") != "pass" or build_benchmark() != report:
        raise ValueError("focused SAE input-lineage benchmark is not reproducible")
    return {
        "schema": "sch.vpd-focused-sae-input-lineage-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "scenarios": len(report["scenarios"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the model-free focused SAE input-lineage benchmark."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("benchmark generation requires --output")
        result = build_benchmark()
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
