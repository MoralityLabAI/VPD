from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-prompt-frozen-sae-quality-audit.v1"
SITES = ("model.layers.11", "model.layers.21", "model.norm")
PASSES = (0, 1)
MINIMUM_HELD_OUT_EXPLAINED_VARIANCE = 0.20
SITE_TO_DICTIONARY = {
    "model.layers.11": "sae_00.npz",
    "model.layers.21": "sae_01.npz",
    "model.norm": "sae_02.npz",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sources(
    prompt_dir: Path, original_calibration_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_manifest = _load(prompt_dir / "manifest.json")
    transfer_dir = prompt_dir / "frozen_sae_transfer"
    transfer_manifest = _load(transfer_dir / "manifest.json")
    summary = _load(transfer_dir / "summary.json")
    original_manifest_path = original_calibration_dir / "manifest.json"
    original_manifest = _load(original_manifest_path)

    if prompt_manifest.get("status") != "prompt_capture_complete":
        raise ValueError("prompt capture is not complete")
    if prompt_manifest.get("new_sae_fitted") is not False:
        raise ValueError("prompt capture unexpectedly reports fitting a new SAE")
    records = prompt_manifest.get("records")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("prompt capture must contain eight records")
    if {str(row["split"]) for row in records} != {"train", "eval"}:
        raise ValueError("prompt capture split labels differ")
    if sum(str(row["split"]) == "eval" for row in records) != 4:
        raise ValueError("prompt capture must contain four held-out records")
    for row in records:
        if sha256_file(prompt_dir / str(row["archive"])) != row["sha256"]:
            raise ValueError("prompt activation archive hash differs")

    if transfer_manifest.get("status") != "transfer_analysis_complete":
        raise ValueError("frozen-SAE transfer is not complete")
    if transfer_manifest.get("no_new_fitting") is not True:
        raise ValueError("frozen-SAE transfer unexpectedly fitted a dictionary")
    if (
        transfer_manifest.get("original_calibration_manifest_sha256")
        != sha256_file(original_manifest_path)
    ):
        raise ValueError("original calibration manifest hash differs")
    for name, expected in transfer_manifest["files"].items():
        if sha256_file(transfer_dir / name) != expected:
            raise ValueError(f"frozen-SAE transfer artifact hash differs: {name}")
    for site, dictionary_name in SITE_TO_DICTIONARY.items():
        expected = original_manifest["artifacts"][dictionary_name]
        if transfer_manifest["dictionaries"].get(site) != expected:
            raise ValueError(f"frozen dictionary identity differs: {site}")
        if sha256_file(original_calibration_dir / dictionary_name) != expected:
            raise ValueError(f"original frozen dictionary hash differs: {site}")

    if not isinstance(summary, list) or len(summary) != len(records) * len(SITES):
        raise ValueError("frozen-SAE transfer summary coverage differs")
    expected_pairs = {(str(row["id"]), site) for row in records for site in SITES}
    observed_pairs = {(str(row["id"]), str(row["site"])) for row in summary}
    if observed_pairs != expected_pairs or len(observed_pairs) != len(summary):
        raise ValueError("frozen-SAE transfer summary rows differ")
    records_by_id = {str(row["id"]): row for row in records}
    for row in summary:
        record = records_by_id[str(row["id"])]
        if row["split"] != record["split"]:
            raise ValueError("transfer summary split differs from capture")
        if transfer_manifest["files"].get(row["archive"]) != row["sha256"]:
            raise ValueError("transfer feature archive binding differs")
        for pass_index in PASSES:
            quality = row[f"pass_{pass_index}_quality"]
            if int(quality["rows"]) != 32:
                raise ValueError("prompt feature archive row count differs")
            values = np.asarray(
                [quality["mse"], quality["explained_variance"], quality["mean_l0"]],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError("prompt frozen-SAE quality contains nonfinite values")
    reload_controls = _load(transfer_dir / "dictionary_reload_controls.json")
    if len(reload_controls) != len(SITES) * len(PASSES):
        raise ValueError("dictionary reload controls have incomplete coverage")
    if max(float(row["explained_variance_abs_error"]) for row in reload_controls) > 2e-8:
        raise ValueError("dictionary reload controls exceed tolerance")
    return prompt_manifest, transfer_manifest, summary, reload_controls


def build_report(prompt_dir: Path, original_calibration_dir: Path) -> dict[str, Any]:
    prompt_dir = prompt_dir.resolve()
    original_calibration_dir = original_calibration_dir.resolve()
    prompt_manifest, transfer_manifest, summary, reload_controls = _verify_sources(
        prompt_dir, original_calibration_dir
    )
    held_out = [row for row in summary if row["split"] == "eval"]
    sites = []
    for site in SITES:
        site_rows = [row for row in held_out if row["site"] == site]
        minimum_by_pass = {
            f"pass_{pass_index}": min(
                float(row[f"pass_{pass_index}_quality"]["explained_variance"])
                for row in site_rows
            )
            for pass_index in PASSES
        }
        minimum = min(minimum_by_pass.values())
        qualified = minimum >= MINIMUM_HELD_OUT_EXPLAINED_VARIANCE
        sites.append(
            {
                "site": site,
                "held_out_examples": len(site_rows),
                "minimum_held_out_explained_variance_by_pass": minimum_by_pass,
                "minimum_held_out_explained_variance": minimum,
                "quality_threshold": MINIMUM_HELD_OUT_EXPLAINED_VARIANCE,
                "feature_interpretation_quality_gate": "pass" if qualified else "fail",
                "pass_feature_strata_admission": "fail",
            }
        )
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "sources": {
            "prompt_capture": {
                "path": prompt_dir.as_posix(),
                "manifest_sha256": sha256_file(prompt_dir / "manifest.json"),
                "records": len(prompt_manifest["records"]),
            },
            "frozen_sae_transfer": {
                "path": (prompt_dir / "frozen_sae_transfer").as_posix(),
                "manifest_sha256": sha256_file(
                    prompt_dir / "frozen_sae_transfer" / "manifest.json"
                ),
                "no_new_fitting": transfer_manifest["no_new_fitting"],
            },
            "original_calibration": {
                "path": original_calibration_dir.as_posix(),
                "manifest_sha256": sha256_file(original_calibration_dir / "manifest.json"),
            },
        },
        "design": {
            "purpose": "gate_prompt_phase_feature_interpretation_before_cross_phase_compartment_claims",
            "held_out_unit": "source_designated_eval_example",
            "held_out_examples": 4,
            "minimum_held_out_explained_variance": MINIMUM_HELD_OUT_EXPLAINED_VARIANCE,
            "site_must_pass_for_both_recurrent_passes": True,
            "dictionary_reload_max_explained_variance_abs_error": max(
                float(row["explained_variance_abs_error"]) for row in reload_controls
            ),
        },
        "sites": sites,
        "summary": {
            "site_checks": len(sites),
            "feature_interpretation_quality_admissions": sum(
                row["feature_interpretation_quality_gate"] == "pass" for row in sites
            ),
            "prompt_phase_pass_feature_strata_admissions": 0,
            "quality_rejected_sites": [row["site"] for row in sites],
        },
        "scientific_boundary": {
            "frozen_feature_archives_exist": True,
            "archive_existence_is_not_feature_interpretation_validity": True,
            "prompt_phase_feature_interpretation_quality_admitted": False,
            "cross_phase_pass_feature_strata_admitted": False,
            "new_sae_fitted": False,
            "new_model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "paid_compute_authorized": False,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
        "provenance": {"runner_sha256": sha256_file(Path(__file__))},
    }
    return {**body, "report_id": content_sha256(body)}


def verify_report(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    report_id = str(body.pop("report_id", ""))
    if content_sha256(body) != report_id:
        raise ValueError("prompt frozen-SAE quality report content address differs")
    expected = build_report(
        Path(observed["sources"]["prompt_capture"]["path"]),
        Path(observed["sources"]["original_calibration"]["path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("prompt frozen-SAE quality report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-prompt-frozen-sae-quality-verification.v1",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--original-calibration-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if args.prompt_dir is None or args.original_calibration_dir is None or args.output is None:
            parser.error("both source directories and --output are required unless verifying")
        result = build_report(args.prompt_dir, args.original_calibration_dir)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
