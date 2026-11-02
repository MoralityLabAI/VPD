"""Read-only positional compatibility audit of the recovered BF16 reasoning pack."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.nanbeige_reasoning.positions import capture_positions

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/nanbeige_reasoning/prime-20260906T204058Z-561/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration"
MANIFEST_SHA256 = "7c5205eeb3c1e7c025ac1aababc6e0526bbe82048f633a52e73bfefa07c8a546"
SITES = ("model.layers.11", "model.norm")


def position_coverage(prompt_tokens, total_tokens, saved_positions):
    saved = np.asarray(saved_positions)
    if (type(prompt_tokens) is not int or type(total_tokens) is not int or not 0 < prompt_tokens < total_tokens
            or saved.ndim != 1 or saved.dtype.kind not in "iu" or len(set(saved.tolist())) != len(saved)
            or np.any(saved < 0) or np.any(saved >= total_tokens)):
        raise ValueError("valid prompt/continuation and unique saved absolute positions required")
    desired, regions = capture_positions(prompt_tokens, total_tokens)
    valid_layout = len(desired) == 32 and all(end-start == 8 for start, end in regions.values())
    have = set(saved.tolist())
    return {"q2_window_layout_possible": valid_layout, "saved_rows": len(saved), "required_rows": len(desired),
            "missing_positions": sorted(set(desired)-have), "extra_saved_positions": sorted(have-set(desired)),
            "region_coverage": {name: {"required": end-start, "available": sum(p in have for p in range(start, end))}
                                for name, (start, end) in regions.items()},
            "q2_windows_present": valid_layout and set(desired).issubset(have)}


def build_report():
    manifest_path = SOURCE/"manifest.json"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("pinned recovered manifest differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["schema"] != "vpd.multipass.calibration.v1" or manifest["status"] != "calibration_complete"
            or manifest["passes"] != 2 or manifest["pass_index_base"] != 0):
        raise ValueError("unsupported recovered calibration contract")
    if sha256_file(SOURCE/"examples.json") != manifest["artifacts"]["examples.json"]:
        raise ValueError("source examples hash differs")
    examples = json.loads((SOURCE/"examples.json").read_text(encoding="utf-8"))
    if len({r["id"] for r in examples}) != len(examples) or {r["id"] for r in examples} != set(manifest["baseline_archives"]):
        raise ValueError("source ID inventory differs")
    records = []
    for example in examples:
        if example["split"] not in ("train", "eval") or not isinstance(example["family"], str) or not example["family"]:
            raise ValueError("invalid source family/split")
        tokens = np.asarray(example["token_ids"])
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 0):
            raise ValueError("invalid full source token sequence")
        filename = manifest["baseline_archives"][example["id"]]
        path = (SOURCE/filename).resolve()
        if path.parent != SOURCE.resolve() or sha256_file(path) != manifest["artifacts"][filename]:
            raise ValueError("source archive path/hash differs")
        arrays_checked = []
        with np.load(path, allow_pickle=False) as archive:
            positions = np.asarray(archive["token_positions"])
            result = position_coverage(example["prompt_tokens"], len(tokens), positions)
            if not np.array_equal(archive["token_ids"], tokens[positions]):
                raise ValueError("saved token tail differs from full source token positions")
            for site in SITES:
                for p in (0, 1):
                    key = f"{site}__pass_{p}"
                    values = np.asarray(archive[key])
                    if values.shape != (len(positions), 3072) or values.dtype.kind != "f" or not np.isfinite(values).all():
                        raise ValueError("invalid baseline activation array")
                    arrays_checked.append({"key": key, "shape": list(values.shape), "storage_dtype": str(values.dtype)})
        records.append({"id": example["id"], "family": example["family"], "split": example["split"],
                        "archive": filename, "archive_sha256": manifest["artifacts"][filename],
                        "position_coverage": result, "arrays_checked": arrays_checked})
    training = [r for r in records if r["split"] == "train"]
    evaluation = [r for r in records if r["split"] == "eval"]
    if len(training) != manifest["train_examples"] or len(evaluation) != manifest["eval_examples"]:
        raise ValueError("source split counts differ")
    family_counts = dict(sorted(Counter(r["family"] for r in training).items()))
    body = {"schema": "sch.vpd-bf16-q2-capture-coverage.v1", "source": {"path": str(SOURCE), "manifest_sha256": MANIFEST_SHA256},
            "records": records,
            "summary": {"examples": len(records), "training_examples": len(training), "evaluation_examples": len(evaluation),
                        "checked_site_pass_arrays": sum(len(r["arrays_checked"]) for r in records),
                        "training_family_example_counts": family_counts,
                        "training_inventory_matches_q2": len(family_counts) == 10 and set(family_counts.values()) == {4},
                        "examples_with_all_q2_windows": sum(r["position_coverage"]["q2_windows_present"] for r in records),
                        "missing_training_positions_per_site_pass": sum(len(r["position_coverage"]["missing_positions"]) for r in training),
                        "missing_all_positions_per_site_pass": sum(len(r["position_coverage"]["missing_positions"]) for r in records),
                        "declared_model_dtype": manifest["model"]["dtype"],
                        "checkpoint_revision_verified_by_source": manifest["model"]["revision_verified"],
                        "source_declares_family_disjoint": manifest["family_disjoint"]},
            "decision": {"q2_import_eligible": False, "reason": "Historical tail-only small calibration is not a fresh four-window Q2 pack; missing activations cannot be reconstructed by relabeling.",
                         "allowed_use": "historical activation/VPD diagnostics with existing provenance limits"},
            "boundary": {"real_saved_arrays_read": True, "capture_reexecuted": False, "training_performed": False,
                         "weights_modified": False, "freshness_verified": False, "paid_compute_launched": False, "operative_admissions": 0},
            "limitations": ["Only the pinned recovered BF16 reasoning calibration was audited, not every local or recovery capture.",
                            "Float32 array storage is not proof of FP32 model inference; declared BF16 execution comes from the source manifest and its revision remains unverified.",
                            "Hash/shape/token checks do not independently prove that the original producer attached the correct model hook or pass label.",
                            "No interpolation, row duplication or archive relabeling can replace the missing observed activation positions."],
            "provenance": {"runner_sha256": sha256_file(Path(__file__)), "window_recipe_sha256": sha256_file(ROOT/"experiments/nanbeige_reasoning/positions.py")}}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("refusing to overwrite existing receipt")
    report = build_report()
    if args.verify:
        if report != json.loads(args.verify.read_text(encoding="utf-8")):
            raise ValueError("BF16 positional coverage receipt does not reproduce")
    else:
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "summary": report["summary"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
