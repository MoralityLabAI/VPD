"""Exploratory training-trajectory coverage using saved activations and SAE codes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp import audit_q4_sae_phase_quality_v1 as phase_audit
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json


def coverage_scores(training, evaluation, training_ids):
    training, evaluation = (np.asarray(x, dtype=float) for x in (training, evaluation))
    ids = np.asarray(training_ids)
    if (training.ndim != 2 or evaluation.ndim != 2 or not len(training) or not len(evaluation)
            or training.shape[1] == 0 or training.shape[1] != evaluation.shape[1]
            or not np.isfinite(training).all() or not np.isfinite(evaluation).all()
            or ids.shape != (len(training),)):
        raise ValueError("finite aligned nonempty activation matrices and training IDs required")
    unique = sorted(set(ids.tolist()))
    if len(unique) < 2 or any(not isinstance(i, str) or not i for i in unique):
        raise ValueError("at least two named training trajectories required")
    if len({int(np.sum(ids == i)) for i in unique}) != 1:
        raise ValueError("equal rows per training trajectory required for matched reference pools")
    center = training.mean(0)
    training, evaluation = training-center, evaluation-center
    scale = float(np.mean(np.sum(training*training, axis=1)))
    if scale <= 1e-12:
        raise ValueError("degenerate training variance")

    def distances_to_trajectories(query):
        squared = (np.sum(query*query, axis=1)[:, None] + np.sum(training*training, axis=1)[None, :]
                   - 2 * query @ training.T)
        squared = np.maximum(squared, 0.0) / scale
        return np.stack([squared[:, ids == i].min(1) for i in unique], axis=1)

    train_distances = distances_to_trajectories(training)
    eval_distances = distances_to_trajectories(evaluation)
    for index, example_id in enumerate(unique):
        train_distances[ids == example_id, index] = np.inf
    training_loto = train_distances.min(1)
    # Each evaluation row is compared to every (N-1)-trajectory reference pool.
    evaluation_matched = np.mean([
        np.delete(eval_distances, i, axis=1).min(1) for i in range(len(unique))
    ], axis=0)
    return {"training_scale": scale, "training_loto": training_loto,
            "evaluation_matched": evaluation_matched,
            "evaluation_full_pool": eval_distances.min(1),
            "reference_trajectories": len(unique)-1}


def descriptive_correlation(left, right):
    left, right = (np.asarray(v, dtype=float) for v in (left, right))
    if (left.ndim != 1 or left.shape != right.shape or not np.isfinite(left).all()
            or not np.isfinite(right).all()):
        raise ValueError("aligned finite vectors required")
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_phase(training, evaluation, residuals, training_ids, evaluation_ids, families):
    training_ids, evaluation_ids, families = (np.asarray(v) for v in (training_ids, evaluation_ids, families))
    residuals = np.asarray(residuals, dtype=float)
    if (residuals.shape != evaluation.shape or not np.isfinite(residuals).all()
            or evaluation_ids.shape != (len(evaluation),) or families.shape != evaluation_ids.shape
            or set(training_ids.tolist()) & set(evaluation_ids.tolist())):
        raise ValueError("aligned disjoint train/evaluation metadata and residuals required")
    scores = coverage_scores(training, evaluation, training_ids)
    error = np.sum(residuals*residuals, axis=1) / scores["training_scale"]
    training_examples = [{"id": i, "rows": int(np.sum(training_ids == i)),
                          "mean_loto_distance_squared_over_training_variance": float(scores["training_loto"][training_ids == i].mean())}
                         for i in sorted(set(training_ids.tolist()))]
    examples = []
    for example_id in sorted(set(evaluation_ids.tolist())):
        keep = evaluation_ids == example_id
        labels = set(families[keep].tolist())
        if len(labels) != 1:
            raise ValueError("one family per evaluation trajectory required")
        examples.append({"id": example_id, "family": next(iter(labels)), "rows": int(keep.sum()),
                         "mean_matched_distance_squared_over_training_variance": float(scores["evaluation_matched"][keep].mean()),
                         "mean_full_pool_distance_squared_over_training_variance": float(scores["evaluation_full_pool"][keep].mean()),
                         "mean_reconstruction_error_squared_over_training_variance": float(error[keep].mean())})
    train_median = float(np.median([r["mean_loto_distance_squared_over_training_variance"] for r in training_examples]))
    eval_median = float(np.median([r["mean_matched_distance_squared_over_training_variance"] for r in examples]))
    family_rows = []
    for family in sorted(set(families.tolist())):
        group = [r for r in examples if r["family"] == family]
        family_rows.append({"family": family, "examples": len(group),
                            "mean_example_matched_distance": float(np.mean([r["mean_matched_distance_squared_over_training_variance"] for r in group])),
                            "mean_example_reconstruction_error": float(np.mean([r["mean_reconstruction_error_squared_over_training_variance"] for r in group]))})
    return {"training_variance_per_row": scores["training_scale"],
            "reference_trajectories_per_comparison": scores["reference_trajectories"],
            "training_examples": training_examples, "evaluation_examples": examples, "families": family_rows,
            "median_training_example_loto_distance": train_median,
            "median_evaluation_example_matched_distance": eval_median,
            "evaluation_over_training_median_distance_ratio": eval_median/train_median if train_median > 1e-12 else None,
            "descriptive_evaluation_example_pearson_r": descriptive_correlation(
                [r["mean_matched_distance_squared_over_training_variance"] for r in examples],
                [r["mean_reconstruction_error_squared_over_training_variance"] for r in examples])}


def audit_archive(source):
    directory = Path(source["path"])
    capture_directory = directory.parent / "q4_activations"
    if (sha256_file(directory / "manifest.json") != source["sae_manifest_sha256"]
            or sha256_file(capture_directory / "manifest.json") != source["capture_manifest_sha256"]):
        raise ValueError("source manifest hash differs")
    manifest, quality, _ = phase_audit._verify_and_load(directory)
    cm, records = phase_audit.load_metadata(capture_directory)
    if set(records) != set(manifest["training_ids"] + manifest["evaluation_ids"]):
        raise ValueError("capture/SAE IDs differ")
    rows = sorted(records.values(), key=lambda r: r["id"])
    if any(r["split"] != ("train" if r["id"] in manifest["training_ids"] else "eval") for r in rows):
        raise ValueError("capture/SAE splits differ")
    captures = {r["id"]: phase_audit.load_capture(capture_directory, cm, r) for r in rows}
    ids = np.concatenate([[r["id"]]*len(r["positions"]) for r in rows])
    families = np.concatenate([[r["family"]]*len(r["positions"]) for r in rows])
    splits = np.concatenate([[r["split"]]*len(r["positions"]) for r in rows])
    phases = np.concatenate([phase_audit.phase_labels(r["positions"], r["prompt_tokens"], len(captures[r["id"]]["token_ids"])) for r in rows])
    sites = []
    for site in phase_audit.SITES:
        entry = quality[site]
        with np.load(directory / entry["dictionary"], allow_pickle=False) as data:
            decoder, bias = np.asarray(data["decoder"], dtype=float), np.asarray(data["pre_bias"], dtype=float)
        if decoder.shape != (6144, 3072) or bias.shape != (3072,) or not np.isfinite(decoder).all() or not np.isfinite(bias).all():
            raise ValueError("invalid dictionary")
        by_id = {r["id"]: r for r in entry["differences"]}
        codes = [[], []]
        for row in rows:
            with np.load(directory / by_id[row["id"]]["archive"], allow_pickle=False) as data:
                if not np.array_equal(data["token_positions"], row["positions"]):
                    raise ValueError("code positions differ")
                for p in (0, 1):
                    codes[p].append(np.asarray(data[f"pass_{p}"], dtype=float))
        results = []
        for p in (0, 1):
            x = np.concatenate([captures[r["id"]][f"{site}__pass_{p}"].astype(float) for r in rows])
            reconstructed = np.concatenate(codes[p]) @ decoder + bias
            for split in ("train", "eval"):
                keep = splits == split
                if not np.isclose(phase_audit.explained_variance(x[keep], reconstructed[keep]),
                                  entry["quality"][split][f"pass_{p}"]["explained_variance"], atol=1e-5, rtol=0):
                    raise ValueError("recorded mixed-phase quality does not reproduce")
            for phase in ("prompt", "generated"):
                train = (splits == "train") & (phases == phase)
                evaluation = (splits == "eval") & (phases == phase)
                result = summarize_phase(x[train], x[evaluation], (x-reconstructed)[evaluation],
                                         ids[train], ids[evaluation], families[evaluation])
                results.append({"pass": p, "phase": phase, **result})
        sites.append({"site": site, "results": results})
    return {"path": str(directory), "sites": sites}


def build_report(source_path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source.get("schema") != "sch.vpd-q4-sae-family-contributions.v1"
            or source.get("report_id") != content_sha256({k: v for k, v in source.items() if k != "report_id"})):
        raise ValueError("source receipt integrity differs")
    archives = [audit_archive(a) for a in source["archives"]]
    dependencies = [Path(__file__), Path(phase_audit.__file__),
                    Path(__file__).with_name("audit_q4_common_capture_rows_v1.py"),
                    Path(__file__).with_name("benchmark_nanbeige_q4_sae_pass_selectivity_v1.py")]
    body = {"schema": "sch.vpd-q4-sae-training-coverage.v1", "archives": archives,
            "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path), "report_id": source["report_id"]},
            "design": {"scope": "exploratory, phase/pass/site-specific local saved-array diagnostic",
                       "distance": "squared Euclidean nearest-row distance divided by same-phase/pass training variance per row",
                       "training_reference": "exclude the entire query trajectory, not merely its own row",
                       "evaluation_reference": "average nearest distance across all leave-one-training-trajectory-out pools, matching reference size",
                       "aggregation": "mean rows within each example; median examples for train/evaluation ratio; descriptive Pearson correlation over evaluation examples only"},
            "limitations": ["Tiny development archives, overlapping family labels, and same seed; no independent replication or significance test.",
                            "Euclidean distance includes activation magnitude and direction and is not a semantic or density estimate.",
                            "Training LOTO distances are not out-of-sample SAE errors: the SAE was fitted on these training rows.",
                            "Distance-error association is descriptive and confounded by family/content/scale; it cannot establish that broader training will repair quality.",
                            "Nearest captured rows do not measure coverage of the full activation manifold; only retained windows are observed."],
            "boundary": {"training_performed": False, "encoder_recomputed": False, "model_inference_performed": False,
                         "paid_compute_launched": False, "operative_admissions": 0},
            "provenance": {p.name: sha256_file(p) for p in dependencies}}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        report = build_report(Path(observed["source"]["path"]))
        if observed != report:
            raise ValueError("coverage receipt does not reproduce")
    else:
        if not args.source:
            parser.error("--source required")
        if args.output.exists():
            raise ValueError("refusing to overwrite an existing receipt")
        report = build_report(args.source)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "summary": [
        {"path": a["path"], "site": s["site"], "phase": r["phase"], "pass": r["pass"],
         "distance_ratio": r["evaluation_over_training_median_distance_ratio"],
         "descriptive_r": r["descriptive_evaluation_example_pearson_r"]}
        for a in report["archives"] for s in a["sites"] for r in s["results"]]}, indent=2))


if __name__ == "__main__":
    main()
