"""Offline removal of feature contributions from saved Q4 SAE reconstructions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import _assign_features, _verify_and_load
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

SITES = ("model.layers.11", "model.norm")
RANDOM_GROUPS = 10
SEED = 260919


def matched_groups(train_mass, selected, seed):
    """Match counts in training-only mass/support bins; permit null overlap."""
    mass = train_mass.sum(axis=(0, 1))
    support = np.any(train_mass > 0, axis=1).sum(axis=0)
    active = mass > 0
    edges = np.quantile(mass[active], [1/3, 2/3])
    mass_bin = np.searchsorted(edges, mass, side="right")
    support_bin = np.minimum(support // 3, 2)
    bins = mass_bin * 3 + support_bin
    bins[~active] = -1
    if np.any(selected & ~active):
        raise ValueError("group includes inactive training feature")
    rng = np.random.default_rng(seed)
    groups = []
    for _ in range(RANDOM_GROUPS):
        control = np.zeros_like(selected)
        for b in np.unique(bins[selected]):
            pool = np.flatnonzero(bins == b)
            count = int(np.sum(selected & (bins == b)))
            control[rng.choice(pool, count, replace=False)] = True
        groups.append(control)
    return groups, bins


def removal_delta_sse(residual, contribution):
    # residual = observed - full reconstruction; removal adds contribution.
    return float(np.sum(contribution**2) + 2 * np.sum(residual * contribution))


def removal_components(residual, contribution, denominator):
    centered_r = residual - residual.mean(axis=0)
    centered_c = contribution - contribution.mean(axis=0)
    total = removal_delta_sse(residual, contribution) / denominator
    varying = removal_delta_sse(centered_r, centered_c) / denominator
    return {"total": total, "within_pass_variation": varying,
            "pass_mean_component": total - varying}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_archive(sae_dir):
    manifest, quality, features = _verify_and_load(sae_dir)
    capture_dir = sae_dir.parent / "q4_activations"
    capture_manifest = _json(capture_dir / "manifest.json")
    records_path = capture_dir / "records.jsonl"
    if sha256_file(records_path) != capture_manifest["files"]["records.jsonl"]:
        raise ValueError("capture records hash differs")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(manifest["training_ids"] + manifest["evaluation_ids"]):
        raise ValueError("capture IDs differ")
    captures = {}
    for record in records:
        if record["split"] != ("train" if record["id"] in manifest["training_ids"] else "eval"):
            raise ValueError("capture split differs")
        if record["split"] != "eval":
            continue
        path = capture_dir / record["archive"]
        if sha256_file(path) != record["sha256"] or capture_manifest["files"].get(record["archive"]) != record["sha256"]:
            raise ValueError("capture array hash differs")
        with np.load(path, allow_pickle=False) as a:
            captures[record["id"]] = {k: np.asarray(a[k]) for k in ["token_positions"] + [f"{s}__pass_{p}" for s in SITES for p in (0, 1)]}
        if not np.array_equal(captures[record["id"]]["token_positions"], record["positions"]):
            raise ValueError("capture positions differ")
    site_results = []
    for site_index, site in enumerate(SITES):
        entry = quality[site]
        with np.load(sae_dir / entry["dictionary"], allow_pickle=False) as a:
            decoder = np.asarray(a["decoder"], dtype=np.float64)
            bias = np.asarray(a["pre_bias"], dtype=np.float64)
        if decoder.shape != (6144, 3072) or bias.shape != (3072,) or not np.isfinite(decoder).all() or not np.isfinite(bias).all():
            raise ValueError("invalid dictionary arrays")
        masks = _assign_features(features[site]["train"])
        control_sets = [matched_groups(features[site]["train"], mask, SEED + site_index * 10 + p) for p, mask in enumerate(masks)]
        all_rows = [r for r in entry["differences"] if r["split"] == "eval"]
        xs, zs = [[], []], [[], []]
        for row in all_rows:
            with np.load(sae_dir / row["archive"], allow_pickle=False) as a:
                if not np.array_equal(a["token_positions"], captures[row["id"]]["token_positions"]):
                    raise ValueError("feature/capture positions differ")
                for p in (0, 1):
                    x = np.asarray(captures[row["id"]][f"{site}__pass_{p}"], dtype=np.float64)
                    z = np.asarray(a[f"pass_{p}"], dtype=np.float64)
                    if x.shape != (32, 3072) or z.shape != (32, 6144) or not np.isfinite(x).all():
                        raise ValueError("invalid held-out rows")
                    xs[p].append(x)
                    zs[p].append(z)
        xs = [np.concatenate(x) for x in xs]
        zs = [np.concatenate(z) for z in zs]
        residuals = [x - (z @ decoder + bias) for x, z in zip(xs, zs)]
        denominators = [float(np.sum((x - x.mean(axis=0))**2)) for x in xs]
        if min(denominators) <= 0:
            raise ValueError("zero held-out variance")
        evs = [1 - float(np.sum(r**2)) / den for r, den in zip(residuals, denominators)]
        expected = [entry["quality"]["eval"][f"pass_{p}"]["explained_variance"] for p in (0, 1)]
        if not np.allclose(evs, expected, atol=1e-5, rtol=0):
            raise ValueError("saved codes do not reproduce recorded reconstruction quality")
        groups = []
        for group_pass, mask in enumerate(masks):
            controls, bins = control_sets[group_pass]
            components = []
            for group in [mask] + controls:
                components.append([removal_components(r, z[:, group] @ decoder[group], den)
                                   for r, z, den in zip(residuals, zs, denominators)])
            scores = [[c["total"] for c in row] for row in components]
            score_array = np.asarray(scores)
            effects = score_array[:, group_pass] - score_array[:, 1-group_pass]
            varying = np.asarray([[c["within_pass_variation"] for c in row] for row in components])
            varying_effects = varying[:, group_pass] - varying[:, 1-group_pass]
            groups.append({
                "assigned_pass": group_pass, "selected_features": int(mask.sum()),
                "removal_delta_error_over_pass_variance": scores[0],
                "random_removal_delta_error_over_pass_variance": scores[1:],
                "target_minus_other_removal_effect": float(effects[0]),
                "random_target_minus_other_effects": effects[1:].tolist(),
                "effect_minus_random_mean": float(effects[0] - effects[1:].mean()),
                "selected_removal_decomposition_by_pass": components[0],
                "random_removal_decomposition_by_pass": components[1:],
                "within_pass_variation_target_minus_other": float(varying_effects[0]),
                "within_pass_variation_effect_minus_random_mean": float(varying_effects[0] - varying_effects[1:].mean()),
                "random_controls_exceeding_or_equal": int(np.sum(effects[1:] >= effects[0])),
                "random_overlap_with_selected_fraction": [float(np.sum(c & mask) / mask.sum()) for c in controls],
                "matching_bin_counts": [{"bin": int(b), "selected": int(np.sum(mask & (bins == b))), "pool": int(np.sum(bins == b))} for b in np.unique(bins[mask])],
            })
        site_results.append({"site": site, "held_out_examples": len(all_rows),
                             "held_out_explained_variance": evs,
                             "quality_qualified": min(evs) >= 0.20, "groups": groups})
    return {"sae_directory": str(sae_dir.resolve()), "manifest_sha256": sha256_file(sae_dir / "manifest.json"),
            "capture_manifest_sha256": sha256_file(capture_dir / "manifest.json"),
            "held_out_families": sorted({r["family"] for r in records if r["split"] == "eval"}),
            "seed": manifest["seed"], "sites": site_results}


def build_report(directories):
    archives = [audit_archive(path) for path in directories]
    body = {"schema": "sch.vpd-q4-sae-group-reconstruction-audit.v1",
            "archives": archives,
            "design": {"random_groups_per_selected_group": RANDOM_GROUPS, "seed": SEED,
                       "random_matching": "training_mass_tertiles_and_training_example_support_bins",
                       "positive_removal_effect_means_reconstruction_worsens": True,
                       "denominator": "pooled_held_out_variance_separately_for_each_pass",
                       "mean_decomposition": "evaluation_pass_mean_used_only_for_descriptive_algebraic_decomposition_not_feature_selection_or_retraining",
                       "random_groups_can_overlap_selected_group": True,
                       "inference": "descriptive_only_ten_controls_no_significance_or_admission",
                       "input_alignment": "within_archive_feature_positions_equal_capture_positions"},
            "boundary": {"reconstruction_diagnostic_is_not_model_behavior": True,
                         "cross_budget_same_input_comparison_performed": False,
                         "encoder_recomputed": False, "saved_codes_used": True,
                         "new_training_performed": False, "model_weights_loaded": False,
                         "sae_decoder_weights_read": True, "paid_compute_used": False,
                         "operative_admissions": 0},
            "provenance": {"runner_sha256": sha256_file(Path(__file__)),
                           "source_loader_sha256": sha256_file(Path(__file__).with_name("benchmark_nanbeige_q4_sae_pass_selectivity_v1.py"))}}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-directory", type=Path, action="append")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = _json(args.verify)
        report = build_report([Path(a["sae_directory"]) for a in observed["archives"]])
        if report != observed:
            raise ValueError("group reconstruction audit does not reproduce")
    else:
        if not args.sae_directory:
            parser.error("--sae-directory is required")
        report = build_report(args.sae_directory)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "archives": [
        {"path": a["sae_directory"], "sites": [{"site": s["site"], "ev": s["held_out_explained_variance"],
         "effects": [g["target_minus_other_removal_effect"] for g in s["groups"]],
         "minus_random_mean": [g["effect_minus_random_mean"] for g in s["groups"]]} for s in a["sites"]]} for a in report["archives"]]}, indent=2))


if __name__ == "__main__":
    main()
