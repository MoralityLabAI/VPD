"""Exploratory family-balanced reconstruction audit on existing Q4 captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from experiments.authz_mechinterp.audit_q4_sae_group_reconstruction_v1 import (
    SITES, SEED, _json, matched_groups, removal_components,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import _verify_and_load, _assign_features
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, content_sha256, write_json


def summarize_family_advantages(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite independent-family scores required")
    sd = float(values.std(ddof=1))
    return {"families": len(values), "equal_family_mean_advantage": float(values.mean()),
            "family_advantage_sample_sd": sd,
            "mean_over_sample_sd_descriptive": float(values.mean()/sd) if sd > 0 else None,
            "minimum_family_advantage": float(values.min()),
            "families_with_positive_advantage": int(np.sum(values > 0)),
            "minimum_leave_one_family_out_mean": float(min(np.delete(values, i).mean() for i in range(len(values))))}


def audit(sae_dir):
    manifest, quality, features = _verify_and_load(sae_dir)
    capture_dir = sae_dir.parent / "q4_activations"
    cm = _json(capture_dir / "manifest.json")
    if sha256_file(capture_dir / "records.jsonl") != cm["files"]["records.jsonl"]:
        raise ValueError("capture metadata hash differs")
    records = [json.loads(line) for line in (capture_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = {r["id"] for r in records}
    if len(ids) != len(records) or ids != set(manifest["training_ids"] + manifest["evaluation_ids"]):
        raise ValueError("capture IDs differ")
    captures = {}
    for r in records:
        if r["split"] != ("train" if r["id"] in manifest["training_ids"] else "eval"):
            raise ValueError("capture split differs")
        path = capture_dir / r["archive"]
        if sha256_file(path) != r["sha256"] or cm["files"].get(r["archive"]) != r["sha256"]:
            raise ValueError("capture hash differs")
        with np.load(path, allow_pickle=False) as a:
            if not np.array_equal(a["token_positions"], r["positions"]):
                raise ValueError("capture row positions differ")
            captures[r["id"]] = {f"{s}__pass_{p}": np.asarray(a[f"{s}__pass_{p}"], dtype=float) for s in SITES for p in (0, 1)}
        if any(x.shape != (32, 3072) or not np.isfinite(x).all() for x in captures[r["id"]].values()):
            raise ValueError("invalid captured array")
    training = [r for r in records if r["split"] == "train"]
    evaluation = [r for r in records if r["split"] == "eval"]
    families = sorted({r["family"] for r in evaluation})
    sites = []
    for si, site in enumerate(SITES):
        entry = quality[site]
        with np.load(sae_dir / entry["dictionary"], allow_pickle=False) as a:
            decoder, bias = np.asarray(a["decoder"], dtype=float), np.asarray(a["pre_bias"], dtype=float)
        if decoder.shape != (6144, 3072) or bias.shape != (3072,) or not np.isfinite(decoder).all() or not np.isfinite(bias).all():
            raise ValueError("invalid dictionary")
        training_variances = []
        for p in (0, 1):
            x = np.concatenate([captures[r["id"]][f"{site}__pass_{p}"] for r in training])
            training_variances.append(float(np.sum((x-x.mean(0))**2)/len(x)))
        if min(training_variances) <= 1e-12:
            raise ValueError("training normalization is degenerate")
        by_id = {r["id"]: r for r in entry["differences"]}
        zs = [[], []]
        for r in evaluation:
            with np.load(sae_dir / by_id[r["id"]]["archive"], allow_pickle=False) as a:
                if not np.array_equal(a["token_positions"], r["positions"]):
                    raise ValueError("feature row positions differ")
                for p in (0, 1):
                    zs[p].append(np.asarray(a[f"pass_{p}"], dtype=float))
        zs = [np.concatenate(z) for z in zs]
        xs = [np.concatenate([captures[r["id"]][f"{site}__pass_{p}"] for r in evaluation]) for p in (0, 1)]
        residuals = [x - (z @ decoder + bias) for x, z in zip(xs, zs)]
        pooled_ev = [1-float(np.sum(r*r))/float(np.sum((x-x.mean(0))**2)) for r, x in zip(residuals, xs)]
        if not np.allclose(pooled_ev, [entry["quality"]["eval"][f"pass_{p}"]["explained_variance"] for p in (0, 1)], atol=1e-5, rtol=0):
            raise ValueError("reconstruction quality does not reproduce")
        row_families = np.repeat([r["family"] for r in evaluation], 32)
        family_quality = []
        for family in families:
            keep = row_families == family
            ev = []
            for p in (0, 1):
                xx, rr = xs[p][keep], residuals[p][keep]
                variance = float(np.sum((xx-xx.mean(0))**2))
                ev.append(1-float(np.sum(rr*rr))/variance if variance > 1e-12 else None)
            family_quality.append({"family": family, "examples": int(keep.sum()/32), "per_pass_ev": ev})
        groups = []
        for p, mask in enumerate(_assign_features(features[site]["train"])):
            controls, _ = matched_groups(features[site]["train"], mask, SEED + si*10 + p)
            family_scores = {f: [] for f in families}
            for group in [mask] + controls:
                contributions = [z[:, group] @ decoder[group] for z in zs]
                for family in families:
                    keep = row_families == family
                    score = [removal_components(r[keep], c[keep], int(keep.sum())*v)["within_pass_variation"]
                             for r, c, v in zip(residuals, contributions, training_variances)]
                    family_scores[family].append(score[p] - score[1-p])
            family_rows = [{"family": f, "selected_effect": scores[0], "random_effects": scores[1:],
                            "advantage_over_random_mean": float(scores[0]-np.mean(scores[1:]))}
                           for f, scores in family_scores.items()]
            groups.append({"assigned_pass": p, "selected_features": int(mask.sum()), "family_scores": family_rows,
                           **summarize_family_advantages([r["advantage_over_random_mean"] for r in family_rows])})
        sites.append({"site": site, "normalizer_training_variance_per_row": training_variances,
                      "pooled_ev": pooled_ev, "family_reconstruction_quality": family_quality, "groups": groups})
    return {"path": str(sae_dir.resolve()), "sae_manifest_sha256": sha256_file(sae_dir / "manifest.json"),
            "capture_manifest_sha256": sha256_file(capture_dir / "manifest.json"),
            "training_families": sorted({r["family"] for r in training}), "evaluation_families": families,
            "families_shared_between_train_and_eval": sorted({r["family"] for r in training} & set(families)),
            "sites": sites}


def build_report(paths):
    body = {"schema": "sch.vpd-q4-sae-family-contributions.v1", "archives": [audit(p) for p in paths],
            "design": {"centering": "within_evaluation_family_and_pass_for_descriptive_variation",
                       "denominator": "family_row_count_times_training_only_variance_per_row",
                       "aggregation": "equal_weight_per_family", "random_groups": 10,
                       "thresholds_fitted_from_evaluation": False},
            "limitations": ["Only four broad source family labels; no estimate of population family variance is established.",
                            "ARC grid and math appear in both training and evaluation.",
                            "Changing centering, denominators, and family weighting means these scores differ from pooled V1.",
                            "Same seeds and example IDs across archives; no independent replication or model-behavior claim."],
            "provenance": {"runner_sha256": sha256_file(Path(__file__)),
                           "group_audit_sha256": sha256_file(Path(__file__).with_name("audit_q4_sae_group_reconstruction_v1.py")),
                           "loader_sha256": sha256_file(Path(__file__).with_name("benchmark_nanbeige_q4_sae_pass_selectivity_v1.py"))},
            "authority": {"operative_admissions": 0, "training_performed": False, "model_inference_performed": False}}
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
        report = build_report([Path(a["path"]) for a in observed["archives"]])
        if observed != report:
            raise ValueError("family contribution report does not reproduce")
    else:
        if not args.sae_directory:
            parser.error("--sae-directory required")
        report = build_report(args.sae_directory)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "groups": [
        {"archive": Path(a["path"]).parent.name, "site": s["site"], **{k: v for k, v in g.items() if k != "family_scores"}}
        for a in report["archives"] for s in a["sites"] for g in s["groups"]]}, indent=2))


if __name__ == "__main__":
    main()
