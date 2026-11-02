"""Post-hoc family and influence controls on the saved coverage diagnostic."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp import audit_q4_sae_training_coverage_v1 as coverage
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

DISTANCE = "mean_matched_distance_squared_over_training_variance"
ERROR = "mean_reconstruction_error_squared_over_training_variance"


def family_controls(examples):
    if not examples or len({r["id"] for r in examples}) != len(examples):
        raise ValueError("nonempty unique example IDs required")
    if any(not isinstance(r["family"], str) or not r["family"] for r in examples):
        raise ValueError("named families required")
    rows = sorted(examples, key=lambda r: r["id"])
    x, y = (np.asarray([r[key] for r in rows], dtype=float) for key in (DISTANCE, ERROR))
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x < 0) or np.any(y < 0):
        raise ValueError("finite nonnegative distance and error required")
    labels = np.asarray([r["family"] for r in rows])
    groups = [(f, np.flatnonzero(labels == f)) for f in sorted(set(labels.tolist()))]
    permutation_count = math.prod(math.factorial(len(indices)) for _, indices in groups)
    if permutation_count > 100_000:
        raise ValueError("exact permutation design exceeds bounded local limit")
    xw, yw = np.empty_like(x), np.empty_like(y)
    family_rows = []
    for family, indices in groups:
        xw[indices], yw[indices] = x[indices]-x[indices].mean(), y[indices]-y[indices].mean()
        family_rows.append({"family": family, "examples": len(indices),
                            "descriptive_pearson_r": coverage.descriptive_correlation(x[indices], y[indices])})
    within_r = coverage.descriptive_correlation(xw, yw)
    total_cov = float(np.dot(x-x.mean(), y-y.mean()))
    within_cov = float(np.dot(xw, yw))
    between_cov = float(np.dot(x-xw-x.mean(), y-yw-y.mean()))
    if not np.isclose(total_cov, within_cov+between_cov, atol=1e-12, rtol=1e-12):
        raise ValueError("covariance decomposition does not reconstruct")
    if within_r is None:
        upper_p = None
    else:
        choices = [list(itertools.permutations(indices.tolist())) for _, indices in groups]
        upper_count = 0
        for assignments in itertools.product(*choices):
            permuted = np.empty_like(yw)
            for (_, indices), source_indices in zip(groups, assignments):
                permuted[indices] = yw[list(source_indices)]
            # Family-preserving permutations leave the correlation denominator fixed.
            upper_count += float(np.dot(xw, permuted)) >= within_cov - 1e-12
        upper_p = upper_count/permutation_count
    return {"evaluation_examples": len(rows), "family_sizes": family_rows,
            "families_with_within_family_variation_possible": sum(len(i) >= 2 for _, i in groups),
            "examples_in_nonsingleton_families": sum(len(i) for _, i in groups if len(i) >= 2),
            "pooled_pearson_r": coverage.descriptive_correlation(x, y),
            "family_centered_pearson_r": within_r,
            "total_centered_cross_product": total_cov, "within_family_cross_product": within_cov,
            "between_family_cross_product": between_cov,
            "within_family_fraction_of_total_cross_product": within_cov/total_cov if abs(total_cov) > 1e-12 else None,
            "exact_within_family_permutations": permutation_count,
            "posthoc_conditional_upper_tail_p": upper_p,
            "leave_one_example_out": [{"omitted_id": r["id"], "pearson_r": coverage.descriptive_correlation(np.delete(x, i), np.delete(y, i))}
                                      for i, r in enumerate(rows)],
            "leave_one_family_out": [{"omitted_family": f, "remaining_examples": int(np.sum(labels != f)),
                                       "pearson_r": coverage.descriptive_correlation(x[labels != f], y[labels != f])}
                                      for f, _ in groups]}


def analyze_source(source):
    if (source.get("schema") != "sch.vpd-q4-sae-training-coverage.v1"
            or source.get("report_id") != content_sha256({k: v for k, v in source.items() if k != "report_id"})):
        raise ValueError("coverage receipt integrity differs")
    rows = []
    keys = set()
    for archive in source["archives"]:
        for site in archive["sites"]:
            for result in site["results"]:
                key = (archive["path"], site["site"], result["phase"], result["pass"])
                if key in keys or result["phase"] not in ("prompt", "generated") or result["pass"] not in (0, 1):
                    raise ValueError("invalid or duplicated archive/site/phase/pass check")
                keys.add(key)
                control = family_controls(result["evaluation_examples"])
                if control["pooled_pearson_r"] != result["descriptive_evaluation_example_pearson_r"]:
                    raise ValueError("source pooled correlation does not reproduce")
                rows.append({"path": key[0], "site": key[1], "phase": key[2], "pass": key[3], **control})
    if len(rows) != 16:
        raise ValueError("this archive-specific audit requires all sixteen phase/pass/site/archive checks")
    for row in rows:
        p = row["posthoc_conditional_upper_tail_p"]
        row["bonferroni_all_16_posthoc_p"] = min(1.0, 16*p) if p is not None else None
    return rows


def build_report(source_path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    results = analyze_source(source)
    if source != coverage.build_report(Path(source["source"]["path"])):
        raise ValueError("coverage source does not reproduce from saved arrays")
    body = {"schema": "sch.vpd-q4-sae-coverage-family-controls.v1", "results": results,
            "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path), "report_id": source["report_id"]},
            "design": {"status": "post-hoc exploratory controls after inspecting the pooled correlations",
                       "family_control": "subtract each observed family mean from both example-mean distance and error",
                       "conditional_null": "enumerate all error assignments within each family, holding distances and family labels fixed",
                       "multiplicity": "Bonferroni across all 16 related archive/site/pass/phase checks",
                       "influence": "descriptive leave-one-example and leave-one-family-out pooled correlations; no selection"},
            "limitations": ["Conditional permutation calibration assumes within-family example exchangeability; repeated templates/content may violate it.",
                            "Post-hoc p-values are descriptive sensitivity results, not confirmatory evidence or launch/admission gates.",
                            "Only ARC (six) and math (two) have multiple evaluation examples; singleton families contribute zero centered residuals.",
                            "Family centering does not control activation magnitude, token/content similarity or other within-family confounding.",
                            "Neither a surviving nor a vanishing association identifies a causal repair or validates SAE compartments."],
            "boundary": {"training_performed": False, "model_inference_performed": False, "paid_compute_launched": False,
                         "fresh_holdout_used": False, "operative_admissions": 0},
            "provenance": {"runner_sha256": sha256_file(Path(__file__)), "coverage_runner_sha256": sha256_file(Path(coverage.__file__))}}
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
            raise ValueError("family-control report does not reproduce")
    else:
        if not args.source:
            parser.error("--source required")
        if args.output.exists():
            raise ValueError("refusing to overwrite existing receipt")
        report = build_report(args.source)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "generated_summary": [
        {k: r[k] for k in ("path", "site", "pass", "family_centered_pearson_r", "within_family_fraction_of_total_cross_product", "bonferroni_all_16_posthoc_p")}
        for r in report["results"] if r["phase"] == "generated"]}, indent=2))


if __name__ == "__main__":
    main()
