"""Prospective sensitivity analysis for a paired family-level functional test."""
from __future__ import annotations

import argparse
import json
from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

ALPHA = 0.05 / 18
TRIALS = 512
SEED = 260920
FAMILY_COUNTS = (4, 8, 10, 12)
EXCESS_EFFECTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def family_means(values, family_ids):
    values = np.asarray(values, dtype=np.float64)
    ids = np.asarray(family_ids)
    if values.ndim != 1 or ids.shape != values.shape or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("finite nonempty aligned examples and families required")
    return np.asarray([values[ids == label].mean() for label in np.unique(ids)])


def sign_matrix(n):
    if not 1 <= n <= 12:
        raise ValueError("exact enumerator supports 1 through 12 independent families")
    bits = (np.arange(1 << n, dtype=np.uint32)[:, None] >> np.arange(n, dtype=np.uint32)) & 1
    return bits.astype(np.float64) * 2 - 1


def exact_upper_p_values(differences):
    """Each row is one experiment, each column one independent family.

    Under the null, validity requires joint invariance to family sign flips.
    Zero expectation alone is insufficient. The caller subtracts the margin.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty experiment-by-family matrix required")
    signs = sign_matrix(values.shape[1])
    output = []
    for start in range(0, len(values), 128):
        chunk = values[start:start + 128]
        observed = chunk.sum(axis=1)
        null = signs @ chunk.T
        tolerance = 32 * np.finfo(float).eps * np.maximum(1.0, np.abs(chunk).sum(axis=1))
        output.extend(np.mean(null >= observed[None, :] - tolerance[None, :], axis=0).tolist())
    return np.asarray(output)


def wilson_interval(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    radius = z * sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def build_report():
    rng = np.random.default_rng(SEED)
    rows = []
    for n in FAMILY_COUNTS:
        for distribution in ("gaussian", "student_t_3_unit_variance"):
            noise = rng.normal(size=(TRIALS, n)) if distribution == "gaussian" else rng.standard_t(3, size=(TRIALS, n)) / sqrt(3)
            for effect in EXCESS_EFFECTS:
                p = exact_upper_p_values(noise + effect)
                successes = int(np.sum(p <= ALPHA))
                rows.append({"families": n, "distribution": distribution,
                             "excess_over_useful_margin_in_family_sd": effect,
                             "rejections": successes, "simulations": TRIALS,
                             "estimated_rejection_probability": successes/TRIALS,
                             "monte_carlo_wilson_95_interval": wilson_interval(successes, TRIALS),
                             "minimum_exact_p": 1/(1 << n)})
    sensitivity = []
    for n, distribution in product(FAMILY_COUNTS, ("gaussian", "student_t_3_unit_variance")):
        cases = [r for r in rows if r["families"] == n and r["distribution"] == distribution]
        point = [r["excess_over_useful_margin_in_family_sd"] for r in cases if r["estimated_rejection_probability"] >= 0.8]
        supported = [r["excess_over_useful_margin_in_family_sd"] for r in cases if r["monte_carlo_wilson_95_interval"][0] >= 0.8]
        sensitivity.append({"families": n, "distribution": distribution,
                            "smallest_tested_effect_with_point_power_at_least_0_8": min(point) if point else None,
                            "smallest_tested_effect_with_mc_lower_bound_at_least_0_8": min(supported) if supported else None})
    body = {"schema": "sch.vpd-sae-family-test-power.v1",
            "design": {"seed": SEED, "simulations_per_cell": TRIALS, "family_counts": list(FAMILY_COUNTS),
                       "one_sided_alpha": ALPHA, "planned_comparisons": 18,
                       "family_unit": "equal_weight_mean_of_paired_example_effects",
                       "excess_effect_grid_in_family_sd": list(EXCESS_EFFECTS),
                       "margin_in_original_units": None,
                       "mde_is_coarse_grid_sensitivity_not_exact_estimate": True},
            "results": rows, "sensitivity": sensitivity,
            "limitations": ["Synthetic location-family assumptions, not empirical power or measured family variance.",
                            "Sign-flip validity needs independent symmetric family errors conditional on frozen fits and controls.",
                            "Does not simulate training-label permutation, feature reassignment, all-pair success, or quality gate failures.",
                            "Heavy tails are sensitivity cases; independence and symmetry remain unverified on model data.",
                            "Intervals quantify Monte Carlo sampling uncertainty, not uncertainty in the model assumptions."],
            "provenance": {"runner_sha256": sha256_file(Path(__file__)), "kind": "new_synthetic_design_analysis"},
            "authority": {"training_performed": False, "capture_performed": False, "execution_authorized": False}}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    report = build_report()
    if args.verify:
        if json.loads(args.verify.read_text(encoding="utf-8")) != report:
            raise ValueError("family power simulation does not reproduce")
    else:
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "sensitivity": report["sensitivity"]}, indent=2))


if __name__ == "__main__":
    main()
