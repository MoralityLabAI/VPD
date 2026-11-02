"""Analytic all-required-success bounds; not measured or calibrated Q2 power."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json


def joint_bounds(probabilities):
    ps = list(probabilities)
    if not ps or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in ps):
        raise ValueError("nonempty finite marginal probabilities in [0,1] required")
    return {"dependence_unrestricted_lower": max(0.0, 1-sum(1-p for p in ps)),
            "dependence_unrestricted_upper": min(ps),
            "independent_events_only": math.prod(ps)}


def equal_marginal_requirement(count, target):
    if type(count) is not int or count < 1 or type(target) not in (int, float) or not 0 < target < 1:
        raise ValueError("positive count and target in (0,1) required")
    return {"required_successes": count, "joint_target": target,
            "sufficient_equal_marginal_for_arbitrary_dependence": 1-(1-target)/count,
            "equal_marginal_if_independent": target**(1/count)}


def build_report(source_path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source.get("schema") != "sch.vpd-sae-family-test-power.v1"
            or source.get("report_id") != content_sha256({k: v for k, v in source.items() if k != "report_id"})):
        raise ValueError("marginal power receipt integrity differs")
    if source["design"]["planned_comparisons"] != 18 or source["design"]["one_sided_alpha"] != 0.05/18:
        raise ValueError("source testing design differs")
    scenarios = []
    for row in source["results"]:
        if row["families"] not in (10, 12) or row["excess_over_useful_margin_in_family_sd"] not in (1.0, 1.5, 2.0):
            continue
        for count in (6, 18):
            p = row["estimated_rejection_probability"]
            if p != row["rejections"]/row["simulations"]:
                raise ValueError("source marginal count differs")
            scenarios.append({"families": row["families"], "distribution": row["distribution"],
                              "excess_effect_in_family_sd": row["excess_over_useful_margin_in_family_sd"],
                              "hypothetical_equal_marginal_events": count, "marginal_power_estimate": p,
                              "plug_in_joint_bounds": joint_bounds([p]*count)})
    if len(scenarios) != 24:
        raise ValueError("expected marginal sensitivity grid differs")
    body = {"schema": "sch.vpd-sae-joint-success-power-audit.v1",
            "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path), "report_id": source["report_id"]},
            "exact_illustrations": [{"events": n, "equal_marginal_probability": p, **joint_bounds([p]*n)} for n in (6, 18) for p in (0.8, 0.95, 0.99)],
            "target_requirements": [equal_marginal_requirement(n, 0.8) for n in (6, 18)],
            "source_marginal_sensitivity": scenarios,
            "limitations": ["Bounds are exact for known marginals; source scenario bounds plug in Monte Carlo estimates and are not confidence bounds.",
                            "Six and eighteen are illustrative conjunction sizes, not newly frozen Q2 endpoint counts. Source marginals still use the original alpha 0.05/18.",
                            "Shared training data, evaluation families, schedules and initialization designs make event independence unverified. Product probabilities are conditional illustrations only.",
                            "The source simulation was not rerun here; its receipt hash and stored count arithmetic were checked. It is synthetic, not empirical variance calibration.",
                            "Quality prerequisites, model failures, censoring and missing captures are absent from the source simulation. Adding required gates cannot improve all-required-success probability.",
                            "No useful-effect margin in EV units, prompt non-degradation rule, final multiplicity scope or calibrated joint Q2 power is established."],
            "boundary": {"new_simulation_performed": False, "training_performed": False, "capture_performed": False,
                         "launch_authorized": False, "clustering_admission": False},
            "runner_sha256": sha256_file(Path(__file__))}
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
            raise ValueError("joint-success audit does not reproduce")
    else:
        if not args.source or args.output.exists():
            parser.error("source required; refuse existing output")
        report = build_report(args.source)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "exact_illustrations": report["exact_illustrations"], "target_requirements": report["target_requirements"]}, indent=2))


if __name__ == "__main__":
    main()
