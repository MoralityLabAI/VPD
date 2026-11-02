"""Fail-closed reconstruction prerequisite; not a clustering admission test.

The CLI evaluates an existing family-audit receipt. It checks receipt integrity,
not the underlying arrays: run the source audit's --verify separately first.
Applying this rule to archived results is exploratory, not preregistration.
"""
from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


def _names(values):
    if not isinstance(values, list) or not values or any(
        not isinstance(v, str) or not v.strip() for v in values
    ) or len(set(values)) != len(values):
        raise ValueError("expected families must be a nonempty unique list of names")
    return values


def _pair(values, threshold):
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("exactly two pass EV values required")
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, Real)
                             or not math.isfinite(v) or v > 1) for v in values):
        raise ValueError("EV must be finite, at most one, or null for undefined variance")
    if any(v is None for v in values):
        return "unevaluable"
    return "pass" if min(values) >= threshold else "fail"


def evaluate_quality(pooled_ev, family_rows, expected_families, threshold=0.20):
    """Apply a declared all-family prerequisite, preserving every failed family.

    This function deliberately accepts no group-contribution or geometry score.
    Missing declared families are unevaluable; extras/duplicates are input errors.
    Passing quality alone never admits clustering or separate SAE training.
    """
    expected_families = _names(expected_families)
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    if not isinstance(family_rows, list):
        raise ValueError("family rows must be a list")
    observed = {}
    for row in family_rows:
        family = row["family"]
        if family not in expected_families or family in observed:
            raise ValueError("unexpected or duplicate family")
        if type(row["examples"]) is not int or row["examples"] < 1:
            raise ValueError("each observed family needs a positive example count")
        observed[family] = row
    pooled_status = _pair(pooled_ev, threshold)
    decisions = []
    for family in expected_families:
        row = observed.get(family)
        decisions.append({"family": family, "examples": row["examples"] if row else 0,
                          "status": _pair(row["per_pass_ev"], threshold) if row else "unevaluable",
                          "reason": "both_pass_threshold" if row else "missing_expected_family"})
    statuses = [pooled_status] + [r["status"] for r in decisions]
    return {"threshold": threshold, "pooled_status": pooled_status, "families": decisions,
            "quality_prerequisite_pass": all(s == "pass" for s in statuses),
            "failed_families": [r["family"] for r in decisions if r["status"] == "fail"],
            "unevaluable_families": [r["family"] for r in decisions if r["status"] == "unevaluable"],
            "clustering_admission": False}


def evaluate_receipt(receipt, expected_families, threshold=0.20):
    if receipt.get("schema") != "sch.vpd-q4-sae-family-contributions.v1":
        raise ValueError("unsupported source receipt")
    if receipt.get("report_id") != content_sha256({k: v for k, v in receipt.items() if k != "report_id"}):
        raise ValueError("source receipt content hash differs")
    _names(expected_families)
    if not receipt["archives"]:
        raise ValueError("empty archive inventory")
    fits, seen = [], set()
    for archive in receipt["archives"]:
        if set(_names(archive["evaluation_families"])) != set(expected_families):
            raise ValueError("source family inventory differs from declared scope")
        if not archive["sites"]:
            raise ValueError("empty site inventory")
        for site in archive["sites"]:
            key = (archive["path"], site["site"])
            if key in seen:
                raise ValueError("duplicate archive/site fit")
            seen.add(key)
            fits.append({"archive": key[0], "site": key[1], **evaluate_quality(
                site["pooled_ev"], site["family_reconstruction_quality"], expected_families, threshold)})
    return {"source_report_id": receipt["report_id"], "scope": "archived_exploratory_quality_only",
            "source_arrays_verified_by_this_command": False, "fits": fits,
            "summary": {"fits": len(fits),
                        "pooled_passes": sum(f["pooled_status"] == "pass" for f in fits),
                        "all_family_quality_passes": sum(f["quality_prerequisite_pass"] for f in fits),
                        "family_checks": sum(len(f["families"]) for f in fits),
                        "family_passes": sum(r["status"] == "pass" for f in fits for r in f["families"])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-family", required=True, action="append")
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    result = evaluate_receipt(json.loads(args.source.read_text(encoding="utf-8")), args.expected_family, args.threshold)
    print(json.dumps(result, indent=2, allow_nan=False))
    # A scientifically negative result is a successful diagnostic, not a tool error.


if __name__ == "__main__":
    main()
