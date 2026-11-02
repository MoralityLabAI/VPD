"""Offline finite-catalog bounds. No fitting, model access, or outcome selection.

Caller supplies independent family-block means, conditional on frozen fits and
selection. Distinct block IDs do not prove statistical independence. Hoeffding
plus a union bound covers 2K utility/benign means simultaneously within a track.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _mean_interval(values, n, radius, rho):
    if values is None:
        return {"mean": None, "interval": [0.0, 1.0], "missing_blocks": n}
    if len(values) != n:
        raise ValueError("every score column must match the declared block IDs")
    for value in values:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("scores must be finite numbers in [0,1] or None")
    missing = sum(value is None for value in values)
    if missing:
        return {"mean": None, "interval": [0.0, 1.0], "missing_blocks": missing}
    mean = math.fsum(values) / n
    return {
        "mean": mean,
        "interval": [max(0.0, mean - radius - rho), min(1.0, mean + radius + rho)],
        "missing_blocks": 0,
    }


def _difference(left, right, *, same=False):
    if same:
        return [0.0, 0.0]
    return [left[0] - right[1], left[1] - right[0]]


def evaluate_selection(
    *,
    block_ids: Sequence[str],
    utility: Mapping[str, Sequence[float | None] | None],
    benign: Mapping[str, Sequence[float | None] | None],
    selected: str,
    baseline: str,
    alpha: float = 0.05,
    error_radius: float = 0.0,
) -> dict:
    """Bound a frozen choice over ALL supplied feasible candidates, even missing ones.

    Mapping insertion order is the preregistered catalog order. Supply None for
    wholly unmeasured candidates and None cells for missing scores. Feasibility
    must be settled independently of outcomes before constructing these mappings.
    This helper does not enforce a study's empirical gates or certify independence.
    """
    if not block_ids or any(not isinstance(x, str) or not x for x in block_ids):
        raise ValueError("nonempty string family-block IDs required")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("duplicate family blocks are not independent observations")
    if not utility or set(utility) != set(benign):
        raise ValueError("task and benign tables must retain the same entire catalog")
    if any(not isinstance(x, str) or not x for x in utility):
        raise ValueError("nonempty string candidate IDs required")
    if selected not in utility or baseline not in utility:
        raise ValueError("selected and baseline candidates must be in the catalog")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if not math.isfinite(error_radius) or not 0 <= error_radius <= 1:
        raise ValueError("error radius must be finite and in [0,1]")
    n, k = len(block_ids), len(utility)
    radius = math.sqrt(math.log(4 * k / alpha) / (2 * n))
    intervals = {
        name: {
            "utility": _mean_interval(utility[name], n, radius, error_radius),
            "benign": _mean_interval(benign[name], n, radius, error_radius),
        }
        for name in utility
    }
    q = intervals[selected]
    b = intervals[baseline]
    challengers = []
    for name, row in intervals.items():
        if name == selected:
            continue
        advantage = _difference(row["utility"]["interval"], q["utility"]["interval"])
        challengers.append({"id": name, "advantage_interval": advantage})
    lower = max([0.0] + [x["advantage_interval"][0] for x in challengers])
    upper = max([0.0] + [x["advantage_interval"][1] for x in challengers])
    means = {name: row["utility"]["mean"] for name, row in intervals.items()}
    measured = {name: value for name, value in means.items() if value is not None}
    observed_best = max(measured, key=measured.get) if measured else None
    all_measured = len(measured) == k
    return {
        "scope": "conditional_within_track_simultaneous_bounds_not_empirical_admission",
        "selected": selected,
        "baseline": baseline,
        "catalog": list(utility),
        "independent_family_blocks_assumed": n,
        "alpha": alpha,
        "simultaneous_mean_count": 2 * k,
        "hoeffding_radius": radius,
        "error_radius": error_radius,
        "intervals": intervals,
        "baseline_improvement": {
            "point": None if means[selected] is None or means[baseline] is None
            else means[selected] - means[baseline],
            "interval": _difference(
                q["utility"]["interval"], b["utility"]["interval"], same=selected == baseline
            ),
        },
        "regret": {
            "point": max(measured.values()) - means[selected] if all_measured else None,
            "interval": [lower, upper],
            "observed_best_candidate": observed_best,
            "possible_missed_challengers": [
                x["id"] for x in challengers if x["advantage_interval"][1] > 0
            ],
            "lower_bound_attainers": [
                x["id"] for x in challengers if x["advantage_interval"][0] == lower
            ] if lower > 0 else [selected],
            "upper_bound_attainers": [
                x["id"] for x in challengers if x["advantage_interval"][1] == upper
            ] if upper > 0 else [selected],
        },
        "benign_loss_interval": _difference(
            b["benign"]["interval"], q["benign"]["interval"], same=selected == baseline
        ),
        "challengers": challengers,
    }
