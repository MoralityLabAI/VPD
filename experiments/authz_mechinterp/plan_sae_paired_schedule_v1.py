"""Metadata-only Q2 paired minibatch schedule; never fits weights or loads captures."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

from experiments.authz_mechinterp import plan_sae_trajectory_breadth_v1 as breadth
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

SCHEDULE_SEED = 20260911
UPDATES = 1000
ROWS_PER_PASS_PER_UPDATE = 64
ARMS = ("Q2-N", "Q2-B")


def aligned_slots(plan):
    if set(plan["arms"]) != set(ARMS):
        raise ValueError("exactly Q2-N and Q2-B required")
    mappings, factors = {}, {}
    for arm in ARMS:
        description = plan["arms"][arm]
        roster = description["roster"]
        if (len(roster) != 320 or description["rows_per_pass"] != 320
                or content_sha256(roster) != description["roster_sha256"]):
            raise ValueError("Q2 retained roster budget or hash differs")
        if len({(r["id"], r["position"]) for r in roster}) != len(roster):
            raise ValueError("duplicate retained capture row")
        if any(r["region"] not in breadth.REGIONS or r["phase"] != ("prompt" if r["region"] == "prompt" else "generated") for r in roster):
            raise ValueError("invalid region/phase association")
        counts = Counter((r["family"], r["region"]) for r in roster)
        if len(counts) != 40 or set(counts.values()) != {8} or len({r["family"] for r in roster}) != 10:
            raise ValueError("ten families with eight rows per region required")
        order = sorted(range(len(roster)), key=lambda i: (
            roster[i]["family"], breadth.REGIONS.index(roster[i]["region"]), roster[i]["id"], roster[i]["position"]))
        mappings[arm] = order
        factors[arm] = [(roster[i]["family"], roster[i]["region"], roster[i]["phase"]) for i in order]
    if factors[ARMS[0]] != factors[ARMS[1]]:
        raise ValueError("paired family/region/phase slots differ")
    return mappings, factors[ARMS[0]]


def draw_slots(seed=SCHEDULE_SEED, updates=UPDATES, batch_rows=ROWS_PER_PASS_PER_UPDATE):
    if (type(seed) is not int or not 0 <= seed < 2**64 or type(updates) is not int
            or type(batch_rows) is not int or not 1 <= updates <= UPDATES
            or not 1 <= batch_rows <= ROWS_PER_PASS_PER_UPDATE):
        raise ValueError("bounded integer seed, updates and batch rows required")
    domain_seed = int(content_sha256({"domain": "Q2-paired-optimization-schedule-v1", "seed": seed}), 16)
    generator = random.Random(domain_seed)
    return [[generator.randrange(320) for _ in range(batch_rows)] for _ in range(updates)]


def paired_batch_indices(mappings, slot_batch):
    if not slot_batch or any(type(i) is not int or not 0 <= i < 320 for i in slot_batch):
        raise ValueError("nonempty in-bounds integer slot batch required")
    if set(mappings) != set(ARMS) or any(sorted(mappings[a]) != list(range(320)) for a in ARMS):
        raise ValueError("each slot map must be a bijection of the retained row indices")
    # Both passes use the same retained-row indices; actual capture alignment is a separate prerequisite.
    return {a: {"pass_0": [mappings[a][i] for i in slot_batch], "pass_1": [mappings[a][i] for i in slot_batch]} for a in ARMS}


def schedule_summary(plan, mappings, slots):
    counts, family_counts, region_counts = {}, {}, {}
    for arm in ARMS:
        roster = plan["arms"][arm]["roster"]
        observed = Counter(mappings[arm][slot] for batch in slots for slot in batch)
        counts[arm] = {"unique_rows_drawn_per_pass": len(observed),
                       "minimum_draws_per_retained_row_per_pass": min(observed.get(i, 0) for i in range(320)),
                       "maximum_draws_per_retained_row_per_pass": max(observed.values()),
                       "trajectory_draws_per_pass": dict(sorted(Counter(
                           roster[mappings[arm][slot]]["id"] for batch in slots for slot in batch).items()))}
        family_counts[arm] = dict(sorted(Counter(roster[mappings[arm][slot]]["family"] for batch in slots for slot in batch).items()))
        region_counts[arm] = dict(sorted(Counter(roster[mappings[arm][slot]]["region"] for batch in slots for slot in batch).items()))
    matched = True
    for batch in slots:
        paired = paired_batch_indices(mappings, batch)
        for arm in ARMS:
            if paired[arm]["pass_0"] != paired[arm]["pass_1"]:
                raise ValueError("pass schedule differs")
        for left, right in zip(paired[ARMS[0]]["pass_0"], paired[ARMS[1]]["pass_0"]):
            lrow, rrow = plan["arms"][ARMS[0]]["roster"][left], plan["arms"][ARMS[1]]["roster"][right]
            matched &= all(lrow[k] == rrow[k] for k in ("family", "phase", "region"))
    if not matched or family_counts[ARMS[0]] != family_counts[ARMS[1]] or region_counts[ARMS[0]] != region_counts[ARMS[1]]:
        raise ValueError("paired sampled factors differ")
    region = region_counts[ARMS[0]]
    return {"arms": counts, "matched_family_draws_per_arm_per_pass": family_counts[ARMS[0]],
            "matched_region_draws_per_arm_per_pass": region,
            "matched_phase_draws_per_arm_per_pass": {"prompt": region.get("prompt", 0),
                                                      "generated": sum(v for k, v in region.items() if k != "prompt")},
            "all_draws_factor_matched": matched, "all_draws_pass_index_matched": True}


def load_verified_plan(path):
    source = json.loads(path.read_text(encoding="utf-8"))
    if (source.get("schema") != "sch.vpd-sae-trajectory-breadth-plan.v1"
            or source.get("report_id") != content_sha256({k: v for k, v in source.items() if k != "report_id"})):
        raise ValueError("source row plan integrity differs")
    kind = source["source"]["kind"]
    if kind not in ("synthetic_demo", "metadata_file"):
        raise ValueError("unsupported plan source")
    reconstructed = breadth.build_report(Path(source["source"]["path"]) if kind == "metadata_file" else None)
    if reconstructed != source:
        raise ValueError("source row plan does not reproduce from metadata")
    return source


def build_report(plan_path):
    source = load_verified_plan(plan_path)
    plan = source["plan"]
    mappings, factors = aligned_slots(plan)
    slots = draw_slots()
    body = {"schema": "sch.vpd-sae-paired-schedule.v1", "candidate": "Q2-v1-schedule-draft1",
            "source_plan": {"path": str(plan_path.resolve()), "sha256": sha256_file(plan_path),
                            "report_id": source["report_id"], "kind": source["source"]["kind"]},
            "schedule": {"seed": SCHEDULE_SEED, "updates": UPDATES, "rows_per_pass_per_update": ROWS_PER_PASS_PER_UPDATE,
                         "draws_per_arm_per_pass": UPDATES*ROWS_PER_PASS_PER_UPDATE,
                         "file": "slot_indices.json", "slot_indices_content_sha256": content_sha256(slots),
                         "slot_to_native_roster_index": mappings,
                         "aligned_slot_factors_content_sha256": content_sha256(factors)},
            "summary": schedule_summary(plan, mappings, slots),
            "design": {"sampling": "uniform with replacement over 320 slots; each arm's slot map is a bijection",
                       "coupling": "sort slots by family, region, example ID and position; pair family/region/phase on every draw",
                       "seed_scope": "one independent domain-separated schedule reused across both arms, both passes, both sites and all three initialization seeds",
                       "initialization_seeds": [1707, 250829, 260829],
                       "pass_mapping": "native roster row index is identical on both passes; arrays must later be verified against that roster"},
            "limitations": ["Metadata-only: no captured arrays, initialization tensors, optimizer or trainer were executed or verified.",
                            "Actual phase/family draw totals fluctuate around expectations but are exactly matched between arms.",
                            "Seed effects are conditional on one fixed optimization schedule, not independent schedule replications.",
                            "A synthetic source plan is not a real fresh pack; role completeness, semantics and exposure require separate checks."],
            "boundary": {"training_performed": False, "activation_arrays_loaded": False, "initialization_equality_verified": False,
                         "fresh_pack_verified": False, "trainer_integration_verified": False, "paid_compute_launched": False,
                         "launch_authorized": False, "operative_admissions": 0},
            "provenance": {"runner_sha256": sha256_file(Path(__file__)), "row_planner_sha256": sha256_file(Path(breadth.__file__)),
                           "python_version": sys.version}}
    return {**body, "report_id": content_sha256(body)}, slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-directory", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        report, slots = build_report(Path(observed["source_plan"]["path"]))
        if observed != report:
            raise ValueError("paired schedule report does not reproduce")
        saved = json.loads((args.verify.parent/report["schedule"]["file"]).read_text(encoding="utf-8"))
        if saved != slots:
            raise ValueError("saved draw slots do not reproduce")
    else:
        if not args.plan:
            parser.error("--plan required")
        if args.output_directory.exists():
            raise ValueError("refusing to overwrite existing schedule directory")
        report, slots = build_report(args.plan)
        args.output_directory.mkdir(parents=True, exist_ok=False)
        write_json(args.output_directory/"slot_indices.json", slots)
        write_json(args.output_directory/"report.json", report)
    print(json.dumps({"report_id": report["report_id"], "source_kind": report["source_plan"]["kind"],
                      "updates": report["schedule"]["updates"], "draws_per_arm_per_pass": report["schedule"]["draws_per_arm_per_pass"],
                      "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
