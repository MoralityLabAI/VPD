"""Q2 activation-row binding and synthetic consumer check; no model or optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp import plan_sae_trajectory_breadth_v1 as breadth
from experiments.authz_mechinterp import plan_sae_paired_schedule_v1 as schedule
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

SITES = ("model.layers.11", "model.norm")
PASSES = ("pass_0", "pass_1")


def array_digest(values):
    values = np.asarray(values)
    if values.dtype.kind != "f" or values.dtype.itemsize not in (4, 8):
        raise ValueError("float32/float64 capture storage required")
    header = content_sha256({"dtype": values.dtype.str, "shape": list(values.shape)})
    return hashlib.sha256(bytes.fromhex(header)+np.ascontiguousarray(values).tobytes()).hexdigest()


def bind_rows(metadata, manifest, arrays, *, expected_width):
    plan = breadth.build_plan(metadata)
    if type(expected_width) is not int or not 1 <= expected_width <= 3072:
        raise ValueError("explicit bounded expected activation width required")
    if (manifest.get("schema") != "sch.vpd-q2-array-binding-input.v1"
            or manifest["metadata_content_sha256"] != content_sha256(metadata)
            or manifest["sites"] != list(SITES) or manifest["activation_width"] != expected_width):
        raise ValueError("capture binding contract differs")
    training = sorted((r for r in metadata["records"] if r["split"] == "train"), key=lambda r: (r["family"], r["id"]))
    expected_ids = {r["id"] for r in training}
    if set(manifest["records"]) != expected_ids or set(arrays) != expected_ids:
        raise ValueError("exact training-only array inventory required")
    complete = []
    ordered = {s: {p: [] for p in PASSES} for s in SITES}
    for row in training:
        entry = manifest["records"][row["id"]]
        if entry["family"] != row["family"] or entry["split"] != "train":
            raise ValueError("capture family/role identity differs")
        if set(entry["sites"]) != set(SITES) or set(arrays[row["id"]]) != set(SITES):
            raise ValueError("capture site inventory differs")
        positions = []
        for region in breadth.REGIONS:
            start, end = row["regions"][region]
            for position in range(start, end):
                positions.append(position)
                complete.append({"id": row["id"], "family": row["family"], "position": position,
                                 "region": region, "phase": "prompt" if region == "prompt" else "generated"})
        for site in SITES:
            if set(entry["sites"][site]) != set(PASSES) or set(arrays[row["id"]][site]) != set(PASSES):
                raise ValueError("capture pass inventory differs")
            for p in PASSES:
                info = entry["sites"][site][p]
                stored_positions = info["positions"]
                if (not isinstance(stored_positions, list) or any(type(i) is not int for i in stored_positions)
                        or len(stored_positions) != 32 or len(set(stored_positions)) != 32
                        or set(stored_positions) != set(row["positions"])):
                    raise ValueError("pass-specific positions differ from frozen capture window")
                values = np.asarray(arrays[row["id"]][site][p])
                if values.shape != (32, expected_width) or not np.isfinite(values).all():
                    raise ValueError("invalid activation shape or nonfinite values")
                if array_digest(values) != info["array_content_sha256"]:
                    raise ValueError("frozen per-ID/site/pass array hash differs")
                lookup = {position: i for i, position in enumerate(stored_positions)}
                ordered[site][p].append(values[[lookup[position] for position in positions]])
    if (len(complete) != plan["common_preprocessing_pool"]["rows_per_pass"]
            or content_sha256(complete) != plan["common_preprocessing_pool"]["roster_sha256"]):
        raise ValueError("complete preprocessing pool differs")
    pool_index = {(r["id"], r["position"]): i for i, r in enumerate(complete)}
    views, biases = {}, {}
    for site in SITES:
        pool = {p: np.concatenate(ordered[site][p]) for p in PASSES}
        # Compute once from the complete training-only pool, never an arm's retained subset.
        biases[site] = np.mean([pool[p].mean(0, dtype=np.float64) for p in PASSES], axis=0).astype(np.float32)
        biases[site].setflags(write=False)
        views[site] = {}
        for arm in schedule.ARMS:
            indices = [pool_index[(r["id"], r["position"])] for r in plan["arms"][arm]["roster"]]
            views[site][arm] = {p: pool[p][indices].copy() for p in PASSES}
            for p in PASSES:
                views[site][arm][p].setflags(write=False)
    return {"plan": plan, "views": views, "common_pre_bias": biases,
            "complete_pool_rows_per_pass": len(complete)}


def iter_paired_batches(bound, mappings, slots):
    if mappings != schedule.aligned_slots(bound["plan"])[0]:
        raise ValueError("batch maps differ from the bound plan's factor-aligned maps")
    for step, batch in enumerate(slots):
        indices = schedule.paired_batch_indices(mappings, batch)
        yield step, {s: {a: {p: bound["views"][s][a][p][indices[a][p]] for p in PASSES}
                         for a in schedule.ARMS} for s in SITES}


def synthetic_fixture():
    metadata = breadth.synthetic_metadata()
    records, arrays = {}, {}
    for row in metadata["records"]:
        if row["split"] != "train":
            continue
        family_index = int(row["family"].rsplit("-", 1)[1])
        example_index = int(row["id"].rsplit("-", 1)[1])
        entries, values = {}, {}
        for si, site in enumerate(SITES):
            entries[site], values[site] = {}, {}
            for pi, p in enumerate(PASSES):
                x = np.array([[family_index, example_index, si, pi, pos] for pos in row["positions"]], dtype=np.float32)
                values[site][p] = x
                entries[site][p] = {"positions": row["positions"][:], "array_content_sha256": array_digest(x)}
        arrays[row["id"]] = values
        records[row["id"]] = {"family": row["family"], "split": "train", "sites": entries}
    manifest = {"schema": "sch.vpd-q2-array-binding-input.v1", "source_kind": "synthetic_identity_sentinel",
                "metadata_content_sha256": content_sha256(metadata), "activation_width": 5, "sites": list(SITES), "records": records}
    return metadata, manifest, arrays


def build_report(schedule_path):
    source = json.loads(schedule_path.read_text(encoding="utf-8"))
    expected, slots = schedule.build_report(Path(source["source_plan"]["path"]))
    if source != expected or source["source_plan"]["kind"] != "synthetic_demo":
        raise ValueError("reproducible synthetic schedule required for this benchmark")
    if json.loads((schedule_path.parent/source["schedule"]["file"]).read_text(encoding="utf-8")) != slots:
        raise ValueError("exported schedule differs")
    metadata, manifest, arrays = synthetic_fixture()
    bound = bind_rows(metadata, manifest, arrays, expected_width=5)
    if content_sha256(bound["plan"]) != content_sha256(schedule.load_verified_plan(Path(source["source_plan"]["path"]))["plan"]):
        raise ValueError("fixture and schedule plans differ")
    mappings = source["schedule"]["slot_to_native_roster_index"]
    trace = {s: {a: {p: hashlib.sha256() for p in PASSES} for a in schedule.ARMS} for s in SITES}
    batches, draws = 0, 0
    for step, batch in iter_paired_batches(bound, mappings, slots):
        for si, site in enumerate(SITES):
            for arm in schedule.ARMS:
                for pi, p in enumerate(PASSES):
                    indices = [mappings[arm][i] for i in slots[step]]
                    rows = [bound["plan"]["arms"][arm]["roster"][i] for i in indices]
                    oracle = np.array([[int(r["family"].rsplit("-", 1)[1]), int(r["id"].rsplit("-", 1)[1]),
                                        si, pi, r["position"]] for r in rows], dtype=np.float32)
                    if not np.array_equal(batch[site][arm][p], oracle):
                        raise ValueError("consumed activation identity does not match scheduled row")
                    trace[site][arm][p].update(batch[site][arm][p].tobytes())
        batches += 1
        draws += len(slots[step])
    body = {"schema": "sch.vpd-sae-paired-activation-binding-benchmark.v1", "status": "synthetic_binding_verified",
            "source": {"schedule_path": str(schedule_path.resolve()), "schedule_sha256": sha256_file(schedule_path),
                       "schedule_report_id": source["report_id"], "fixture_kind": manifest["source_kind"],
                       "fixture_manifest_content_sha256": content_sha256(manifest)},
            "results": {"consumed_batches": batches, "draws_per_site_arm_pass": draws,
                        "complete_pool_rows_per_pass": bound["complete_pool_rows_per_pass"], "fixture_width": 5,
                        "all_consumed_rows_match_identity_oracle": True,
                        "consumption_trace_sha256": {s: {a: {p: h.hexdigest() for p, h in passes.items()} for a, passes in arms.items()} for s, arms in trace.items()},
                        "common_pre_bias_sha256": {s: array_digest(x) for s, x in bound["common_pre_bias"].items()}},
            "limitations": ["Five-dimensional synthetic identity sentinels, not model activations or a fresh 3072-wide capture.",
                            "Per-array hashes detect changes relative to a frozen producer manifest; they cannot prove the producer originally labeled model hooks/passes correctly.",
                            "Bias is computed once from the complete training pool using float64 accumulation then float32 storage; no model parameter receives it here.",
                            "No model initialization, encoder, decoder, loss, optimizer or provider termination control is exercised."],
            "boundary": {"synthetic_arrays_loaded": True, "real_activation_arrays_loaded": False, "common_preprocessing_computed": True,
                         "training_performed": False, "initialization_equality_verified": False, "trainer_integration_verified": False,
                         "fresh_pack_verified": False, "paid_compute_launched": False, "launch_authorized": False, "operative_admissions": 0},
            "provenance": {"runner_sha256": sha256_file(Path(__file__)), "schedule_runner_sha256": sha256_file(Path(schedule.__file__)),
                           "row_planner_sha256": sha256_file(Path(breadth.__file__)), "numpy_version": np.__version__}}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        report = build_report(Path(observed["source"]["schedule_path"]))
        if observed != report:
            raise ValueError("activation binding report does not reproduce")
    else:
        if not args.schedule:
            parser.error("--schedule required")
        if args.output.exists():
            raise ValueError("refusing to overwrite an existing receipt")
        report = build_report(args.schedule)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "status": report["status"],
                      "consumed_batches": report["results"]["consumed_batches"],
                      "draws_per_site_arm_pass": report["results"]["draws_per_site_arm_pass"]}, indent=2))


if __name__ == "__main__":
    main()
