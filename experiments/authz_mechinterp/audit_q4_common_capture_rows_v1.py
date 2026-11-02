"""Saved-capture alignment prerequisite; no SAE or model evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

SITES = ("model.layers.11", "model.norm")


def common_prefix_rows(left_tokens, left_positions, right_tokens, right_positions):
    token_arrays = [np.asarray(t) for t in (left_tokens, right_tokens)]
    position_arrays = [np.asarray(p) for p in (left_positions, right_positions)]
    for tokens, positions in zip(token_arrays, position_arrays):
        if tokens.ndim != 1 or not len(tokens) or tokens.dtype.kind not in "iu" or np.any(tokens < 0):
            raise ValueError("nonempty integer token vector required")
        if positions.ndim != 1 or positions.dtype.kind not in "iu" or len(set(positions.tolist())) != len(positions) or np.any(positions < 0) or np.any(positions >= len(tokens)):
            raise ValueError("unique in-bounds integer positions required")
    count = min(len(t) for t in token_arrays)
    differences = np.flatnonzero(token_arrays[0][:count] != token_arrays[1][:count])
    prefix_length = int(differences[0]) if len(differences) else count
    left_map, right_map = ({int(p): i for i, p in enumerate(positions)} for positions in position_arrays)
    same_position = sorted(set(left_map) & set(right_map))
    positions = [p for p in same_position if p < prefix_length]
    return {"shared_prefix_tokens": prefix_length, "position_intersection": same_position,
            "prefix_verified_positions": positions,
            "left_row_indices": [left_map[p] for p in positions],
            "right_row_indices": [right_map[p] for p in positions]}


def load_metadata(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "q4_reasoning_capture_complete" or sha256_file(directory / "records.jsonl") != manifest["files"]["records.jsonl"]:
        raise ValueError("capture metadata integrity differs")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    if len({r["id"] for r in records}) != len(records) or any(r["split"] not in ("train", "eval") for r in records):
        raise ValueError("invalid record inventory")
    return manifest, {r["id"]: r for r in records}


def load_capture(directory, manifest, row):
    path = directory / row["archive"]
    if sha256_file(path) != row["sha256"] or manifest["files"].get(row["archive"]) != row["sha256"]:
        raise ValueError("capture hash differs")
    keys = ["token_ids", "token_positions"] + [f"{s}__pass_{p}" for s in SITES for p in (0, 1)]
    with np.load(path, allow_pickle=False) as a:
        result = {k: np.asarray(a[k]) for k in keys}
    if not np.array_equal(result["token_positions"], row["positions"]) or len(result["token_ids"]) != row["prompt_tokens"]+row["generated_tokens"]:
        raise ValueError("token metadata differs")
    for key in keys[2:]:
        if result[key].shape != (len(row["positions"]), 3072) or not np.isfinite(result[key]).all():
            raise ValueError("invalid activation matrix")
    return result


def build_report(left, right):
    lm, lr = load_metadata(left)
    rm, rr = load_metadata(right)
    if set(lr) != set(rr) or not lr:
        raise ValueError("paired example inventories differ or are empty")
    if lm["generation_contract"]["model_sha256"] != rm["generation_contract"]["model_sha256"]:
        raise ValueError("recorded model revisions differ")
    phase_counts = [{split: {"prompt": 0, "generated": 0} for split in ("train", "eval")} for _ in (0, 1)]
    for records, counts in zip((lr, rr), phase_counts):
        for row in records.values():
            for pos in row["positions"]:
                counts[row["split"]]["prompt" if pos < row["prompt_tokens"] else "generated"] += 1
    examples = []
    for id, lrow in lr.items():
        rrow = rr[id]
        if any(lrow[k] != rrow[k] for k in ("family", "split", "prompt_tokens")):
            raise ValueError("paired record identity differs")
        if lrow["split"] != "eval":
            continue
        la, ra = load_capture(left, lm, lrow), load_capture(right, rm, rrow)
        aligned = common_prefix_rows(la["token_ids"], la["token_positions"], ra["token_ids"], ra["token_positions"])
        positions = aligned["prefix_verified_positions"]
        activation_checks = []
        common_exact = np.ones(len(positions), dtype=bool)
        for site in SITES:
            for p in (0, 1):
                lx = la[f"{site}__pass_{p}"][aligned["left_row_indices"]]
                rx = ra[f"{site}__pass_{p}"][aligned["right_row_indices"]]
                equal = np.all(lx == rx, axis=1)
                common_exact &= equal
                activation_checks.append({"site": site, "pass": p,
                                          "exact_equal_rows": int(equal.sum()),
                                          "exact_equal_generated_rows": int(sum(bool(eq) and pos >= lrow["prompt_tokens"] for eq, pos in zip(equal, positions))),
                                          "max_absolute_difference": float(np.max(np.abs(lx.astype(float)-rx))) if len(lx) else None})
        examples.append({"id": id, "family": lrow["family"], **aligned,
                         "prefix_verified_prompt_rows": sum(p < lrow["prompt_tokens"] for p in positions),
                         "prefix_verified_generated_rows": sum(p >= lrow["prompt_tokens"] for p in positions),
                         "exact_common_positions_all_sites_passes": [pos for pos, eq in zip(positions, common_exact) if eq],
                         "exact_common_generated_positions_all_sites_passes": [pos for pos, eq in zip(positions, common_exact) if eq and pos >= lrow["prompt_tokens"]],
                         "activation_checks": activation_checks})
    body = {"schema": "sch.vpd-q4-common-capture-rows.v1",
            "sources": [{"path": str(d.resolve()), "manifest_sha256": sha256_file(d / "manifest.json"),
                         "recorded_phase_counts_per_pass": counts} for d, counts in zip((left, right), phase_counts)],
            "examples": examples,
            "summary": {"evaluation_examples": len(examples), "prefix_verified_rows_per_pass": sum(len(r["prefix_verified_positions"]) for r in examples),
                        "prefix_verified_prompt_rows_per_pass": sum(r["prefix_verified_prompt_rows"] for r in examples),
                        "prefix_verified_generated_rows_per_pass": sum(r["prefix_verified_generated_rows"] for r in examples),
                        "exact_common_rows_all_sites_passes": sum(len(r["exact_common_positions_all_sites_passes"]) for r in examples),
                        "exact_common_generated_rows_all_sites_passes": sum(len(r["exact_common_generated_positions_all_sites_passes"]) for r in examples),
                        "families_without_exact_common_generated_rows": sorted({r["family"] for r in examples}-{r["family"] for r in examples if r["exact_common_generated_positions_all_sites_passes"]})},
            "limits": ["Exact token-prefix agreement is necessary but does not imply equal numerical activations.",
                       "Position equality or matching current tokens after prefix divergence is insufficient.",
                       "Common-row selection changes the evaluation population; report phase and family coverage.",
                       "Train phase counts are metadata-only; held-out token and activation arrays are hash-checked.",
                       "No saved SAE codes or decoder contributions compared by this audit; no functional admission."],
            "boundary": {"model_inference_performed": False, "training_performed": False, "operative_admissions": 0},
            "runner_sha256": sha256_file(Path(__file__))}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        report = build_report(*(Path(s["path"]) for s in observed["sources"]))
        if report != observed:
            raise ValueError("common-row audit does not reproduce")
    else:
        if not args.left or not args.right:
            parser.error("--left and --right required")
        report = build_report(args.left, args.right)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
