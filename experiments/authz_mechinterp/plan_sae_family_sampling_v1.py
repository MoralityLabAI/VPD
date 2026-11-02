"""Metadata-only sampling control: no torch, activations, fitting, or inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json


def sampling_distribution(records, scheme):
    if scheme not in ("uniform_rows", "equal_family_example_row"):
        raise ValueError("unknown sampling scheme")
    if not records or any(r.get("split") not in ("train", "eval") for r in records):
        raise ValueError("explicit train/eval splits required")
    ids = [r["id"] for r in records]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("unique nonempty IDs required")
    training = [r for r in records if r["split"] == "train"]
    if not training:
        raise ValueError("no training examples")
    for r in training:
        if not isinstance(r["family"], str) or not r["family"]:
            raise ValueError("training family required")
        positions = r["positions"]
        if not isinstance(positions, list) or not positions or any(type(p) is not int or p < 0 for p in positions) or len(set(positions)) != len(positions):
            raise ValueError("unique nonnegative training positions required")
    counts = Counter(r["family"] for r in training)
    roster, probabilities = [], []
    total_rows = sum(len(r["positions"]) for r in training)
    for r in training:
        weight = 1/total_rows if scheme == "uniform_rows" else 1/(len(counts)*counts[r["family"]]*len(r["positions"]))
        for position in r["positions"]:
            roster.append({"id": r["id"], "family": r["family"], "position": position})
            probabilities.append(weight)
    return roster, np.asarray(probabilities, dtype=float)


def sample_indices(probabilities, *, steps, rows_per_pass, seed):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p <= 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("positive normalized probability vector required")
    if any(type(v) is not int or v < 1 for v in (steps, rows_per_pass)) or type(seed) is not int or seed < 0:
        raise ValueError("positive integer budget and nonnegative integer seed required")
    cdf = np.cumsum(p)
    cdf[-1] = 1.
    return np.searchsorted(cdf, np.random.default_rng(seed).random((steps, rows_per_pass)), side="right")


def compare_sampling(records, *, steps=1000, rows_per_pass=64, seed=42):
    arms, distributions = [], []
    for scheme in ("uniform_rows", "equal_family_example_row"):
        roster, p = sampling_distribution(records, scheme)
        indices = sample_indices(p, steps=steps, rows_per_pass=rows_per_pass, seed=seed)
        counts = Counter(roster[int(i)]["family"] for i in indices.ravel())
        families = sorted({r["family"] for r in roster})
        arms.append({"scheme": scheme, "roster_sha256": content_sha256(roster),
                     "schedule_sha256_int64_le": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                     "training_rows": len(roster), "sampled_rows_per_pass": int(indices.size),
                     "family_expected_probability": {f: float(sum(v for r, v in zip(roster, p) if r["family"] == f)) for f in families},
                     "family_sampled_rows_per_pass": {f: counts[f] for f in families}})
        distributions.append(p)
    max_difference = float(np.max(np.abs(distributions[0]-distributions[1])))
    return {"arms": arms, "steps": steps, "rows_per_pass": rows_per_pass, "seed": seed,
            "maximum_row_probability_difference": max_difference,
            "distinct_sampling_distribution": max_difference > 1e-14,
            "identical_dry_run_schedule": arms[0]["schedule_sha256_int64_le"] == arms[1]["schedule_sha256_int64_le"],
            "training_families": sorted({r["family"] for r in records if r["split"] == "train"}),
            "evaluation_families_absent_from_training": sorted({r["family"] for r in records if r["split"] == "eval"}-{r["family"] for r in records if r["split"] == "train"})}


def build_report(capture_directory):
    manifest_path = capture_directory / "manifest.json"
    records_path = capture_directory / "records.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "q4_reasoning_capture_complete" or sha256_file(records_path) != manifest["files"]["records.jsonl"]:
        raise ValueError("capture metadata integrity differs")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    # This is a synthetic roster with the F1 proposed training counts, NOT a fresh pack.
    balanced = [{"id": f"synthetic-family-{f}/example-{e}", "family": f"synthetic-family-{f}",
                 "split": "train", "positions": list(range(32))} for f in range(10) for e in range(4)]
    body = {"schema": "sch.vpd-sae-family-sampling-plan.v1", "candidate": "Q1-sampling-only-draft-v1",
            "source": {"capture_directory": str(capture_directory.resolve()), "manifest_sha256": sha256_file(manifest_path), "records_sha256": sha256_file(records_path)},
            "archived_metadata_dry_run": compare_sampling(records),
            "synthetic_balanced_F1_roster_dry_run": compare_sampling(balanced),
            "design": {"pass_alignment": "Reuse each sampled index vector on both aligned recurrent passes.",
                       "paired_randomness": "Both arms invert their own CDF on the same uniform variates; this does not replay historical randint batches.",
                       "candidate_change": "Only row sampling probabilities may differ; keep initialization and training-derived pre-bias identical across arms.",
                       "schedule_seeds_here": "Seed 42 is a metadata demonstration only, not a fresh training seed or fitted result."},
            "limits": ["No evidence that balancing improves SAE quality.", "No absent family can be supplied by reweighting.",
                       "Equal families, examples, and row counts make these two distributions identical.",
                       "Effective sampling weights are not independent experimental sample sizes.",
                       "Metadata hashes checked; activation arrays intentionally not loaded."],
            "boundary": {"training_performed": False, "model_inference_performed": False, "launch_authorized": False},
            "runner_sha256": sha256_file(Path(__file__))}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        report = build_report(Path(observed["source"]["capture_directory"]))
        if report != observed:
            raise ValueError("sampling plan does not reproduce")
    else:
        if not args.capture_directory:
            parser.error("--capture-directory required")
        report = build_report(args.capture_directory)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "archived_distinct": report["archived_metadata_dry_run"]["distinct_sampling_distribution"],
                      "balanced_F1_distinct": report["synthetic_balanced_F1_roster_dry_run"]["distinct_sampling_distribution"]}, indent=2))


if __name__ == "__main__":
    main()
