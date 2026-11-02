"""Saved-code training versus evaluation phase quality; no new fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from experiments.authz_mechinterp import audit_q4_sae_phase_quality_v1 as phase_audit
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json


def compare_common_families(training_phases, evaluation_phases):
    comparisons = []
    for phase in ("prompt", "generated"):
        train = next(r for r in training_phases if r["phase"] == phase)
        evaluation = next(r for r in evaluation_phases if r["phase"] == phase)
        train_by = {r["family"]: r for r in train["family_reconstruction_quality"]}
        eval_by = {r["family"]: r for r in evaluation["family_reconstruction_quality"]}
        shared = sorted(set(train_by) & set(eval_by))
        rows = []
        for family in shared:
            t, e = train_by[family], eval_by[family]
            rows.append({"family": family, "training_examples": t["examples"], "evaluation_examples": e["examples"],
                         "training_ev": t["per_pass_ev"], "evaluation_ev": e["per_pass_ev"],
                         "descriptive_training_minus_evaluation_ev": [a-b if a is not None and b is not None else None for a, b in zip(t["per_pass_ev"], e["per_pass_ev"])]})
        comparisons.append({"phase": phase, "common_family_labels": rows,
                            "evaluation_only_families": sorted(set(eval_by)-set(train_by)),
                            "training_only_families": sorted(set(train_by)-set(eval_by))})
    return comparisons


def training_archive(directory):
    manifest, quality, _ = phase_audit._verify_and_load(directory)
    capture_dir = directory.parent / "q4_activations"
    cm, records = phase_audit.load_metadata(capture_dir)
    if set(records) != set(manifest["training_ids"]+manifest["evaluation_ids"]):
        raise ValueError("capture IDs differ")
    if any(r["split"] != ("train" if r["id"] in manifest["training_ids"] else "eval") for r in records.values()):
        raise ValueError("capture splits differ")
    rows = [r for r in records.values() if r["split"] == "train"]
    captures = {r["id"]: phase_audit.load_capture(capture_dir, cm, r) for r in rows}
    families = np.concatenate([[r["family"]]*len(r["positions"]) for r in rows])
    ids = np.concatenate([[r["id"]]*len(r["positions"]) for r in rows])
    phases = np.concatenate([phase_audit.phase_labels(r["positions"], r["prompt_tokens"], len(captures[r["id"]]["token_ids"])) for r in rows])
    sites = []
    for site in phase_audit.SITES:
        entry = quality[site]
        with np.load(directory / entry["dictionary"], allow_pickle=False) as a:
            decoder, bias = np.asarray(a["decoder"], dtype=float), np.asarray(a["pre_bias"], dtype=float)
        if decoder.shape != (6144, 3072) or bias.shape != (3072,) or not np.isfinite(decoder).all() or not np.isfinite(bias).all():
            raise ValueError("invalid dictionary")
        by_id = {r["id"]: r for r in entry["differences"]}
        zs = [[], []]
        for r in rows:
            with np.load(directory / by_id[r["id"]]["archive"], allow_pickle=False) as a:
                if not np.array_equal(a["token_positions"], r["positions"]):
                    raise ValueError("code positions differ")
                for p in (0, 1):
                    zs[p].append(np.asarray(a[f"pass_{p}"], dtype=float))
        xs = [np.concatenate([captures[r["id"]][f"{site}__pass_{p}"].astype(float) for r in rows]) for p in (0, 1)]
        reconstruction = [np.concatenate(z) @ decoder+bias for z in zs]
        mixed_ev = [phase_audit.explained_variance(x, r) for x, r in zip(xs, reconstruction)]
        if not np.allclose(mixed_ev, [entry["quality"]["train"][f"pass_{p}"]["explained_variance"] for p in (0, 1)], atol=1e-5, rtol=0):
            raise ValueError("recorded training EV does not reproduce")
        sites.append({"site": site, "mixed_training_ev": mixed_ev,
                      "phases": phase_audit.phase_quality(xs, reconstruction, families, phases, ids, sorted(set(families)))})
    return sites


def build_report(source_path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source != phase_audit.build_report(Path(source["source"]["path"])):
        raise ValueError("evaluation phase receipt does not reproduce from arrays")
    archives = []
    for archive in source["archives"]:
        train_sites = training_archive(Path(archive["path"]))
        sites = []
        for train in train_sites:
            evaluation = next(s for s in archive["sites"] if s["site"] == train["site"])
            sites.append({"site": train["site"], "training": train, "evaluation": evaluation,
                          "common_family_comparisons": compare_common_families(train["phases"], evaluation["phases"])})
        archives.append({"path": archive["path"], "sites": sites})
    body = {"schema": "sch.vpd-q4-sae-training-phase-quality.v1", "archives": archives,
            "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path), "report_id": source["report_id"]},
            "interpretation": "Training/evaluation EV uses each subset's own variance; gaps are descriptive, not a paired causal or identically distributed test.",
            "limitations": ["Train and evaluation have different family mixes; ARC/math labels are shared but examples differ.",
                            "High training quality does not prove memorization as a mechanism or rule out optimization improvements.",
                            "Only eight training examples and ten evaluation examples; same seed across archives.",
                            "Training quality cannot qualify held-out feature compartments."],
            "boundary": {"training_performed": False, "encoder_recomputed": False, "model_inference_performed": False, "operative_admissions": 0},
            "provenance": {"runner_sha256": sha256_file(Path(__file__)), "phase_audit_sha256": sha256_file(Path(phase_audit.__file__))}}
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
        if report != observed:
            raise ValueError("training/evaluation audit does not reproduce")
    else:
        if not args.source:
            parser.error("--source required")
        report = build_report(args.source)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "archives": len(report["archives"])}, indent=2))


if __name__ == "__main__":
    main()
