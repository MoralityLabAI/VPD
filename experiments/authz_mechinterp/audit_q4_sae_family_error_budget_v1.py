"""Decompose archived SAE family error; held-out mean correction is oracle-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from experiments.authz_mechinterp.audit_q4_sae_group_reconstruction_v1 import SITES, _json
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import _verify_and_load
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json


def error_budget(x, reconstruction):
    x, reconstruction = np.asarray(x, dtype=float), np.asarray(reconstruction, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or x.shape != reconstruction.shape or not np.isfinite(x).all() or not np.isfinite(reconstruction).all():
        raise ValueError("aligned finite nonempty matrices required")
    residual = x - reconstruction
    total = float(np.sum(residual**2))
    mean_error = float(len(x) * np.sum(residual.mean(0)**2))
    varying_error = float(np.sum((residual-residual.mean(0))**2))
    variance = float(np.sum((x-x.mean(0))**2))
    if not np.isclose(total, mean_error+varying_error, rtol=1e-10, atol=1e-10):
        raise ValueError("error decomposition fails")
    return {"rows": len(x), "sse": total, "mean_offset_sse": mean_error,
            "within_family_residual_sse": varying_error, "within_family_tss": variance,
            "mean_offset_fraction_of_sse": mean_error/total if total else None,
            "raw_ev": 1-total/variance if variance > 1e-12 else None,
            "oracle_mean_corrected_ev": 1-varying_error/variance if variance > 1e-12 else None}


def audit_archive(source):
    sae_dir = Path(source["path"])
    capture_dir = sae_dir.parent / "q4_activations"
    if sha256_file(sae_dir / "manifest.json") != source["sae_manifest_sha256"] or sha256_file(capture_dir / "manifest.json") != source["capture_manifest_sha256"]:
        raise ValueError("source manifests changed")
    manifest, quality, _ = _verify_and_load(sae_dir)
    cm = _json(capture_dir / "manifest.json")
    if sha256_file(capture_dir / "records.jsonl") != cm["files"]["records.jsonl"]:
        raise ValueError("capture record hash differs")
    records = [json.loads(line) for line in (capture_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    if len({r["id"] for r in records}) != len(records) or {r["id"] for r in records} != set(manifest["training_ids"]+manifest["evaluation_ids"]):
        raise ValueError("capture IDs differ")
    if any(r["split"] != ("train" if r["id"] in manifest["training_ids"] else "eval") for r in records):
        raise ValueError("capture split differs")
    rows = [r for r in records if r["split"] == "eval"]
    captures = {}
    for r in rows:
        path = capture_dir / r["archive"]
        if sha256_file(path) != r["sha256"] or cm["files"].get(r["archive"]) != r["sha256"]:
            raise ValueError("capture array hash differs")
        with np.load(path, allow_pickle=False) as a:
            if not np.array_equal(a["token_positions"], r["positions"]):
                raise ValueError("capture positions differ")
            captures[r["id"]] = {f"{s}__pass_{p}": np.asarray(a[f"{s}__pass_{p}"], dtype=float) for s in SITES for p in (0, 1)}
        if any(x.shape != (32, 3072) or not np.isfinite(x).all() for x in captures[r["id"]].values()):
            raise ValueError("invalid captured matrix")
    families = sorted({r["family"] for r in rows})
    if families != sorted(source["evaluation_families"]):
        raise ValueError("family inventory differs")
    labels = np.repeat([r["family"] for r in rows], 32)
    sites = []
    for site in SITES:
        entry = quality[site]
        baseline = next(s for s in source["sites"] if s["site"] == site)
        with np.load(sae_dir / entry["dictionary"], allow_pickle=False) as a:
            decoder, bias = np.asarray(a["decoder"], dtype=float), np.asarray(a["pre_bias"], dtype=float)
        if decoder.shape != (6144, 3072) or bias.shape != (3072,) or not np.isfinite(decoder).all() or not np.isfinite(bias).all():
            raise ValueError("invalid dictionary")
        by_id = {r["id"]: r for r in entry["differences"]}
        zs = [[], []]
        for r in rows:
            with np.load(sae_dir / by_id[r["id"]]["archive"], allow_pickle=False) as a:
                if not np.array_equal(a["token_positions"], r["positions"]):
                    raise ValueError("code positions differ")
                for p in (0, 1):
                    zs[p].append(np.asarray(a[f"pass_{p}"], dtype=float))
        passes = []
        for p in (0, 1):
            x = np.concatenate([captures[r["id"]][f"{site}__pass_{p}"] for r in rows])
            reconstruction = np.concatenate(zs[p]) @ decoder + bias
            pooled = error_budget(x, reconstruction)
            if not np.isclose(pooled["raw_ev"], baseline["pooled_ev"][p], atol=1e-5, rtol=0):
                raise ValueError("pooled EV does not reproduce")
            family_results = []
            for family in families:
                keep = labels == family
                budget = error_budget(x[keep], reconstruction[keep])
                old = next(f for f in baseline["family_reconstruction_quality"] if f["family"] == family)
                if not np.isclose(budget["raw_ev"], old["per_pass_ev"][p], atol=1e-5, rtol=0):
                    raise ValueError("family EV does not reproduce")
                family_results.append({"family": family, "examples": int(keep.sum()/32), **budget})
            within_tss = sum(f["within_family_tss"] for f in family_results)
            between_tss = sum(float((labels == f).sum()*np.sum((x[labels == f].mean(0)-x.mean(0))**2)) for f in families)
            if not np.isclose(pooled["within_family_tss"], within_tss+between_tss, rtol=1e-10, atol=1e-10):
                raise ValueError("variance decomposition fails")
            passes.append({"pass": p, "pooled_raw_ev": pooled["raw_ev"],
                           "between_family_fraction_of_pooled_tss": between_tss/pooled["within_family_tss"],
                           "families": family_results})
        sites.append({"site": site, "passes": passes})
    return {"path": str(sae_dir), "sites": sites}


def build_report(source_path):
    source = _json(source_path)
    if source.get("schema") != "sch.vpd-q4-sae-family-contributions.v1" or source.get("report_id") != content_sha256({k: v for k, v in source.items() if k != "report_id"}):
        raise ValueError("source receipt integrity differs")
    body = {"schema": "sch.vpd-q4-sae-family-error-budget.v1",
            "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path), "report_id": source["report_id"]},
            "archives": [audit_archive(a) for a in source["archives"]],
            "interpretation": "Exploratory algebraic diagnostic. Oracle correction uses held-out residual means, is not deployable, and cannot qualify a fit.",
            "boundary": {"training_performed": False, "encoder_recomputed": False, "model_inference_performed": False, "operative_admissions": 0},
            "provenance": {"runner_sha256": sha256_file(Path(__file__)),
                           "loader_sha256": sha256_file(Path(__file__).with_name("benchmark_nanbeige_q4_sae_pass_selectivity_v1.py"))}}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = _json(args.verify)
        report = build_report(Path(observed["source"]["path"]))
        if report != observed:
            raise ValueError("error budget receipt does not reproduce")
    else:
        if not args.source:
            parser.error("--source required")
        report = build_report(args.source)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "archives": len(report["archives"])}, indent=2))


if __name__ == "__main__":
    main()
