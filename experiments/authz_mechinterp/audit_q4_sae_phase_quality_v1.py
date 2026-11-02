"""Exploratory phase/family quality of mixed-phase SAEs, using saved codes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from experiments.authz_mechinterp.audit_q4_common_capture_rows_v1 import load_metadata, load_capture, SITES
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import _verify_and_load
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.authz_mechinterp.evaluate_sae_family_quality_v1 import evaluate_quality


def phase_labels(positions, prompt_tokens, total_tokens):
    positions = np.asarray(positions)
    if type(prompt_tokens) is not int or type(total_tokens) is not int or not 0 < prompt_tokens <= total_tokens:
        raise ValueError("valid prompt and total lengths required")
    if positions.ndim != 1 or positions.dtype.kind not in "iu" or np.any(positions < 0) or np.any(positions >= total_tokens) or len(set(positions.tolist())) != len(positions):
        raise ValueError("unique integer in-bounds positions required")
    return np.where(positions < prompt_tokens, "prompt", "generated")


def explained_variance(x, reconstruction):
    x, reconstruction = np.asarray(x, dtype=float), np.asarray(reconstruction, dtype=float)
    if x.ndim != 2 or x.shape != reconstruction.shape or not np.isfinite(x).all() or not np.isfinite(reconstruction).all():
        raise ValueError("aligned finite matrices required")
    if not len(x):
        return None
    variance = float(np.sum((x-x.mean(0))**2))
    return 1-float(np.sum((x-reconstruction)**2))/variance if variance > 1e-12 else None


def phase_quality(xs, reconstructions, families, phases, ids, expected_families):
    if len(xs) != 2 or len(reconstructions) != 2:
        raise ValueError("two passes required")
    families, phases, ids = (np.asarray(v) for v in (families, phases, ids))
    if any(v.ndim != 1 for v in (families, phases, ids)) or not len(ids) or len(set(map(len, (families, phases, ids)))) != 1:
        raise ValueError("nonempty aligned row metadata required")
    if any(x.shape[0] != len(ids) for x in xs+reconstructions) or not set(phases).issubset({"prompt", "generated"}) or not set(families).issubset(set(expected_families)):
        raise ValueError("row counts or labels differ")
    results = []
    for phase in ("prompt", "generated"):
        keep = phases == phase
        evs = [explained_variance(x[keep], r[keep]) for x, r in zip(xs, reconstructions)]
        family_rows = []
        for family in expected_families:
            selected = keep & (families == family)
            if not selected.any():
                continue
            family_rows.append({"family": family, "rows_per_pass": int(selected.sum()),
                                "examples": len(set(ids[selected])),
                                "per_pass_ev": [explained_variance(x[selected], r[selected]) for x, r in zip(xs, reconstructions)]})
        results.append({"phase": phase, "rows_per_pass": int(keep.sum()), "pooled_ev": evs,
                        "family_reconstruction_quality": family_rows,
                        "quality_decision": evaluate_quality(evs, family_rows, expected_families)})
    return results


def audit_archive(source):
    sae_dir = Path(source["path"])
    capture_dir = sae_dir.parent / "q4_activations"
    if sha256_file(sae_dir / "manifest.json") != source["sae_manifest_sha256"] or sha256_file(capture_dir / "manifest.json") != source["capture_manifest_sha256"]:
        raise ValueError("source manifests differ")
    manifest, quality, _ = _verify_and_load(sae_dir)
    cm, records = load_metadata(capture_dir)
    if set(records) != set(manifest["training_ids"]+manifest["evaluation_ids"]):
        raise ValueError("capture IDs differ")
    if any(r["split"] != ("train" if r["id"] in manifest["training_ids"] else "eval") for r in records.values()):
        raise ValueError("capture splits differ")
    rows = [r for r in records.values() if r["split"] == "eval"]
    captures = {r["id"]: load_capture(capture_dir, cm, r) for r in rows}
    families = np.concatenate([[r["family"]]*len(r["positions"]) for r in rows])
    ids = np.concatenate([[r["id"]]*len(r["positions"]) for r in rows])
    phases = np.concatenate([phase_labels(r["positions"], r["prompt_tokens"], len(captures[r["id"]]["token_ids"])) for r in rows])
    if set(families) != set(source["evaluation_families"]):
        raise ValueError("evaluation families differ")
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
        xs = [np.concatenate([captures[r["id"]][f"{site}__pass_{p}"].astype(float) for r in rows]) for p in (0, 1)]
        reconstructed = [np.concatenate(z) @ decoder + bias for z in zs]
        mixed_ev = [explained_variance(x, r) for x, r in zip(xs, reconstructed)]
        if not np.allclose(mixed_ev, baseline["pooled_ev"], atol=1e-5, rtol=0):
            raise ValueError("mixed-phase EV does not reproduce")
        sites.append({"site": site, "mixed_phase_ev": mixed_ev,
                      "phases": phase_quality(xs, reconstructed, families, phases, ids, source["evaluation_families"])})
    return {"path": str(sae_dir), "sites": sites}


def build_report(source_path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema") != "sch.vpd-q4-sae-family-contributions.v1" or source.get("report_id") != content_sha256({k: v for k, v in source.items() if k != "report_id"}):
        raise ValueError("source receipt integrity differs")
    archives = [audit_archive(a) for a in source["archives"]]
    summary = []
    for phase in ("prompt", "generated"):
        decisions = [p["quality_decision"] for a in archives for s in a["sites"] for p in s["phases"] if p["phase"] == phase]
        summary.append({"phase": phase, "fits": len(decisions),
                        "pooled_quality_passes": sum(d["pooled_status"] == "pass" for d in decisions),
                        "all_family_quality_passes": sum(d["quality_prerequisite_pass"] for d in decisions),
                        "family_checks": sum(len(d["families"]) for d in decisions),
                        "family_passes": sum(f["status"] == "pass" for d in decisions for f in d["families"])})
    dependencies = [Path(__file__), Path(__file__).with_name("audit_q4_common_capture_rows_v1.py"),
                    Path(__file__).with_name("benchmark_nanbeige_q4_sae_pass_selectivity_v1.py"),
                    Path(__file__).with_name("evaluate_sae_family_quality_v1.py")]
    body = {"schema": "sch.vpd-q4-sae-phase-quality.v1", "archives": archives, "summary": summary,
            "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path), "report_id": source["report_id"]},
            "design": {"phase_rule": "prompt if absolute position < prompt token count, otherwise generated",
                       "variance": "center observed activations separately inside each scored phase/family subset; keep full uncorrected reconstruction error",
                       "threshold": 0.20, "analysis": "exploratory reanalysis of saved codes from mixed-phase trained SAEs"},
            "limitations": ["No phase-specific dictionary was trained and no pass group was reassigned.",
                            "32 retained windows per example are not all trajectory tokens.",
                            "Tiny, partly training-overlapping families and same-seed archives cannot establish independent replication.",
                            "A phase quality pass alone does not establish functional groups or a split-dictionary advantage."],
            "boundary": {"training_performed": False, "encoder_recomputed": False, "model_inference_performed": False, "operative_admissions": 0},
            "provenance": {p.name: sha256_file(p) for p in dependencies}}
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
            raise ValueError("phase quality receipt does not reproduce")
    else:
        if not args.source:
            parser.error("--source required")
        report = build_report(args.source)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
