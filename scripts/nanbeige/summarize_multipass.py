"""Summarize measured pass effects without assigning semantics to SAE features."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def norm(value):
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def summarize(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "calibration_complete":
        raise ValueError("Only completed experiment artifacts can be summarized")
    records = json.loads((root / "effects.json").read_text())
    quality = json.loads((root / "sae_quality.json").read_text())
    examples = json.loads((root / "examples.json").read_text())
    baseline_rows = defaultdict(lambda: [[], []])
    for example in examples:
        if example["split"] != "eval":
            continue
        path = root / manifest["baseline_archives"][example["id"]]
        with np.load(path, allow_pickle=False) as arrays:
            for site in manifest["sites"]:
                for invocation in range(2):
                    baseline_rows[site][invocation].append(arrays[f"{site}__pass_{invocation}"].astype(np.float64))
    baseline_comparisons = {}
    for site, rows in baseline_rows.items():
        first, second = [np.concatenate(values) for values in rows]
        first_norm, second_norm = norm(first), norm(second)
        baseline_comparisons[site] = {"first_norm": first_norm, "second_norm": second_norm,
            "second_to_first_norm_ratio": ratio(second_norm, first_norm),
            "pooled_cosine": ratio(float((first * second).sum()), first_norm * second_norm),
            "pass_difference_norm": norm(second - first), "held_out_token_rows": len(first)}
    measurements = []
    groups = defaultdict(list)
    for record in records:
        path = root / record["archive"]
        if path.parent.resolve() != root.resolve():
            raise ValueError("Effect archive must be directly inside the run directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Effect hash mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as arrays:
            for site in manifest["sites"]:
                for invocation in range(2):
                    for kind in ("h", "z"):
                        key = f"{site}__pass_{invocation}__{kind}"
                        baseline = arrays[f"baseline__{key}"]
                        first = arrays[f"first__{key}__positive_effect"]
                        second = arrays[f"second__{key}__positive_effect"]
                        both = arrays[f"both__{key}__positive_effect"]
                        interaction = arrays[f"interaction__{key}"]
                        base_norm = norm(baseline)
                        row = {
                            "example_id": record["example_id"],
                            "physical_module": record["physical_module"],
                            "component": record["component"], "alpha": record["alpha"],
                            "site": site, "pass": invocation, "space": kind,
                            "baseline_norm": base_norm,
                            "first_norm": norm(first), "second_norm": norm(second),
                            "both_norm": norm(both), "interaction_norm": norm(interaction),
                            "both_relative_to_baseline": ratio(norm(both), base_norm),
                            "interaction_relative_to_additive_effect": ratio(norm(interaction), norm(first + second)),
                            "both_exactly_zero": not bool(np.any(both)),
                            "both_changed_coordinate_fraction": float(np.count_nonzero(both) / both.size),
                        }
                        if kind == "z":
                            magnitude = np.abs(both).mean(axis=0)
                            selected = np.argsort(magnitude)[::-1][:10]
                            row["largest_mean_absolute_feature_changes"] = [
                                {"feature": int(i), "mean_absolute_change": float(magnitude[i])}
                                for i in selected if magnitude[i] > 0]
                            row["top_k_support_change_fraction"] = float(np.mean(
                                (arrays[f"both_plus__{key}"] > 0) != (baseline > 0)))
                        measurements.append(row)
                        if site == "model.norm" and invocation == 1 and kind == "h":
                            groups[(record["physical_module"], record["component"])].append(row)
    pooled = []
    for (module, component), rows in groups.items():
        sums = {name: float(np.sqrt(sum(row[name] ** 2 for row in rows)))
                for name in ("baseline_norm", "first_norm", "second_norm", "both_norm", "interaction_norm")}
        pooled.append({"physical_module": module, "component": component, "examples": len(rows),
                       **sums, "both_relative_to_baseline": ratio(sums["both_norm"], sums["baseline_norm"]),
                       "all_both_effects_exactly_zero": all(r["both_exactly_zero"] for r in rows)})
    output = {"model_kind": manifest["model"]["kind"], "effect_records": len(records),
              "held_out_baseline_pass_comparisons": baseline_comparisons,
              "sae_quality": quality, "final_norm_pass_2_pooled": pooled,
              "measurements": measurements,
              "interpretation_limits": [
                  "Raw finite-coefficient effects; BF16 rounding and nonlinearities can affect interactions.",
                  "A zero stored difference means unresolved at the execution precision, not proven causal absence.",
                  "Feature IDs identify this fitted dictionary only; they have no validated semantic labels.",
                  "This calibration does not establish the tuck hypothesis or Astra internals."]}
    (root / "effect_summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    lines = ["# Measured multipass effects", "",
             f"Source: {output['model_kind']}. {len(records)} held-out component-effect records.", "",
             "Final normalization output, second pass. Norms pool all held-out token rows; "
             "coefficients and exact conditions are recorded in effects.json.", "",
             "| Physical matrix | Component | First-only norm | Second-only norm | Both norm | Interaction norm |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in pooled:
        lines.append(f"| {row['physical_module']} | {row['component']} | " + " | ".join(
            f"{row[name]:.6g}" for name in ("first_norm", "second_norm", "both_norm", "interaction_norm")) + " |")
    lines += ["", "## SAE reconstruction on held-out data", "",
              "| Site | First-pass explained variance | Second-pass explained variance |",
              "| --- | ---: | ---: |"]
    for site, entry in quality.items():
        values = [entry["held_out_quality"][f"pass_{p}"]["explained_variance"] for p in (0, 1)]
        lines.append(f"| {site} | " + " | ".join("undefined" if v is None else f"{v:.6g}" for v in values) + " |")
    lines += ["", "Negative explained variance means worse reconstruction than the held-out mean. "
              "Such feature differences should not receive mechanistic interpretation.", ""]
    lines += ["## Baseline pass comparison on held-out tokens", "",
              "| Site | Second/first activation norm | Pooled cosine |",
              "| --- | ---: | ---: |"]
    for site, entry in baseline_comparisons.items():
        values = [entry["second_to_first_norm_ratio"], entry["pooled_cosine"]]
        lines.append(f"| {site} | " + " | ".join("undefined" if v is None else f"{v:.6g}" for v in values) + " |")
    lines.append("")
    lines += ["- " + item for item in output["interpretation_limits"]]
    (root / "EFFECT_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": str(root / "EFFECT_SUMMARY.md"), "effect_records": len(records),
                      "final_site_groups": len(pooled)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    summarize(parser.parse_args().run_dir.resolve())
