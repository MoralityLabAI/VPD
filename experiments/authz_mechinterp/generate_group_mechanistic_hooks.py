from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    receipts = {row["group_id"]: row for row in summary["receipt_checks"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for group in summary["groups"]:
            group_id = str(group["group_id"])
            row = {
                "schema_version": "1.0",
                "hook_id": f"bonsai8b-authz-v4-{group_id}",
                "kind": "vpd",
                "metric_id": "decision_logit_margin",
                "probe_family_id": "v4-target-logit-perturbation",
                "model_id": summary["frozen_identity"]["model_id"],
                "same_model": True,
                "split": "development",
                "causal_intervention": True,
                "intervention": "ablate",
                "target_metric_delta": -float(
                    group["target_invalid_margin_reduction"]
                ),
                "target_improvement_direction": "decrease",
                "minimum_target_effect": float(
                    summary["gate"]["minimum_target_reduction"]
                ),
                "guardrail_delta": float(group["target_valid_pose_logprob_effect"]),
                "max_guardrail_regression": abs(
                    float(summary["gate"]["minimum_valid_pose_logprob_effect"])
                ),
                "matched_random_control_delta": -float(
                    group["control_invalid_margin_reduction"]
                ),
                "control_margin": float(
                    summary["gate"]["minimum_control_advantage"]
                ),
                "module": "prespecified_multi_module_group",
                "component_id": group_id,
                "artifact_uri": str(
                    (args.input_dir / f"group_{group_id}.jsonl").resolve()
                ),
                "sha256": receipts[group_id]["output_sha256"],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
