from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.authz_mechinterp.logit_margin_metric import (
    candidate_argmax_valid,
    decision_logit_margin,
    token_logprob,
)


SCHEMA = "1.0"


def row(**values: object) -> dict[str, object]:
    return {"schema_version": SCHEMA, "split": "development", **values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    margin_baseline = decision_logit_margin(1.0, 0.0)
    pose_baseline_logits = [-5.0, 0.0, 0.0]
    observations = [
        row(
            probe_id="invariant-v4-common-offset-01",
            probe_family_id="v4-common-logit-offset",
            metric_id="decision_logit_margin",
            probe_type="invariance",
            expected_behavior="stable",
            baseline_value=margin_baseline,
            staged_value=decision_logit_margin(6.0, 5.0),
            tolerance=0.000001,
        ),
        row(
            probe_id="invariant-v4-common-offset-01",
            probe_family_id="v4-common-logit-offset",
            metric_id="pose_continuation_logprob",
            probe_type="invariance",
            expected_behavior="stable",
            baseline_value=token_logprob(pose_baseline_logits, 0),
            staged_value=token_logprob([0.0, 5.0, 5.0], 0),
            tolerance=0.000001,
        ),
        row(
            probe_id="sensitivity-v4-target-logit-01",
            probe_family_id="v4-target-logit-perturbation",
            metric_id="decision_logit_margin",
            probe_type="sensitivity",
            expected_behavior="increase",
            baseline_value=margin_baseline,
            staged_value=decision_logit_margin(1.5, 0.0),
            required_delta=0.49,
        ),
        row(
            probe_id="sensitivity-v4-target-logit-01",
            probe_family_id="v4-target-logit-perturbation",
            metric_id="pose_continuation_logprob",
            probe_type="sensitivity",
            expected_behavior="increase",
            baseline_value=token_logprob(pose_baseline_logits, 0),
            staged_value=token_logprob([-4.5, 0.0, 0.0], 0),
            required_delta=0.49,
        ),
        row(
            probe_id="monotonic-v4-target-grid-01",
            probe_family_id="v4-target-logit-grid",
            metric_id="decision_logit_margin",
            probe_type="monotonicity",
            expected_behavior="increase",
            baseline_value=margin_baseline,
            staged_value=decision_logit_margin(1.2, 0.0),
            required_delta=0.19,
        ),
        row(
            probe_id="monotonic-v4-target-grid-01",
            probe_family_id="v4-target-logit-grid",
            metric_id="pose_continuation_logprob",
            probe_type="monotonicity",
            expected_behavior="increase",
            baseline_value=token_logprob(pose_baseline_logits, 0),
            staged_value=token_logprob([-4.8, 0.0, 0.0], 0),
            required_delta=0.19,
        ),
        row(
            probe_id="anti-gaming-v4-unrelated-argmax-01",
            probe_family_id="v4-unrelated-token-argmax",
            metric_id="candidate_argmax_valid",
            probe_type="anti_gaming",
            expected_behavior="detect_violation",
            observed_violation=not candidate_argmax_valid([2.0, 1.0, 3.0], 0, 1),
        ),
        row(
            probe_id="clean-control-v4-candidate-argmax-01",
            probe_family_id="v4-clean-candidate-argmax",
            metric_id="candidate_argmax_valid",
            probe_type="clean_control",
            expected_behavior="no_violation",
            observed_violation=not candidate_argmax_valid([2.0, 1.0, 0.0], 0, 1),
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for observation in observations:
            handle.write(json.dumps(observation, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
