from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.capture_activations import (
    prompt_window_metadata_rows,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.verify_activation_capture_semantics import (
    validate_prompt_window_metadata_against_episodes,
)


SCHEMA = "sch.vpd-activation-capture-semantic-benchmark.v1"


def fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    episodes = [
        {
            "episode_id": "episode-a",
            "pair_id": "pair-a",
            "condition": "replayed_host_visible",
            "template_group": "calendar_reschedule",
            "prompt_version": "v3",
        },
        {
            "episode_id": "episode-b",
            "pair_id": "pair-b",
            "condition": "valid_visible",
            "template_group": "document_share",
            "prompt_version": "v3",
        },
    ]
    split = {
        "activation_category_split_id": "1" * 64,
        "activation_category_split_role": "sae_training",
    }
    metadata: list[dict[str, Any]] = []
    for episode in episodes:
        metadata.extend(
            prompt_window_metadata_rows(
                episode=episode,
                split_provenance=split,
                model_variant="base",
                prompt_length=8,
                positions={"proposal_start": 7},
                window_size=4,
                first_row_index=len(metadata),
            )
        )
    sampling = {
        "schema": "sch.vpd-sae-training-sampling.v1",
        "mode": "prompt_context_window",
        "generated_tokens": 0,
        "window_end_anchor": "proposal_start",
        "activation_rows_per_episode": 4,
        "token_offset_start": -3,
        "token_offset_end": 0,
        "exact_rows_per_episode": True,
        "episode_count": 2,
        "activation_row_count": 8,
        "episode_id_set_sha256": content_sha256(["episode-a", "episode-b"]),
        "pair_id_set_sha256": content_sha256(["pair-a", "pair-b"]),
    }
    body = {
        "source": {"activation_category_split": split},
        "model": {"model_variant": "base"},
        "capture": {"rows": 8, "sae_training_sampling": sampling},
    }
    return episodes, metadata, {**body, "capture_id": content_sha256(body)}


def _observe(action: Callable[[], Any]) -> str:
    try:
        action()
    except ValueError:
        return "reject"
    return "pass"


def build_benchmark() -> dict[str, Any]:
    episodes, exact, manifest = fixture()
    reordered = exact[4:] + exact[:4]
    relabelled = copy.deepcopy(exact)
    relabelled[0]["activation_category"] = "valid_authority"
    shifted = copy.deepcopy(exact)
    shifted[2]["sampled_token_position"] += 1
    substituted_digest = copy.deepcopy(exact)
    substituted_digest[-1]["source_episode_sha256"] = "0" * 64
    duplicated = exact[:4] + exact[:4]
    misstated_sampling = copy.deepcopy(manifest)
    misstated_sampling["capture"]["sae_training_sampling"][
        "pair_id_set_sha256"
    ] = "f" * 64

    cases = [
        ("exact_source_order_category_and_window_pass", "pass", exact, manifest),
        ("reordered_episode_windows_reject", "reject", reordered, manifest),
        ("relabeled_activation_category_reject", "reject", relabelled, manifest),
        ("shifted_token_coordinate_reject", "reject", shifted, manifest),
        ("substituted_source_episode_digest_reject", "reject", substituted_digest, manifest),
        ("duplicated_episode_window_reject", "reject", duplicated, manifest),
        ("misstated_sampling_identity_reject", "reject", exact, misstated_sampling),
    ]
    scenarios = []
    for scenario_id, expected, metadata, capture in cases:
        observed = _observe(
            lambda metadata=metadata, capture=capture: (
                validate_prompt_window_metadata_against_episodes(
                    metadata=metadata,
                    episodes=episodes,
                    capture_manifest=capture,
                )
            )
        )
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "expected_status": expected,
                "observed_status": observed,
                "status": "pass" if observed == expected else "fail",
            }
        )
    body = {
        "schema": SCHEMA,
        "status": "pass" if all(row["status"] == "pass" for row in scenarios) else "fail",
        "source": {
            "semantic_verifier_sha256": sha256_file(
                Path(__file__).with_name("verify_activation_capture_semantics.py")
            ),
            "benchmark_runner_sha256": sha256_file(Path(__file__)),
        },
        "contract": {
            "episode_window_order_bound_to_source_dataset": True,
            "observable_activation_category_recomputed_from_source_episode": True,
            "token_window_coordinates_reproduced_without_model_inference": True,
            "source_episode_content_address_required_per_metadata_row": True,
            "sampling_identity_receipt_recomputed": True,
        },
        "scenarios": scenarios,
        "scientific_boundary": {
            "synthetic_metadata_only": True,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_cluster_quality_claimed": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = build_benchmark()
    if observed != expected:
        raise ValueError("activation capture semantic benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("activation capture semantic benchmark failed")
    return {
        "schema": "sch.vpd-activation-capture-semantic-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": str(observed["benchmark_id"]),
        "scenarios": len(observed["scenarios"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fail-closed semantic admission for prompt-window captures."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        report = build_benchmark()
        if args.output is not None:
            write_json(args.output, report)
        result = report
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
