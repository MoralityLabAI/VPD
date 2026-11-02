from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import activation_category
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.finalize_activation_category_pack_v2 import (
    verify_pack_manifest,
)
from experiments.authz_mechinterp.finalize_activation_category_capacity_pack_v3 import (
    SCHEMA as CAPACITY_PACK_SCHEMA,
    verify_capacity_pack_manifest,
)


CAPACITY_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "legacy_anchor_default",
        "expansion": 8,
        "top_k": 32,
        "activation_rows_per_episode": 1,
        "purpose": "audit_only_current_single_anchor_default",
    },
    {
        "id": "bounded_context_pilot",
        "expansion": 2,
        "top_k": 16,
        "activation_rows_per_episode": 128,
        "purpose": "minimum_compartment_pilot_candidate",
    },
    {
        "id": "overcomplete_context_study",
        "expansion": 8,
        "top_k": 32,
        "activation_rows_per_episode": 256,
        "purpose": "overcomplete_compartment_study_candidate",
    },
)


def verify_source_pack(pack_dir: Path) -> dict[str, Any]:
    manifest = json.loads(
        (pack_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") == CAPACITY_PACK_SCHEMA:
        return verify_capacity_pack_manifest(pack_dir)
    return verify_pack_manifest(pack_dir)


def required_observations(
    *,
    dictionary_features: int,
    top_k: int,
    minimum_training_observations_per_feature: int,
    minimum_expected_training_firings_per_feature: int,
    training_fraction: float,
) -> dict[str, int]:
    if dictionary_features < 1 or not 1 <= top_k <= dictionary_features:
        raise ValueError("invalid SAE dictionary width or top-k")
    if minimum_training_observations_per_feature < 1:
        raise ValueError("minimum observations per feature must be positive")
    if minimum_expected_training_firings_per_feature < 1:
        raise ValueError("minimum expected firings per feature must be positive")
    if not 0.0 < training_fraction < 1.0:
        raise ValueError("training fraction must be in (0, 1)")
    by_observation_ratio = (
        dictionary_features * minimum_training_observations_per_feature
    )
    by_expected_firings = math.ceil(
        dictionary_features
        * minimum_expected_training_firings_per_feature
        / top_k
    )
    required_training = max(by_observation_ratio, by_expected_firings)
    return {
        "by_observation_to_feature_ratio": by_observation_ratio,
        "by_expected_feature_firings": by_expected_firings,
        "required_training_observations": required_training,
        "required_total_observations_before_train_validation_split": math.ceil(
            required_training / training_fraction
        ),
    }


def _support_result(
    *,
    episodes: int,
    unique_pairs: int,
    activation_rows_per_episode: int,
    requirements: dict[str, int],
    dictionary_features: int,
    top_k: int,
    minimum_unique_pairs: int,
    training_fraction: float,
) -> dict[str, Any]:
    observations = episodes * activation_rows_per_episode
    training_observations = math.floor(observations * training_fraction)
    validation_observations = observations - training_observations
    observations_per_feature = training_observations / dictionary_features
    expected_firings_per_feature = (
        training_observations * top_k / dictionary_features
    )
    checks = {
        "total_observations": observations
        >= requirements["required_total_observations_before_train_validation_split"],
        "unique_pairs": unique_pairs >= minimum_unique_pairs,
        "nonempty_validation": validation_observations > 0,
    }
    return {
        "status": "ready" if all(checks.values()) else "insufficient",
        "episodes": episodes,
        "unique_pairs": unique_pairs,
        "activation_rows_per_episode": activation_rows_per_episode,
        "total_observations": observations,
        "training_observations": training_observations,
        "validation_observations": validation_observations,
        "training_observations_per_feature": observations_per_feature,
        "expected_training_firings_per_feature_under_uniform_usage": (
            expected_firings_per_feature
        ),
        "checks": checks,
        "additional_observations_required": max(
            0,
            requirements[
                "required_total_observations_before_train_validation_split"
            ]
            - observations,
        ),
        "additional_unique_pairs_required": max(
            0, minimum_unique_pairs - unique_pairs
        ),
    }


def build_capacity_plan(
    pack_dir: Path,
    *,
    split_dir: Path | None = None,
    hidden_size: int = 4096,
    minimum_training_observations_per_feature: int = 8,
    minimum_expected_training_firings_per_feature: int = 32,
    minimum_unique_pairs: int = 32,
    training_fraction: float = 0.90,
) -> dict[str, Any]:
    pack_verification = verify_source_pack(pack_dir)
    pack_manifest_path = pack_dir / "artifact_manifest.json"
    pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    effective_split_dir = split_dir or (pack_dir / "splits")
    split_manifest_path = effective_split_dir / "split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    training_entry = split_manifest["splits"]["sae_training"]
    training_path = effective_split_dir / training_entry["artifact"]["path"]
    rows = load_jsonl(training_path)
    categories = sorted({activation_category(row) for row in rows})
    category_rows = Counter(activation_category(row) for row in rows)
    category_pairs: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        category_pairs[activation_category(row)].add(str(row["pair_id"]))
    training_template_groups = len(training_entry["template_group_counts"])
    total_template_groups = int(pack_manifest["coverage"]["template_groups"])
    conditions_per_pair = int(pack_manifest["generation"]["conditions_per_pair"])
    pairs_per_family_current = int(pack_manifest["generation"]["pairs_per_family"])
    profile_rows: list[dict[str, Any]] = []
    for profile in CAPACITY_PROFILES:
        dictionary_features = hidden_size * int(profile["expansion"])
        top_k = int(profile["top_k"])
        activation_rows_per_episode = int(profile["activation_rows_per_episode"])
        requirements = required_observations(
            dictionary_features=dictionary_features,
            top_k=top_k,
            minimum_training_observations_per_feature=(
                minimum_training_observations_per_feature
            ),
            minimum_expected_training_firings_per_feature=(
                minimum_expected_training_firings_per_feature
            ),
            training_fraction=training_fraction,
        )
        specific_support = {
            category: _support_result(
                episodes=int(category_rows[category]),
                unique_pairs=len(category_pairs[category]),
                activation_rows_per_episode=activation_rows_per_episode,
                requirements=requirements,
                dictionary_features=dictionary_features,
                top_k=top_k,
                minimum_unique_pairs=minimum_unique_pairs,
                training_fraction=training_fraction,
            )
            for category in categories
        }
        balanced_rows_per_category = min(category_rows.values())
        shared_episodes = balanced_rows_per_category * len(categories)
        shared_pairs = len({str(row["pair_id"]) for row in rows})
        shared_support = _support_result(
            episodes=shared_episodes,
            unique_pairs=shared_pairs,
            activation_rows_per_episode=activation_rows_per_episode,
            requirements=requirements,
            dictionary_features=dictionary_features,
            top_k=top_k,
            minimum_unique_pairs=minimum_unique_pairs,
            training_fraction=training_fraction,
        )
        minimum_conditions_per_pair = min(
            category_rows[category] / len(category_pairs[category])
            for category in categories
        )
        required_pairs_per_training_family = max(
            math.ceil(
                requirements[
                    "required_total_observations_before_train_validation_split"
                ]
                / (
                    training_template_groups
                    * minimum_conditions_per_pair
                    * activation_rows_per_episode
                )
            ),
            math.ceil(minimum_unique_pairs / training_template_groups),
        )
        profile_status = (
            "ready"
            if all(row["status"] == "ready" for row in specific_support.values())
            and shared_support["status"] == "ready"
            else "insufficient"
        )
        profile_rows.append(
            {
                **profile,
                "status": profile_status,
                "dictionary_features": dictionary_features,
                "requirements": requirements,
                "specific_compartments": specific_support,
                "shared_compartment": {
                    **shared_support,
                    "balance_rule": "equal_rows_per_activation_category",
                    "balanced_rows_per_category": balanced_rows_per_category,
                },
                "minimum_pairs_per_training_template_group_for_fresh_pack": (
                    required_pairs_per_training_family
                ),
                "estimated_full_pack_pairs": (
                    total_template_groups * required_pairs_per_training_family
                ),
                "estimated_full_pack_rows": (
                    total_template_groups
                    * required_pairs_per_training_family
                    * conditions_per_pair
                ),
                "current_pairs_per_template_group": pairs_per_family_current,
            }
        )
    body = {
        "schema": "sch.vpd-compartment-sae-capacity-plan.v1",
        "status": (
            "ready"
            if any(row["status"] == "ready" for row in profile_rows)
            else "insufficient_current_pack"
        ),
        "source": {
            "pack_id": pack_verification["pack_id"],
            "pack_manifest_sha256": sha256_file(pack_manifest_path),
            **(
                {"source_pack_split_id": pack_verification["split_id"]}
                if split_dir is not None
                else {}
            ),
            "split_id": split_manifest["split_id"],
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "sae_training_artifact_sha256": sha256_file(training_path),
            "planner_sha256": sha256_file(Path(__file__)),
        },
        "model_space": {
            "hidden_size": hidden_size,
            "activation_space": "layer_35_residual_stream",
            "dictionary_scope": "fixed_base_per_compartment",
        },
        "frozen_support_policy": {
            "minimum_training_observations_per_dictionary_feature": (
                minimum_training_observations_per_feature
            ),
            "minimum_expected_training_firings_per_feature_under_uniform_usage": (
                minimum_expected_training_firings_per_feature
            ),
            "minimum_unique_prompt_pairs_per_compartment": minimum_unique_pairs,
            "training_fraction": training_fraction,
            "claim_boundary": (
                "Conservative design-floor arithmetic, not a guarantee of SAE quality. "
                "Empirical reconstruction, sparsity, dead-feature, and held-out behavior "
                "checks remain mandatory."
            ),
        },
        "observed_training_role": {
            "rows": len(rows),
            "unique_pairs": len({str(row["pair_id"]) for row in rows}),
            "template_groups": training_template_groups,
            "category_episode_counts": dict(sorted(category_rows.items())),
            "category_unique_pair_counts": {
                category: len(category_pairs[category]) for category in categories
            },
        },
        "profiles": profile_rows,
        "invariants": {
            "no_profile_declared_ready_without_all_specific_and_shared_support": True,
            "capacity_plan_does_not_authorize_inference_or_training": True,
            "single_anchor_current_default_is_audited_not_grandfathered": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "current_pack_trainability_claimed": False,
        },
    }
    return {**body, "capacity_plan_id": content_sha256(body)}


def verify_capacity_plan(
    path: Path, *, pack_dir: Path, split_dir: Path | None = None
) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    body = dict(plan)
    plan_id = str(body.pop("capacity_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("SAE capacity plan content address mismatch")
    expected = build_capacity_plan(
        pack_dir,
        split_dir=split_dir,
        hidden_size=int(plan["model_space"]["hidden_size"]),
        minimum_training_observations_per_feature=int(
            plan["frozen_support_policy"][
                "minimum_training_observations_per_dictionary_feature"
            ]
        ),
        minimum_expected_training_firings_per_feature=int(
            plan["frozen_support_policy"][
                "minimum_expected_training_firings_per_feature_under_uniform_usage"
            ]
        ),
        minimum_unique_pairs=int(
            plan["frozen_support_policy"][
                "minimum_unique_prompt_pairs_per_compartment"
            ]
        ),
        training_fraction=float(plan["frozen_support_policy"]["training_fraction"]),
    )
    if expected != plan:
        raise ValueError("SAE capacity plan is not reproducible")
    return {
        "schema": "sch.vpd-compartment-sae-capacity-verification.v1",
        "status": "pass",
        "capacity_plan_id": plan_id,
        "plan_status": plan["status"],
        "profiles": len(plan["profiles"]),
        "ready_profiles": sum(row["status"] == "ready" for row in plan["profiles"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or verify per-compartment SAE dictionary capacity support."
    )
    parser.add_argument("--pack-dir", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--hidden-size", type=int, default=4096)
    args = parser.parse_args()
    if args.verify is not None:
        if args.pack_dir is None:
            parser.error("--verify requires --pack-dir")
        print(
            json.dumps(
                verify_capacity_plan(
                    args.verify,
                    pack_dir=args.pack_dir,
                    split_dir=args.split_dir,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.pack_dir is None or args.output is None:
        parser.error("planning requires --pack-dir and --output")
    plan = build_capacity_plan(
        args.pack_dir,
        split_dir=args.split_dir,
        hidden_size=args.hidden_size,
    )
    write_json(args.output, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
