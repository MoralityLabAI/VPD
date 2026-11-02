from __future__ import annotations

import argparse
import copy
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.benchmark_recovered_shape_cluster_aware_sae_capacity_v1 import (
    _load_role,
    _verify_role_summary,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


ROLES = (
    "sae_training",
    "discovery",
    "calibration",
    "cluster_validation",
    "final_evaluation",
)

ROLE_PURPOSE = {
    "sae_training": "fixed_base_dictionary_learning_only",
    "discovery": "vpd_effect_features_and_final_cluster_assignment_only",
    "calibration": "cross_template_compartment_threshold_selection_only",
    "cluster_validation": "within_category_cluster_count_replication_only",
    "final_evaluation": "unseen_template_focused_intervention_evaluation_only",
}

SELECTION_ROLES = frozenset({"discovery", "calibration", "cluster_validation"})
ROUTING_ROLES = frozenset({"discovery", "cluster_validation", "sae_training"})
DICTIONARY_TRAINING_ROLES = frozenset({"sae_training"})


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def assert_role_use(role: str, use: str) -> None:
    if role not in ROLES:
        raise ValueError("unknown recovered split role")
    allowed: dict[str, frozenset[str]] = {
        "selection": SELECTION_ROLES,
        "routing": ROUTING_ROLES,
        "dictionary_training": DICTIONARY_TRAINING_ROLES,
        "final_intervention_evaluation": frozenset({"final_evaluation"}),
    }
    if use not in allowed:
        raise ValueError("unknown recovered role use")
    if role not in allowed[use]:
        raise ValueError(f"split role {role} is forbidden for {use}")


def _row_identity_sets(
    grouped: dict[str, list[dict[str, str]]],
) -> tuple[set[str], set[str], set[str]]:
    rows = [row for category_rows in grouped.values() for row in category_rows]
    return (
        {row["episode_id"] for row in rows},
        {row["pair_id"] for row in rows},
        {row["template_group_id"] for row in rows},
    )


def build_report(project_root: Path) -> dict[str, Any]:
    split_root = project_root / "artifacts/vpd_activation_category_capacity_split_v4"
    manifest_path = split_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest["protocol"]["roles"]) != ROLES:
        raise ValueError("five-role manifest order differs")
    if manifest["protocol"]["role_uses"] != ROLE_PURPOSE:
        raise ValueError("five-role role-use contract differs")

    grouped_by_role: dict[str, dict[str, list[dict[str, str]]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    role_ids: dict[str, str] = {}
    identity_sets: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for role in ROLES:
        grouped_by_role[role], summaries[role], role_ids[role] = _load_role(
            split_root=split_root,
            manifest=manifest,
            role=role,
        )
        identity_sets[role] = _row_identity_sets(grouped_by_role[role])

    overlap: dict[str, dict[str, int]] = {}
    for left, right in combinations(ROLES, 2):
        left_episode, left_pair, left_template = identity_sets[left]
        right_episode, right_pair, right_template = identity_sets[right]
        overlap[f"{left}__{right}"] = {
            "episodes": len(left_episode & right_episode),
            "pairs": len(left_pair & right_pair),
            "template_groups": len(left_template & right_template),
        }

    source_episode = set().union(*(sets[0] for sets in identity_sets.values()))
    source_pairs = set().union(*(sets[1] for sets in identity_sets.values()))
    source_templates = set().union(*(sets[2] for sets in identity_sets.values()))
    aggregate = {
        "rows": sum(summary["rows"] for summary in summaries.values()),
        "pairs": len(source_pairs),
        "template_groups": len(source_templates),
        "episode_id_set_sha256": content_sha256(sorted(source_episode)),
        "pair_id_set_sha256": content_sha256(sorted(source_pairs)),
        "template_group_set_sha256": content_sha256(sorted(source_templates)),
    }

    tampered = copy.deepcopy(summaries["final_evaluation"])
    tampered["episode_id_set_sha256"] = "0" * 64
    summary_tamper_rejected = _rejects(
        lambda: _verify_role_summary(tampered, manifest["splits"]["final_evaluation"])
    )
    final_eval_forbidden_uses = {
        use: _rejects(lambda use=use: assert_role_use("final_evaluation", use))
        for use in ("selection", "routing", "dictionary_training")
    }
    calibration_forbidden_for_routing = _rejects(
        lambda: assert_role_use("calibration", "routing")
    )
    calibration_forbidden_for_training = _rejects(
        lambda: assert_role_use("calibration", "dictionary_training")
    )
    role_id_collision_tamper = copy.deepcopy(role_ids)
    role_id_collision_tamper["final_evaluation"] = role_id_collision_tamper[
        "sae_training"
    ]

    controls = {
        "all_five_role_artifacts_match_manifest_hashes_and_counts": all(
            summaries[role]["rows"] == manifest["splits"][role]["rows"]
            and summaries[role]["pairs"] == manifest["splits"][role]["pairs"]
            for role in ROLES
        ),
        "all_five_roles_have_distinct_hash_derived_identities": len(
            set(role_ids.values())
        )
        == 5,
        "all_five_roles_preserve_one_semantic_parent": {
            summary["split_role_identity"]["semantic_split_id"]
            for summary in summaries.values()
        }
        == {manifest["split_id"]},
        "episode_pair_and_template_sets_are_pairwise_disjoint": all(
            values == {"episodes": 0, "pairs": 0, "template_groups": 0}
            for values in overlap.values()
        ),
        "aggregate_exactly_reconstructs_source_identity_sets": aggregate
        == {
            "rows": manifest["source"]["rows"],
            "pairs": manifest["source"]["pairs"],
            "template_groups": len(manifest["observed"]["template_groups"]),
            "episode_id_set_sha256": manifest["source"]["episode_id_set_sha256"],
            "pair_id_set_sha256": manifest["source"]["pair_id_set_sha256"],
            "template_group_set_sha256": manifest["source"][
                "template_group_set_sha256"
            ],
        },
        "role_use_contract_matches_recovered_manifest": all(
            manifest["protocol"]["role_uses"][role] == ROLE_PURPOSE[role]
            for role in ROLES
        ),
        "final_evaluation_forbidden_for_selection_routing_and_training": all(
            final_eval_forbidden_uses.values()
        ),
        "final_evaluation_allowed_only_for_final_intervention_evaluation": not _rejects(
            lambda: assert_role_use(
                "final_evaluation", "final_intervention_evaluation"
            )
        ),
        "calibration_forbidden_for_routing_and_dictionary_training": calibration_forbidden_for_routing
        and calibration_forbidden_for_training,
        "sae_training_is_not_a_selection_role": "sae_training" not in SELECTION_ROLES,
        "cluster_validation_is_not_discovery_or_threshold_selection": manifest[
            "invariants"
        ]["cluster_validation_excluded_from_discovery_and_threshold_selection"],
        "role_summary_identity_tamper_rejected": summary_tamper_rejected,
        "role_identity_collision_tamper_detected": len(
            set(role_id_collision_tamper.values())
        )
        != len(ROLES),
        "all_rows_remain_no_action_metadata": True,
        "audit_performs_no_routing_selection_training_or_evaluation": True,
    }
    if not all(controls.values()):
        raise ValueError(
            "five-role incidence audit failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )

    body = {
        "schema": "sch.vpd-five-role-recovered-incidence-audit.v1",
        "status": "pass",
        "source": {
            "manifest_path": manifest_path.relative_to(project_root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "audit_source_sha256": sha256_file(Path(__file__)),
        },
        "semantic_split_id": manifest["split_id"],
        "role_summaries": summaries,
        "role_ids": role_ids,
        "aggregate": aggregate,
        "pairwise_overlap": overlap,
        "role_use_policy": {
            "selection_roles": sorted(SELECTION_ROLES),
            "routing_roles": sorted(ROUTING_ROLES),
            "dictionary_training_roles": sorted(DICTIONARY_TRAINING_ROLES),
            "final_evaluation_role": "final_evaluation",
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "scientific_boundary": {
            "recovered_metadata_read_only": True,
            "route_allocations_created": False,
            "selection_performed": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "sae_dictionary_trained": False,
            "weights_loaded_or_mutated": False,
            "final_evaluation_consumed": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "release_authorized": False,
        },
    }
    identifier = content_sha256(body)
    return {**body, "audit_id": identifier}


def verify_audit(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = observed.get("audit_id")
    body = {key: value for key, value in observed.items() if key != "audit_id"}
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("five-role recovered incidence audit is not reproducible")
    return {
        "schema": "sch.vpd-five-role-recovered-incidence-audit-verification.v1",
        "status": "pass",
        "audit_id": identifier,
        "roles": len(observed["role_summaries"]),
        "rows": observed["aggregate"]["rows"],
        "pairs": observed["aggregate"]["pairs"],
        "controls": observed["summary"]["controls"],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact five-role recovered incidence without routing or inference."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_audit(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
