from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_activation_category_splits import (
    SPLIT_ROLES,
    verify_split_bundle,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    write_json,
)


REQUIRED_ARTIFACTS = {
    "episodes": "episodes.jsonl",
    "gated_episodes": "gated_episodes.jsonl",
    "dataset_validation": "dataset_validation.json",
    "split_manifest": "splits/split_manifest.json",
}


def _artifact_entry(pack_dir: Path, relative_path: str) -> dict[str, Any]:
    path = pack_dir / relative_path
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_pack_manifest(
    pack_dir: Path, *, generation_seed: int, pairs_per_family: int
) -> dict[str, Any]:
    raw_path = pack_dir / REQUIRED_ARTIFACTS["episodes"]
    gated_path = pack_dir / REQUIRED_ARTIFACTS["gated_episodes"]
    validation_path = pack_dir / REQUIRED_ARTIFACTS["dataset_validation"]
    split_dir = pack_dir / "splits"
    split_verification = verify_split_bundle(split_dir, source_path=gated_path)
    raw_rows = load_jsonl(raw_path)
    gated_rows = load_jsonl(gated_path)
    if [row["episode_id"] for row in raw_rows] != [
        row["episode_id"] for row in gated_rows
    ]:
        raise ValueError("raw and gated activation-category episode identities differ")
    by_pair_conditions: dict[str, set[str]] = {}
    for row in raw_rows:
        by_pair_conditions.setdefault(str(row["pair_id"]), set()).add(
            str(row["condition"])
        )
    condition_sets = {tuple(sorted(values)) for values in by_pair_conditions.values()}
    if len(condition_sets) != 1:
        raise ValueError("activation-category pairs have different condition coverage")
    conditions = next(iter(condition_sets))
    if len(raw_rows) != pairs_per_family * 4 * len(conditions):
        raise ValueError("activation-category pack row count differs from design")
    if any(
        bool(row.get("actual_execution")) or bool(row.get("execution_path_exposed"))
        for row in gated_rows
    ):
        raise ValueError("activation-category pack exposes or records execution")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "pass":
        raise ValueError("activation-category dataset validation did not pass")
    if validation.get("input_sha256") != sha256_file(gated_path):
        raise ValueError("activation-category validation input hash differs")
    if int(validation.get("rows", -1)) != len(gated_rows):
        raise ValueError("activation-category validation row count differs")
    if int(validation.get("exact_episode_id_overlap", -1)) != 0 or int(
        validation.get("exact_pair_id_overlap", -1)
    ) != 0 or int(validation.get("exact_message_overlap", -1)) != 0:
        raise ValueError("activation-category validation reports prior overlap")
    byte_witnesses = validation.get("corrupt_prior_byte_witnesses", [])
    if not byte_witnesses or any(
        int(row.get("case_reference_occurrences", -1)) != 0 for row in byte_witnesses
    ):
        raise ValueError("activation-category corrupt-prior witnesses are incomplete")
    split_manifest = json.loads(
        (split_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    project_root = Path(__file__).resolve().parents[2]
    source_files = {
        "generator": project_root
        / "experiments/authz_mechinterp/generate_activation_category_pack.py",
        "candidate_gate": project_root
        / "experiments/authz_mechinterp/run_candidate_gate_dataset.py",
        "validator": project_root
        / "experiments/authz_mechinterp/validate_independent_authority_pack_v2.py",
        "split_builder": project_root
        / "experiments/authz_mechinterp/build_activation_category_splits.py",
    }
    body = {
        "schema": "sch.vpd-activation-category-pack-manifest.v1",
        "status": "locally_validated_no_model_outcomes",
        "generation": {
            "seed": generation_seed,
            "pairs_per_family": pairs_per_family,
            "action_families": 4,
            "conditions_per_pair": len(conditions),
            "conditions": list(conditions),
            "prompt_variant_rule": (
                "pair_constant_label_independent_non_authority_case_reference"
            ),
        },
        "coverage": {
            "rows": len(gated_rows),
            "pairs": len({str(row["pair_id"]) for row in gated_rows}),
            "activation_categories": split_verification["activation_categories"],
            "template_groups": split_verification["template_groups"],
        },
        "split": {
            "split_id": split_verification["split_id"],
            "roles": list(SPLIT_ROLES),
            "rows_per_role": {
                role: int(split_manifest["splits"][role]["rows"])
                for role in SPLIT_ROLES
            },
            "exact_partition_verified": True,
        },
        "prior_overlap": {
            "parseable_prior_paths": validation["prior_paths"],
            "parseable_prior_rows_checked": validation["prior_rows_checked"],
            "corrupt_prior_byte_witnesses": byte_witnesses,
            "episode_id_overlap": 0,
            "pair_id_overlap": 0,
            "exact_message_overlap": 0,
        },
        "artifacts": {
            name: _artifact_entry(pack_dir, relative_path)
            for name, relative_path in REQUIRED_ARTIFACTS.items()
        },
        "source_code": {
            name: {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in source_files.items()
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "model_outcomes_recorded": False,
            "weights_mutated": False,
            "execution_paths_exposed": False,
            "externally_authenticated": False,
            "final_evaluation_is_algorithmically_held_out_not_secret": True,
            "pack_supports_future_study_but_is_not_an_empirical_result": True,
        },
    }
    manifest = {**body, "pack_id": content_sha256(body)}
    write_json(pack_dir / "artifact_manifest.json", manifest)
    return manifest


def verify_pack_manifest(pack_dir: Path) -> dict[str, Any]:
    manifest = json.loads(
        (pack_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    body = dict(manifest)
    pack_id = str(body.pop("pack_id", ""))
    if content_sha256(body) != pack_id:
        raise ValueError("activation-category pack manifest content address mismatch")
    for name, entry in manifest["artifacts"].items():
        path = pack_dir / entry["path"]
        if sha256_file(path) != entry["sha256"] or path.stat().st_size != int(
            entry["bytes"]
        ):
            raise ValueError(f"activation-category pack artifact differs: {name}")
    project_root = Path(__file__).resolve().parents[2]
    for name, entry in manifest["source_code"].items():
        if sha256_file(project_root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"activation-category pack source differs: {name}")
    split = verify_split_bundle(
        pack_dir / "splits",
        source_path=pack_dir / REQUIRED_ARTIFACTS["gated_episodes"],
    )
    if split["split_id"] != manifest["split"]["split_id"]:
        raise ValueError("activation-category pack split ID differs")
    validation = json.loads(
        (pack_dir / REQUIRED_ARTIFACTS["dataset_validation"]).read_text(
            encoding="utf-8"
        )
    )
    if validation.get("status") != "pass":
        raise ValueError("activation-category pack validation status differs")
    return {
        "schema": "sch.vpd-activation-category-pack-verification.v1",
        "status": "pass",
        "pack_id": pack_id,
        "split_id": split["split_id"],
        "rows": split["rows"],
        "pairs": split["pairs"],
        "activation_categories": split["activation_categories"],
        "template_groups": split["template_groups"],
        "model_outcomes": 0,
        "exact_partition": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize or verify the local activation-category episode pack."
    )
    parser.add_argument("--pack-dir", type=Path)
    parser.add_argument("--generation-seed", type=int, default=290829)
    parser.add_argument("--pairs-per-family", type=int, default=24)
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        print(
            json.dumps(
                verify_pack_manifest(args.verify_output), indent=2, sort_keys=True
            )
        )
        return
    if args.pack_dir is None:
        parser.error("--pack-dir is required unless verifying")
    print(
        json.dumps(
            build_pack_manifest(
                args.pack_dir,
                generation_seed=args.generation_seed,
                pairs_per_family=args.pairs_per_family,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
