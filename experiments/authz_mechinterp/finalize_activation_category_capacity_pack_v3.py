from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_activation_category_splits_v2 import (
    verify_semantic_split_bundle,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.finalize_activation_category_pack_v2 import (
    REQUIRED_ARTIFACTS,
    build_pack_manifest as build_v2_pack_manifest,
)


SCHEMA = "sch.vpd-activation-category-capacity-pack-manifest.v3"


def build_capacity_pack_manifest(
    pack_dir: Path, *, generation_seed: int, pairs_per_family: int
) -> dict[str, Any]:
    v2_manifest = build_v2_pack_manifest(
        pack_dir,
        generation_seed=generation_seed,
        pairs_per_family=pairs_per_family,
    )
    body = dict(v2_manifest)
    body.pop("pack_id")
    body["schema"] = SCHEMA
    body["status"] = "capacity_ready_design_locally_validated_no_model_outcomes"
    body["predecessor"] = {
        "path": "artifacts/vpd_activation_category_pack_v2/artifact_manifest.json",
        "pack_id": "dc5224d9838d8661764f45342543a4f3ea35152cc16be60dd7d2d11b3589910d",
        "relationship": "identity_disjoint_capacity_scaled_successor",
    }
    body["generation"] = {
        **body["generation"],
        "artifact_version": "activation-category-capacity-v3",
        "identity_namespace": f"semantic-capacity-v3-s{generation_seed}",
        "identity_namespace_includes_seed": True,
        "prompt_variant_rule": (
            "pair_constant_label_independent_non_authority_case_reference_capacity_v3"
        ),
    }
    project_root = Path(__file__).resolve().parents[2]
    v2_generator = body["source_code"]["generator"]
    v3_generator_path = (
        project_root
        / "experiments/authz_mechinterp/generate_activation_category_capacity_pack_v3.py"
    )
    body["source_code"]["generator_parent_v2"] = v2_generator
    body["source_code"]["generator"] = {
        "path": v3_generator_path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(v3_generator_path),
    }
    finalizer_path = Path(__file__)
    body["source_code"]["finalizer"] = {
        "path": finalizer_path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(finalizer_path),
    }
    body["capacity_design"] = {
        "target_profile": "bounded_context_pilot",
        "hidden_size": 4096,
        "expansion": 2,
        "dictionary_features": 8192,
        "top_k": 16,
        "prompt_context_rows_per_episode": 128,
        "minimum_pairs_per_training_template_group": 143,
        "support_floor_is_not_a_trainability_or_mechanism_claim": True,
        "capture_or_training_authorized": False,
    }
    manifest = {**body, "pack_id": content_sha256(body)}
    write_json(pack_dir / "artifact_manifest.json", manifest)
    return manifest


def verify_capacity_pack_manifest(pack_dir: Path) -> dict[str, Any]:
    manifest_path = pack_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    pack_id = str(body.pop("pack_id", ""))
    if content_sha256(body) != pack_id:
        raise ValueError("capacity pack manifest content address mismatch")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected activation-category capacity pack schema")
    for name, entry in manifest["artifacts"].items():
        path = pack_dir / str(entry["path"])
        if sha256_file(path) != entry["sha256"] or path.stat().st_size != int(
            entry["bytes"]
        ):
            raise ValueError(f"capacity pack artifact differs: {name}")
    project_root = Path(__file__).resolve().parents[2]
    for name, entry in manifest["source_code"].items():
        if sha256_file(project_root / str(entry["path"])) != entry["sha256"]:
            raise ValueError(f"capacity pack source differs: {name}")
    split = verify_semantic_split_bundle(
        pack_dir / "splits",
        source_path=pack_dir / REQUIRED_ARTIFACTS["gated_episodes"],
    )
    if split["split_id"] != manifest["split"]["split_id"]:
        raise ValueError("capacity pack split ID differs")
    validation = json.loads(
        (pack_dir / REQUIRED_ARTIFACTS["dataset_validation"]).read_text(
            encoding="utf-8"
        )
    )
    if validation.get("status") != "pass":
        raise ValueError("capacity pack validation status differs")
    if manifest["capacity_design"]["capture_or_training_authorized"] is not False:
        raise ValueError("capacity pack may not authorize capture or training")
    return {
        "schema": "sch.vpd-activation-category-capacity-pack-verification.v3",
        "status": "pass",
        "pack_id": pack_id,
        "split_id": split["split_id"],
        "rows": split["rows"],
        "pairs": split["pairs"],
        "activation_categories": split["activation_categories"],
        "template_groups": split["template_groups"],
        "model_outcomes": 0,
        "exact_partition": True,
        "template_group_sets_pairwise_disjoint": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize or verify the capacity-sized activation-category pack."
    )
    parser.add_argument("--pack-dir", type=Path)
    parser.add_argument("--generation-seed", type=int, default=320829)
    parser.add_argument("--pairs-per-family", type=int, default=143)
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        print(
            json.dumps(
                verify_capacity_pack_manifest(args.verify_output),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.pack_dir is None:
        parser.error("--pack-dir is required unless verifying")
    print(
        json.dumps(
            build_capacity_pack_manifest(
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
