from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp import assemble_canonical_vpd_effect_shard_v1 as v1
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


SCHEMA = "sch.vpd-canonical-effect-shard-assembly.v2"
ALLOWED_SPLIT_ROLES = {
    "discovery",
    "calibration",
    "cluster_validation",
    "sae_training",
}


def _validate_split_role_extension(
    split_provenance: dict[str, Any],
    *,
    episode_ids: list[str],
    pair_ids: list[str],
    template_groups: list[str],
) -> dict[str, Any]:
    if not isinstance(split_provenance, dict):
        raise ValueError("canonical effect assembly V2 requires split provenance")
    role = str(split_provenance.get("activation_category_split_role", ""))
    if role not in ALLOWED_SPLIT_ROLES:
        raise ValueError("canonical effect assembly V2 split role is forbidden")
    compatibility = copy.deepcopy(split_provenance)
    compatibility["activation_category_split_role"] = "calibration"
    validated = v1._validate_split_provenance(
        compatibility,
        episode_ids=episode_ids,
        pair_ids=pair_ids,
        template_groups=template_groups,
    )
    validated["activation_category_split_role"] = role
    return validated


def assemble_canonical_effect_shard(
    *,
    episodes_by_category: dict[str, list[dict[str, Any]]],
    component_results: dict[str, dict[str, Any]],
    dataset_sha256: str,
    split_provenance: dict[str, Any],
    requested_uncertainty_blocks: int,
    countsketch_dimension: int,
    countsketch_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _, _, all_rows = v1._validate_episode_rows(episodes_by_category)
    verified_split = _validate_split_role_extension(
        split_provenance,
        episode_ids=[row["episode_id"] for row in all_rows],
        pair_ids=[row["pair_id"] for row in all_rows],
        template_groups=[row["template_group"] for row in all_rows],
    )
    compatibility = copy.deepcopy(verified_split)
    compatibility["activation_category_split_role"] = "calibration"
    arrays, predecessor = v1.assemble_canonical_effect_shard(
        episodes_by_category=episodes_by_category,
        component_results=component_results,
        dataset_sha256=dataset_sha256,
        split_provenance=compatibility,
        requested_uncertainty_blocks=requested_uncertainty_blocks,
        countsketch_dimension=countsketch_dimension,
        countsketch_seed=countsketch_seed,
    )
    body = {
        key: copy.deepcopy(value)
        for key, value in predecessor.items()
        if key != "canonical_effect_shard_assembly_id"
    }
    body["schema"] = SCHEMA
    body["status"] = "complete_model_free_role_extended_assembly_v2"
    body["inputs"]["split_provenance"] = verified_split
    body["policy"]["predecessor_array_kernel_schema"] = v1.SCHEMA
    body["policy"]["predecessor_array_kernel_role_independent"] = True
    body["policy"]["sae_training_role_extension"] = (
        verified_split["activation_category_split_role"] == "sae_training"
    )
    body["scientific_boundary"]["split_role_extended_without_model_access"] = True
    body["scientific_boundary"]["sae_dictionary_training_authorized"] = False
    return arrays, {
        **body,
        "canonical_effect_shard_assembly_id": content_sha256(body),
    }


def write_canonical_effect_shard(
    *,
    output_path: Path,
    receipt_path: Path,
    arrays: dict[str, np.ndarray],
    assembly_receipt: dict[str, Any],
) -> dict[str, Any]:
    if assembly_receipt.get("schema") != SCHEMA:
        raise ValueError("canonical effect shard V2 assembly schema differs")
    return v1.write_canonical_effect_shard(
        output_path=output_path,
        receipt_path=receipt_path,
        arrays=arrays,
        assembly_receipt=assembly_receipt,
    )


def verify_canonical_effect_shard(receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assembly = receipt.get("assembly", {})
    if assembly.get("schema") != SCHEMA:
        raise ValueError("canonical effect shard V2 assembly schema differs")
    role = str(
        assembly.get("inputs", {})
        .get("split_provenance", {})
        .get("activation_category_split_role", "")
    )
    if role not in ALLOWED_SPLIT_ROLES:
        raise ValueError("canonical effect shard V2 split role differs")
    if assembly.get("scientific_boundary", {}).get(
        "sae_dictionary_training_authorized"
    ) is not False:
        raise ValueError("canonical effect shard V2 training boundary differs")
    predecessor = v1.verify_canonical_effect_shard(receipt_path)
    return {
        **predecessor,
        "schema": "sch.vpd-canonical-effect-shard-artifact-verification.v2",
        "split_role": role,
        "sae_dictionary_training_authorized": False,
    }
