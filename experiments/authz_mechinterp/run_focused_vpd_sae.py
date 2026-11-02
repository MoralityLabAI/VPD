from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    stable_hash,
    write_json,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import load_effect_archive
from experiments.authz_mechinterp.activation_categories import (
    CATEGORY_PROTOCOLS,
    COMPONENT_COMPARTMENT_PROTOCOLS,
    activation_category,
    category_contract,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import load_packets, require_digest
from experiments.authz_mechinterp.activation_category_split_contract import (
    verify_activation_category_split_role,
)
from experiments.authz_mechinterp.score_sae_features import (
    apply_residualizer,
    encode_topk,
    fit_residualizer,
    permute_within_blocks,
    read_label,
    roc_auc,
)
from experiments.authz_mechinterp.evaluate_compartment_sae_quality import (
    verify_quality_report,
)


INTERVENTION_KINDS = (
    "activation",
    "parameter",
    "matched_norm_component_control",
)
INTERVENTION_COEFFICIENTS = (-1.0, 0.0, 1.0)


def require_finite_array(value: np.ndarray, name: str, *, rank: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != rank:
        raise ValueError(f"{name} must be rank-{rank}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def load_dictionary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "activation_key",
            "decoder",
            "encoder_weight",
            "encoder_bias",
            "pre_bias",
            "top_k",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"SAE dictionary is missing arrays: {sorted(missing)}")
        activation_key = str(np.asarray(archive["activation_key"]).item())
        decoder = require_finite_array(archive["decoder"], "decoder", rank=2)
        encoder_weight = require_finite_array(
            archive["encoder_weight"], "encoder_weight", rank=2
        )
        encoder_bias = require_finite_array(archive["encoder_bias"], "encoder_bias", rank=1)
        pre_bias = require_finite_array(archive["pre_bias"], "pre_bias", rank=1)
        top_k = int(np.asarray(archive["top_k"]).item())
    feature_count, hidden_size = decoder.shape
    if encoder_weight.shape != (feature_count, hidden_size):
        raise ValueError("SAE encoder and decoder shapes differ")
    if encoder_bias.shape != (feature_count,) or pre_bias.shape != (hidden_size,):
        raise ValueError("SAE bias shapes do not match the dictionary")
    if not 0 < top_k <= feature_count:
        raise ValueError("SAE top_k is outside the dictionary width")
    decoder_norms = np.linalg.norm(decoder, axis=1)
    if np.any(decoder_norms <= 1e-8):
        raise ValueError("SAE dictionary contains a zero decoder direction")
    return {
        "activation_key": activation_key,
        "decoder": decoder,
        "encoder_weight": encoder_weight,
        "encoder_bias": encoder_bias,
        "pre_bias": pre_bias,
        "top_k": top_k,
        "feature_count": feature_count,
        "hidden_size": hidden_size,
    }


def rank_columns(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(values.shape, dtype=np.float64)
    for feature_index in range(values.shape[1]):
        column = values[:, feature_index]
        order = np.argsort(column, kind="mergesort")
        sorted_values = column[order]
        start = 0
        while start < len(column):
            end = start + 1
            while end < len(column) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end], feature_index] = 0.5 * (start + 1 + end)
            start = end
    return ranks


def aucs_from_ranks(labels: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return np.full(ranks.shape[1], np.nan, dtype=np.float64)
    rank_sums = ranks[labels].sum(axis=0)
    return (rank_sums - positives * (positives + 1) / 2) / (positives * negatives)


def stable_top_features(strength: np.ndarray, count: int) -> list[int]:
    finite_strength = np.nan_to_num(strength, nan=-1.0)
    return sorted(
        range(len(finite_strength)),
        key=lambda index: (-float(finite_strength[index]), index),
    )[: min(count, len(finite_strength))]


def focused_screen_endpoint(
    *,
    metadata: list[dict[str, Any]],
    label_name: str,
    compartment_category: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Freeze the screening target without leaking a global outcome into a specific compartment."""
    if compartment_category is not None and compartment_category not in (
        COMPONENT_COMPARTMENT_PROTOCOLS
    ):
        raise ValueError("focused SAE screen uses an unknown activation compartment")
    if compartment_category not in (None, "shared_cross_category"):
        labels = np.asarray(
            [
                int(activation_category(row) == compartment_category)
                for row in metadata
            ],
            dtype=np.int8,
        )
        body = {
            "schema": "sch.vpd-focused-sae-screen-endpoint.v1",
            "kind": "observable_activation_category_membership",
            "positive_activation_category": compartment_category,
            "activation_category_contract_sha256": content_sha256(
                category_contract()
            ),
            "model_outcome_label_used_for_selection": False,
            "heldout_intervention_endpoint": "signed_authorization_decision_margin",
        }
    else:
        labels = np.asarray(
            [read_label(row, label_name) for row in metadata], dtype=np.int8
        )
        body = {
            "schema": "sch.vpd-focused-sae-screen-endpoint.v1",
            "kind": "behavior_label",
            "label_name": label_name,
            "activation_compartment": compartment_category,
            "model_outcome_label_used_for_selection": True,
            "heldout_intervention_endpoint": "signed_authorization_decision_margin",
        }
    receipt = {
        **body,
        "rows": len(metadata),
        "positive_rows": int(labels.sum()),
        "negative_rows": int(len(labels) - labels.sum()),
        "endpoint_id": content_sha256(body),
    }
    return labels, receipt


def nested_group_scores(
    *,
    labels: np.ndarray,
    features: np.ndarray,
    covariates: np.ndarray,
    groups: np.ndarray,
    candidate_indices: list[int],
    selection_width: int,
) -> dict[int, dict[str, float]]:
    scores: dict[int, dict[str, float]] = {index: {} for index in candidate_indices}
    candidate_set = set(candidate_indices)
    for group in sorted(set(str(value) for value in groups.tolist())):
        test = groups.astype(str) == group
        train = ~test
        if train.sum() < 4 or test.sum() < 2:
            continue
        if len(set(labels[train].tolist())) < 2 or len(set(labels[test].tolist())) < 2:
            continue
        beta = fit_residualizer(features[train], covariates[train])
        train_residual = apply_residualizer(features[train], covariates[train], beta)
        test_residual = apply_residualizer(features[test], covariates[test], beta)
        train_aucs = aucs_from_ranks(labels[train], rank_columns(train_residual))
        selected = set(stable_top_features(np.abs(train_aucs - 0.5), selection_width))
        for feature_index in sorted(candidate_set & selected):
            test_auc = roc_auc(labels[test], test_residual[:, feature_index])
            if not math.isfinite(test_auc):
                continue
            oriented = test_auc if train_aucs[feature_index] >= 0.5 else 1.0 - test_auc
            scores[feature_index][group] = float(oriented)
    return scores


def behavior_screen(
    *,
    labels: np.ndarray,
    features: np.ndarray,
    covariates: np.ndarray,
    groups: np.ndarray,
    blocks: np.ndarray,
    seed: int,
    permutations: int,
    top_features: int,
) -> list[dict[str, Any]]:
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    beta = fit_residualizer(features, covariates)
    residual = apply_residualizer(features, covariates, beta)
    residual_ranks = rank_columns(residual)
    raw_aucs = aucs_from_ranks(labels, rank_columns(features))
    residual_aucs = aucs_from_ranks(labels, residual_ranks)
    observed_strength = np.abs(residual_aucs - 0.5)
    candidate_indices = stable_top_features(observed_strength, top_features)
    candidate_null = np.empty((permutations, len(candidate_indices)), dtype=np.float64)
    null_max = np.empty(permutations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for permutation_index in range(permutations):
        shuffled = permute_within_blocks(labels, blocks, rng)
        null_aucs = aucs_from_ranks(shuffled, residual_ranks)
        null_strength = np.abs(null_aucs - 0.5)
        candidate_null[permutation_index] = null_strength[candidate_indices]
        null_max[permutation_index] = float(np.nanmax(null_strength))
    nested = nested_group_scores(
        labels=labels,
        features=features,
        covariates=covariates,
        groups=groups,
        candidate_indices=candidate_indices,
        selection_width=top_features,
    )
    rows: list[dict[str, Any]] = []
    for local_index, feature_index in enumerate(candidate_indices):
        strength = float(observed_strength[feature_index])
        feature_null = candidate_null[:, local_index]
        shuffle_p = float((1 + np.sum(feature_null >= strength)) / (1 + permutations))
        max_p = float((1 + np.sum(null_max >= strength)) / (1 + permutations))
        fold_scores = nested[feature_index]
        values = np.asarray(list(fold_scores.values()), dtype=np.float64)
        rows.append(
            {
                "feature_index": feature_index,
                "raw_auc": float(raw_aucs[feature_index]),
                "residualized_auc": float(residual_aucs[feature_index]),
                "shuffle_p": shuffle_p,
                "shuffle_max_stat_p": max_p,
                "cv_held_out_template_auc": fold_scores,
                "cv_mean_oriented_auc": float(values.mean()) if len(values) else 0.5,
                "cv_min_oriented_auc": float(values.min()) if len(values) else 0.5,
                "held_out_prompt_families": len(values),
            }
        )
    return rows


def load_effect_inputs(
    *, effect_archive_path: Path, effect_manifest_path: Path, alignment_key: str
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    manifest = json.loads(effect_manifest_path.read_text(encoding="utf-8"))
    merge_id = str(manifest.pop("merge_id"))
    if content_sha256(manifest) != merge_id:
        raise ValueError("effect merge manifest content address mismatch")
    artifact = manifest["artifacts"]["activation_effects"]
    if sha256_file(effect_archive_path) != str(artifact["sha256"]):
        raise ValueError("effect archive does not match the merge manifest")
    identity = dict(manifest["identity"])
    if str(identity["effect_space_id"]) != alignment_key:
        raise ValueError("effect archive and SAE dictionary use different activation spaces")
    component_ids, delta_h = load_effect_archive(effect_archive_path)
    return component_ids, delta_h, identity


def validate_packet_effect_lineage(
    *, packet: dict[str, Any], effect_archive_path: Path, effect_manifest_path: Path
) -> dict[str, Any]:
    """Bind focused alignment to the exact effects that produced the packet."""
    source = packet.get("source")
    if not isinstance(source, dict):
        raise ValueError("focused SAE packet lacks source provenance")
    expected_archive = source.get("effects_sha256")
    expected_manifest = source.get("effects_merge_manifest_sha256")
    activation_compartment = packet.get("sae_study", {}).get(
        "activation_compartment"
    )
    if expected_archive is None and expected_manifest is None:
        if activation_compartment is not None:
            raise ValueError(
                "category packet lacks packet-bound discovery effect provenance"
            )
        return {
            "schema": "sch.vpd-focused-sae-packet-effect-lineage.v1",
            "status": "unattested_legacy_metadata_pilot",
            "packet_bound_effects_required_for_category_studies": True,
            "empirical_category_claim_allowed": False,
        }
    if not isinstance(expected_archive, str) or len(expected_archive) != 64:
        raise ValueError("focused SAE packet has incomplete discovery effect provenance")
    if not isinstance(expected_manifest, str) or len(expected_manifest) != 64:
        raise ValueError(
            "focused SAE packet has incomplete discovery merge provenance"
        )
    observed_archive = sha256_file(effect_archive_path)
    observed_manifest = sha256_file(effect_manifest_path)
    if observed_archive != expected_archive:
        raise ValueError("focused SAE effect archive differs from its packet lineage")
    if observed_manifest != expected_manifest:
        raise ValueError(
            "focused SAE effect merge manifest differs from its packet lineage"
        )
    return {
        "schema": "sch.vpd-focused-sae-packet-effect-lineage.v1",
        "status": "pass",
        "effect_archive_sha256": observed_archive,
        "effect_merge_manifest_sha256": observed_manifest,
        "packet_bound_effects_required_for_category_studies": True,
        "empirical_category_claim_allowed": activation_compartment is not None,
    }


def validate_capture_manifest(
    *,
    manifest_path: Path,
    activation_capture_path: Path,
    metadata_path: Path,
    alignment_key: str,
    effect_identity: dict[str, Any],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "sch.vpd-activation-capture-manifest.v1":
        raise ValueError("unsupported activation capture manifest schema")
    body = dict(manifest)
    capture_id = str(body.pop("capture_id"))
    if content_sha256(body) != capture_id:
        raise ValueError("activation capture manifest content address mismatch")
    if sha256_file(activation_capture_path) != str(
        manifest["artifacts"]["activations"]["sha256"]
    ):
        raise ValueError("activation capture differs from its manifest")
    if sha256_file(metadata_path) != str(manifest["artifacts"]["metadata"]["sha256"]):
        raise ValueError("activation metadata differs from its manifest")
    model = manifest["model"]
    if model.get("adapter_sha256") is not None:
        raise ValueError("focused SAE dictionary evaluation requires fixed-base activations")
    if str(model["model_id"]) != str(effect_identity["model_id"]) or str(
        model["revision"]
    ) != str(effect_identity["revision"]):
        raise ValueError("activation capture and VPD effects use different base models")
    if alignment_key not in set(manifest["capture"]["activation_keys"]):
        raise ValueError("activation capture manifest omits the alignment key")
    alignment_layer = int(alignment_key.split("__", 1)[0].removeprefix("layer_"))
    if alignment_layer not in {int(value) for value in manifest["capture"]["captured_layers"]}:
        raise ValueError("activation capture manifest omits the alignment layer")
    if not 0 <= alignment_layer < int(manifest["capture"]["n_layers"]):
        raise ValueError("activation capture manifest has an invalid layer count")
    if str(effect_identity["phase"]) not in set(manifest["capture"]["phases"]):
        raise ValueError("activation capture manifest omits the effect phase")
    return manifest


def validate_dictionary_manifest(
    *,
    manifest_path: Path,
    dictionary_path: Path,
    evaluation_activation_path: Path,
    alignment_key: str,
    effect_identity: dict[str, Any],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "sch.vpd-fixed-sae-dictionary-manifest.v1":
        raise ValueError("unsupported fixed SAE dictionary manifest schema")
    body = dict(manifest)
    dictionary_id = str(body.pop("dictionary_id"))
    if content_sha256(body) != dictionary_id:
        raise ValueError("fixed SAE dictionary manifest content address mismatch")
    if sha256_file(dictionary_path) != str(
        manifest["artifacts"]["dictionary"]["sha256"]
    ):
        raise ValueError("fixed SAE dictionary differs from its manifest")
    if str(manifest["dictionary"]["activation_key"]) != alignment_key:
        raise ValueError("fixed SAE dictionary manifest uses a different activation key")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("adapter_sha256") is not None:
        raise ValueError("SAE dictionary must be trained on manifest-verified base activations")
    if str(model["model_id"]) != str(effect_identity["model_id"]) or str(
        model["revision"]
    ) != str(effect_identity["revision"]):
        raise ValueError("SAE dictionary and VPD effects use different base models")
    source = manifest["source"]
    require_digest(source.get("capture_manifest_sha256"), "dictionary capture manifest")
    training_capture_sha256 = require_digest(
        source.get("activation_capture_sha256"), "dictionary training activation capture"
    )
    if training_capture_sha256 == sha256_file(evaluation_activation_path):
        raise ValueError("SAE dictionary training and focused evaluation captures must differ")
    compartment = manifest["dictionary"].get("activation_compartment")
    if compartment is not None:
        if not isinstance(compartment, dict):
            raise ValueError("SAE dictionary activation compartment is malformed")
        capacity = source.get("capacity_authorization")
        if not isinstance(capacity, dict) or capacity.get("status") != "pass":
            raise ValueError(
                "compartment SAE dictionary lacks passed capacity authorization"
            )
        require_digest(
            source.get("capacity_plan_sha256"), "dictionary capacity plan"
        )
        require_digest(capacity.get("capacity_plan_id"), "dictionary capacity plan ID")
        if not str(capacity.get("profile_id", "")):
            raise ValueError("compartment SAE dictionary lacks a capacity profile")
        if str(capacity.get("activation_category")) != str(
            compartment.get("activation_category")
        ):
            raise ValueError(
                "dictionary capacity receipt uses a different activation compartment"
            )
        if compartment.get("capacity_authorization") != capacity:
            raise ValueError(
                "dictionary compartment and source capacity receipts differ"
            )
    return manifest


def validate_compartment_quality_report(
    *,
    report_path: Path,
    dictionary_path: Path,
    dictionary_manifest_path: Path,
    dictionary_manifest: dict[str, Any],
) -> dict[str, Any]:
    report = verify_quality_report(report_path, require_pass=True)
    dictionary_id = str(dictionary_manifest["dictionary_id"])
    sources = {
        str(row["dictionary_id"]): row for row in report["dictionaries"]
    }
    source = sources.get(dictionary_id)
    if source is None:
        raise ValueError("quality report omits the selected SAE dictionary")
    if str(source.get("dictionary_manifest_sha256")) != sha256_file(
        dictionary_manifest_path
    ):
        raise ValueError("quality report binds a different dictionary manifest")
    if str(source.get("dictionary_sha256")) != sha256_file(dictionary_path):
        raise ValueError("quality report binds a different dictionary artifact")
    compartment = dictionary_manifest["dictionary"]["activation_compartment"]
    category = str(compartment["activation_category"])
    if str(source.get("compartment")) != category:
        raise ValueError("quality report uses a different activation compartment")
    if source.get("capacity_authorization") != dictionary_manifest["source"].get(
        "capacity_authorization"
    ):
        raise ValueError("quality report uses a different capacity authorization")
    dictionary_split = dictionary_manifest["source"].get(
        "activation_category_split"
    )
    quality_split = report["source"]["activation_category_split"]
    if not isinstance(dictionary_split, dict) or str(
        dictionary_split.get("activation_category_split_id")
    ) != str(quality_split.get("activation_category_split_id")):
        raise ValueError("quality report uses a different activation-category split")
    result = report["result"]
    result_rows = {
        str(row["dictionary_id"]): row for row in result["dictionary_results"]
    }
    if result_rows[dictionary_id].get("status") != "pass":
        raise ValueError("selected SAE dictionary failed its quality checks")
    if result["compartment_status"].get(category) != "pass":
        raise ValueError("selected SAE compartment failed replica quality checks")
    return {
        "schema": "sch.vpd-compartment-sae-quality-authorization.v1",
        "status": "pass",
        "quality_report_id": str(report["quality_report_id"]),
        "quality_report_sha256": sha256_file(report_path),
        "dictionary_id": dictionary_id,
        "activation_compartment": category,
        "activation_category_split_id": str(
            quality_split["activation_category_split_id"]
        ),
        "activation_category_split_role": "calibration",
        "holdout_capture": dict(report["source"]["holdout_capture"]),
    }


def validate_study_split_provenance(
    *,
    split_manifest_path: Path,
    intervention_dataset_path: Path,
    dictionary_manifest: dict[str, Any],
    capture_manifest: dict[str, Any],
    effect_identity: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    intervention_provenance = verify_activation_category_split_role(
        intervention_dataset_path,
        split_manifest_path,
        "final_evaluation",
    )
    dictionary_provenance = dictionary_manifest["source"].get(
        "activation_category_split"
    )
    capture_provenance = capture_manifest["source"].get(
        "activation_category_split"
    )
    if not isinstance(dictionary_provenance, dict) or dictionary_provenance.get(
        "activation_category_split_role"
    ) != "sae_training":
        raise ValueError("SAE dictionary lacks the sae_training split role")
    if not isinstance(capture_provenance, dict) or capture_provenance.get(
        "activation_category_split_role"
    ) != "final_evaluation":
        raise ValueError("focused activation capture lacks the final_evaluation split role")
    if str(effect_identity.get("activation_category_split_role")) != "discovery":
        raise ValueError("focused VPD effects lack the discovery split role")
    split_ids = {
        str(dictionary_provenance.get("activation_category_split_id")),
        str(capture_provenance.get("activation_category_split_id")),
        str(intervention_provenance.get("activation_category_split_id")),
        str(effect_identity.get("activation_category_split_id")),
        str(packet.get("source", {}).get("activation_category_split_id")),
    }
    if len(split_ids) != 1 or "None" in split_ids:
        raise ValueError("focused SAE inputs do not share one activation-category split")
    for name, provenance in (
        ("dictionary", dictionary_provenance),
        ("capture", capture_provenance),
        ("intervention", intervention_provenance),
    ):
        if provenance.get("activation_category_split_verification_status") != "pass":
            raise ValueError(f"focused SAE {name} split provenance was not verified")
    split_schemas = {
        str(
            provenance.get(
                "activation_category_split_schema",
                "sch.vpd-activation-category-split-manifest.v1",
            )
        )
        for provenance in (
            dictionary_provenance,
            capture_provenance,
            intervention_provenance,
            effect_identity,
        )
    }
    if len(split_schemas) != 1:
        raise ValueError("focused SAE inputs use different split schemas")
    split_schema = next(iter(split_schemas))
    semantic_evidence: dict[str, Any] = {
        "split_schema": split_schema,
        "status": "not_provided_v1",
    }
    if split_schema in {
        "sch.vpd-activation-category-split-manifest.v2",
        "sch.vpd-activation-category-split-manifest.v3",
    }:
        provenances = {
            "dictionary": dictionary_provenance,
            "capture": capture_provenance,
            "intervention": intervention_provenance,
            "vpd_effects": effect_identity,
        }
        if any(
            provenance.get("activation_category_split_semantic_disjointness")
            != "pass"
            for provenance in provenances.values()
        ):
            raise ValueError("focused SAE semantic split disjointness was not verified")
        template_hashes = {
            name: require_digest(
                provenance.get(
                    "activation_category_split_template_group_set_sha256"
                ),
                f"focused SAE {name} template-group set hash",
            )
            for name, provenance in provenances.items()
        }
        if template_hashes["capture"] != template_hashes["intervention"]:
            raise ValueError(
                "focused activation capture and interventions use different final templates"
            )
        packet_role_hashes = packet.get("source", {}).get(
            "activation_category_template_group_set_sha256_by_role"
        )
        expected_packet_roles = {"discovery", "calibration"}
        if split_schema == "sch.vpd-activation-category-split-manifest.v3":
            expected_packet_roles.add("cluster_validation")
        if not isinstance(packet_role_hashes, dict) or set(packet_role_hashes) != (
            expected_packet_roles
        ):
            raise ValueError("focused SAE packet omits semantic effect-role sets")
        packet_role_hashes = {
            role: require_digest(value, f"focused SAE packet {role} template set")
            for role, value in packet_role_hashes.items()
        }
        if packet_role_hashes["discovery"] != template_hashes["vpd_effects"]:
            raise ValueError("focused SAE packet discovery template set differs")
        all_role_hashes = {
            "sae_training": template_hashes["dictionary"],
            "discovery": packet_role_hashes["discovery"],
            "calibration": packet_role_hashes["calibration"],
            **(
                {
                    "cluster_validation": packet_role_hashes[
                        "cluster_validation"
                    ]
                }
                if "cluster_validation" in packet_role_hashes
                else {}
            ),
            "final_evaluation": template_hashes["capture"],
        }
        if len(set(all_role_hashes.values())) != len(all_role_hashes):
            raise ValueError(
                "focused SAE role template sets are not disjoint"
            )
        semantic_evidence = {
            "split_schema": split_schema,
            "status": "verified",
            "template_group_set_sha256_by_input": template_hashes,
            "template_group_set_sha256_by_role": all_role_hashes,
        }
    elif split_schema != "sch.vpd-activation-category-split-manifest.v1":
        raise ValueError("focused SAE uses an unsupported split schema")
    return {
        "schema": "sch.vpd-focused-sae-split-provenance.v1",
        "status": "verified",
        "activation_category_split_id": next(iter(split_ids)),
        "activation_category_split_manifest_sha256": sha256_file(
            split_manifest_path
        ),
        "roles": {
            "dictionary": "sae_training",
            "vpd_effects": "discovery",
            "threshold_calibration": "calibration",
            **(
                {"cluster_count_validation": "cluster_validation"}
                if split_schema == "sch.vpd-activation-category-split-manifest.v3"
                else {}
            ),
            "activation_capture": "final_evaluation",
            "interventions": "final_evaluation",
        },
        "semantic_template_disjointness": semantic_evidence,
    }


def choose_control(
    *,
    control_ids: list[str],
    effect_index: dict[str, int],
    delta_h: np.ndarray,
    decoder_direction: np.ndarray,
    target_norm: float,
) -> tuple[str, float, float]:
    candidates: list[tuple[bool, float, float, str]] = []
    direction = decoder_direction / max(float(np.linalg.norm(decoder_direction)), 1e-12)
    for component_id in control_ids:
        index = effect_index.get(component_id)
        if index is None:
            continue
        vector = delta_h[index]
        norm = float(np.linalg.norm(vector))
        cosine = float(vector @ direction / max(norm, 1e-12))
        norm_gap = abs(norm - target_norm) / max(target_norm, 1e-12)
        candidates.append((abs(cosine) > 0.10, norm_gap, abs(cosine), component_id))
    if not candidates:
        raise ValueError("packet has no matched controls present in the effect archive")
    _, norm_gap, absolute_cosine, component_id = min(candidates)
    return component_id, norm_gap, absolute_cosine


def prepare_study(
    *,
    bundle_dir: Path,
    packet_id: str,
    dictionary_path: Path,
    dictionary_manifest_path: Path,
    activation_capture_path: Path,
    activation_capture_manifest_path: Path,
    metadata_path: Path,
    effect_archive_path: Path,
    effect_manifest_path: Path,
    intervention_dataset_path: Path,
    label_name: str,
    seed: int,
    permutations: int,
    top_features: int,
    max_pairs: int,
    maximum_max_stat_p: float,
    minimum_absolute_cosine: float,
    minimum_pairs_per_family: int,
    intervention_pairs_per_family: int,
    activation_category_split_manifest_path: Path | None = None,
    dictionary_quality_report_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if seed < 0 or permutations <= 0 or top_features <= 0 or max_pairs <= 0:
        raise ValueError("seed and focused SAE search sizes are invalid")
    if not 0.0 <= maximum_max_stat_p <= 1.0:
        raise ValueError("maximum_max_stat_p must be in [0, 1]")
    if not 0.0 <= minimum_absolute_cosine <= 1.0:
        raise ValueError("minimum_absolute_cosine must be in [0, 1]")
    if minimum_pairs_per_family <= 0 or intervention_pairs_per_family < minimum_pairs_per_family:
        raise ValueError("intervention pair counts are invalid")
    packets, packet_hashes = load_packets(bundle_dir)
    if packet_id not in packets:
        raise ValueError(f"unknown packet: {packet_id}")
    packet = packets[packet_id]
    alignment_key = str(packet["sae_study"]["alignment_activation_key"])
    dictionary = load_dictionary(dictionary_path)
    if dictionary["activation_key"] != alignment_key:
        raise ValueError("fixed SAE dictionary uses a different activation key")
    metadata_all = load_jsonl(metadata_path)
    with np.load(activation_capture_path, allow_pickle=False) as archive:
        if alignment_key not in archive.files:
            raise ValueError("activation capture omits the packet alignment key")
        raw_activations = np.asarray(archive[alignment_key], dtype=np.float32)
    if raw_activations.ndim != 2:
        raise ValueError("activation capture must be rank-2")
    if len(raw_activations) != len(metadata_all):
        raise ValueError("activation rows and metadata rows do not align")
    if raw_activations.shape[1] != int(dictionary["hidden_size"]):
        raise ValueError("activation and SAE dictionary hidden dimensions differ")
    component_ids, delta_h, effect_identity = load_effect_inputs(
        effect_archive_path=effect_archive_path,
        effect_manifest_path=effect_manifest_path,
        alignment_key=alignment_key,
    )
    packet_effect_lineage = validate_packet_effect_lineage(
        packet=packet,
        effect_archive_path=effect_archive_path,
        effect_manifest_path=effect_manifest_path,
    )
    dictionary_manifest = validate_dictionary_manifest(
        manifest_path=dictionary_manifest_path,
        dictionary_path=dictionary_path,
        evaluation_activation_path=activation_capture_path,
        alignment_key=alignment_key,
        effect_identity=effect_identity,
    )
    capture_manifest = validate_capture_manifest(
        manifest_path=activation_capture_manifest_path,
        activation_capture_path=activation_capture_path,
        metadata_path=metadata_path,
        alignment_key=alignment_key,
        effect_identity=effect_identity,
    )
    if int(capture_manifest["capture"]["rows"]) != len(raw_activations):
        raise ValueError("activation capture row count differs from its manifest")
    if int(capture_manifest["capture"]["hidden_size"]) != raw_activations.shape[1]:
        raise ValueError("activation capture hidden size differs from its manifest")
    if delta_h.shape[1] != int(dictionary["hidden_size"]):
        raise ValueError("VPD effects and SAE dictionary hidden dimensions differ")
    activation_compartment = packet["sae_study"].get("activation_compartment")
    compartment_category: str | None = None
    study_split_provenance: dict[str, Any] | None = None
    quality_authorization: dict[str, Any] | None = None
    if dictionary_manifest["dictionary"].get("activation_compartment") is not None:
        if dictionary_quality_report_path is None:
            raise ValueError(
                "compartment SAE dictionary requires a passed calibration quality report"
            )
        quality_authorization = validate_compartment_quality_report(
            report_path=dictionary_quality_report_path,
            dictionary_path=dictionary_path,
            dictionary_manifest_path=dictionary_manifest_path,
            dictionary_manifest=dictionary_manifest,
        )
    if activation_compartment is not None:
        if not isinstance(activation_compartment, dict):
            raise ValueError("packet activation compartment is malformed")
        compartment_category = str(activation_compartment["category"])
        if compartment_category not in COMPONENT_COMPARTMENT_PROTOCOLS:
            raise ValueError("packet activation compartment category is unknown")
        dictionary_compartment = dictionary_manifest["dictionary"].get(
            "activation_compartment"
        )
        if not isinstance(dictionary_compartment, dict):
            raise ValueError("category packet requires a compartment-trained SAE dictionary")
        if str(dictionary_compartment.get("activation_category")) != compartment_category:
            raise ValueError("SAE dictionary belongs to a different activation compartment")
        if str(dictionary_compartment.get("protocol")) != COMPONENT_COMPARTMENT_PROTOCOLS[
            compartment_category
        ]:
            raise ValueError("SAE dictionary uses the wrong activation-category protocol")
        if str(dictionary_compartment.get("category_contract_sha256")) != content_sha256(
            category_contract()
        ):
            raise ValueError("SAE dictionary uses a different activation-category contract")
        effect_categories = set(
            str(value) for value in effect_identity.get("activation_categories", [])
        )
        if compartment_category == "shared_cross_category" and len(effect_categories) < 2:
            raise ValueError("shared compartment requires multiple effect categories")
        if compartment_category != "shared_cross_category" and compartment_category not in effect_categories:
            raise ValueError("effect archive omits the packet activation category")
        threshold_calibration = activation_compartment.get("threshold_calibration")
        requires_verified_split = (
            isinstance(threshold_calibration, dict)
            and threshold_calibration.get("status") == "pass"
        )
        if requires_verified_split and activation_category_split_manifest_path is None:
            raise ValueError(
                "production category packet requires an activation-category split manifest"
            )
        if activation_category_split_manifest_path is not None:
            study_split_provenance = validate_study_split_provenance(
                split_manifest_path=activation_category_split_manifest_path,
                intervention_dataset_path=intervention_dataset_path,
                dictionary_manifest=dictionary_manifest,
                capture_manifest=capture_manifest,
                effect_identity=effect_identity,
                packet=packet,
            )
    phase = str(effect_identity["phase"])
    intervention_rows = load_jsonl(intervention_dataset_path)
    intervention_by_id: dict[str, dict[str, Any]] = {}
    for row in intervention_rows:
        episode_id = str(row["episode_id"])
        if episode_id in intervention_by_id:
            raise ValueError(f"duplicate intervention episode ID: {episode_id}")
        messages_field = "messages" if phase == "pre_gate" else "messages_post_gate"
        if messages_field not in row:
            raise ValueError(f"intervention episode omits {messages_field}: {episode_id}")
        intervention_by_id[episode_id] = row
    keep = np.asarray(
        [str(row.get("phase")) == phase for row in metadata_all], dtype=bool
    )
    keep &= np.isfinite(raw_activations).all(axis=1)
    if int(keep.sum()) < 16:
        raise ValueError("focused SAE study has fewer than 16 aligned phase rows")
    activations = raw_activations[keep]
    metadata = [row for row, retain in zip(metadata_all, keep.tolist()) if retain]
    activation_category_counts = dict(
        sorted(Counter(activation_category(row) for row in metadata).items())
    )
    if compartment_category is not None:
        if compartment_category == "shared_cross_category":
            if len(activation_category_counts) < 2 or min(
                activation_category_counts.values()
            ) < 4:
                raise ValueError(
                    "shared focused SAE study requires at least four rows per category"
                )
        else:
            target_rows = activation_category_counts.get(compartment_category, 0)
            cross_rows = len(metadata) - target_rows
            if target_rows < 4 or cross_rows < 4:
                raise ValueError(
                    "focused SAE category study requires target and cross-category evaluation rows"
                )
    features = encode_topk(
        activations,
        pre_bias=dictionary["pre_bias"],
        encoder_weight=dictionary["encoder_weight"],
        encoder_bias=dictionary["encoder_bias"],
        top_k=int(dictionary["top_k"]),
    )
    reconstruction = features @ dictionary["decoder"] + dictionary["pre_bias"]
    residual_sum = float(np.sum((reconstruction - activations) ** 2))
    total_sum = float(np.sum((activations - activations.mean(axis=0)) ** 2))
    quality = {
        "reconstruction_fvu": residual_sum / max(total_sum, 1e-12),
        "dead_feature_rate": float(np.mean(np.max(features, axis=0) <= 0.0)),
        "mean_l0": float(np.mean(np.sum(features > 0.0, axis=1))),
    }
    labels, screen_endpoint = focused_screen_endpoint(
        metadata=metadata,
        label_name=label_name,
        compartment_category=compartment_category,
    )
    if len(set(labels.tolist())) != 2:
        raise ValueError("focused SAE screen endpoint must contain both classes")
    anchor = alignment_key.split("__", 1)[1]
    prompt_tokens: list[float] = []
    anchor_positions: list[float] = []
    for row in metadata:
        if "prompt_tokens" not in row or anchor not in row.get("positions", {}):
            raise ValueError("metadata omits prompt length or aligned anchor position")
        prompt_tokens.append(float(row["prompt_tokens"]))
        anchor_positions.append(float(row["positions"][anchor]))
    covariates = np.column_stack([prompt_tokens, anchor_positions]).astype(np.float64)
    groups = np.asarray([str(row["template_group"]) for row in metadata])
    blocks = np.asarray([str(row["pair_id"]) for row in metadata])
    scores = behavior_screen(
        labels=labels,
        features=features,
        covariates=covariates,
        groups=groups,
        blocks=blocks,
        seed=seed,
        permutations=permutations,
        top_features=top_features,
    )
    effect_index = {component_id: index for index, component_id in enumerate(component_ids)}
    missing_components = sorted(set(packet["component_ids"]) - set(effect_index))
    if missing_components:
        raise ValueError(f"effect archive omits packet components: {len(missing_components)}")
    decoder = dictionary["decoder"]
    decoder_norm = decoder / np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-12)
    aligned_candidates: list[dict[str, Any]] = []
    for score in scores:
        if float(score["shuffle_max_stat_p"]) > maximum_max_stat_p:
            continue
        if int(score["held_out_prompt_families"]) < 2:
            continue
        feature_index = int(score["feature_index"])
        direction = decoder_norm[feature_index]
        for component_id in packet["component_ids"]:
            vector = delta_h[effect_index[str(component_id)]]
            norm = float(np.linalg.norm(vector))
            cosine = float(vector @ direction / max(norm, 1e-12))
            if abs(cosine) < minimum_absolute_cosine:
                continue
            control_id, norm_gap, control_cosine = choose_control(
                control_ids=[
                    str(value)
                    for value in packet["sae_study"]["matched_control_component_ids"]
                ],
                effect_index=effect_index,
                delta_h=delta_h,
                decoder_direction=decoder[feature_index],
                target_norm=norm,
            )
            intervention_families = sorted(
                str(value) for value in score["cv_held_out_template_auc"]
            )
            intervention_episodes: list[dict[str, Any]] = []
            for family in intervention_families:
                family_rows = [
                    row
                    for row in intervention_rows
                    if str(row.get("template_group")) == family
                    and (
                        compartment_category is None
                        or compartment_category == "shared_cross_category"
                        or activation_category(row) == compartment_category
                    )
                ]
                if compartment_category == "shared_cross_category":
                    by_category: dict[str, list[dict[str, Any]]] = {}
                    for row in family_rows:
                        by_category.setdefault(activation_category(row), []).append(row)
                    if len(by_category) < 2:
                        raise ValueError(
                            f"shared intervention family has fewer than two categories: {family}"
                        )
                    selected_rows = []
                    for category, values in sorted(by_category.items()):
                        values.sort(
                            key=lambda row: stable_hash(
                                seed,
                                "focused-sae-shared-intervention",
                                packet_id,
                                family,
                                category,
                                str(row["episode_id"]),
                            )
                        )
                        if len(values) < intervention_pairs_per_family:
                            raise ValueError(
                                "shared intervention dataset has too few episodes for "
                                f"family/category: {family}/{category}"
                            )
                        selected_rows.extend(values[:intervention_pairs_per_family])
                else:
                    family_rows.sort(
                        key=lambda row: stable_hash(
                            seed,
                            "focused-sae-intervention",
                            packet_id,
                            family,
                            str(row["episode_id"]),
                        )
                    )
                    selected_rows = family_rows[:intervention_pairs_per_family]
                    if len(selected_rows) < intervention_pairs_per_family:
                        raise ValueError(
                            f"intervention dataset has too few episodes for family: {family}"
                        )
                messages_field = "messages" if phase == "pre_gate" else "messages_post_gate"
                intervention_episodes.extend(
                    {
                        "episode_id": str(row["episode_id"]),
                        "template_group": family,
                        "prompt_messages_sha256": content_sha256(row[messages_field]),
                        **(
                            {}
                            if compartment_category is None
                            else {"activation_category": activation_category(row)}
                        ),
                    }
                    for row in selected_rows
                )
            candidate_body = {
                "component_id": str(component_id),
                "activation_key": alignment_key,
                "feature_index": feature_index,
                "cosine": cosine,
                "behavior": score,
                "matched_control_component_id": control_id,
                "matched_control_effect_norm_relative_gap": norm_gap,
                "matched_control_absolute_cosine": control_cosine,
                "intervention_families": intervention_families,
                "intervention_episodes": intervention_episodes,
                "activation_category": compartment_category,
            }
            aligned_candidates.append(
                {**candidate_body, "candidate_id": content_sha256(candidate_body)}
            )
    aligned_candidates.sort(
        key=lambda row: (
            float(row["behavior"]["shuffle_max_stat_p"]),
            -abs(float(row["cosine"])),
            str(row["component_id"]),
            int(row["feature_index"]),
        )
    )
    aligned_candidates = aligned_candidates[:max_pairs]
    study_inputs = {
        "packet_sha256": packet_hashes[packet_id],
        "sae_dictionary_sha256": sha256_file(dictionary_path),
        "sae_dictionary_manifest_sha256": sha256_file(dictionary_manifest_path),
        "activation_capture_sha256": sha256_file(activation_capture_path),
        "activation_capture_manifest_sha256": sha256_file(
            activation_capture_manifest_path
        ),
        "dataset_sha256": sha256_file(metadata_path),
        "effect_archive_sha256": sha256_file(effect_archive_path),
        "effect_merge_manifest_sha256": sha256_file(effect_manifest_path),
        "packet_effect_lineage": packet_effect_lineage,
        "intervention_dataset_sha256": sha256_file(intervention_dataset_path),
        "executor_sha256": sha256_file(Path(__file__)),
        "adapter_sha256": str(effect_identity["adapter_sha256"]),
        "model_id": str(effect_identity["model_id"]),
        "revision": str(effect_identity["revision"]),
        "alignment_activation_key": alignment_key,
        "label": label_name,
        "screen_endpoint": screen_endpoint,
        "phase": phase,
        "permutation_seed": seed,
        "permutation_count": permutations,
        "activation_category": compartment_category,
        "activation_category_contract_sha256": (
            content_sha256(category_contract())
            if compartment_category is not None
            else None
        ),
        "sae_capacity_authorization": dictionary_manifest["source"].get(
            "capacity_authorization"
        ),
        "sae_capacity_plan_sha256": dictionary_manifest["source"].get(
            "capacity_plan_sha256"
        ),
        "sae_quality_authorization": quality_authorization,
        "activation_category_split_provenance": study_split_provenance,
    }
    plan_body = {
        "schema": "sch.vpd-focused-sae-intervention-plan.v1",
        "packet_id": packet_id,
        "cluster_id": str(packet["cluster_id"]),
        "worker_id": str(packet["distribution"]["assigned_worker"]),
        "study_inputs": study_inputs,
        "sae_quality": quality,
        "screen": {
            "endpoint": screen_endpoint,
            "features_tested": int(dictionary["feature_count"]),
            "features_retained": len(scores),
            "maximum_max_stat_p": maximum_max_stat_p,
            "minimum_absolute_cosine": minimum_absolute_cosine,
            "candidate_pairs": len(aligned_candidates),
            "activation_category_counts": activation_category_counts,
            "target_activation_category": compartment_category,
        },
        "intervention_contract": {
            "kinds": list(INTERVENTION_KINDS),
            "coefficients": list(INTERVENTION_COEFFICIENTS),
            "minimum_pairs_per_family": minimum_pairs_per_family,
            "pairs_per_family": intervention_pairs_per_family,
            "shared_pairs_rule": (
                "pairs_per_family_per_activation_category"
                if compartment_category == "shared_cross_category"
                else None
            ),
            "endpoint": "signed_authorization_decision_margin",
            "effect_estimator": "mean_within_pair_slope",
            "parameter_effect_control": "subtract_matched_norm_component_slope",
            "guardrail_rule": "zero_guardrail_violations",
        },
        "candidates": aligned_candidates,
    }
    plan = {**plan_body, "plan_id": content_sha256(plan_body)}
    context = {"packet": packet, "packet_hash": packet_hashes[packet_id]}
    return plan, context


def slope(rows: list[dict[str, Any]]) -> float:
    coefficients = np.asarray([float(row["coefficient"]) for row in rows], dtype=np.float64)
    endpoints = np.asarray([float(row["endpoint_value"]) for row in rows], dtype=np.float64)
    centered = coefficients - coefficients.mean()
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise ValueError("intervention coefficients do not identify a slope")
    return float(centered @ (endpoints - endpoints.mean()) / denominator)


def finalize_study(
    *, plan: dict[str, Any], packet: dict[str, Any], intervention_trace_path: Path
) -> dict[str, Any]:
    trace = json.loads(intervention_trace_path.read_text(encoding="utf-8"))
    if trace.get("schema") != "sch.vpd-focused-sae-intervention-trace.v1":
        raise ValueError("unsupported focused SAE intervention trace schema")
    if str(trace.get("plan_id")) != str(plan["plan_id"]):
        raise ValueError("intervention trace belongs to a different study plan")
    if str(trace.get("worker_id")) != str(plan["worker_id"]):
        raise ValueError("intervention trace came from a different worker")
    if not bool(trace.get("factor_restoration_verified", False)):
        raise ValueError("intervention worker did not verify LoRA factor restoration")
    trace_inputs = trace.get("inputs")
    if not isinstance(trace_inputs, dict):
        raise ValueError("intervention trace omits its frozen inputs")
    expected_trace_inputs = {
        "dictionary_sha256": plan["study_inputs"]["sae_dictionary_sha256"],
        "dataset_sha256": plan["study_inputs"]["intervention_dataset_sha256"],
        "model_id": plan["study_inputs"]["model_id"],
        "revision": plan["study_inputs"]["revision"],
        "adapter_sha256": plan["study_inputs"]["adapter_sha256"],
        "alignment_activation_key": plan["study_inputs"]["alignment_activation_key"],
        "phase": plan["study_inputs"]["phase"],
    }
    for field, expected in expected_trace_inputs.items():
        if str(trace_inputs.get(field)) != str(expected):
            raise ValueError(f"intervention trace changed frozen input: {field}")
    intervention_runner_sha256 = require_digest(
        trace_inputs.get("runner_sha256"), "intervention_runner_sha256"
    )
    decision_tokens = trace.get("decision_tokens")
    if not isinstance(decision_tokens, dict):
        raise ValueError("intervention trace omits decision token identities")
    request_ids = [int(value) for value in decision_tokens["request_first_token_ids"]]
    propose_ids = [int(value) for value in decision_tokens["propose_first_token_ids"]]
    if not request_ids or not propose_ids or set(request_ids) & set(propose_ids):
        raise ValueError("intervention decision token groups are empty or overlapping")
    if str(decision_tokens.get("margin")) != "logsumexp(REQUEST)-logsumexp(PROPOSE)":
        raise ValueError("intervention trace changed its decision margin")
    trials = trace.get("trials")
    if not isinstance(trials, list):
        raise ValueError("intervention trace trials must be a list")
    planned = {str(row["candidate_id"]): row for row in plan["candidates"]}
    grouped: dict[str, list[dict[str, Any]]] = {candidate_id: [] for candidate_id in planned}
    for row in trials:
        if not isinstance(row, dict):
            raise ValueError("intervention trial must be an object")
        if row.get("schema") != "sch.vpd-focused-sae-intervention-trial.v1":
            raise ValueError("unsupported focused SAE intervention trial schema")
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in planned:
            raise ValueError("intervention trace contains an unplanned candidate")
        grouped[candidate_id].append(row)
    candidates: list[dict[str, Any]] = []
    contract = plan["intervention_contract"]
    coefficients = [float(value) for value in contract["coefficients"]]
    kinds = [str(value) for value in contract["kinds"]]
    minimum_pairs = int(contract["minimum_pairs_per_family"])
    for candidate_id, candidate in planned.items():
        rows = grouped[candidate_id]
        planned_episodes = {
            str(row["episode_id"]): {
                "template_group": str(row["template_group"]),
                "activation_category": row.get("activation_category"),
            }
            for row in candidate["intervention_episodes"]
        }
        by_pair: dict[str, list[dict[str, Any]]] = {}
        pair_family: dict[str, str] = {}
        pair_category: dict[str, str | None] = {}
        for row in rows:
            pair_id = str(row["pair_id"])
            family = str(row["template_group"])
            if (
                pair_id not in planned_episodes
                or planned_episodes[pair_id]["template_group"] != family
            ):
                raise ValueError("intervention trial uses an unplanned episode")
            expected_category = planned_episodes[pair_id]["activation_category"]
            observed_category = row.get("activation_category")
            if expected_category is not None and str(observed_category) != str(
                expected_category
            ):
                raise ValueError("intervention trial changed activation category")
            if family not in set(candidate["intervention_families"]):
                raise ValueError("intervention trial uses a non-held-out prompt family")
            prior_family = pair_family.setdefault(pair_id, family)
            if prior_family != family:
                raise ValueError("one intervention pair spans multiple prompt families")
            prior_category = pair_category.setdefault(
                pair_id, None if expected_category is None else str(expected_category)
            )
            if prior_category != (
                None if expected_category is None else str(expected_category)
            ):
                raise ValueError("one intervention pair spans activation categories")
            kind = str(row["intervention_kind"])
            if kind not in kinds:
                raise ValueError("intervention trace contains an unsupported kind")
            coefficient = float(row["coefficient"])
            if coefficient not in coefficients:
                raise ValueError("intervention trace contains an unfrozen coefficient")
            endpoint = float(row["endpoint_value"])
            if not math.isfinite(endpoint):
                raise ValueError("intervention endpoint must be finite")
            request_logit = float(row["request_group_logit"])
            propose_logit = float(row["propose_group_logit"])
            if not math.isfinite(request_logit) or not math.isfinite(propose_logit):
                raise ValueError("intervention decision group logits must be finite")
            if not math.isclose(endpoint, request_logit - propose_logit, abs_tol=1e-5):
                raise ValueError("intervention endpoint differs from its decision logits")
            predicted = "REQUEST" if endpoint >= 0.0 else "PROPOSE"
            if str(row["predicted_decision"]) != predicted:
                raise ValueError("intervention predicted decision contradicts its margin")
            expected_decision = str(row["expected_decision"])
            if expected_decision not in {"REQUEST", "PROPOSE"}:
                raise ValueError("intervention trial has an invalid expected decision")
            expected_violation = expected_decision == "REQUEST" and predicted == "PROPOSE"
            if bool(row["guardrail_violation"]) != expected_violation:
                raise ValueError("intervention guardrail flag contradicts its decisions")
            expected_target = {
                "activation": f"sae_feature:{candidate['feature_index']}",
                "parameter": str(candidate["component_id"]),
                "matched_norm_component_control": str(
                    candidate["matched_control_component_id"]
                ),
            }[kind]
            if str(row["intervention_target_id"]) != expected_target:
                raise ValueError("intervention trial changed its frozen target")
            by_pair.setdefault(pair_id, []).append(row)
        family_counts: dict[str, int] = {}
        activation_slopes: list[float] = []
        parameter_slopes: list[float] = []
        category_activation_slopes: dict[str, list[float]] = {}
        category_parameter_slopes: dict[str, list[float]] = {}
        guardrail_passed = True
        expected_cells = {(kind, coefficient) for kind in kinds for coefficient in coefficients}
        if set(by_pair) != set(planned_episodes):
            raise ValueError("intervention trace does not cover the exact frozen episode set")
        for pair_id, pair_rows in sorted(by_pair.items()):
            observed_cells = [
                (str(row["intervention_kind"]), float(row["coefficient"]))
                for row in pair_rows
            ]
            if len(observed_cells) != len(set(observed_cells)):
                raise ValueError("intervention pair contains duplicate cells")
            if set(observed_cells) != expected_cells:
                raise ValueError("intervention pair does not cover the frozen factorial cells")
            by_kind = {
                kind: [row for row in pair_rows if row["intervention_kind"] == kind]
                for kind in kinds
            }
            activation_value = slope(by_kind["activation"])
            parameter_value = slope(by_kind["parameter"]) - slope(
                by_kind["matched_norm_component_control"]
            )
            activation_slopes.append(activation_value)
            parameter_slopes.append(parameter_value)
            category = pair_category[pair_id]
            if category is not None:
                category_activation_slopes.setdefault(category, []).append(
                    activation_value
                )
                category_parameter_slopes.setdefault(category, []).append(
                    parameter_value
                )
            family = pair_family[pair_id]
            family_counts[family] = family_counts.get(family, 0) + 1
            guardrail_passed = guardrail_passed and not any(
                bool(row["guardrail_violation"]) for row in pair_rows
            )
        complete_families = sorted(
            family for family, count in family_counts.items() if count >= minimum_pairs
        )
        if set(complete_families) != set(candidate["intervention_families"]):
            raise ValueError("intervention trace does not cover every frozen held-out family")
        if not activation_slopes:
            raise ValueError("planned candidate has no intervention trials")
        behavior = candidate["behavior"]
        candidates.append(
            {
                "component_id": str(candidate["component_id"]),
                "activation_key": str(candidate["activation_key"]),
                "feature_index": int(candidate["feature_index"]),
                "cosine": float(candidate["cosine"]),
                "shuffle_max_stat_p": float(behavior["shuffle_max_stat_p"]),
                "cv_mean_oriented_auc": float(behavior["cv_mean_oriented_auc"]),
                "activation_intervention_effect": float(np.mean(activation_slopes)),
                "parameter_intervention_effect": float(np.mean(parameter_slopes)),
                "guardrail_passed": guardrail_passed,
                "held_out_prompt_families": len(complete_families),
                "n_pairs": len(by_pair),
                "activation_category_effects": {
                    category: {
                        "activation_intervention_effect": float(
                            np.mean(category_activation_slopes[category])
                        ),
                        "parameter_intervention_effect": float(
                            np.mean(category_parameter_slopes[category])
                        ),
                        "n_pairs": len(category_activation_slopes[category]),
                    }
                    for category in sorted(category_activation_slopes)
                },
            }
        )
    inputs = {
        **plan["study_inputs"],
        "intervention_trace_sha256": sha256_file(intervention_trace_path),
        "intervention_runner_sha256": intervention_runner_sha256,
    }
    return {
        "schema": "sch.vpd-sae-worker-result.v1",
        "packet_id": str(plan["packet_id"]),
        "cluster_id": str(plan["cluster_id"]),
        "worker_id": str(plan["worker_id"]),
        "run_id": f"focused-sae-{plan['plan_id'][:16]}",
        "status": "complete",
        "inputs": inputs,
        "controls_executed": list(packet["sae_study"]["required_controls"]),
        "sae_quality": dict(plan["sae_quality"]),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan and finalize a packet-scoped focused VPD/SAE study."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--dictionary-manifest", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--activation-capture-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--effect-merge-manifest", type=Path, required=True)
    parser.add_argument("--intervention-dataset", type=Path, required=True)
    parser.add_argument("--activation-category-split-manifest", type=Path)
    parser.add_argument("--dictionary-quality-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intervention-trace", type=Path)
    parser.add_argument("--label", default="semantic_authorization")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--top-features", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--maximum-max-stat-p", type=float, default=0.05)
    parser.add_argument("--minimum-absolute-cosine", type=float, default=0.20)
    parser.add_argument("--minimum-pairs-per-family", type=int, default=2)
    parser.add_argument("--intervention-pairs-per-family", type=int, default=2)
    args = parser.parse_args()
    plan, context = prepare_study(
        bundle_dir=args.bundle,
        packet_id=args.packet_id,
        dictionary_path=args.dictionary,
        dictionary_manifest_path=args.dictionary_manifest,
        activation_capture_path=args.activations,
        activation_capture_manifest_path=args.activation_capture_manifest,
        metadata_path=args.metadata,
        effect_archive_path=args.effects,
        effect_manifest_path=args.effect_merge_manifest,
        intervention_dataset_path=args.intervention_dataset,
        label_name=args.label,
        seed=args.seed,
        permutations=args.permutations,
        top_features=args.top_features,
        max_pairs=args.max_pairs,
        maximum_max_stat_p=args.maximum_max_stat_p,
        minimum_absolute_cosine=args.minimum_absolute_cosine,
        minimum_pairs_per_family=args.minimum_pairs_per_family,
        intervention_pairs_per_family=args.intervention_pairs_per_family,
        activation_category_split_manifest_path=(
            args.activation_category_split_manifest
        ),
        dictionary_quality_report_path=args.dictionary_quality_report,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "intervention_plan.json", plan)
    result: dict[str, Any] = {
        "schema": "sch.vpd-focused-sae-planning-output.v1",
        "plan_id": plan["plan_id"],
        "candidate_pairs": len(plan["candidates"]),
        "plan_path": "intervention_plan.json",
    }
    if args.intervention_trace is not None:
        unsigned = finalize_study(
            plan=plan,
            packet=context["packet"],
            intervention_trace_path=args.intervention_trace,
        )
        write_json(args.output_dir / "unsigned_result.json", unsigned)
        result["unsigned_result_path"] = "unsigned_result.json"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
