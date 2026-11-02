from __future__ import annotations

import argparse
import base64
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    canonical_bytes,
    content_sha256,
    load_jsonl,
    sha256_file,
    verify_bundle,
    write_json,
    write_jsonl,
)


HEX_256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_POLICY: dict[str, Any] = {
    "schema": "sch.vpd-sae-merge-policy.v1",
    "maximum_reconstruction_fvu": 0.20,
    "maximum_dead_feature_rate": 0.20,
    "minimum_absolute_cosine": 0.20,
    "maximum_shuffle_max_stat_p": 0.05,
    "minimum_cv_mean_oriented_auc": 0.70,
    "minimum_held_out_prompt_families": 2,
    "minimum_absolute_activation_effect": 0.02,
    "minimum_absolute_parameter_effect": 0.02,
    "require_signed_effect_agreement": True,
    "require_guardrail_pass": True,
    "require_assigned_worker": True,
}
STUDY_DIGEST_FIELDS = (
    "sae_dictionary_sha256",
    "sae_dictionary_manifest_sha256",
    "activation_capture_sha256",
    "activation_capture_manifest_sha256",
    "dataset_sha256",
    "effect_archive_sha256",
    "effect_merge_manifest_sha256",
    "intervention_dataset_sha256",
    "intervention_trace_sha256",
    "executor_sha256",
    "intervention_runner_sha256",
    "adapter_sha256",
)
SHARED_STUDY_INPUT_FIELDS = (
    "sae_dictionary_sha256",
    "sae_dictionary_manifest_sha256",
    "activation_capture_sha256",
    "activation_capture_manifest_sha256",
    "dataset_sha256",
    "effect_archive_sha256",
    "effect_merge_manifest_sha256",
    "intervention_dataset_sha256",
    "executor_sha256",
    "intervention_runner_sha256",
    "adapter_sha256",
    "model_id",
    "revision",
    "alignment_activation_key",
    "label",
    "phase",
    "permutation_seed",
    "permutation_count",
)


def crypto_types() -> tuple[Any, Any, Any, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise SystemExit("Ed25519 receipts require the cryptography package") from exc
    return Ed25519PrivateKey, Ed25519PublicKey, serialization, serialization.Encoding


def result_body(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"result_id", "signature"}
    }


def sign_result(result: dict[str, Any], private_key: Any) -> dict[str, Any]:
    body = result_body(result)
    result_id = content_sha256(body)
    signature = private_key.sign(canonical_bytes(body))
    return {
        **body,
        "result_id": result_id,
        "signature": {
            "algorithm": "ed25519",
            "value_base64": base64.b64encode(signature).decode("ascii"),
        },
    }


def public_key_base64(private_key: Any) -> str:
    _, _, serialization, _ = crypto_types()
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def load_private_key(path: Path) -> Any:
    _, _, serialization, _ = crypto_types()
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def generate_worker_key(worker_id: str, private_path: Path, public_path: Path) -> None:
    Ed25519PrivateKey, _, serialization, _ = crypto_types()
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(private_bytes)
    write_json(
        public_path,
        {
            "schema": "sch.vpd-sae-worker-public-record.v1",
            "worker_id": worker_id,
            "algorithm": "ed25519",
            "public_key_base64": public_key_base64(private_key),
            "enabled": True,
        },
    )


def require_digest(value: Any, field: str) -> str:
    result = str(value).lower()
    if HEX_256_RE.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def require_finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("worker registry contains null bytes")
    registry = json.loads(data.decode("utf-8"))
    if registry.get("schema") != "sch.vpd-sae-worker-registry.v1":
        raise ValueError("unsupported worker registry schema")
    workers: dict[str, dict[str, Any]] = {}
    for row in registry["workers"]:
        worker_id = str(row["worker_id"])
        if worker_id in workers:
            raise ValueError(f"duplicate worker registry entry: {worker_id}")
        if str(row.get("algorithm")) != "ed25519":
            raise ValueError(f"unsupported worker key algorithm: {worker_id}")
        raw = base64.b64decode(str(row["public_key_base64"]), validate=True)
        if len(raw) != 32:
            raise ValueError(f"invalid Ed25519 public key length: {worker_id}")
        workers[worker_id] = dict(row)
    return workers


def load_packets(bundle_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    verify_bundle(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    packets: dict[str, dict[str, Any]] = {}
    packet_hashes: dict[str, str] = {}
    for entry in manifest["artifacts"]["packets"]:
        packet = json.loads((bundle_dir / entry["packet_path"]).read_text(encoding="utf-8"))
        packet_id = str(packet["packet_id"])
        if packet_id in packets:
            raise ValueError(f"duplicate packet ID in bundle: {packet_id}")
        packets[packet_id] = packet
        packet_hashes[packet_id] = str(entry["packet_sha256"])
    return packets, packet_hashes


def verify_signature(result: dict[str, Any], worker: dict[str, Any]) -> None:
    _, Ed25519PublicKey, _, _ = crypto_types()
    signature = result.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        raise ValueError("result has no supported signature")
    body = result_body(result)
    result_id = content_sha256(body)
    if result_id != str(result.get("result_id")):
        raise ValueError("worker result content address mismatch")
    try:
        signature_bytes = base64.b64decode(str(signature["value_base64"]), validate=True)
        public_bytes = base64.b64decode(str(worker["public_key_base64"]), validate=True)
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(
            signature_bytes, canonical_bytes(body)
        )
    except Exception as exc:
        raise ValueError("worker result signature verification failed") from exc


def validate_packet_effect_input_lineage(
    *, packet: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any]:
    """Independently enforce packet-bound discovery inputs at merge time."""
    source = packet.get("source")
    if not isinstance(source, dict):
        raise ValueError("worker packet lacks source provenance")
    expected_archive = source.get("effects_sha256")
    expected_manifest = source.get("effects_merge_manifest_sha256")
    activation_compartment = packet.get("sae_study", {}).get(
        "activation_compartment"
    )
    reported = inputs.get("packet_effect_lineage")
    if expected_archive is None and expected_manifest is None:
        if activation_compartment is not None:
            raise ValueError(
                "category worker packet lacks packet-bound discovery effect provenance"
            )
        if reported is not None and (
            not isinstance(reported, dict)
            or reported.get("status") != "unattested_legacy_metadata_pilot"
            or reported.get("empirical_category_claim_allowed") is not False
        ):
            raise ValueError("legacy worker result misstates packet/effect lineage")
        return {
            "schema": "sch.vpd-focused-sae-packet-effect-lineage.v1",
            "status": "unattested_legacy_metadata_pilot",
            "empirical_category_claim_allowed": False,
            "independently_verified_by_reducer": True,
        }
    expected_archive = require_digest(
        expected_archive, "packet discovery effect archive"
    )
    expected_manifest = require_digest(
        expected_manifest, "packet discovery effect merge manifest"
    )
    observed_archive = require_digest(
        inputs.get("effect_archive_sha256"), "effect_archive_sha256"
    )
    observed_manifest = require_digest(
        inputs.get("effect_merge_manifest_sha256"),
        "effect_merge_manifest_sha256",
    )
    if observed_archive != expected_archive:
        raise ValueError("worker effect archive differs from its packet lineage")
    if observed_manifest != expected_manifest:
        raise ValueError("worker effect merge manifest differs from its packet lineage")
    if (
        not isinstance(reported, dict)
        or reported.get("status") != "pass"
        or reported.get("effect_archive_sha256") != expected_archive
        or reported.get("effect_merge_manifest_sha256") != expected_manifest
        or reported.get("empirical_category_claim_allowed")
        is not (activation_compartment is not None)
    ):
        raise ValueError("worker result lacks exact packet/effect lineage receipt")
    return {
        "schema": "sch.vpd-focused-sae-packet-effect-lineage.v1",
        "status": "pass",
        "effect_archive_sha256": expected_archive,
        "effect_merge_manifest_sha256": expected_manifest,
        "empirical_category_claim_allowed": activation_compartment is not None,
        "independently_verified_by_reducer": True,
    }


def validate_quality(quality: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    reconstruction_fvu = require_finite(quality["reconstruction_fvu"], "reconstruction_fvu")
    dead_feature_rate = require_finite(quality["dead_feature_rate"], "dead_feature_rate")
    mean_l0 = require_finite(quality["mean_l0"], "mean_l0")
    if reconstruction_fvu < 0.0:
        raise ValueError("reconstruction_fvu must be non-negative")
    if not 0.0 <= dead_feature_rate <= 1.0:
        raise ValueError("dead_feature_rate must be in [0, 1]")
    if mean_l0 < 0.0:
        raise ValueError("mean_l0 must be non-negative")
    return {
        "reconstruction_fvu": reconstruction_fvu,
        "dead_feature_rate": dead_feature_rate,
        "mean_l0": mean_l0,
        "quality_gate_passed": (
            reconstruction_fvu <= float(policy["maximum_reconstruction_fvu"])
            and dead_feature_rate <= float(policy["maximum_dead_feature_rate"])
        ),
    }


def evaluate_candidate(
    row: dict[str, Any],
    *,
    packet: dict[str, Any],
    quality_passed: bool,
    policy: dict[str, Any],
) -> dict[str, Any]:
    component_id = str(row["component_id"])
    if component_id not in set(packet["component_ids"]):
        raise ValueError(f"candidate component is outside packet: {component_id}")
    activation_key = str(row["activation_key"])
    if activation_key not in set(packet["focus"]["recommended_activation_keys"]):
        raise ValueError(f"candidate activation key is outside packet: {activation_key}")
    feature_index = int(row["feature_index"])
    if feature_index < 0:
        raise ValueError("feature_index must be non-negative")
    cosine = require_finite(row["cosine"], "cosine")
    p_value = require_finite(row["shuffle_max_stat_p"], "shuffle_max_stat_p")
    cv_auc = require_finite(row["cv_mean_oriented_auc"], "cv_mean_oriented_auc")
    activation_effect = require_finite(
        row["activation_intervention_effect"], "activation_intervention_effect"
    )
    parameter_effect = require_finite(
        row["parameter_intervention_effect"], "parameter_intervention_effect"
    )
    held_out_families = int(row["held_out_prompt_families"])
    n_pairs = int(row["n_pairs"])
    if not -1.0 <= cosine <= 1.0:
        raise ValueError("cosine must be in [-1, 1]")
    if not 0.0 <= p_value <= 1.0 or not 0.0 <= cv_auc <= 1.0:
        raise ValueError("probability and AUC values must be in [0, 1]")
    if held_out_families < 0 or n_pairs <= 0:
        raise ValueError("held-out family and pair counts are invalid")
    guardrail_passed = bool(row["guardrail_passed"])
    sign_agreement = activation_effect * parameter_effect > 0.0
    screen_checks = {
        "sae_quality": quality_passed,
        "absolute_cosine": abs(cosine) >= float(policy["minimum_absolute_cosine"]),
        "max_stat": p_value <= float(policy["maximum_shuffle_max_stat_p"]),
        "cross_validation": cv_auc >= float(policy["minimum_cv_mean_oriented_auc"]),
        "held_out_families": held_out_families
        >= int(policy["minimum_held_out_prompt_families"]),
    }
    causal_checks = {
        "activation_effect": abs(activation_effect)
        >= float(policy["minimum_absolute_activation_effect"]),
        "parameter_effect": abs(parameter_effect)
        >= float(policy["minimum_absolute_parameter_effect"]),
        "signed_effect_agreement": (
            sign_agreement if bool(policy["require_signed_effect_agreement"]) else True
        ),
        "guardrail": guardrail_passed if bool(policy["require_guardrail_pass"]) else True,
    }
    screen_passed = all(screen_checks.values())
    causal_passed = screen_passed and all(causal_checks.values())
    return {
        "component_id": component_id,
        "activation_key": activation_key,
        "feature_index": feature_index,
        "cosine": cosine,
        "absolute_cosine": abs(cosine),
        "shuffle_max_stat_p": p_value,
        "cv_mean_oriented_auc": cv_auc,
        "activation_intervention_effect": activation_effect,
        "parameter_intervention_effect": parameter_effect,
        "signed_effect_agreement": sign_agreement,
        "guardrail_passed": guardrail_passed,
        "held_out_prompt_families": held_out_families,
        "n_pairs": n_pairs,
        "screen_checks": screen_checks,
        "causal_checks": causal_checks,
        "passes_sae_screen": screen_passed,
        "passes_causal_gate": causal_passed,
    }


def validate_result(
    result: dict[str, Any],
    *,
    packets: dict[str, dict[str, Any]],
    packet_hashes: dict[str, str],
    workers: dict[str, dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if result.get("schema") != "sch.vpd-sae-worker-result.v1":
        raise ValueError("unsupported worker result schema")
    packet_id = str(result["packet_id"])
    if packet_id not in packets:
        raise ValueError(f"worker result references unknown packet: {packet_id}")
    packet = packets[packet_id]
    if str(result["cluster_id"]) != str(packet["cluster_id"]):
        raise ValueError("worker result cluster does not match packet")
    worker_id = str(result["worker_id"])
    worker = workers.get(worker_id)
    if worker is None or not bool(worker.get("enabled", False)):
        raise ValueError(f"worker is absent or disabled: {worker_id}")
    assigned_worker = packet["distribution"].get("assigned_worker")
    if bool(policy["require_assigned_worker"]) and worker_id != assigned_worker:
        raise ValueError(f"result came from unassigned worker: {worker_id}")
    verify_signature(result, worker)
    inputs = result["inputs"]
    if require_digest(inputs["packet_sha256"], "packet_sha256") != packet_hashes[packet_id]:
        raise ValueError("worker used a different packet artifact")
    for field in STUDY_DIGEST_FIELDS:
        require_digest(inputs[field], field)
    packet_effect_lineage = validate_packet_effect_input_lineage(
        packet=packet,
        inputs=inputs,
    )
    for field in ("model_id", "revision", "alignment_activation_key", "label", "phase"):
        if not str(inputs.get(field, "")).strip():
            raise ValueError(f"{field} must be non-empty")
    if int(inputs["permutation_seed"]) < 0:
        raise ValueError("permutation_seed must be non-negative")
    if int(inputs["permutation_count"]) <= 0:
        raise ValueError("permutation_count must be positive")
    alignment_key = str(packet["sae_study"]["alignment_activation_key"])
    if str(inputs["alignment_activation_key"]) != alignment_key:
        raise ValueError("worker used a different SAE/effect alignment activation key")
    controls = sorted(str(value) for value in result["controls_executed"])
    required_controls = sorted(str(value) for value in packet["sae_study"]["required_controls"])
    if controls != required_controls:
        raise ValueError("worker controls differ from frozen packet controls")
    status = str(result["status"])
    if status not in {"complete", "failed"}:
        raise ValueError(f"unsupported worker result status: {status}")
    quality = validate_quality(result["sae_quality"], policy)
    candidates: list[dict[str, Any]] = []
    seen_candidates: set[tuple[str, str, int]] = set()
    for candidate in result.get("candidates", []):
        evaluated = evaluate_candidate(
            candidate,
            packet=packet,
            quality_passed=bool(quality["quality_gate_passed"]),
            policy=policy,
        )
        key = (
            str(evaluated["component_id"]),
            str(evaluated["activation_key"]),
            int(evaluated["feature_index"]),
        )
        if key in seen_candidates:
            raise ValueError(f"duplicate candidate in worker result: {key}")
        seen_candidates.add(key)
        candidates.append(evaluated)
    if status == "failed" and candidates:
        raise ValueError("failed worker result must not contain candidates")
    receipt = {
        "schema": "sch.vpd-sae-accepted-result.v1",
        "result_id": str(result["result_id"]),
        "run_id": str(result["run_id"]),
        "packet_id": packet_id,
        "cluster_id": str(packet["cluster_id"]),
        "worker_id": worker_id,
        "status": status,
        "inputs": dict(inputs),
        "packet_effect_lineage": packet_effect_lineage,
        "sae_quality": quality,
        "candidate_count": len(candidates),
        "signature_verified": True,
    }
    return receipt, candidates


def load_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_POLICY)
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema") != "sch.vpd-sae-merge-policy.v1":
        raise ValueError("unsupported merge policy schema")
    missing = set(DEFAULT_POLICY) - set(policy)
    if missing:
        raise ValueError(f"merge policy is missing fields: {sorted(missing)}")
    return policy


def shared_study_identity(inputs: dict[str, Any]) -> dict[str, Any]:
    return {field: inputs[field] for field in SHARED_STUDY_INPUT_FIELDS}


def merge_results(
    *,
    bundle_dir: Path,
    registry_path: Path,
    result_paths: list[Path],
    output_dir: Path,
    policy_path: Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    packets, packet_hashes = load_packets(bundle_dir)
    workers = load_registry(registry_path)
    policy = load_policy(policy_path)
    accepted: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    by_worker_packet: dict[tuple[str, str], str] = {}
    study_identities: dict[str, dict[str, Any]] = {}
    duplicate_receipts = 0
    for path in sorted(result_paths, key=lambda item: item.as_posix()):
        result = json.loads(path.read_text(encoding="utf-8"))
        receipt, candidates = validate_result(
            result,
            packets=packets,
            packet_hashes=packet_hashes,
            workers=workers,
            policy=policy,
        )
        key = (str(receipt["packet_id"]), str(receipt["worker_id"]))
        prior = by_worker_packet.get(key)
        if prior is not None:
            if prior != receipt["result_id"]:
                raise ValueError(f"conflicting worker results for packet: {key}")
            duplicate_receipts += 1
            continue
        by_worker_packet[key] = str(receipt["result_id"])
        identity = shared_study_identity(receipt["inputs"])
        study_identities[content_sha256(identity)] = identity
        accepted.append(receipt)
        for candidate in candidates:
            evidence.append(
                {
                    "schema": "sch.vpd-sae-candidate-evidence.v1",
                    "result_id": receipt["result_id"],
                    "packet_id": receipt["packet_id"],
                    "cluster_id": receipt["cluster_id"],
                    "worker_id": receipt["worker_id"],
                    **candidate,
                }
            )
    if len(study_identities) != 1:
        raise ValueError("worker results do not share one frozen focused-SAE study identity")
    completed_packet_ids = {
        str(row["packet_id"]) for row in accepted if row["status"] == "complete"
    }
    missing_packet_ids = sorted(set(packets) - completed_packet_ids)
    if require_complete and missing_packet_ids:
        raise ValueError(f"merge is missing completed packets: {len(missing_packet_ids)}")
    accepted.sort(key=lambda row: (row["packet_id"], row["worker_id"], row["result_id"]))
    evidence.sort(
        key=lambda row: (
            not bool(row["passes_causal_gate"]),
            not bool(row["passes_sae_screen"]),
            float(row["shuffle_max_stat_p"]),
            -float(row["absolute_cosine"]),
            str(row["component_id"]),
            int(row["feature_index"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted_results.jsonl"
    evidence_path = output_dir / "candidate_evidence.jsonl"
    write_jsonl(accepted_path, accepted)
    write_jsonl(evidence_path, evidence)
    counts = Counter(str(row["worker_id"]) for row in accepted)
    merge_body = {
        "schema": "sch.vpd-sae-decentralized-merge.v1",
        "status": "complete" if not missing_packet_ids else "incomplete",
        "source": {
            "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
            "worker_registry_sha256": sha256_file(registry_path),
            "policy_sha256": content_sha256(policy),
        },
        "policy": policy,
        "study_identity": next(iter(study_identities.values())),
        "coverage": {
            "expected_packets": len(packets),
            "completed_packets": len(completed_packet_ids),
            "missing_packet_ids": missing_packet_ids,
            "accepted_results": len(accepted),
            "duplicate_receipts_ignored": duplicate_receipts,
            "worker_result_counts": dict(sorted(counts.items())),
        },
        "evidence": {
            "candidate_rows": len(evidence),
            "sae_screen_passed": sum(bool(row["passes_sae_screen"]) for row in evidence),
            "causal_gate_passed": sum(bool(row["passes_causal_gate"]) for row in evidence),
        },
        "artifacts": {
            "accepted_results": {
                "path": accepted_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(accepted_path),
            },
            "candidate_evidence": {
                "path": evidence_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(evidence_path),
            },
        },
    }
    manifest = {**merge_body, "merge_id": content_sha256(merge_body)}
    write_json(output_dir / "merge_manifest.json", manifest)
    return manifest


def verify_merge(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "merge_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    merge_id = str(manifest.pop("merge_id"))
    if content_sha256(manifest) != merge_id:
        raise ValueError("merge manifest content address mismatch")
    for name, entry in manifest["artifacts"].items():
        if sha256_file(output_dir / entry["path"]) != entry["sha256"]:
            raise ValueError(f"merged artifact hash mismatch: {name}")
    accepted = load_jsonl(output_dir / manifest["artifacts"]["accepted_results"]["path"])
    evidence_path = output_dir / manifest["artifacts"]["candidate_evidence"]["path"]
    evidence = (
        load_jsonl(evidence_path) if evidence_path.stat().st_size else []
    )
    if len(accepted) != int(manifest["coverage"]["accepted_results"]):
        raise ValueError("accepted result count mismatch")
    if len(evidence) != int(manifest["evidence"]["candidate_rows"]):
        raise ValueError("candidate evidence count mismatch")
    return {
        "schema": "sch.vpd-sae-merge-verification.v1",
        "status": "pass",
        "merge_status": manifest["status"],
        "accepted_results": len(accepted),
        "candidate_rows": len(evidence),
        "merge_id": merge_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sign, verify, and deterministically merge decentralized VPD/SAE results."
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--verify-output", type=Path)
    parser.add_argument("--generate-key")
    parser.add_argument("--private-key-out", type=Path)
    parser.add_argument("--public-record-out", type=Path)
    parser.add_argument("--sign-result", type=Path)
    parser.add_argument("--private-key", type=Path)
    parser.add_argument("--signed-output", type=Path)
    args = parser.parse_args()
    if args.generate_key:
        if args.private_key_out is None or args.public_record_out is None:
            parser.error("key generation requires --private-key-out and --public-record-out")
        generate_worker_key(args.generate_key, args.private_key_out, args.public_record_out)
        return
    if args.sign_result is not None:
        if args.private_key is None or args.signed_output is None:
            parser.error("result signing requires --private-key and --signed-output")
        result = json.loads(args.sign_result.read_text(encoding="utf-8"))
        write_json(args.signed_output, sign_result(result, load_private_key(args.private_key)))
        return
    if args.verify_output is not None:
        print(json.dumps(verify_merge(args.verify_output), indent=2, sort_keys=True))
        return
    if any(value is None for value in (args.bundle, args.registry, args.results_dir, args.output_dir)):
        parser.error("merge requires --bundle, --registry, --results-dir, and --output-dir")
    result_paths = sorted(args.results_dir.glob("*.json"))
    if not result_paths:
        raise ValueError("results directory contains no JSON receipts")
    manifest = merge_results(
        bundle_dir=args.bundle,
        registry_path=args.registry,
        result_paths=result_paths,
        output_dir=args.output_dir,
        policy_path=args.policy,
        require_complete=args.require_complete,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
