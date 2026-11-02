from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.authz_mechinterp.cluster_vpd_for_sae import build_bundle, content_sha256
from experiments.authz_mechinterp.merge_vpd_sae_results import (
    DEFAULT_POLICY,
    merge_results,
    sign_result,
    validate_result,
    verify_merge,
)
from tests.test_vpd_sae_clusters import synthetic_rows


def public_base64(private_key: Ed25519PrivateKey) -> str:
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def make_bundle(root: Path) -> tuple[Path, dict[str, dict], dict[str, str]]:
    rows = synthetic_rows()
    bundle = root / "bundle"
    manifest = build_bundle(
        rows,
        inventory_sha256=content_sha256(rows),
        effects_path=None,
        output_dir=bundle,
        cluster_count=2,
        seed=13,
        projection_dim=3,
        metadata_weight=0.2,
        max_iterations=50,
        anchors=["proposal_start"],
        workers=["worker-a", "worker-b"],
    )
    packets = {}
    hashes = {}
    for entry in manifest["artifacts"]["packets"]:
        packet = json.loads((bundle / entry["packet_path"]).read_text())
        packets[packet["packet_id"]] = packet
        hashes[packet["packet_id"]] = entry["packet_sha256"]
    return bundle, packets, hashes


def unsigned_result(packet: dict, packet_sha256: str, worker_id: str) -> dict:
    return {
        "schema": "sch.vpd-sae-worker-result.v1",
        "packet_id": packet["packet_id"],
        "cluster_id": packet["cluster_id"],
        "worker_id": worker_id,
        "run_id": f"run-{packet['cluster_id']}",
        "status": "complete",
        "inputs": {
            "packet_sha256": packet_sha256,
            "sae_dictionary_sha256": "1" * 64,
            "sae_dictionary_manifest_sha256": "f" * 64,
            "activation_capture_sha256": "2" * 64,
            "activation_capture_manifest_sha256": "0" * 64,
            "dataset_sha256": "3" * 64,
            "effect_archive_sha256": "4" * 64,
            "effect_merge_manifest_sha256": "5" * 64,
            "intervention_dataset_sha256": "8" * 64,
            "intervention_trace_sha256": "6" * 64,
            "executor_sha256": "7" * 64,
            "intervention_runner_sha256": "9" * 64,
            "adapter_sha256": "a" * 64,
            "model_id": "example/model",
            "revision": "revision-1",
            "alignment_activation_key": packet["sae_study"]["alignment_activation_key"],
            "label": "semantic_authorization",
            "phase": "pre_gate",
            "permutation_seed": 17,
            "permutation_count": 999,
        },
        "controls_executed": packet["sae_study"]["required_controls"],
        "sae_quality": {
            "reconstruction_fvu": 0.08,
            "dead_feature_rate": 0.03,
            "mean_l0": 28.0,
        },
        "candidates": [
            {
                "component_id": packet["component_ids"][0],
                "activation_key": packet["focus"]["recommended_activation_keys"][0],
                "feature_index": 7,
                "cosine": 0.61,
                "shuffle_max_stat_p": 0.01,
                "cv_mean_oriented_auc": 0.79,
                "activation_intervention_effect": -0.08,
                "parameter_intervention_effect": -0.05,
                "guardrail_passed": True,
                "held_out_prompt_families": 3,
                "n_pairs": 48,
            }
        ],
    }


class VpdSaeResultMergeTests(unittest.TestCase):
    def prepare(self, root: Path) -> tuple[Path, Path, dict[str, dict], dict[str, str], dict]:
        bundle, packets, hashes = make_bundle(root)
        worker_ids = {packet["distribution"]["assigned_worker"] for packet in packets.values()}
        keys = {worker_id: Ed25519PrivateKey.generate() for worker_id in worker_ids}
        registry = {
            "schema": "sch.vpd-sae-worker-registry.v1",
            "workers": [
                {
                    "worker_id": worker_id,
                    "algorithm": "ed25519",
                    "public_key_base64": public_base64(key),
                    "enabled": True,
                }
                for worker_id, key in sorted(keys.items())
            ],
        }
        registry_path = root / "registry.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        return bundle, registry_path, packets, hashes, keys

    def test_signed_complete_merge_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, packets, hashes, keys = self.prepare(root)
            paths = []
            for index, packet in enumerate(packets.values()):
                worker = packet["distribution"]["assigned_worker"]
                signed = sign_result(unsigned_result(packet, hashes[packet["packet_id"]], worker), keys[worker])
                path = root / f"result-{index}.json"
                path.write_text(json.dumps(signed), encoding="utf-8")
                paths.append(path)
            manifest = merge_results(
                bundle_dir=bundle,
                registry_path=registry,
                result_paths=paths,
                output_dir=root / "merged",
                require_complete=True,
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["coverage"]["completed_packets"], 2)
            self.assertEqual(manifest["evidence"]["causal_gate_passed"], 2)
            self.assertEqual(verify_merge(root / "merged")["status"], "pass")

    def test_tampered_signature_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, packets, hashes, keys = self.prepare(root)
            packet = next(iter(packets.values()))
            worker = packet["distribution"]["assigned_worker"]
            signed = sign_result(unsigned_result(packet, hashes[packet["packet_id"]], worker), keys[worker])
            signed["sae_quality"]["reconstruction_fvu"] = 0.01
            path = root / "tampered.json"
            path.write_text(json.dumps(signed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content address mismatch"):
                merge_results(
                    bundle_dir=bundle,
                    registry_path=registry,
                    result_paths=[path],
                    output_dir=root / "merged",
                )

    def test_component_outside_packet_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, packets, hashes, keys = self.prepare(root)
            packet_list = list(packets.values())
            packet = packet_list[0]
            outside = packet_list[1]["component_ids"][0]
            worker = packet["distribution"]["assigned_worker"]
            result = unsigned_result(packet, hashes[packet["packet_id"]], worker)
            result["candidates"][0]["component_id"] = outside
            signed = sign_result(result, keys[worker])
            path = root / "outside.json"
            path.write_text(json.dumps(signed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside packet"):
                merge_results(
                    bundle_dir=bundle,
                    registry_path=registry,
                    result_paths=[path],
                    output_dir=root / "merged",
                )

    def test_conflicting_duplicate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, packets, hashes, keys = self.prepare(root)
            packet = next(iter(packets.values()))
            worker = packet["distribution"]["assigned_worker"]
            first = unsigned_result(packet, hashes[packet["packet_id"]], worker)
            second = unsigned_result(packet, hashes[packet["packet_id"]], worker)
            second["run_id"] += "-conflict"
            paths = []
            for name, result in (("a", first), ("b", second)):
                path = root / f"{name}.json"
                path.write_text(json.dumps(sign_result(result, keys[worker])), encoding="utf-8")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "conflicting worker results"):
                merge_results(
                    bundle_dir=bundle,
                    registry_path=registry,
                    result_paths=paths,
                    output_dir=root / "merged",
                )

    def test_signed_category_result_requires_exact_packet_effect_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, packets, _, keys = self.prepare(root)
            packet = dict(next(iter(packets.values())))
            packet["source"] = {
                **packet["source"],
                "effects_sha256": "4" * 64,
                "effects_merge_manifest_sha256": "5" * 64,
            }
            packet["sae_study"] = {
                **packet["sae_study"],
                "activation_compartment": {"category": "replayed"},
            }
            packet_id = str(packet["packet_id"])
            worker_id = str(packet["distribution"]["assigned_worker"])
            packet_hash = "f" * 64
            result = unsigned_result(packet, packet_hash, worker_id)
            result["inputs"]["packet_effect_lineage"] = {
                "schema": "sch.vpd-focused-sae-packet-effect-lineage.v1",
                "status": "pass",
                "effect_archive_sha256": "4" * 64,
                "effect_merge_manifest_sha256": "5" * 64,
                "empirical_category_claim_allowed": True,
            }
            worker = {
                "worker_id": worker_id,
                "algorithm": "ed25519",
                "public_key_base64": public_base64(keys[worker_id]),
                "enabled": True,
            }
            signed = sign_result(result, keys[worker_id])
            receipt, _ = validate_result(
                signed,
                packets={packet_id: packet},
                packet_hashes={packet_id: packet_hash},
                workers={worker_id: worker},
                policy=dict(DEFAULT_POLICY),
            )
            self.assertEqual(receipt["packet_effect_lineage"]["status"], "pass")
            self.assertTrue(
                receipt["packet_effect_lineage"][
                    "independently_verified_by_reducer"
                ]
            )

            for field, message in (
                ("effect_archive_sha256", "effect archive differs"),
                ("effect_merge_manifest_sha256", "merge manifest differs"),
            ):
                substituted = unsigned_result(packet, packet_hash, worker_id)
                substituted["inputs"]["packet_effect_lineage"] = dict(
                    result["inputs"]["packet_effect_lineage"]
                )
                substituted["inputs"][field] = "e" * 64
                resigned = sign_result(substituted, keys[worker_id])
                with self.assertRaisesRegex(ValueError, message):
                    validate_result(
                        resigned,
                        packets={packet_id: packet},
                        packet_hashes={packet_id: packet_hash},
                        workers={worker_id: worker},
                        policy=dict(DEFAULT_POLICY),
                    )


if __name__ == "__main__":
    unittest.main()
