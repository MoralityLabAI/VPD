from __future__ import annotations

import base64
import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file
from experiments.authz_mechinterp.merge_vpd_effect_shards import write_deterministic_npz
from experiments.authz_mechinterp.merge_vpd_effect_shards import verify_effect_merge
from experiments.authz_mechinterp.merge_vpd_sae_results import sign_result, verify_merge
from experiments.authz_mechinterp.vpd_web_exchange import (
    SubmissionConflict,
    VpdWebExchange,
    make_handler,
)
from experiments.authz_mechinterp.vpd_web_worker import (
    submit_effect as worker_submit_effect,
    submit_sae as worker_submit_sae,
    sync_packets,
)
from tests.test_vpd_sae_result_merge import (
    make_bundle,
    public_base64,
    unsigned_result,
)


class VpdWebExchangeTests(unittest.TestCase):
    def prepare(
        self, root: Path
    ) -> tuple[
        Path,
        Path,
        dict[str, dict],
        dict[str, str],
        dict[str, Ed25519PrivateKey],
    ]:
        bundle, packets, hashes = make_bundle(root)
        worker_ids = {
            str(packet["distribution"]["assigned_worker"])
            for packet in packets.values()
        }
        keys = {worker_id: Ed25519PrivateKey.generate() for worker_id in worker_ids}
        registry_path = root / "registry.json"
        registry_path.write_text(
            json.dumps(
                {
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
            ),
            encoding="utf-8",
        )
        return bundle, registry_path, packets, hashes, keys

    def effect_submission(
        self,
        root: Path,
        packet: dict,
        packet_sha256: str,
        key: Ed25519PrivateKey,
        *,
        run_suffix: str = "",
    ) -> tuple[dict, bytes, str]:
        component_ids = sorted(str(value) for value in packet["component_ids"])
        vectors = np.asarray(
            [[float(index), float(index + 1), 1.0] for index in range(len(component_ids))],
            dtype=np.float32,
        )
        shard_path = root / f"{packet['packet_id']}{run_suffix}.npz"
        write_deterministic_npz(
            shard_path,
            {
                "component_ids": np.asarray(component_ids),
                "delta_h": vectors,
            },
        )
        shard_bytes = shard_path.read_bytes()
        worker_id = str(packet["distribution"]["assigned_worker"])
        receipt = {
            "schema": "sch.vpd-effect-shard-receipt.v1",
            "packet_id": packet["packet_id"],
            "cluster_id": packet["cluster_id"],
            "worker_id": worker_id,
            "run_id": f"effect-run{run_suffix}",
            "inputs": {
                "packet_sha256": packet_sha256,
                "model_id": "example/model",
                "revision": "revision-1",
                "adapter_sha256": "a" * 64,
                "dataset_sha256": "b" * 64,
                "runner_sha256": "c" * 64,
                "effect_space_id": packet["effect_refinement"]["effect_space_id"],
                "phase": "pre_gate",
                "estimator": "mean_central_difference_lora_rank_multiplier",
                "perturbation_epsilon": 0.1,
                "selection_seed": 1707,
                "episode_order_sha256": "d" * 64,
                "episodes": 16,
                    "generated_tokens": 0,
            },
            "shard_filename": shard_path.name,
            "shard_sha256": sha256_file(shard_path),
            "component_id_set_sha256": content_sha256(component_ids),
            "rows": len(component_ids),
            "hidden_size": 3,
            "dtype": "float32",
            "factor_restoration_verified": True,
        }
        signed = sign_result(receipt, key)
        submission = {
            "schema": "sch.vpd-effect-web-submission.v1",
            "receipt": signed,
            "shard_base64": base64.b64encode(shard_bytes).decode("ascii"),
        }
        return submission, shard_bytes, str(receipt["shard_sha256"])

    @staticmethod
    def request(
        server: ThreadingHTTPServer,
        method: str,
        path: str,
        body: dict | None = None,
    ) -> tuple[int, dict | bytes, dict[str, str]]:
        connection = http.client.HTTPConnection(
            server.server_address[0], server.server_address[1], timeout=10
        )
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        headers = {} if encoded is None else {"Content-Type": "application/json"}
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        data = response.read()
        response_headers = {name.lower(): value for name, value in response.getheaders()}
        connection.close()
        if response_headers.get("content-type") == "application/json":
            return response.status, json.loads(data), response_headers
        return response.status, data, response_headers

    def test_real_http_discovery_admission_conflict_and_object_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, packets, hashes, keys = self.prepare(root)
            exchange = VpdWebExchange(
                bundle_dir=bundle,
                registry_path=registry,
                state_dir=root / "state",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(exchange))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                exchange_url = f"http://127.0.0.1:{server.server_address[1]}"
                packet = next(iter(packets.values()))
                packet_id = str(packet["packet_id"])
                worker_id = str(packet["distribution"]["assigned_worker"])
                status, network, _ = self.request(server, "GET", "/v1/network")
                self.assertEqual(status, 200)
                self.assertEqual(network["network_id"], exchange.network()["network_id"])

                sync = sync_packets(
                    exchange_url=exchange_url,
                    worker_id=worker_id,
                    output_dir=root / "worker-sync",
                )
                synced_packet = root / "worker-sync" / "packets" / f"{packet_id}.json"
                self.assertTrue(synced_packet.is_file())
                self.assertEqual(sha256_file(synced_packet), hashes[packet_id])
                self.assertEqual(sync["network_id"], network["network_id"])

                status, work, _ = self.request(
                    server, "GET", f"/v1/workers/{worker_id}/work"
                )
                self.assertEqual(status, 200)
                selected = next(row for row in work["packets"] if row["packet_id"] == packet_id)
                self.assertEqual(selected["effect_status"], "pending")

                submission, shard_bytes, shard_sha256 = self.effect_submission(
                    root, packet, hashes[packet_id], keys[worker_id]
                )
                receipt_path = root / "signed-effect-receipt.json"
                receipt_path.write_text(json.dumps(submission["receipt"]), encoding="utf-8")
                admission = worker_submit_effect(
                    exchange_url=exchange_url,
                    receipt_path=receipt_path,
                    shard_path=root / submission["receipt"]["shard_filename"],
                )
                self.assertEqual(admission["status"], "accepted")

                status, duplicate, _ = self.request(
                    server, "POST", "/v1/submissions/effect", submission
                )
                self.assertEqual(status, 200)
                self.assertEqual(duplicate["status"], "idempotent")
                self.assertEqual(duplicate["admission_id"], admission["admission_id"])

                conflict, _, _ = self.effect_submission(
                    root, packet, hashes[packet_id], keys[worker_id], run_suffix="-other"
                )
                status, error, _ = self.request(
                    server, "POST", "/v1/submissions/effect", conflict
                )
                self.assertEqual(status, 409)
                self.assertIn("different effect result", error["error"])

                status, fetched, headers = self.request(
                    server, "GET", f"/v1/objects/sha256/{shard_sha256}"
                )
                self.assertEqual(status, 200)
                self.assertEqual(fetched, shard_bytes)
                self.assertEqual(headers["etag"], f'"sha256:{shard_sha256}"')
                self.assertEqual(headers["cache-control"], "public, immutable")

                sae = sign_result(
                    unsigned_result(packet, hashes[packet_id], worker_id), keys[worker_id]
                )
                sae_path = root / "signed-sae-result.json"
                sae_path.write_text(json.dumps(sae), encoding="utf-8")
                sae_admission = worker_submit_sae(
                    exchange_url=exchange_url,
                    result_path=sae_path,
                )
                self.assertEqual(sae_admission["status"], "accepted")

                status, snapshot, _ = self.request(server, "GET", "/v1/snapshot")
                self.assertEqual(status, 200)
                self.assertEqual(snapshot["coverage"]["effect_packets"], 1)
                self.assertEqual(snapshot["coverage"]["sae_packets"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=10)

    def test_mirrors_derive_same_snapshot_independent_of_arrival_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, packets, hashes, keys = self.prepare(root)
            exchanges = [
                VpdWebExchange(
                    bundle_dir=bundle,
                    registry_path=registry,
                    state_dir=root / f"mirror-{index}",
                )
                for index in range(2)
            ]
            effect_submissions = []
            sae_submissions = []
            for packet in packets.values():
                packet_id = str(packet["packet_id"])
                worker_id = str(packet["distribution"]["assigned_worker"])
                submission, _, _ = self.effect_submission(
                    root, packet, hashes[packet_id], keys[worker_id]
                )
                effect_submissions.append(submission)
                sae_submissions.append(
                    {
                        "schema": "sch.vpd-sae-web-submission.v1",
                        "result": sign_result(
                            unsigned_result(packet, hashes[packet_id], worker_id),
                            keys[worker_id],
                        ),
                    }
                )
            for submission in effect_submissions:
                exchanges[0].submit_effect(submission)
            for submission in sae_submissions:
                exchanges[0].submit_sae(submission)
            for submission in reversed(effect_submissions):
                exchanges[1].submit_effect(submission)
            for submission in reversed(sae_submissions):
                exchanges[1].submit_sae(submission)
            first = exchanges[0].snapshot()
            second = exchanges[1].snapshot()
            self.assertEqual(first, second)
            self.assertEqual(first["coverage"]["effect_missing_packet_ids"], [])
            self.assertEqual(first["coverage"]["sae_missing_packet_ids"], [])

            effect_manifests = [
                exchange.reduce_effects(root / f"effect-merge-{index}")
                for index, exchange in enumerate(exchanges)
            ]
            sae_manifests = [
                exchange.reduce_sae(root / f"sae-merge-{index}")
                for index, exchange in enumerate(exchanges)
            ]
            self.assertEqual(effect_manifests[0]["merge_id"], effect_manifests[1]["merge_id"])
            self.assertEqual(sae_manifests[0]["merge_id"], sae_manifests[1]["merge_id"])
            self.assertEqual(verify_effect_merge(root / "effect-merge-0")["status"], "pass")
            self.assertEqual(verify_merge(root / "sae-merge-0")["status"], "pass")

    def test_direct_conflicting_result_is_rejected_without_snapshot_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, packets, hashes, keys = self.prepare(root)
            exchange = VpdWebExchange(
                bundle_dir=bundle,
                registry_path=registry,
                state_dir=root / "state",
            )
            packet = next(iter(packets.values()))
            packet_id = str(packet["packet_id"])
            worker_id = str(packet["distribution"]["assigned_worker"])
            first, _, _ = self.effect_submission(
                root, packet, hashes[packet_id], keys[worker_id]
            )
            second, _, _ = self.effect_submission(
                root, packet, hashes[packet_id], keys[worker_id], run_suffix="-conflict"
            )
            exchange.submit_effect(first)
            snapshot_id = exchange.snapshot()["snapshot_id"]
            with self.assertRaises(SubmissionConflict):
                exchange.submit_effect(second)
            self.assertEqual(exchange.snapshot()["snapshot_id"], snapshot_id)


if __name__ == "__main__":
    unittest.main()
