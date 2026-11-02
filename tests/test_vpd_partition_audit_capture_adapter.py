from __future__ import annotations

import base64
import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.authz_mechinterp.attest_partition_audit_effect_receipt import (
    attest_partition_audit_receipt,
    build_dataset_attestation,
    verify_attested_partition_audit_receipt,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    stable_hash,
    write_json,
)
from experiments.authz_mechinterp.effect_uncertainty import uncertainty_contract
from experiments.authz_mechinterp.compute_vpd_effect_shard_v2 import (
    corrected_stable_hash,
)
from experiments.authz_mechinterp.merge_vpd_partition_audit_effect_shards import (
    validate_signed_partition_audit_receipts,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import (
    crypto_types,
    sign_result,
)


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "artifacts" / "vpd_partition_audit_pack_v1"
GATE_PLAN = ROOT / "artifacts" / "vpd_partition_audit_gate_v1" / "plan.json"
GATE_PLAN_V2 = ROOT / "artifacts" / "vpd_partition_audit_gate_v2" / "plan.json"


def public_key_base64(key: object) -> str:
    _, _, serialization, _ = crypto_types()
    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def fixture_base_receipt(
    gate_plan_path: Path = GATE_PLAN,
) -> tuple[dict, dict]:
    plan = json.loads(gate_plan_path.read_text(encoding="utf-8"))
    work_unit = plan["capture_work_units"][0]
    dataset = build_dataset_attestation(
        pack_dir=PACK,
        selection_seed=1707,
        maximum_episodes=0,
    )
    receipt = {
        "schema": "sch.vpd-effect-shard-receipt.v5",
        "packet_id": work_unit["packet_id"],
        "cluster_id": work_unit["cluster_id"],
        "worker_id": "worker-a",
        "run_id": "fixture-partition-audit",
        "inputs": {
            "packet_sha256": work_unit["packet_sha256"],
            "model_id": "fixture/model",
            "revision": "fixture-revision",
            "adapter_sha256": "1" * 64,
            "dataset_sha256": dataset["dataset_sha256"],
            "runner_sha256": sha256_file(
                ROOT / "experiments" / "authz_mechinterp" / "compute_vpd_effect_shard.py"
            ),
            "effect_space_id": "layer_35__proposal_start",
            "phase": "pre_gate",
            "estimator": "mean_central_difference_lora_rank_multiplier",
            "perturbation_epsilon": 0.1,
            "selection_seed": 1707,
            "episode_order_sha256": dataset["episode_order_sha256"],
            "episodes": dataset["episodes"],
            "generated_tokens": 0,
            "activation_category_contract_sha256": dataset[
                "activation_category_contract_sha256"
            ],
            "activation_categories": dataset["activation_categories"],
            "activation_category_episode_counts": dataset[
                "activation_category_episode_counts"
            ],
            "activation_category_episode_ids_sha256": dataset[
                "activation_category_episode_ids_sha256"
            ],
        },
        "shard_filename": "fixture.npz",
        "shard_sha256": "2" * 64,
        "component_id_set_sha256": work_unit["component_id_set_sha256"],
        "rows": work_unit["component_count"],
        "hidden_size": 4,
        "activation_category_count": 8,
        "dtype": "float32",
        "factor_restoration_verified": True,
        "claim_boundary": "base worker fixture",
    }
    if plan["schema"] == "sch.vpd-partition-audit-gate-plan.v2":
        parameters = plan["source"]["discovery_effect_capture_campaign"][
            "capture_parameters"
        ]
        receipt["hidden_size"] = parameters["expected_hidden_size"]
        receipt["inputs"]["activation_category_uncertainty"] = uncertainty_contract(
            block_count=parameters["uncertainty_blocks"],
            sketch_dimension=parameters["uncertainty_sketch_dimension"],
                seed=(
                    int(
                        stable_hash(
                            parameters["selection_seed"],
                            "category-effect-countsketch",
                            parameters["effect_space_id"],
                        ),
                        16,
                    )
                    % (2**32)
                ),
        )
    return receipt, work_unit


class VpdPartitionAuditCaptureAdapterTests(unittest.TestCase):
    def test_v2_worker_wrapper_only_converts_countsketch_seed_hash(self) -> None:
        ordinary = corrected_stable_hash(1707, "episode", "fixture")
        countsketch = corrected_stable_hash(
            1707, "category-effect-countsketch", "layer_35__proposal_start"
        )
        self.assertIsInstance(ordinary, str)
        self.assertIsInstance(countsketch, int)
        self.assertEqual(countsketch % (2**32), 3799294060)

    def test_v2_receipt_binds_verified_campaign_measurement_contract(self) -> None:
        receipt, work_unit = fixture_base_receipt(GATE_PLAN_V2)
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.json"
            write_json(base_path, receipt)
            adapted = attest_partition_audit_receipt(
                base_receipt_path=base_path,
                pack_dir=PACK,
                gate_plan_path=GATE_PLAN_V2,
                work_unit_id=work_unit["work_unit_id"],
            )
        verification = verify_attested_partition_audit_receipt(
            adapted,
            pack_dir=PACK,
            gate_plan_path=GATE_PLAN_V2,
            require_signed=False,
        )
        provenance = adapted["inputs"]["partition_audit_provenance"]
        contract = provenance["measurement_contract"]
        self.assertEqual(
            provenance["schema"], "sch.vpd-partition-audit-effect-provenance.v2"
        )
        self.assertEqual(
            verification["measurement_contract_id"],
            contract["measurement_contract_id"],
        )
        self.assertEqual(contract["capture_parameters"]["dtype"], "float16")
        self.assertIn("dtype", contract["gate_command_bound_fields"])
        self.assertFalse(contract["worker_receipt_observes_model_load_dtype"])
        self.assertEqual(
            contract["model_load_dtype_evidence"],
            "content_addressed_gate_command_and_worker_hash",
        )

    def test_v2_receipt_rejects_campaign_measurement_drift(self) -> None:
        mutations = {
            "epsilon": lambda receipt: receipt["inputs"].__setitem__(
                "perturbation_epsilon", 0.2
            ),
            "hidden_size": lambda receipt: receipt.__setitem__("hidden_size", 2048),
            "uncertainty": lambda receipt: receipt["inputs"][
                "activation_category_uncertainty"
            ].__setitem__("block_count", 4),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt, work_unit = fixture_base_receipt(GATE_PLAN_V2)
                mutate(receipt)
                with tempfile.TemporaryDirectory() as directory:
                    base_path = Path(directory) / "base.json"
                    write_json(base_path, receipt)
                    with self.assertRaisesRegex(ValueError, "campaign"):
                        attest_partition_audit_receipt(
                            base_receipt_path=base_path,
                            pack_dir=PACK,
                            gate_plan_path=GATE_PLAN_V2,
                            work_unit_id=work_unit["work_unit_id"],
                        )

    def test_unsigned_receipt_is_attested_before_signing(self) -> None:
        receipt, work_unit = fixture_base_receipt()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.json"
            write_json(base_path, receipt)
            adapted = attest_partition_audit_receipt(
                base_receipt_path=base_path,
                pack_dir=PACK,
                gate_plan_path=GATE_PLAN,
                work_unit_id=work_unit["work_unit_id"],
            )
        verification = verify_attested_partition_audit_receipt(
            adapted,
            pack_dir=PACK,
            gate_plan_path=GATE_PLAN,
            require_signed=False,
        )
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(verification["work_unit_id"], work_unit["work_unit_id"])
        provenance = adapted["inputs"]["partition_audit_provenance"]
        self.assertEqual(provenance["role"], "partition_audit")
        self.assertFalse(provenance["final_evaluation_consumed"])
        self.assertNotIn("activation_category_split_role", adapted["inputs"])

    def test_signed_adapter_receipt_passes_dedicated_front_end(self) -> None:
        receipt, work_unit = fixture_base_receipt()
        Ed25519PrivateKey, _, _, _ = crypto_types()
        key = Ed25519PrivateKey.generate()
        receipt["worker_id"] = "worker-a"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.json"
            signed_path = root / "signed.json"
            registry_path = root / "registry.json"
            write_json(base_path, receipt)
            adapted = attest_partition_audit_receipt(
                base_receipt_path=base_path,
                pack_dir=PACK,
                gate_plan_path=GATE_PLAN,
                work_unit_id=work_unit["work_unit_id"],
            )
            write_json(signed_path, sign_result(adapted, key))
            write_json(
                registry_path,
                {
                    "schema": "sch.vpd-sae-worker-registry.v1",
                    "workers": [
                        {
                            "worker_id": "worker-a",
                            "algorithm": "ed25519",
                            "public_key_base64": public_key_base64(key),
                            "enabled": True,
                        }
                    ],
                },
            )
            summary = validate_signed_partition_audit_receipts(
                receipt_paths=[signed_path],
                registry_path=registry_path,
                pack_dir=PACK,
                gate_plan_path=GATE_PLAN,
                require_complete=False,
            )
            self.assertEqual(summary["verified_work_units"], 1)
            self.assertTrue(summary["all_signatures_verified"])
            self.assertEqual(summary["status"], "incomplete")
            with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                validate_signed_partition_audit_receipts(
                    receipt_paths=[signed_path],
                    registry_path=registry_path,
                    pack_dir=PACK,
                    gate_plan_path=GATE_PLAN,
                    require_complete=True,
                )

    def test_readdressed_policy_tampering_is_rejected_even_when_resigned(self) -> None:
        receipt, work_unit = fixture_base_receipt()
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.json"
            write_json(base_path, receipt)
            adapted = attest_partition_audit_receipt(
                base_receipt_path=base_path,
                pack_dir=PACK,
                gate_plan_path=GATE_PLAN,
                work_unit_id=work_unit["work_unit_id"],
            )
        tampered = copy.deepcopy(adapted)
        provenance = tampered["inputs"]["partition_audit_provenance"]
        provenance["final_evaluation_consumed"] = True
        body = dict(provenance)
        body.pop("provenance_id")
        provenance["provenance_id"] = content_sha256(body)
        with self.assertRaisesRegex(ValueError, "policy differs"):
            verify_attested_partition_audit_receipt(
                tampered,
                pack_dir=PACK,
                gate_plan_path=GATE_PLAN,
                require_signed=False,
            )

    def test_five_role_split_provenance_cannot_be_reused(self) -> None:
        receipt, work_unit = fixture_base_receipt()
        receipt["inputs"]["activation_category_split_role"] = "final_evaluation"
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.json"
            write_json(base_path, receipt)
            with self.assertRaisesRegex(ValueError, "may not masquerade"):
                attest_partition_audit_receipt(
                    base_receipt_path=base_path,
                    pack_dir=PACK,
                    gate_plan_path=GATE_PLAN,
                    work_unit_id=work_unit["work_unit_id"],
                )


if __name__ == "__main__":
    unittest.main()
