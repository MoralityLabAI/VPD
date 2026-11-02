from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.authz_mechinterp.audit_frozen_vpd_category_partition import (
    verify_partition_audit_report,
)
from experiments.authz_mechinterp.benchmark_frozen_vpd_partition_audit import (
    _base_inputs,
    build_benchmark,
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v5 import (
    verify_protocol,
)
from experiments.authz_mechinterp.finalize_partition_audit_pack_v1 import (
    verify_partition_audit_manifest,
)
from experiments.authz_mechinterp.generate_partition_audit_pack_v1 import (
    generate_partition_audit_episodes,
)
from experiments.authz_mechinterp.plan_vpd_partition_audit_gate import (
    verify_gate_plan,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline_v3 import (
    canonical_inputs as v5_lineage_inputs,
    verify_receipt as verify_v5_lineage_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "artifacts" / "vpd_frozen_partition_audit_benchmark_v1" / "report.json"
)
PACK = ROOT / "artifacts" / "vpd_partition_audit_pack_v1"
GATE_PLAN = ROOT / "artifacts" / "vpd_partition_audit_gate_v1" / "plan.json"
V4_PROTOCOL = ROOT / "configs" / "evals" / "vpd_activation_category_compartments_v4.json"
V5_PROTOCOL = ROOT / "configs" / "evals" / "vpd_activation_category_compartments_v5.json"
BUNDLE = ROOT / "artifacts" / "bonsai8b_authz_v8" / "vpd_sae_clusters"
V5_RECEIPT = (
    ROOT / "artifacts" / "vpd_unified_compartment_pipeline_v3" / "receipt.json"
)


class VpdPartitionAuditTests(unittest.TestCase):
    def test_fresh_generator_has_four_new_balanced_semantic_families(self) -> None:
        rows = list(generate_partition_audit_episodes(pairs_per_family=8))
        self.assertEqual(len(rows), 352)
        self.assertEqual(len({row["pair_id"] for row in rows}), 32)
        self.assertEqual(len({row["template_group"] for row in rows}), 4)
        self.assertEqual(len({row["condition"] for row in rows}), 11)
        self.assertTrue(
            all(
                row["partition_audit_pack"]["use"]
                == "post_freeze_reject_only_partition_audit"
                for row in rows
            )
        )

    def test_durable_model_free_benchmark_reproduces(self) -> None:
        result = verify_benchmark(BENCHMARK)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["scenarios"], 2)
        self.assertFalse(result["execution_authorized"])

    def test_fresh_durable_pack_and_gate_plan_reproduce(self) -> None:
        pack = verify_partition_audit_manifest(PACK)
        self.assertEqual(
            pack["pack_id"],
            "e65c78bb07fcf25c5a008d94a8f447b1b75f01cb8734632e7e2e567c81ca2f23",
        )
        self.assertEqual(pack["rows"], 2816)
        gate = verify_gate_plan(
            GATE_PLAN,
            predecessor_protocol_path=V4_PROTOCOL,
            pack_dir=PACK,
            benchmark_path=BENCHMARK,
            bundle_dir=BUNDLE,
        )
        self.assertEqual(
            gate["gate_plan_id"],
            "e6762af41fcce70bca47b1a6925c9e3b53b8a88e65aee946c5c7451054818358",
        )
        self.assertEqual(gate["work_units"], 16)
        self.assertEqual(gate["episode_forward_equivalents"], 22_708_224)
        self.assertFalse(gate["execution_authorized"])

    def test_v5_protocol_binds_reject_only_audit_without_final_evaluation(self) -> None:
        verification = verify_protocol(
            V5_PROTOCOL,
            predecessor_path=V4_PROTOCOL,
            pack_dir=PACK,
            benchmark_path=BENCHMARK,
            gate_plan_path=GATE_PLAN,
            bundle_dir=BUNDLE,
        )
        self.assertEqual(
            verification["protocol_sha256"],
            "0894ac8e06ba7828bf5cb21ae79e1ae48ae96e2c27cd66e6853899765c1b9c24",
        )
        protocol = json.loads(V5_PROTOCOL.read_text(encoding="utf-8"))
        audit = protocol["component_compartments"]["post_freeze_partition_audit"]
        self.assertFalse(audit["final_evaluation_role_consumed"])
        self.assertFalse(audit["audit_may_select_thresholds_clusters_or_components"])
        self.assertFalse(audit["audit_may_promote_shared_to_specific"])
        self.assertTrue(
            protocol["sae_training"]["partition_audit_gate"][
                "category_specific_training_requires_pass"
            ]
        )

    def test_v5_lineage_amendment_reverifies_predecessor_and_audit(self) -> None:
        verification = verify_v5_lineage_receipt(
            V5_RECEIPT, **v5_lineage_inputs(ROOT)
        )
        self.assertEqual(
            verification["pipeline_receipt_id"],
            "bec00550d7d523f4ece993aafdac61f203a88c4417d8cab7bafc3ef0752831e2",
        )
        self.assertEqual(verification["partition_audit_work_units"], 16)
        self.assertFalse(verification["execution_authorized"])

    def test_report_verifier_rejects_readdressed_decision_tampering(self) -> None:
        inputs = _base_inputs()
        inputs["frozen_categories"] = [
            "category_a",
            "shared_cross_category",
            "shared_cross_category",
            "shared_cross_category",
            "shared_cross_category",
            "shared_cross_category",
        ]
        from experiments.authz_mechinterp.audit_frozen_vpd_category_partition import (
            evaluate_frozen_partition,
        )

        report = evaluate_frozen_partition(**inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json(path, report)
            verification = verify_partition_audit_report(path, **inputs)
            self.assertTrue(
                verification["eligible_for_category_specific_sae_training"]
            )
            tampered = copy.deepcopy(report)
            tampered["summary"]["eligible_for_category_specific_sae_training"] = False
            body = dict(tampered)
            body.pop("partition_audit_id")
            from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256

            tampered["partition_audit_id"] = content_sha256(body)
            write_json(path, tampered)
            with self.assertRaisesRegex(ValueError, "not reproducible"):
                verify_partition_audit_report(path, **inputs)

    def test_benchmark_builder_exposes_all_failure_modes(self) -> None:
        benchmark = build_benchmark()
        scenario = benchmark["scenarios"][
            "orthogonal_failure_modes_and_shared_nonpromotion"
        ]
        self.assertEqual(scenario["decision"], "reject_partition_audit")
        self.assertEqual(scenario["summary"]["specific_failure_count"], 4)
        self.assertFalse(scenario["shared_promoted"])


if __name__ == "__main__":
    unittest.main()
