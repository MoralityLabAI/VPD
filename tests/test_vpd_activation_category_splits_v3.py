from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.authz_mechinterp.activation_category_split_contract import (
    verify_activation_category_split_bundle,
    verify_activation_category_split_role,
)
from experiments.authz_mechinterp.build_activation_category_splits_v3 import (
    SPLIT_ROLES,
    build_semantic_split_bundle,
    parse_split_weights,
    verify_semantic_split_bundle,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_jsonl
from experiments.authz_mechinterp.generate_activation_category_pack_v2 import (
    generate_activation_category_episodes_v2,
)


class ActivationCategorySplitV3Tests(unittest.TestCase):
    def test_five_role_semantic_split_is_exact_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episodes.jsonl"
            rows = list(
                generate_activation_category_episodes_v2(
                    seed=310829,
                    pairs_per_family=2,
                )
            )
            write_jsonl(source, rows)
            output = root / "splits"
            manifest = build_semantic_split_bundle(
                rows,
                source_sha256=sha256_file(source),
                output_dir=output,
                seed=330829,
                minimum_episodes_per_category_per_split=1,
            )
            self.assertEqual(manifest["protocol"]["roles"], list(SPLIT_ROLES))
            self.assertIn("cluster_validation", manifest["splits"])
            self.assertEqual(
                sum(entry["rows"] for entry in manifest["splits"].values()),
                len(rows),
            )
            verification = verify_semantic_split_bundle(
                output,
                source_path=source,
            )
            self.assertEqual(verification["status"], "pass")
            self.assertEqual(verification["roles"], 5)
            self.assertEqual(
                verify_activation_category_split_bundle(
                    output,
                    source_path=source,
                )["split_id"],
                manifest["split_id"],
            )

    def test_cluster_validation_role_provenance_is_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episodes.jsonl"
            rows = list(
                generate_activation_category_episodes_v2(
                    seed=310829,
                    pairs_per_family=2,
                )
            )
            write_jsonl(source, rows)
            output = root / "splits"
            manifest = build_semantic_split_bundle(
                rows,
                source_sha256=sha256_file(source),
                output_dir=output,
                seed=330829,
                minimum_episodes_per_category_per_split=1,
            )
            role_path = output / "cluster_validation_episodes.jsonl"
            provenance = verify_activation_category_split_role(
                role_path,
                output / "split_manifest.json",
                "cluster_validation",
            )
            self.assertEqual(
                provenance["activation_category_split_role"],
                "cluster_validation",
            )
            self.assertEqual(
                provenance["activation_category_split_schema"],
                "sch.vpd-activation-category-split-manifest.v3",
            )
            self.assertEqual(
                provenance["activation_category_split_id"],
                manifest["split_id"],
            )

    def test_v3_weight_parser_requires_every_role(self) -> None:
        self.assertEqual(
            parse_split_weights(
                "sae_training=1,discovery=2,calibration=1,"
                "cluster_validation=2,final_evaluation=1"
            ),
            {
                "sae_training": 1,
                "discovery": 2,
                "calibration": 1,
                "cluster_validation": 2,
                "final_evaluation": 1,
            },
        )
        with self.assertRaisesRegex(ValueError, "must name"):
            parse_split_weights(
                "sae_training=1,discovery=1,calibration=1,final_evaluation=1"
            )


if __name__ == "__main__":
    unittest.main()
