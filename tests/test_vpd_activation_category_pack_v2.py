from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.authz_mechinterp.activation_category_split_contract import (
    verify_activation_category_split_bundle,
    verify_activation_category_split_role,
)
from experiments.authz_mechinterp.build_activation_category_splits_v2 import (
    build_semantic_split_bundle,
    verify_semantic_split_bundle,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    sha256_file,
    write_jsonl,
)
from experiments.authz_mechinterp.finalize_activation_category_pack_v2 import (
    verify_pack_manifest,
)
from experiments.authz_mechinterp.generate_activation_category_pack_v2 import (
    generate_activation_category_episodes_v2,
)
from experiments.authz_mechinterp.run_focused_vpd_sae import (
    validate_study_split_provenance,
)


class ActivationCategoryPackV2Tests(unittest.TestCase):
    @property
    def pack_dir(self) -> Path:
        return (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "vpd_activation_category_pack_v2"
        )

    def test_durable_semantic_pack_verifies(self) -> None:
        self.assertEqual(
            verify_pack_manifest(self.pack_dir),
            {
                "schema": "sch.vpd-activation-category-pack-verification.v2",
                "status": "pass",
                "pack_id": "dc5224d9838d8661764f45342543a4f3ea35152cc16be60dd7d2d11b3589910d",
                "split_id": "bf36a956222f517a68a6d7596033b4d66a59f49e2be37f946d0984f85715c606",
                "rows": 1056,
                "pairs": 96,
                "activation_categories": 8,
                "template_groups": 16,
                "model_outcomes": 0,
                "exact_partition": True,
                "template_group_sets_pairwise_disjoint": True,
            },
        )
        verification = verify_activation_category_split_bundle(
            self.pack_dir / "splits",
            source_path=self.pack_dir / "gated_episodes.jsonl",
        )
        self.assertTrue(verification["template_group_sets_pairwise_disjoint"])

    def test_every_role_has_unique_template_set_and_all_categories(self) -> None:
        manifest = json.loads(
            (self.pack_dir / "splits" / "split_manifest.json").read_text()
        )
        template_sets = [
            set(manifest["splits"][role]["template_group_counts"])
            for role in manifest["protocol"]["roles"]
        ]
        self.assertTrue(
            all(
                not left & right
                for index, left in enumerate(template_sets)
                for right in template_sets[index + 1 :]
            )
        )
        self.assertEqual(
            set.union(*template_sets), set(manifest["observed"]["template_groups"])
        )
        for role in manifest["protocol"]["roles"]:
            self.assertEqual(
                set(manifest["splits"][role]["activation_category_counts"]),
                set(manifest["observed"]["activation_categories"]),
            )

    def test_v2_machine_protocol_matches_durable_pack(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads(
            (
                root
                / "configs"
                / "evals"
                / "vpd_activation_category_compartments_v2.json"
            ).read_text()
        )
        manifest = json.loads(
            (self.pack_dir / "artifact_manifest.json").read_text()
        )
        data_pack = protocol["data_pack"]
        self.assertEqual(data_pack["pack_id"], manifest["pack_id"])
        split_manifest = json.loads(
            (root / data_pack["split_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(data_pack["split_id"], split_manifest["split_id"])
        self.assertEqual(data_pack["split_schema"], split_manifest["schema"])
        self.assertEqual(data_pack["template_groups"], 16)
        for role, role_entry in data_pack["roles"].items():
            self.assertEqual(
                role_entry["template_groups"],
                list(split_manifest["splits"][role]["template_group_counts"]),
            )

    def test_generator_has_unique_pair_constant_references_across_16_families(self) -> None:
        rows = list(
            generate_activation_category_episodes_v2(
                seed=310829, pairs_per_family=2
            )
        )
        self.assertEqual(len(rows), 352)
        self.assertEqual(len({row["template_group"] for row in rows}), 16)
        by_pair: dict[str, list[dict]] = {}
        for row in rows:
            by_pair.setdefault(str(row["pair_id"]), []).append(row)
        references = []
        for pair_rows in by_pair.values():
            pair_references = {
                row["activation_category_pack"]["case_reference"]
                for row in pair_rows
            }
            self.assertEqual(len(pair_references), 1)
            self.assertEqual(len({row["condition"] for row in pair_rows}), 11)
            references.extend(pair_references)
        self.assertEqual(len(references), len(set(references)))

    def test_semantic_split_is_deterministic_and_rejects_template_overlap(self) -> None:
        rows = list(
            generate_activation_category_episodes_v2(
                seed=310829, pairs_per_family=2
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rows.jsonl"
            write_jsonl(source, rows)
            output = root / "splits"
            manifest = build_semantic_split_bundle(
                rows,
                source_sha256=sha256_file(source),
                output_dir=output,
                seed=310829,
                minimum_episodes_per_category_per_split=2,
            )
            first = verify_semantic_split_bundle(output, source_path=source)
            self.assertEqual(first["split_id"], manifest["split_id"])
            reversed_output = root / "reversed"
            second = build_semantic_split_bundle(
                list(reversed(rows)),
                source_sha256=sha256_file(source),
                output_dir=reversed_output,
                seed=310829,
                minimum_episodes_per_category_per_split=2,
            )
            self.assertEqual(
                manifest["observed"]["template_group_role_assignment"],
                second["observed"]["template_group_role_assignment"],
            )

    def test_focused_study_enforces_semantic_role_disjointness(self) -> None:
        split_dir = self.pack_dir / "splits"
        split_manifest = split_dir / "split_manifest.json"
        provenance = {
            role: verify_activation_category_split_role(
                split_dir / f"{role}_episodes.jsonl", split_manifest, role
            )
            for role in (
                "sae_training",
                "discovery",
                "calibration",
                "final_evaluation",
            )
        }
        result = validate_study_split_provenance(
            split_manifest_path=split_manifest,
            intervention_dataset_path=split_dir / "final_evaluation_episodes.jsonl",
            dictionary_manifest={
                "source": {"activation_category_split": provenance["sae_training"]}
            },
            capture_manifest={
                "source": {"activation_category_split": provenance["final_evaluation"]}
            },
            effect_identity=provenance["discovery"],
            packet={
                "source": {
                    "activation_category_split_id": provenance["discovery"][
                        "activation_category_split_id"
                    ],
                    "activation_category_template_group_set_sha256_by_role": {
                        role: provenance[role][
                            "activation_category_split_template_group_set_sha256"
                        ]
                        for role in ("discovery", "calibration")
                    },
                }
            },
        )
        self.assertEqual(
            result["semantic_template_disjointness"]["status"], "verified"
        )
        with self.assertRaisesRegex(
            ValueError, "focused SAE role template sets are not disjoint"
        ):
            validate_study_split_provenance(
                split_manifest_path=split_manifest,
                intervention_dataset_path=split_dir / "final_evaluation_episodes.jsonl",
                dictionary_manifest={
                    "source": {
                        "activation_category_split": provenance["sae_training"]
                    }
                },
                capture_manifest={
                    "source": {
                        "activation_category_split": provenance["final_evaluation"]
                    }
                },
                effect_identity=provenance["discovery"],
                packet={
                    "source": {
                        "activation_category_split_id": provenance["discovery"][
                            "activation_category_split_id"
                        ],
                        "activation_category_template_group_set_sha256_by_role": {
                            "discovery": provenance["discovery"][
                                "activation_category_split_template_group_set_sha256"
                            ],
                            "calibration": provenance["sae_training"][
                                "activation_category_split_template_group_set_sha256"
                            ],
                        },
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
