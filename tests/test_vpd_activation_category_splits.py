from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.authz_mechinterp.build_activation_category_splits import (
    assign_pair_roles,
    build_split_bundle,
    parse_split_weights,
    verify_split_bundle,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_jsonl,
)
from experiments.authz_mechinterp.generate_activation_category_pack import (
    generate_activation_category_episodes,
)
from experiments.authz_mechinterp.finalize_activation_category_pack import (
    verify_pack_manifest,
)
from experiments.authz_mechinterp.build_activation_category_splits import (
    verify_split_artifact_role,
)
from experiments.authz_mechinterp.run_focused_vpd_sae import (
    validate_study_split_provenance,
)


def synthetic_episodes(*, pairs_per_family: int = 8) -> list[dict]:
    rows: list[dict] = []
    for family in ("family-a", "family-b"):
        for pair_index in range(pairs_per_family):
            pair_id = f"{family}-pair-{pair_index:02d}"
            for category in ("valid_authority", "replayed", "signature_invalid"):
                rows.append(
                    {
                        "episode_id": f"{pair_id}-{category}",
                        "pair_id": pair_id,
                        "template_group": family,
                        "activation_category": category,
                        "expected_gate_decision": (
                            "allow" if category == "valid_authority" else "deny"
                        ),
                    }
                )
    return rows


class ActivationCategorySplitTests(unittest.TestCase):
    def test_focused_study_requires_four_verified_roles_from_one_split(self) -> None:
        pack_dir = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "vpd_activation_category_pack_v1"
        )
        split_dir = pack_dir / "splits"
        split_manifest = split_dir / "split_manifest.json"
        sae = verify_split_artifact_role(
            split_dir / "sae_training_episodes.jsonl",
            split_manifest,
            "sae_training",
        )
        discovery = verify_split_artifact_role(
            split_dir / "discovery_episodes.jsonl",
            split_manifest,
            "discovery",
        )
        final = verify_split_artifact_role(
            split_dir / "final_evaluation_episodes.jsonl",
            split_manifest,
            "final_evaluation",
        )
        result = validate_study_split_provenance(
            split_manifest_path=split_manifest,
            intervention_dataset_path=split_dir / "final_evaluation_episodes.jsonl",
            dictionary_manifest={"source": {"activation_category_split": sae}},
            capture_manifest={"source": {"activation_category_split": final}},
            effect_identity=discovery,
            packet={"source": {"activation_category_split_id": sae[
                "activation_category_split_id"
            ]}},
        )
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["roles"]["dictionary"], "sae_training")
        bad_dictionary = {"source": {"activation_category_split": discovery}}
        with self.assertRaisesRegex(ValueError, "sae_training split role"):
            validate_study_split_provenance(
                split_manifest_path=split_manifest,
                intervention_dataset_path=split_dir / "final_evaluation_episodes.jsonl",
                dictionary_manifest=bad_dictionary,
                capture_manifest={"source": {"activation_category_split": final}},
                effect_identity=discovery,
                packet={"source": {"activation_category_split_id": sae[
                    "activation_category_split_id"
                ]}},
            )

    def test_durable_activation_category_pack_verifies(self) -> None:
        pack_dir = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "vpd_activation_category_pack_v1"
        )
        self.assertEqual(
            verify_pack_manifest(pack_dir),
            {
                "schema": "sch.vpd-activation-category-pack-verification.v1",
                "status": "pass",
                "pack_id": "3dbf9bfea845da9e8dcab78d75436dbfccb13ff5c487f66f5f76ba42b6f4a863",
                "split_id": "6fd609e550e6eb6fe11d5e68487e6e2a8e8c23331d4b978d0c3329f0c3de2931",
                "rows": 1056,
                "pairs": 96,
                "activation_categories": 8,
                "template_groups": 4,
                "model_outcomes": 0,
                "exact_partition": True,
            },
        )

    def test_prompt_variants_are_pair_constant_label_independent_and_unique(self) -> None:
        rows = list(
            generate_activation_category_episodes(seed=290829, pairs_per_family=3)
        )
        self.assertEqual(len(rows), 132)
        by_pair: dict[str, list[dict]] = {}
        for row in rows:
            by_pair.setdefault(row["pair_id"], []).append(row)
        case_references: set[str] = set()
        message_hashes: set[str] = set()
        for pair_rows in by_pair.values():
            references = {
                row["activation_category_pack"]["case_reference"] for row in pair_rows
            }
            variants = {
                row["activation_category_pack"]["context_variant_index"]
                for row in pair_rows
            }
            self.assertEqual(len(references), 1)
            self.assertEqual(len(variants), 1)
            self.assertEqual(len({row["condition"] for row in pair_rows}), 11)
            self.assertEqual(
                sum(row["expected_gate_decision"] == "allow" for row in pair_rows),
                1,
            )
            case_references.update(references)
            for row in pair_rows:
                message_hashes.add(
                    hashlib.sha256(
                        json.dumps(
                            row["messages"],
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                    ).hexdigest()
                )
        self.assertEqual(len(case_references), len(by_pair))
        self.assertEqual(len(message_hashes), len(rows))

    def test_split_is_deterministic_exact_pair_partition(self) -> None:
        rows = synthetic_episodes()
        first = assign_pair_roles(rows, seed=19, weights={
            "sae_training": 1,
            "discovery": 1,
            "calibration": 1,
            "final_evaluation": 1,
        })
        second = assign_pair_roles(list(reversed(rows)), seed=19, weights={
            "sae_training": 1,
            "discovery": 1,
            "calibration": 1,
            "final_evaluation": 1,
        })
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {
            "discovery",
            "calibration",
            "final_evaluation",
            "sae_training",
        })
        for family in ("family-a", "family-b"):
            family_roles = [
                role for pair_id, role in first.items() if pair_id.startswith(family)
            ]
            self.assertEqual(
                {role: family_roles.count(role) for role in set(family_roles)},
                {
                    "sae_training": 2,
                    "discovery": 2,
                    "calibration": 2,
                    "final_evaluation": 2,
                },
            )

    def test_bundle_verifier_recomputes_source_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episodes.jsonl"
            rows = synthetic_episodes()
            write_jsonl(source, rows)
            output = root / "splits"
            manifest = build_split_bundle(
                rows,
                source_sha256=sha256_file(source),
                output_dir=output,
                seed=19,
                minimum_episodes_per_category_per_split=1,
            )
            self.assertTrue(manifest["invariants"]["pair_sets_pairwise_disjoint"])
            self.assertEqual(
                verify_split_bundle(output, source_path=source),
                {
                    "schema": "sch.vpd-activation-category-split-verification.v1",
                    "status": "pass",
                    "split_id": manifest["split_id"],
                    "rows": 48,
                    "pairs": 16,
                    "activation_categories": 3,
                    "template_groups": 2,
                    "exact_partition": True,
                },
            )
            for role, entry in manifest["splits"].items():
                self.assertEqual(entry["rows"], 12, role)
                self.assertEqual(entry["pairs"], 4, role)
                self.assertEqual(
                    entry["activation_category_counts"],
                    {"replayed": 4, "signature_invalid": 4, "valid_authority": 4},
                )

    def test_verifier_rejects_tampered_split_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episodes.jsonl"
            rows = synthetic_episodes()
            write_jsonl(source, rows)
            output = root / "splits"
            manifest = build_split_bundle(
                rows,
                source_sha256=sha256_file(source),
                output_dir=output,
                seed=19,
                minimum_episodes_per_category_per_split=1,
            )
            discovery = output / manifest["splits"]["discovery"]["artifact"]["path"]
            discovery.write_text(
                discovery.read_text(encoding="utf-8") + json.dumps(rows[0]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                verify_split_bundle(output, source_path=source)

    def test_rejects_strata_too_small_for_three_way_split(self) -> None:
        rows = synthetic_episodes(pairs_per_family=3)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one pair per split"):
                build_split_bundle(
                    rows,
                    source_sha256=content_sha256(rows),
                    output_dir=Path(directory),
                    seed=19,
                    minimum_episodes_per_category_per_split=1,
                )

    def test_split_weight_parser_is_fail_closed(self) -> None:
        self.assertEqual(
            parse_split_weights(
                "sae_training=1,discovery=2,calibration=1,final_evaluation=1"
            ),
            {
                "sae_training": 1,
                "discovery": 2,
                "calibration": 1,
                "final_evaluation": 1,
            },
        )
        with self.assertRaises(ValueError):
            parse_split_weights("discovery=1,calibration=1")


if __name__ == "__main__":
    unittest.main()
