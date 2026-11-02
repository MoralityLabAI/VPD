from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import (
    categorize_episodes,
    category_contract,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    stable_hash,
    write_json,
)
from experiments.authz_mechinterp.compute_vpd_effect_shard import select_episodes
from experiments.authz_mechinterp.finalize_partition_audit_pack_v1 import (
    verify_partition_audit_manifest,
)
from experiments.authz_mechinterp.effect_uncertainty import uncertainty_contract


PROVENANCE_SCHEMA_V1 = "sch.vpd-partition-audit-effect-provenance.v1"
PROVENANCE_SCHEMA_V2 = "sch.vpd-partition-audit-effect-provenance.v2"
MEASUREMENT_CONTRACT_SCHEMA = "sch.vpd-partition-audit-measurement-contract.v1"
BASE_RECEIPT_SCHEMA = "sch.vpd-effect-shard-receipt.v5"
FORBIDDEN_SPLIT_KEYS = {
    "activation_category_split_id",
    "activation_category_split_manifest_sha256",
    "activation_category_split_role",
    "activation_category_split_episode_id_set_sha256",
    "activation_category_split_pair_id_set_sha256",
    "activation_category_split_template_group_set_sha256",
    "activation_category_split_verification_status",
}


def load_gate_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(plan)
    plan_id = str(body.pop("gate_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("partition-audit gate plan content address mismatch")
    if plan.get("schema") not in {
        "sch.vpd-partition-audit-gate-plan.v1",
        "sch.vpd-partition-audit-gate-plan.v2",
    }:
        raise ValueError("unsupported partition-audit gate plan schema")
    if plan.get("invariants", {}).get("plan_does_not_authorize_execution") is not True:
        raise ValueError("partition-audit gate plan authority boundary differs")
    return plan


def _work_unit(plan: dict[str, Any], work_unit_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in plan.get("capture_work_units", [])
        if str(row.get("work_unit_id")) == work_unit_id
    ]
    if len(matches) != 1:
        raise ValueError("partition-audit work unit is absent or duplicated")
    work_unit = matches[0]
    body = copy.deepcopy(work_unit)
    observed_id = str(body.pop("work_unit_id", ""))
    if content_sha256(body) != observed_id:
        raise ValueError("partition-audit work unit content address mismatch")
    return work_unit


def _command_argument(command: list[str], name: str) -> str:
    if command.count(name) != 1:
        raise ValueError(f"partition-audit gate command omits or duplicates {name}")
    position = command.index(name)
    if position + 1 >= len(command):
        raise ValueError(f"partition-audit gate command has no value for {name}")
    return str(command[position + 1])


def _gate_compute_command(plan: dict[str, Any], *, work_unit_id: str) -> list[str]:
    jobs = [
        row
        for row in plan.get("execution_plan", {}).get("job_groups", [])
        if str(row.get("work_unit_id")) == work_unit_id
    ]
    if len(jobs) != 1:
        raise ValueError("partition-audit gate measurement job is absent or duplicated")
    compute_steps = [
        row
        for row in jobs[0].get("steps", [])
        if row.get("name") == "compute_unsigned_base_receipt"
    ]
    if len(compute_steps) != 1 or not isinstance(compute_steps[0].get("command"), list):
        raise ValueError("partition-audit gate compute command is absent or duplicated")
    return [str(value) for value in compute_steps[0]["command"]]


def gate_measurement_contract(
    plan: dict[str, Any], *, work_unit_id: str
) -> dict[str, Any] | None:
    if plan.get("schema") == "sch.vpd-partition-audit-gate-plan.v1":
        return None
    if plan.get("schema") != "sch.vpd-partition-audit-gate-plan.v2":
        raise ValueError("unsupported partition-audit measurement contract gate")
    campaign = plan.get("source", {}).get("discovery_effect_capture_campaign")
    if not isinstance(campaign, dict):
        raise ValueError("partition-audit gate omits the discovery campaign contract")
    verification = campaign.get("verification")
    parameters = campaign.get("capture_parameters")
    if (
        not isinstance(verification, dict)
        or verification.get("status") != "pass"
        or verification.get("campaign_plan_id") != campaign.get("campaign_plan_id")
        or not isinstance(parameters, dict)
    ):
        raise ValueError("partition-audit discovery campaign contract is invalid")
    command = _gate_compute_command(plan, work_unit_id=work_unit_id)
    expected_arguments = {
        "--maximum-episodes": str(int(parameters["maximum_episodes"])),
        "--selection-seed": str(int(parameters["selection_seed"])),
        "--epsilon": str(float(parameters["epsilon"])),
        "--batch-size": str(int(parameters["batch_size"])),
        "--uncertainty-blocks": str(int(parameters["uncertainty_blocks"])),
        "--uncertainty-sketch-dimension": str(
            int(parameters["uncertainty_sketch_dimension"])
        ),
        "--dtype": str(parameters["dtype"]),
    }
    if any(
        _command_argument(command, name) != expected
        for name, expected in expected_arguments.items()
    ):
        raise ValueError("partition-audit gate command and campaign parameters differ")
    if "DTYPE" in plan.get("execution_plan", {}).get("required_placeholders", []):
        raise ValueError("partition-audit model-load dtype remains runtime configurable")
    body = {
        "schema": MEASUREMENT_CONTRACT_SCHEMA,
        "discovery_campaign_plan_id": campaign["campaign_plan_id"],
        "discovery_campaign_plan_sha256": campaign["sha256"],
        "capture_parameters": copy.deepcopy(parameters),
        "capture_parameters_sha256": content_sha256(parameters),
        "worker_receipt_verified_fields": [
            "effect_space_id",
            "estimator",
            "perturbation_epsilon",
            "selection_seed",
            "generated_tokens",
            "expected_hidden_size",
            "uncertainty_blocks",
            "uncertainty_sketch_dimension",
            "uncertainty_countsketch_seed",
        ],
        "gate_command_bound_fields": [
            "batch_size",
            "dtype",
            "maximum_episodes",
        ],
        "model_load_dtype_evidence": "content_addressed_gate_command_and_worker_hash",
        "worker_receipt_observes_model_load_dtype": False,
        "execution_authorized": False,
    }
    return {**body, "measurement_contract_id": content_sha256(body)}


def _verify_campaign_measurement_fields(
    receipt: dict[str, Any], *, contract: dict[str, Any]
) -> None:
    inputs = receipt.get("inputs", {})
    parameters = contract["capture_parameters"]
    expected_inputs = {
        "effect_space_id": str(parameters["effect_space_id"]),
        "estimator": str(parameters["estimator"]),
        "perturbation_epsilon": float(parameters["epsilon"]),
        "selection_seed": int(parameters["selection_seed"]),
        "generated_tokens": int(parameters["generated_tokens"]),
    }
    if any(inputs.get(key) != value for key, value in expected_inputs.items()):
        raise ValueError("base effect receipt differs from campaign measurement contract")
    if int(receipt.get("hidden_size", -1)) != int(parameters["expected_hidden_size"]):
        raise ValueError("base effect receipt hidden size differs from campaign contract")
    if receipt.get("dtype") != "float32":
        raise ValueError("base effect receipt merge dtype differs")
    expected_uncertainty = uncertainty_contract(
        block_count=int(parameters["uncertainty_blocks"]),
        sketch_dimension=int(parameters["uncertainty_sketch_dimension"]),
        seed=(
            int(
                stable_hash(
                    int(parameters["selection_seed"]),
                    "category-effect-countsketch",
                    str(parameters["effect_space_id"]),
                ),
                16,
            )
            % (2**32)
        ),
    )
    if inputs.get("activation_category_uncertainty") != expected_uncertainty:
        raise ValueError("base effect receipt uncertainty differs from campaign contract")


def build_dataset_attestation(
    *,
    pack_dir: Path,
    selection_seed: int,
    maximum_episodes: int,
) -> dict[str, Any]:
    if maximum_episodes != 0:
        raise ValueError("partition audit requires the full fresh pack")
    verification = verify_partition_audit_manifest(pack_dir)
    manifest_path = pack_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = pack_dir / manifest["artifacts"]["gated_episodes"]["path"]
    rows = load_jsonl(dataset_path)
    selected = select_episodes(rows, maximum=maximum_episodes, seed=selection_seed)
    if len(selected) != len(rows):
        raise ValueError("partition-audit selection does not cover the full pack")
    categories = categorize_episodes(selected)
    category_names = sorted(categories)
    episode_ids = [str(row["episode_id"]) for row in selected]
    pair_ids = sorted({str(row["pair_id"]) for row in selected})
    template_groups = sorted({str(row["template_group"]) for row in selected})
    body = {
        "schema": "sch.vpd-partition-audit-dataset-attestation.v1",
        "role": "partition_audit",
        "pack_id": verification["pack_id"],
        "pack_manifest_sha256": sha256_file(manifest_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_validation_sha256": manifest["artifacts"]["dataset_validation"][
            "sha256"
        ],
        "selection_seed": selection_seed,
        "maximum_episodes": maximum_episodes,
        "episodes": len(selected),
        "pairs": len(pair_ids),
        "episode_order_sha256": content_sha256(episode_ids),
        "episode_id_set_sha256": content_sha256(sorted(episode_ids)),
        "pair_id_set_sha256": content_sha256(pair_ids),
        "template_group_set_sha256": content_sha256(template_groups),
        "template_groups": template_groups,
        "activation_category_contract_sha256": content_sha256(category_contract()),
        "activation_categories": category_names,
        "activation_category_episode_counts": {
            category: len(categories[category]) for category in category_names
        },
        "activation_category_episode_ids_sha256": {
            category: content_sha256(
                [str(row["episode_id"]) for row in categories[category]]
            )
            for category in category_names
        },
        "identity_message_and_semantic_template_disjointness": "pass",
        "final_evaluation_consumed": False,
        "model_outcomes_used": False,
    }
    return {**body, "dataset_attestation_id": content_sha256(body)}


def expected_partition_audit_provenance(
    *,
    pack_dir: Path,
    gate_plan_path: Path,
    work_unit_id: str,
    base_receipt_sha256: str,
    selection_seed: int,
    maximum_episodes: int,
) -> dict[str, Any]:
    plan = load_gate_plan(gate_plan_path)
    work_unit = _work_unit(plan, work_unit_id)
    measurement_contract = gate_measurement_contract(
        plan, work_unit_id=work_unit_id
    )
    dataset = build_dataset_attestation(
        pack_dir=pack_dir,
        selection_seed=selection_seed,
        maximum_episodes=maximum_episodes,
    )
    if work_unit["dataset_sha256"] != dataset["dataset_sha256"]:
        raise ValueError("partition-audit work unit and dataset differ")
    if (
        work_unit["output_contract"]["partition_audit_pack_id_required"]
        != dataset["pack_id"]
    ):
        raise ValueError("partition-audit work unit and pack identity differ")
    body = {
        "schema": (
            PROVENANCE_SCHEMA_V2
            if measurement_contract is not None
            else PROVENANCE_SCHEMA_V1
        ),
        "role": "partition_audit",
        "pack_id": dataset["pack_id"],
        "pack_manifest_sha256": dataset["pack_manifest_sha256"],
        "dataset_attestation_id": dataset["dataset_attestation_id"],
        "dataset_validation_sha256": dataset["dataset_validation_sha256"],
        "episode_id_set_sha256": dataset["episode_id_set_sha256"],
        "pair_id_set_sha256": dataset["pair_id_set_sha256"],
        "template_group_set_sha256": dataset["template_group_set_sha256"],
        "gate_plan_id": plan["gate_plan_id"],
        "gate_plan_sha256": sha256_file(gate_plan_path),
        "work_unit_id": work_unit_id,
        "base_worker_receipt_sha256": base_receipt_sha256,
        "receipt_attestation_adapter_sha256": sha256_file(Path(__file__)),
        "identity_message_and_semantic_template_disjointness": "pass",
        "threshold_cluster_or_component_selection_performed": False,
        "shared_to_specific_promotion_performed": False,
        "final_evaluation_consumed": False,
    }
    if measurement_contract is not None:
        body["measurement_contract"] = measurement_contract
        body["compute_command_sha256"] = content_sha256(
            _gate_compute_command(plan, work_unit_id=work_unit_id)
        )
    return {**body, "provenance_id": content_sha256(body)}


def _verify_base_receipt_against_attestation(
    receipt: dict[str, Any],
    *,
    dataset: dict[str, Any],
    work_unit: dict[str, Any],
    plan: dict[str, Any],
    require_unsigned: bool = True,
) -> None:
    if receipt.get("schema") != BASE_RECEIPT_SCHEMA:
        raise ValueError("partition audit requires an unsigned V5 effect receipt")
    if require_unsigned and ("result_id" in receipt or "signature" in receipt):
        raise ValueError("partition-audit provenance must be added before signing")
    inputs = receipt.get("inputs", {})
    if FORBIDDEN_SPLIT_KEYS & set(inputs):
        raise ValueError("partition audit may not masquerade as a five-role split")
    if require_unsigned and "partition_audit_provenance" in inputs:
        raise ValueError("base receipt is already partition-audit attested")
    expected_scalars = {
        "dataset_sha256": dataset["dataset_sha256"],
        "episode_order_sha256": dataset["episode_order_sha256"],
        "episodes": dataset["episodes"],
        "generated_tokens": 0,
        "activation_category_contract_sha256": dataset[
            "activation_category_contract_sha256"
        ],
    }
    if any(inputs.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError("base effect receipt differs from the fresh audit dataset")
    if inputs.get("activation_categories") != dataset["activation_categories"]:
        raise ValueError("base effect receipt category names differ")
    if inputs.get("activation_category_episode_counts") != dataset[
        "activation_category_episode_counts"
    ]:
        raise ValueError("base effect receipt category counts differ")
    if inputs.get("activation_category_episode_ids_sha256") != dataset[
        "activation_category_episode_ids_sha256"
    ]:
        raise ValueError("base effect receipt category episode identities differ")
    if receipt.get("packet_id") != work_unit["packet_id"]:
        raise ValueError("base effect receipt packet differs from work unit")
    if receipt.get("cluster_id") != work_unit["cluster_id"]:
        raise ValueError("base effect receipt cluster differs from work unit")
    if receipt.get("component_id_set_sha256") != work_unit[
        "component_id_set_sha256"
    ]:
        raise ValueError("base effect receipt component set differs from work unit")
    if int(receipt.get("rows", -1)) != int(work_unit["component_count"]):
        raise ValueError("base effect receipt component count differs from work unit")
    worker_path = Path(__file__).with_name("compute_vpd_effect_shard.py")
    if inputs.get("runner_sha256") != sha256_file(worker_path):
        raise ValueError("base effect receipt was not emitted by the preserved worker")
    measurement_contract = gate_measurement_contract(
        plan, work_unit_id=str(work_unit["work_unit_id"])
    )
    if measurement_contract is not None:
        _verify_campaign_measurement_fields(receipt, contract=measurement_contract)


def attest_partition_audit_receipt(
    *,
    base_receipt_path: Path,
    pack_dir: Path,
    gate_plan_path: Path,
    work_unit_id: str,
) -> dict[str, Any]:
    receipt = json.loads(base_receipt_path.read_text(encoding="utf-8"))
    inputs = receipt.get("inputs", {})
    selection_seed = int(inputs.get("selection_seed", -1))
    maximum_episodes = 0
    plan = load_gate_plan(gate_plan_path)
    work_unit = _work_unit(plan, work_unit_id)
    dataset = build_dataset_attestation(
        pack_dir=pack_dir,
        selection_seed=selection_seed,
        maximum_episodes=maximum_episodes,
    )
    _verify_base_receipt_against_attestation(
        receipt,
        dataset=dataset,
        work_unit=work_unit,
        plan=plan,
        require_unsigned=True,
    )
    provenance = expected_partition_audit_provenance(
        pack_dir=pack_dir,
        gate_plan_path=gate_plan_path,
        work_unit_id=work_unit_id,
        base_receipt_sha256=sha256_file(base_receipt_path),
        selection_seed=selection_seed,
        maximum_episodes=maximum_episodes,
    )
    result = copy.deepcopy(receipt)
    result["inputs"]["partition_audit_provenance"] = provenance
    result["claim_boundary"] = (
        "Signed fresh-pack partition-audit effect vectors only; this reject-only "
        "generalization audit cannot select or promote a category partition."
    )
    return result


def verify_attested_partition_audit_receipt(
    receipt: dict[str, Any],
    *,
    pack_dir: Path,
    gate_plan_path: Path,
    require_signed: bool,
) -> dict[str, Any]:
    if receipt.get("schema") != BASE_RECEIPT_SCHEMA:
        raise ValueError("unexpected partition-audit effect receipt schema")
    if require_signed and (
        not isinstance(receipt.get("result_id"), str)
        or not isinstance(receipt.get("signature"), dict)
    ):
        raise ValueError("partition-audit effect receipt is not signed")
    provenance = receipt.get("inputs", {}).get("partition_audit_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("partition-audit effect provenance is absent")
    body = copy.deepcopy(provenance)
    provenance_id = str(body.pop("provenance_id", ""))
    if content_sha256(body) != provenance_id:
        raise ValueError("partition-audit effect provenance content address mismatch")
    plan = load_gate_plan(gate_plan_path)
    work_unit = _work_unit(plan, str(provenance.get("work_unit_id", "")))
    dataset = build_dataset_attestation(
        pack_dir=pack_dir,
        selection_seed=int(receipt["inputs"]["selection_seed"]),
        maximum_episodes=0,
    )
    _verify_base_receipt_against_attestation(
        receipt,
        dataset=dataset,
        work_unit=work_unit,
        plan=plan,
        require_unsigned=False,
    )
    if provenance.get("pack_id") != dataset["pack_id"]:
        raise ValueError("partition-audit receipt pack identity differs")
    if provenance.get("dataset_attestation_id") != dataset[
        "dataset_attestation_id"
    ]:
        raise ValueError("partition-audit receipt dataset attestation differs")
    if provenance.get("gate_plan_id") != plan["gate_plan_id"]:
        raise ValueError("partition-audit receipt gate plan differs")
    if receipt.get("packet_id") != work_unit["packet_id"]:
        raise ValueError("partition-audit receipt work unit packet differs")
    expected_fixed = {
        "schema": (
            PROVENANCE_SCHEMA_V2
            if plan.get("schema") == "sch.vpd-partition-audit-gate-plan.v2"
            else PROVENANCE_SCHEMA_V1
        ),
        "role": "partition_audit",
        "pack_manifest_sha256": dataset["pack_manifest_sha256"],
        "dataset_validation_sha256": dataset["dataset_validation_sha256"],
        "episode_id_set_sha256": dataset["episode_id_set_sha256"],
        "pair_id_set_sha256": dataset["pair_id_set_sha256"],
        "template_group_set_sha256": dataset["template_group_set_sha256"],
        "gate_plan_sha256": sha256_file(gate_plan_path),
        "receipt_attestation_adapter_sha256": sha256_file(Path(__file__)),
        "identity_message_and_semantic_template_disjointness": "pass",
        "threshold_cluster_or_component_selection_performed": False,
        "shared_to_specific_promotion_performed": False,
        "final_evaluation_consumed": False,
    }
    measurement_contract = gate_measurement_contract(
        plan, work_unit_id=str(work_unit["work_unit_id"])
    )
    if measurement_contract is not None:
        expected_fixed["measurement_contract"] = measurement_contract
        expected_fixed["compute_command_sha256"] = content_sha256(
            _gate_compute_command(
                plan, work_unit_id=str(work_unit["work_unit_id"])
            )
        )
    if any(provenance.get(key) != value for key, value in expected_fixed.items()):
        raise ValueError("partition-audit receipt provenance policy differs")
    return {
        "schema": "sch.vpd-partition-audit-effect-receipt-verification.v1",
        "status": "pass",
        "provenance_id": provenance_id,
        "work_unit_id": work_unit["work_unit_id"],
        "packet_id": work_unit["packet_id"],
        "pack_id": dataset["pack_id"],
        "measurement_contract_id": (
            measurement_contract["measurement_contract_id"]
            if measurement_contract is not None
            else None
        ),
        "signed": require_signed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind fresh partition-audit provenance into an unsigned V5 receipt."
    )
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--gate-plan", type=Path, required=True)
    parser.add_argument("--work-unit-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = attest_partition_audit_receipt(
        base_receipt_path=args.base_receipt,
        pack_dir=args.pack_dir,
        gate_plan_path=args.gate_plan,
        work_unit_id=args.work_unit_id,
    )
    write_json(args.output, result)
    print(
        json.dumps(
            verify_attested_partition_audit_receipt(
                result,
                pack_dir=args.pack_dir,
                gate_plan_path=args.gate_plan,
                require_signed=False,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
