from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import experiments.authz_mechinterp.compute_focused_sae_interventions as predecessor
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    load_jsonl,
    sha256_file,
    verify_bundle,
    write_json,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import load_packets
from experiments.authz_mechinterp.run_focused_vpd_sae_v2 import (
    SCHEMA,
    recompute_v2_candidate_controls,
    validate_v2_plan_control_contract,
)
from experiments.authz_mechinterp.run_focused_vpd_sae import (
    load_dictionary,
    load_effect_archive,
)


def load_and_verify_v2_plan(
    *,
    plan_path: Path,
    bundle_dir: Path,
    dictionary_path: Path,
    effect_archive_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA:
        raise ValueError("focused SAE V2 executor requires a V2 plan")
    bundle_verification = verify_bundle(bundle_dir)
    if bundle_verification.get("status") != "pass":
        raise ValueError("focused SAE V2 executor category bundle verification failed")
    packets, _ = load_packets(bundle_dir)
    packet_id = str(plan.get("packet_id", ""))
    if packet_id not in packets:
        raise ValueError("focused SAE V2 executor plan packet is absent from its bundle")
    packet = packets[packet_id]
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments_path = bundle_dir / manifest["artifacts"]["assignments"]["path"]
    assignments = load_jsonl(assignments_path)
    assignment_categories = {
        str(row["component_id"]): str(row.get("activation_category", ""))
        for row in assignments
    }
    inputs = plan.get("study_inputs", {})
    if (
        inputs.get("category_bundle_manifest_sha256") != sha256_file(manifest_path)
        or inputs.get("category_bundle_assignments_sha256")
        != sha256_file(assignments_path)
    ):
        raise ValueError("focused SAE V2 executor category bundle lineage differs")
    if (
        inputs.get("sae_dictionary_sha256") != sha256_file(dictionary_path)
        or inputs.get("effect_archive_sha256") != sha256_file(effect_archive_path)
    ):
        raise ValueError("focused SAE V2 executor numeric input lineage differs")
    verification = validate_v2_plan_control_contract(
        plan=plan,
        packet=packet,
        assignment_categories=assignment_categories,
    )
    dictionary = load_dictionary(dictionary_path)
    component_ids, delta_h = load_effect_archive(effect_archive_path)
    numeric_verification = recompute_v2_candidate_controls(
        plan=plan,
        packet=packet,
        effect_component_ids=component_ids,
        delta_h=delta_h,
        decoder=dictionary["decoder"],
    )
    return plan, packet, {
        **verification,
        "numeric_selection": numeric_verification,
    }


def run_interventions_v2(
    *,
    plan_path: Path,
    bundle_dir: Path,
    dictionary_path: Path,
    effect_archive_path: Path,
    dataset_path: Path,
    model_id: str,
    revision: str,
    adapter_path: Path,
    expected_adapter_sha256: str,
    worker_id: str,
    output_path: Path,
    tokenizer_id: str | None = None,
    adapter_name: str = "default",
    device: str = "cuda",
    dtype_name: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    plan, _, verification = load_and_verify_v2_plan(
        plan_path=plan_path,
        bundle_dir=bundle_dir,
        dictionary_path=dictionary_path,
        effect_archive_path=effect_archive_path,
    )
    original_loader = predecessor.load_plan

    def verified_loader(observed_path: Path) -> dict[str, Any]:
        if observed_path.resolve() != plan_path.resolve():
            raise ValueError("focused SAE V2 executor attempted to load another plan")
        return plan

    predecessor.load_plan = verified_loader
    try:
        trace = predecessor.run_interventions(
            plan_path=plan_path,
            dictionary_path=dictionary_path,
            dataset_path=dataset_path,
            model_id=model_id,
            revision=revision,
            adapter_path=adapter_path,
            expected_adapter_sha256=expected_adapter_sha256,
            worker_id=worker_id,
            output_path=output_path,
            tokenizer_id=tokenizer_id,
            adapter_name=adapter_name,
            device=device,
            dtype_name=dtype_name,
            trust_remote_code=trust_remote_code,
        )
    finally:
        predecessor.load_plan = original_loader
    trace["inputs"]["control_compartment_gate_runner_sha256"] = sha256_file(
        Path(__file__)
    )
    trace["control_compartment_verification"] = verification
    write_json(output_path, trace)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a same-compartment-control focused SAE V2 plan."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--adapter-name", default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    trace = run_interventions_v2(
        plan_path=args.plan,
        bundle_dir=args.bundle,
        dictionary_path=args.dictionary,
        effect_archive_path=args.effects,
        dataset_path=args.dataset,
        model_id=args.model,
        revision=args.revision,
        adapter_path=args.adapter,
        expected_adapter_sha256=args.expected_adapter_sha256,
        worker_id=args.worker_id,
        output_path=args.output,
        tokenizer_id=args.tokenizer,
        adapter_name=args.adapter_name,
        device=args.device,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    print(
        json.dumps(
            {
                "schema": "sch.vpd-focused-sae-intervention-output.v2",
                "plan_id": trace["plan_id"],
                "trials": len(trace["trials"]),
                "control_compartment_status": "pass",
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
