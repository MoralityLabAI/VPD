from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp import run_compartment_sae_quality as v1
from experiments.authz_mechinterp.admit_compartment_sae_dictionary_bundle_v2 import (
    verify_bundle_for_quality_v2,
)
from experiments.authz_mechinterp.admit_compartment_sae_training_release_v2 import (
    verify_explicit_compute_authorization,
    verify_training_release_receipt_v2,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.run_compartment_sae_training_job_v2 import (
    MANIFEST_V2_SCHEMA,
    load_training_plan_v2,
)


def _load_manifest_v2(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_V2_SCHEMA:
        raise ValueError("quality V2 requires dictionary manifest V2")
    body = dict(manifest)
    identifier = str(body.pop("dictionary_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("quality V2 dictionary manifest address mismatch")
    return manifest


def delegate_quality_v2(
    *,
    plan: dict[str, Any],
    training_plan_path: Path,
    execution_blueprint_path: Path,
) -> None:
    job = plan["quality_job"]
    arguments = [str(value) for value in job.get("quality_arguments", [])]
    if arguments[:2] != [
        "-m",
        "experiments.authz_mechinterp.run_compartment_sae_quality",
    ]:
        raise ValueError("quality V2 does not delegate to the preserved runner")
    original_loader = v1._content_addressed_json
    original_bundle_verifier = v1.verify_bundle_for_quality
    original_descriptor_loader = v1.load_dictionary_descriptors
    original_file = v1.__file__

    def compatible_loader(path: Path, *, schema: str, id_field: str) -> dict[str, Any]:
        if path.name == "sae_dictionary_manifest.json":
            return _load_manifest_v2(path)
        return original_loader(path, schema=schema, id_field=id_field)

    def compatible_bundle_verifier(**kwargs: Any) -> dict[str, Any]:
        return verify_bundle_for_quality_v2(
            receipt_path=kwargs["receipt_path"],
            training_plan_path=kwargs["training_plan_path"],
            execution_blueprint_path=execution_blueprint_path,
            dictionary_sources=kwargs["dictionary_sources"],
        )

    def compatible_descriptor_loader(**kwargs: Any) -> list[dict[str, Any]]:
        descriptors = original_descriptor_loader(**kwargs)
        for descriptor in descriptors:
            execution = descriptor["manifest"]["source"]["training_execution"]
            descriptor["source_receipt"]["training_execution_receipt_id"] = str(
                execution["training_execution_receipt_id"]
            )
        return descriptors

    previous_argv = sys.argv
    try:
        v1._content_addressed_json = compatible_loader
        v1.verify_bundle_for_quality = compatible_bundle_verifier
        v1.load_dictionary_descriptors = compatible_descriptor_loader
        v1.__file__ = str(Path(__file__).resolve())
        sys.argv = [str(Path(v1.__file__).resolve()), *arguments[2:]]
        v1.main()
    finally:
        sys.argv = previous_argv
        v1._content_addressed_json = original_loader
        v1.verify_bundle_for_quality = original_bundle_verifier
        v1.load_dictionary_descriptors = original_descriptor_loader
        v1.__file__ = original_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run release- and user-authorized compartment SAE quality V2."
    )
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--execution-blueprint", type=Path, required=True)
    parser.add_argument("--training-release-receipt", type=Path, required=True)
    parser.add_argument("--compute-authorization-receipt", type=Path, required=True)
    args = parser.parse_args()
    plan = load_training_plan_v2(args.training_plan)
    release_gate_path = Path(
        plan["source"]["training_release_gate_v2"]["path"]
    )
    release_verification = verify_training_release_receipt_v2(
        args.training_release_receipt,
        release_gate_plan_path=release_gate_path,
        training_plan_path=args.training_plan,
    )
    compute_verification = verify_explicit_compute_authorization(
        args.compute_authorization_receipt,
        release_verification=release_verification,
        release_gate_plan_path=release_gate_path,
        training_plan_path=args.training_plan,
        required_job_id=str(plan["quality_job"]["job_id"]),
        required_action="calibration_quality",
    )
    from experiments.authz_mechinterp.run_compartment_sae_capture_job_v2 import (
        select_capture_job,
        validate_capture_execution,
    )

    capture_job = select_capture_job(plan, role="calibration")
    capture_compute_verification = verify_explicit_compute_authorization(
        args.compute_authorization_receipt,
        release_verification=release_verification,
        release_gate_plan_path=release_gate_path,
        training_plan_path=args.training_plan,
        required_job_id=str(capture_job["job_id"]),
        required_action="base_activation_capture",
    )
    capture_manifest = json.loads(
        (
            Path(str(capture_job["output_dir"])) / "capture_manifest.json"
        ).read_text(encoding="utf-8")
    )
    capture_verification = validate_capture_execution(
        manifest=capture_manifest,
        plan=plan,
        training_plan_path=args.training_plan,
        job=capture_job,
    )
    if (
        str(capture_verification["training_release_receipt_id"])
        != str(release_verification["training_release_receipt_id"])
        or str(capture_verification["compute_authorization_id"])
        != str(capture_compute_verification["compute_authorization_id"])
        or str(capture_compute_verification["compute_authorization_id"])
        != str(compute_verification["compute_authorization_id"])
    ):
        raise ValueError("quality capture used different authorizations")
    delegate_quality_v2(
        plan=plan,
        training_plan_path=args.training_plan,
        execution_blueprint_path=args.execution_blueprint,
    )


if __name__ == "__main__":
    main()
