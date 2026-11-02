from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1 import (
    _synthetic_materialization,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_blueprint_budget_admission_v2 import (
    build_blueprint_bound_proposal,
    evaluate_blueprint_bound_budget_proposal,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1 import (
    build_execution_blueprint,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def _readdress_job(job: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(job)
    body = {key: value for key, value in changed.items() if key != "execution_job_id"}
    changed["execution_job_id"] = content_sha256(body)
    return changed


def _readdress_blueprint(blueprint: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(blueprint)
    body = {
        key: value for key, value in changed.items() if key != "execution_blueprint_id"
    }
    changed["execution_blueprint_id"] = content_sha256(body)
    return changed


def _readdress_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(proposal)
    body = {key: value for key, value in changed.items() if key != "proposal_id"}
    changed["proposal_id"] = content_sha256(body)
    return changed


def _readdress_admission(admission: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(admission)
    body = {key: value for key, value in changed.items() if key != "admission_id"}
    changed["admission_id"] = content_sha256(body)
    return changed


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    policy_path = (
        project_root
        / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json"
    )
    policy = _load(policy_path)
    materialization = _synthetic_materialization(project_root)
    blueprint = build_execution_blueprint(materialization=materialization)
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="vpd-blueprint-budget-v2-") as value:
        temporary_path = Path(value)
        blueprint_path = temporary_path / "execution-blueprint.json"
        write_json(blueprint_path, blueprint)
        blueprint_hash = sha256_file(blueprint_path)
        proposal = build_blueprint_bound_proposal(
            policy=policy,
            blueprint=blueprint,
            blueprint_sha256=blueprint_hash,
        )
        admission = evaluate_blueprint_bound_budget_proposal(
            policy=policy,
            blueprint=blueprint,
            blueprint_sha256=blueprint_hash,
            proposal=proposal,
        )

        def rejects_proposal(candidate: dict[str, Any]) -> bool:
            return _rejects(
                lambda: evaluate_blueprint_bound_budget_proposal(
                    policy=policy,
                    blueprint=blueprint,
                    blueprint_sha256=blueprint_hash,
                    proposal=candidate,
                )
            )

        wrong_blueprint_id = copy.deepcopy(proposal)
        wrong_blueprint_id["source"]["execution_blueprint_id"] = "f" * 64
        wrong_blueprint_id = _readdress_proposal(wrong_blueprint_id)
        wrong_blueprint_hash = copy.deepcopy(proposal)
        wrong_blueprint_hash["source"]["execution_blueprint_sha256"] = "f" * 64
        wrong_blueprint_hash = _readdress_proposal(wrong_blueprint_hash)
        wrong_policy_id = copy.deepcopy(proposal)
        wrong_policy_id["source"]["budget_policy_id"] = "f" * 64
        wrong_policy_id = _readdress_proposal(wrong_policy_id)
        authority_broadening = copy.deepcopy(proposal)
        authority_broadening["authority_claims"]["sae_training_authorized"] = True
        authority_broadening = _readdress_proposal(authority_broadening)
        resource_substitution = copy.deepcopy(proposal)
        resource_substitution["global_requested_resources"]["planned_steps"] -= 1
        resource_substitution = _readdress_proposal(resource_substitution)

        widened_blueprint = copy.deepcopy(blueprint)
        widened_job = copy.deepcopy(widened_blueprint["execution_jobs"][0])
        widened_job["profile"]["steps"] += 1
        widened_blueprint["execution_jobs"][0] = _readdress_job(widened_job)
        widened_blueprint = _readdress_blueprint(widened_blueprint)
        widened_path = temporary_path / "widened-blueprint.json"
        write_json(widened_path, widened_blueprint)
        def evaluate_widened_blueprint() -> dict[str, Any]:
            widened_hash = sha256_file(widened_path)
            widened_proposal = build_blueprint_bound_proposal(
                policy=policy,
                blueprint=widened_blueprint,
                blueprint_sha256=widened_hash,
            )
            return evaluate_blueprint_bound_budget_proposal(
                policy=policy,
                blueprint=widened_blueprint,
                blueprint_sha256=widened_hash,
                proposal=widened_proposal,
            )

        widened_rejected = _rejects(evaluate_widened_blueprint)

        category_rebound_blueprint = copy.deepcopy(blueprint)
        specific_compartment_ids = {
            str(row["compartment_id"])
            for row in policy["specific_compartment_budgets"]
        }
        rebound_compartment_id = str(
            next(
                job
                for job in category_rebound_blueprint["execution_jobs"]
                if str(job["compartment_id"]) in specific_compartment_ids
            )["compartment_id"]
        )
        expected_rebound_categories = next(
            list(row["activation_categories"])
            for row in policy["specific_compartment_budgets"]
            if row["compartment_id"] == rebound_compartment_id
        )
        replacement_category = next(
            category
            for row in policy["category_specific_budgets"]
            for category in [str(row["activation_category"])]
            if category not in expected_rebound_categories
        )
        category_rebound_blueprint["execution_jobs"] = [
            _readdress_job(
                {
                    **copy.deepcopy(job),
                    "activation_categories": [replacement_category],
                }
            )
            if str(job["compartment_id"]) == rebound_compartment_id
            else job
            for job in category_rebound_blueprint["execution_jobs"]
        ]
        category_rebound_blueprint = _readdress_blueprint(category_rebound_blueprint)
        rebound_path = temporary_path / "category-rebound-blueprint.json"
        write_json(rebound_path, category_rebound_blueprint)
        category_rebinding_rejected = _rejects(
            lambda: build_blueprint_bound_proposal(
                policy=policy,
                blueprint=category_rebound_blueprint,
                blueprint_sha256=sha256_file(rebound_path),
            )
        )

        reused_output_blueprint = copy.deepcopy(blueprint)
        reused = copy.deepcopy(reused_output_blueprint["execution_jobs"][1])
        reused["output_dir"] = reused_output_blueprint["execution_jobs"][0][
            "output_dir"
        ]
        reused_output_blueprint["execution_jobs"][1] = _readdress_job(reused)
        reused_output_blueprint = _readdress_blueprint(reused_output_blueprint)
        reused_path = temporary_path / "reused-output-blueprint.json"
        write_json(reused_path, reused_output_blueprint)
        output_reuse_rejected = _rejects(
            lambda: build_blueprint_bound_proposal(
                policy=policy,
                blueprint=reused_output_blueprint,
                blueprint_sha256=sha256_file(reused_path),
            )
        )

        tampered_blueprint = copy.deepcopy(blueprint)
        tampered_blueprint["summary"]["execution_jobs"] -= 1
        tampered_path = temporary_path / "unaddressed-blueprint-tamper.json"
        write_json(tampered_path, tampered_blueprint)
        unaddressed_blueprint_tamper_rejected = _rejects(
            lambda: build_blueprint_bound_proposal(
                policy=policy,
                blueprint=tampered_blueprint,
                blueprint_sha256=sha256_file(tampered_path),
            )
        )

        admission_tamper = copy.deepcopy(admission)
        admission_tamper["blueprint_binding_verified"] = False
        admission_readdressed_tamper = _readdress_admission(admission_tamper)
        admission_tamper_rejected = admission_tamper["admission_id"] != content_sha256(
            {
                key: value
                for key, value in admission_tamper.items()
                if key != "admission_id"
            }
        )
        admission_readdressed_tamper_rejected = admission_readdressed_tamper != (
            evaluate_blueprint_bound_budget_proposal(
                policy=policy,
                blueprint=blueprint,
                blueprint_sha256=blueprint_hash,
                proposal=proposal,
            )
        )

    specific = proposal["specific_compartment_requests"]
    controls = {
        "proposal_is_content_addressed": proposal["proposal_id"]
        == content_sha256(
            {key: value for key, value in proposal.items() if key != "proposal_id"}
        ),
        "admission_is_content_addressed": admission["admission_id"]
        == content_sha256(
            {key: value for key, value in admission.items() if key != "admission_id"}
        ),
        "proposal_binds_exact_blueprint_id": proposal["source"]
        ["execution_blueprint_id"]
        == blueprint["execution_blueprint_id"],
        "proposal_binds_exact_blueprint_file_hash": proposal["source"]
        ["execution_blueprint_sha256"]
        == blueprint_hash,
        "admission_repeats_exact_blueprint_binding": admission["source"]
        ["execution_blueprint_id"]
        == blueprint["execution_blueprint_id"]
        and admission["source"]["execution_blueprint_sha256"] == blueprint_hash,
        "exact_34_job_shape_is_derived": proposal["blueprint_shape"][
            "execution_jobs"
        ]
        == 34
        and proposal["global_requested_resources"]["jobs"] == 34,
        "exact_68000_steps_and_20480_rows_are_derived": proposal[
            "global_requested_resources"
        ]["planned_steps"]
        == 68000
        and proposal["global_requested_resources"]["planned_activation_rows"]
        == 20480,
        "all_16_specific_compartments_are_derived": len(specific) == 16,
        "all_8_categories_are_derived": len(
            proposal["category_specific_requests"]
        )
        == 8,
        "one_shared_fallback_is_derived_and_uncharged": proposal[
            "shared_fallback_request"
        ]["charged_to_category_budgets"]
        is False
        and proposal["shared_fallback_request"]["requested_resources"]["jobs"]
        == 2,
        "specific_and_shared_dictionary_widths_remain_distinct": all(
            row["requested_resources"]["dictionary_feature_widths"] == [4096]
            for row in specific
        )
        and proposal["shared_fallback_request"]["requested_resources"]
        ["dictionary_feature_widths"]
        == [8192],
        "all_34_output_directories_are_unique": proposal[
            "global_requested_resources"
        ]["unique_output_directories"]
        == 34,
        "budget_shape_is_admissible": admission["budget_admissible"] is True
        and admission["blueprint_binding_verified"] is True,
        "wrong_blueprint_id_rejected": rejects_proposal(wrong_blueprint_id),
        "wrong_blueprint_file_hash_rejected": rejects_proposal(wrong_blueprint_hash),
        "wrong_policy_id_rejected": rejects_proposal(wrong_policy_id),
        "proposal_authority_broadening_rejected": rejects_proposal(
            authority_broadening
        ),
        "proposal_resource_substitution_rejected": rejects_proposal(
            resource_substitution
        ),
        "readdressed_blueprint_resource_widening_rejected": widened_rejected,
        "readdressed_blueprint_category_rebinding_rejected": category_rebinding_rejected,
        "readdressed_blueprint_output_reuse_rejected": output_reuse_rejected,
        "unaddressed_blueprint_tamper_rejected": unaddressed_blueprint_tamper_rejected,
        "unaddressed_admission_tamper_rejected": admission_tamper_rejected,
        "readdressed_admission_tamper_rejected_by_reproduction": admission_readdressed_tamper_rejected,
        "temporary_blueprint_fixture_removed": bool(
            temporary_path is not None and not temporary_path.exists()
        ),
        "proposal_and_admission_transfer_no_authority": all(
            admission["scientific_boundary"][field] is False
            for field in (
                "budget_admission_is_execution_authority",
                "empirical_evidence_created",
                "model_loaded",
                "model_inference_performed",
                "training_performed",
                "persistent_dictionary_output_created",
                "execution_authorized",
                "paid_compute_authorized",
                "sae_training_authorized",
                "weight_mutation_authorized",
            )
        ),
        "benchmark_imports_no_model_frameworks": before_frameworks
        == (MODEL_FRAMEWORKS & set(sys.modules)),
    }
    if not all(controls.values()):
        raise ValueError(
            "blueprint-bound budget admission benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-blueprint-bound-budget-admission-benchmark.v2",
        "status": "pass",
        "source": {
            "budget_policy_path": policy_path.relative_to(project_root).as_posix(),
            "budget_policy_sha256": sha256_file(policy_path),
            "budget_policy_id": policy["budget_policy_id"],
            "builder_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/build_cluster_aware_sae_blueprint_budget_admission_v2.py"
            ),
            "synthetic_blueprint_benchmark_id": (
                "7612bb0da7e388f029cbfe66b17a104ff93a32fef22763df3625a462db5daa75"
            ),
        },
        "controls": controls,
        "results": {
            "proposal_id": proposal["proposal_id"],
            "admission_id": admission["admission_id"],
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "execution_blueprint_sha256": blueprint_hash,
            "global_requested_resources": proposal["global_requested_resources"],
        },
        "summary": {
            "passed": sum(controls.values()),
            "controls": len(controls),
            "activation_categories": 8,
            "specific_compartments": 16,
            "shared_fallback_compartments": 1,
            "execution_jobs": 34,
            "planned_steps": 68000,
            "planned_activation_rows": 20480,
            "synthetic_blueprint_persisted": False,
        },
        "scientific_boundary": {
            "synthetic_blueprint_fixture_only": True,
            "empirical_blueprint_present": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "weights_loaded_or_mutated": False,
            "persistent_dictionary_output_created": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("blueprint-bound budget admission benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-blueprint-bound-budget-admission-benchmark-verification.v2",
        "status": "pass",
        "benchmark_id": identifier,
        "proposal_id": observed["results"]["proposal_id"],
        "admission_id": observed["results"]["admission_id"],
        "execution_blueprint_id": observed["results"]["execution_blueprint_id"],
        "execution_blueprint_sha256": observed["results"]
        ["execution_blueprint_sha256"],
        "controls": observed["summary"]["controls"],
        "execution_jobs": observed["summary"]["execution_jobs"],
        "planned_steps": observed["summary"]["planned_steps"],
        "planned_activation_rows": observed["summary"]["planned_activation_rows"],
        "synthetic_blueprint_persisted": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify blueprint-bound budget admission benchmark V2."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=root)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result.get("benchmark_id"),
                "controls": result.get("controls", result.get("summary", {}).get("controls")),
                "execution_authorized": result.get(
                    "execution_authorized",
                    result.get("scientific_boundary", {}).get("execution_authorized"),
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
