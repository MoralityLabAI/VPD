from __future__ import annotations

import argparse
import base64
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1 import (
    COMPUTE_ACTION,
    RELEASE_ACTION,
    sign_receipt,
)
from experiments.authz_mechinterp.admit_cluster_aware_sae_training_v2 import (
    RELEASE_RECEIPT_SCHEMA,
    _expected_release_evidence,
    _reproduce_budget_chain,
    build_budget_bound_runtime_admission_from_paths_v2,
    expected_release_scope_v2,
    sign_scientific_release_receipt_v2,
    verify_runtime_admission_v2,
    verify_scientific_release_receipt_v2,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_admission_v1 import (
    AS_OF,
    DEVICE,
    _compute_body,
    _readdress_registry,
    _registry,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_slice_v1 import (
    _build_fixture,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.dispatch_cluster_aware_sae_semantic_slice_v1 import (
    ALLOWED_INPUT_PARENT,
)
from experiments.authz_mechinterp.dispatch_cluster_aware_sae_semantic_slice_v2 import (
    dispatch_semantic_slice_v2,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import crypto_types
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    validate_execution_blueprint_content,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, RuntimeError, TypeError, ValueError):
        return True
    return False


def _readdress(value: dict[str, Any], identifier: str) -> dict[str, Any]:
    changed = copy.deepcopy(value)
    body = {key: item for key, item in changed.items() if key != identifier}
    changed[identifier] = content_sha256(body)
    return changed


def _write_signed_fixture(
    root: Path, files: dict[str, str]
) -> tuple[dict[str, str], dict[str, Any]]:
    paths = {name: root / relative for name, relative in files.items()}
    blueprint = _load(paths["execution_blueprint"])
    policy = _load(paths["budget_policy"])
    proposal = _load(paths["budget_proposal"])
    admission = _load(paths["budget_admission"])
    blueprint_sha256 = sha256_file(paths["execution_blueprint"])
    policy_sha256 = sha256_file(paths["budget_policy"])
    proposal_sha256 = sha256_file(paths["budget_proposal"])
    admission_sha256 = sha256_file(paths["budget_admission"])
    blueprint_check = validate_execution_blueprint_content(blueprint)
    budget_check = _reproduce_budget_chain(
        policy=policy,
        proposal=proposal,
        admission=admission,
        blueprint=blueprint,
        blueprint_sha256=blueprint_sha256,
    )
    Ed25519PrivateKey, _, _, _ = crypto_types()
    release_key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
    compute_key = Ed25519PrivateKey.from_private_bytes(bytes(range(33, 65)))
    release_registry = _registry(
        release_key,
        authority_id="synthetic-release-authority",
        action=RELEASE_ACTION,
    )
    compute_registry = _registry(
        compute_key,
        authority_id="synthetic-compute-authority",
        action=COMPUTE_ACTION,
    )
    release_body = {
        "schema": RELEASE_RECEIPT_SCHEMA,
        "status": "synthetic_fixture_only",
        "synthetic_fixture_only": True,
        "authority_id": "synthetic-release-authority",
        "issued_at": "2030-01-01T00:00:00Z",
        "not_before": "2030-01-01T00:00:00Z",
        "expires_at": "2030-01-01T01:00:00Z",
        "nonce": "3" * 64,
        "explicit_release_statement": (
            "synthetic benchmark only; no scientific release or execution authority"
        ),
        "source": {
            "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
            "execution_blueprint_sha256": blueprint_sha256,
            "budget_policy_id": budget_check["budget_policy_id"],
            "budget_policy_sha256": policy_sha256,
            "budget_proposal_id": budget_check["budget_proposal_id"],
            "budget_proposal_sha256": proposal_sha256,
            "budget_admission_id": budget_check["budget_admission_id"],
            "budget_admission_sha256": admission_sha256,
        },
        "scope": expected_release_scope_v2(
            blueprint=blueprint,
            blueprint_check=blueprint_check,
            budget_check=budget_check,
            policy_sha256=policy_sha256,
            proposal_sha256=proposal_sha256,
            admission_sha256=admission_sha256,
        ),
        "release_evidence": _expected_release_evidence(blueprint),
    }
    release_receipt = sign_scientific_release_receipt_v2(
        release_body, release_key
    )
    release_registry_path = root / "release-authority-registry.json"
    release_receipt_path = root / "scientific-release-receipt-v2.json"
    compute_registry_path = root / "compute-authority-registry.json"
    compute_receipt_path = root / "compute-authorization-receipt.json"
    runtime_admission_path = root / "runtime-admission-v2.json"
    write_json(release_registry_path, release_registry)
    write_json(release_receipt_path, release_receipt)
    write_json(compute_registry_path, compute_registry)
    release_verification = verify_scientific_release_receipt_v2(
        release_receipt_path,
        registry_path=release_registry_path,
        blueprint_path=paths["execution_blueprint"],
        policy_path=paths["budget_policy"],
        proposal_path=paths["budget_proposal"],
        admission_path=paths["budget_admission"],
        as_of=AS_OF,
    )
    job = blueprint["execution_jobs"][0]
    compute_body = _compute_body(
        blueprint=blueprint,
        job=job,
        release_verification=release_verification,
    )
    compute_receipt = sign_receipt(compute_body, compute_key)
    write_json(compute_receipt_path, compute_receipt)
    runtime_admission = build_budget_bound_runtime_admission_from_paths_v2(
        release_receipt_path=release_receipt_path,
        release_registry_path=release_registry_path,
        blueprint_path=paths["execution_blueprint"],
        policy_path=paths["budget_policy"],
        proposal_path=paths["budget_proposal"],
        budget_admission_path=paths["budget_admission"],
        compute_receipt_path=compute_receipt_path,
        compute_registry_path=compute_registry_path,
        execution_job_id=job["execution_job_id"],
        device=DEVICE,
        as_of=AS_OF,
    )
    write_json(runtime_admission_path, runtime_admission)
    added_files = {
        "scientific_release_receipt": release_receipt_path.relative_to(root).as_posix(),
        "scientific_release_registry": release_registry_path.relative_to(root).as_posix(),
        "compute_authorization_receipt": compute_receipt_path.relative_to(root).as_posix(),
        "compute_authority_registry": compute_registry_path.relative_to(root).as_posix(),
        "runtime_admission": runtime_admission_path.relative_to(root).as_posix(),
    }
    return (
        {**files, **added_files},
        {
            "paths": {
                **paths,
                **{name: root / relative for name, relative in added_files.items()},
            },
            "blueprint": blueprint,
            "policy": policy,
            "proposal": proposal,
            "admission": admission,
            "release_key": release_key,
            "compute_key": compute_key,
            "release_registry": release_registry,
            "compute_registry": compute_registry,
            "release_body": release_body,
            "release_receipt": release_receipt,
            "release_verification": release_verification,
            "compute_body": compute_body,
            "compute_receipt": compute_receipt,
            "runtime_admission": runtime_admission,
            "job": job,
        },
    )


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    allowed_parent = project_root / ALLOWED_INPUT_PARENT
    allowed_parent.mkdir(parents=True, exist_ok=True)
    temporary_root: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix="semantic-slice-v2-", dir=allowed_parent
    ) as value:
        root = Path(value)
        temporary_root = root
        files, fixture = _write_signed_fixture(root, _build_fixture(project_root, root))
        paths: dict[str, Path] = fixture["paths"]
        common = {
            "project_root": project_root,
            "artifact_root": root.relative_to(project_root).as_posix(),
            **files,
            "execution_job_id": fixture["job"]["execution_job_id"],
            "device": DEVICE,
            "as_of": AS_OF,
        }
        clean = dispatch_semantic_slice_v2(
            **common, expected_graph_kind="synthetic_benchmark_fixture"
        )

        release_kwargs = {
            "registry_path": paths["scientific_release_registry"],
            "blueprint_path": paths["execution_blueprint"],
            "policy_path": paths["budget_policy"],
            "proposal_path": paths["budget_proposal"],
            "admission_path": paths["budget_admission"],
            "as_of": AS_OF,
        }

        def verify_release() -> dict[str, Any]:
            return verify_scientific_release_receipt_v2(
                paths["scientific_release_receipt"], **release_kwargs
            )

        invalid_signature = copy.deepcopy(fixture["release_receipt"])
        invalid_signature["signature"]["value_base64"] = base64.b64encode(
            b"x" * 64
        ).decode("ascii")
        write_json(paths["scientific_release_receipt"], invalid_signature)
        invalid_signature_rejected = _rejects(verify_release)
        write_json(paths["scientific_release_receipt"], fixture["release_receipt"])

        def signed_release_mutation(
            mutator: Callable[[dict[str, Any]], None]
        ) -> dict[str, Any]:
            changed = copy.deepcopy(fixture["release_body"])
            mutator(changed)
            return sign_scientific_release_receipt_v2(
                changed, fixture["release_key"]
            )

        wrong_admission_id = signed_release_mutation(
            lambda body: (
                body["source"].__setitem__("budget_admission_id", "f" * 64),
                body["scope"].__setitem__("budget_admission_id", "f" * 64),
            )
        )
        write_json(paths["scientific_release_receipt"], wrong_admission_id)
        wrong_admission_id_rejected = _rejects(verify_release)

        wrong_admission_hash = signed_release_mutation(
            lambda body: (
                body["source"].__setitem__("budget_admission_sha256", "e" * 64),
                body["scope"].__setitem__("budget_admission_sha256", "e" * 64),
            )
        )
        write_json(paths["scientific_release_receipt"], wrong_admission_hash)
        wrong_admission_hash_rejected = _rejects(verify_release)

        expired = signed_release_mutation(
            lambda body: body.__setitem__(
                "expires_at", "2030-01-01T00:15:00Z"
            )
        )
        write_json(paths["scientific_release_receipt"], expired)
        expired_release_rejected = _rejects(verify_release)
        write_json(paths["scientific_release_receipt"], fixture["release_receipt"])

        disabled_registry = copy.deepcopy(fixture["release_registry"])
        disabled_registry["authorities"][0]["enabled"] = False
        disabled_registry = _readdress_registry(disabled_registry)
        write_json(paths["scientific_release_registry"], disabled_registry)
        disabled_authority_rejected = _rejects(verify_release)

        revoked_registry = copy.deepcopy(fixture["release_registry"])
        revoked_registry["revoked_receipt_ids"] = [
            fixture["release_receipt"]["receipt_id"]
        ]
        revoked_registry = _readdress_registry(revoked_registry)
        write_json(paths["scientific_release_registry"], revoked_registry)
        revoked_release_rejected = _rejects(verify_release)

        consumed_registry = copy.deepcopy(fixture["release_registry"])
        consumed_registry["consumed_nonces"] = [fixture["release_receipt"]["nonce"]]
        consumed_registry = _readdress_registry(consumed_registry)
        write_json(paths["scientific_release_registry"], consumed_registry)
        consumed_nonce_rejected = _rejects(verify_release)
        write_json(paths["scientific_release_registry"], fixture["release_registry"])

        changed_proposal = copy.deepcopy(fixture["proposal"])
        changed_proposal["authority_claims"]["paid_compute_authorized"] = True
        changed_proposal = _readdress(changed_proposal, "proposal_id")
        write_json(paths["budget_proposal"], changed_proposal)
        readdressed_proposal_rejected = _rejects(verify_release)
        write_json(paths["budget_proposal"], fixture["proposal"])

        changed_admission = copy.deepcopy(fixture["admission"])
        changed_admission["budget_admissible"] = False
        changed_admission = _readdress(changed_admission, "admission_id")
        write_json(paths["budget_admission"], changed_admission)
        readdressed_admission_rejected = _rejects(verify_release)
        write_json(paths["budget_admission"], fixture["admission"])

        wrong_compute_body = copy.deepcopy(fixture["compute_body"])
        wrong_compute_body["source"]["scientific_release_receipt_id"] = "d" * 64
        wrong_compute = sign_receipt(wrong_compute_body, fixture["compute_key"])
        write_json(paths["compute_authorization_receipt"], wrong_compute)
        wrong_compute_release_rejected = _rejects(
            lambda: build_budget_bound_runtime_admission_from_paths_v2(
                release_receipt_path=paths["scientific_release_receipt"],
                release_registry_path=paths["scientific_release_registry"],
                blueprint_path=paths["execution_blueprint"],
                policy_path=paths["budget_policy"],
                proposal_path=paths["budget_proposal"],
                budget_admission_path=paths["budget_admission"],
                compute_receipt_path=paths["compute_authorization_receipt"],
                compute_registry_path=paths["compute_authority_registry"],
                execution_job_id=fixture["job"]["execution_job_id"],
                device=DEVICE,
                as_of=AS_OF,
            )
        )
        write_json(paths["compute_authorization_receipt"], fixture["compute_receipt"])

        changed_runtime = copy.deepcopy(fixture["runtime_admission"])
        changed_runtime["stop_reason"] = "mutated"
        changed_runtime = _readdress(changed_runtime, "runtime_admission_id")
        write_json(paths["runtime_admission"], changed_runtime)
        readdressed_runtime_rejected = _rejects(
            lambda: verify_runtime_admission_v2(
                paths["runtime_admission"],
                release_receipt_path=paths["scientific_release_receipt"],
                release_registry_path=paths["scientific_release_registry"],
                blueprint_path=paths["execution_blueprint"],
                policy_path=paths["budget_policy"],
                proposal_path=paths["budget_proposal"],
                budget_admission_path=paths["budget_admission"],
                compute_receipt_path=paths["compute_authorization_receipt"],
                compute_registry_path=paths["compute_authority_registry"],
                execution_job_id=fixture["job"]["execution_job_id"],
                device=DEVICE,
                as_of=AS_OF,
            )
        )
        write_json(paths["runtime_admission"], fixture["runtime_admission"])

        empirical_kind_rejected = _rejects(
            lambda: dispatch_semantic_slice_v2(
                **common, expected_graph_kind="empirical_candidate"
            )
        )
        traversal_rejected = _rejects(
            lambda: dispatch_semantic_slice_v2(
                **{
                    **common,
                    "scientific_release_receipt": "../outside.json",
                },
                expected_graph_kind="synthetic_benchmark_fixture",
            )
        )
        controls = {
            "seven_real_semantic_verifiers_dispatch": clean["slice_summary"][
                "dispatched_nodes"
            ]
            == 7,
            "five_nodes_compare_full_reproduction": clean["slice_summary"][
                "full_reproduction_comparisons"
            ]
            == 5,
            "two_ed25519_receipts_verified": clean["slice_summary"][
                "cryptographic_receipt_verifications"
            ]
            == 2,
            "exact_34_job_budget_shape_preserved": clean["slice_summary"][
                "execution_jobs"
            ]
            == 34,
            "release_binds_exact_budget_ids_and_file_hashes": all(
                clean["verified_joins"][name]
                == {
                    "budget_admission_to_signed_release": fixture["admission"][
                        "admission_id"
                    ],
                    "budget_admission_file_sha256_to_signed_release": sha256_file(
                        paths["budget_admission"]
                    ),
                }[name]
                for name in (
                    "budget_admission_to_signed_release",
                    "budget_admission_file_sha256_to_signed_release",
                )
            ),
            "release_reproduces_proposal_and_admission": fixture[
                "release_verification"
            ]["budget_proposal_reproduced"]
            and fixture["release_verification"]["budget_admission_reproduced"],
            "runtime_consumes_budget_bound_release": clean["verified_joins"][
                "budget_admission_to_runtime_admission"
            ]
            == fixture["admission"]["admission_id"],
            "two_budget_terminal_joins_closed": clean["closed_terminal_joins"]
            == [
                "budget_admission_to_signed_release",
                "budget_admission_to_runtime_admission",
            ],
            "durable_transition_and_production_attachment_remain_gaps": clean[
                "terminal_gaps"
            ]
            == [
                "runtime_admission_to_durable_transition",
                "production_runtime_attachment",
            ],
            "invalid_release_signature_rejected": invalid_signature_rejected,
            "signed_wrong_budget_admission_id_rejected": wrong_admission_id_rejected,
            "signed_wrong_budget_admission_hash_rejected": wrong_admission_hash_rejected,
            "expired_release_rejected": expired_release_rejected,
            "disabled_release_authority_rejected": disabled_authority_rejected,
            "revoked_release_rejected": revoked_release_rejected,
            "consumed_release_nonce_rejected": consumed_nonce_rejected,
            "readdressed_proposal_substitution_rejected": readdressed_proposal_rejected,
            "readdressed_admission_substitution_rejected": readdressed_admission_rejected,
            "compute_receipt_wrong_release_rejected": wrong_compute_release_rejected,
            "readdressed_runtime_admission_mutation_rejected": readdressed_runtime_rejected,
            "empirical_or_real_authority_kind_rejected": empirical_kind_rejected,
            "path_traversal_rejected": traversal_rejected,
            "slice_transfers_no_authority": all(
                clean["scientific_boundary"][field] is False
                for field in (
                    "real_signed_authority_dispatched",
                    "nonce_consumed",
                    "output_reserved",
                    "durable_transition_dispatched",
                    "production_runtime_attachment_dispatched",
                    "model_framework_imported",
                    "model_loaded",
                    "model_inference_performed",
                    "training_performed",
                    "weights_loaded_or_mutated",
                    "persistent_dictionary_output_created",
                    "execution_authorized",
                    "sae_training_authorized",
                    "paid_compute_authorized",
                )
            ),
        }
        snapshot = {
            "semantic_dispatch_slice_id": clean["semantic_dispatch_slice_id"],
            "predecessor_semantic_dispatch_slice_id": clean["source"][
                "predecessor_semantic_dispatch_slice_id"
            ],
            "dispatched_node_ids": {
                row["node"]: row["id"] for row in clean["dispatched_nodes"]
            },
            "budget_admission_id": fixture["admission"]["admission_id"],
            "scientific_release_receipt_id": fixture["release_receipt"][
                "receipt_id"
            ],
            "compute_authorization_id": fixture["compute_receipt"]["receipt_id"],
            "runtime_admission_id": fixture["runtime_admission"][
                "runtime_admission_id"
            ],
            "execution_job_id": fixture["job"]["execution_job_id"],
            "dispatched_nodes": clean["slice_summary"]["dispatched_nodes"],
            "full_reproduction_comparisons": clean["slice_summary"][
                "full_reproduction_comparisons"
            ],
            "cryptographic_receipt_verifications": clean["slice_summary"][
                "cryptographic_receipt_verifications"
            ],
            "execution_jobs": clean["slice_summary"]["execution_jobs"],
            "terminal_gap_count": clean["slice_summary"]["terminal_gap_count"],
        }
        boundary = copy.deepcopy(clean["scientific_boundary"])
    temporary_fixture_removed = temporary_root is not None and not temporary_root.exists()
    after_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    controls["benchmark_imports_no_model_frameworks"] = (
        not before_frameworks and not after_frameworks
    )
    controls["temporary_fixture_root_removed"] = temporary_fixture_removed
    if not all(controls.values()):
        raise ValueError(
            "semantic dispatch slice V2 benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-semantic-dispatch-slice-benchmark.v2",
        "status": "pass",
        "source": {
            "release_and_runtime_verifier_sha256": sha256_file(
                Path(__file__).with_name("admit_cluster_aware_sae_training_v2.py")
            ),
            "dispatcher_sha256": sha256_file(
                Path(__file__).with_name(
                    "dispatch_cluster_aware_sae_semantic_slice_v2.py"
                )
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "synthetic_ed25519_receipts_only": True,
            "full_budget_chain_reproduced": True,
            "budget_admission_id_and_file_hash_signed": True,
            "runtime_admission_reproducible": True,
            "real_signed_authority_allowed": False,
            "durable_transition_performed": False,
            "production_runtime_attached": False,
        },
        "controls": controls,
        "summary": {
            "controls": len(controls),
            "passed": sum(controls.values()),
            "dispatched_nodes": snapshot["dispatched_nodes"],
            "full_reproduction_comparisons": snapshot[
                "full_reproduction_comparisons"
            ],
            "cryptographic_receipt_verifications": snapshot[
                "cryptographic_receipt_verifications"
            ],
            "execution_jobs": snapshot["execution_jobs"],
            "terminal_gap_count": snapshot["terminal_gap_count"],
            "synthetic_fixture_persisted": False,
        },
        "results": snapshot,
        "scientific_boundary": boundary,
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("semantic dispatch slice V2 benchmark address differs")
    expected = build_report(project_root)
    if observed != expected:
        raise ValueError("semantic dispatch slice V2 benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-semantic-dispatch-slice-benchmark-verification.v2",
        "status": "pass",
        "benchmark_id": identifier,
        "semantic_dispatch_slice_id": observed["results"][
            "semantic_dispatch_slice_id"
        ],
        "controls": int(observed["summary"]["controls"]),
        "dispatched_nodes": int(observed["results"]["dispatched_nodes"]),
        "full_reproduction_comparisons": int(
            observed["results"]["full_reproduction_comparisons"]
        ),
        "cryptographic_receipt_verifications": int(
            observed["results"]["cryptographic_receipt_verifications"]
        ),
        "execution_jobs": int(observed["results"]["execution_jobs"]),
        "terminal_gap_count": int(observed["results"]["terminal_gap_count"]),
        "synthetic_fixture_persisted": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the synthetic budget-bound semantic dispatcher V2."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
