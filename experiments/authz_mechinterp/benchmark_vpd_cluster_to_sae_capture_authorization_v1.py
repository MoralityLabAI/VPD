from __future__ import annotations

import argparse
import base64
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 import (
    REQUIRED_EXECUTION_BINDINGS,
    build_preflight,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import crypto_types
from experiments.authz_mechinterp.verify_vpd_cluster_to_sae_capture_authorization_v1 import (
    ACTION,
    RECEIPT_SCHEMA,
    REGISTRY_SCHEMA,
    _required_scope,
    sign_authorization_receipt,
    verify_authorization_receipt,
)


MODEL_FRAMEWORKS = {"torch", "transformers", "peft", "accelerate"}
AS_OF = "2030-01-01T00:30:00Z"


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def _bindings() -> dict[str, Any]:
    return {
        "model_id": "synthetic/model-not-loaded",
        "model_revision": "synthetic-revision-not-loaded",
        "model_files_manifest_sha256": "1" * 64,
        "adapter_path": "synthetic/adapter-not-opened",
        "adapter_weights_sha256": "2" * 64,
        "adapter_config_sha256": "3" * 64,
        "device": "synthetic-device-not-used",
        "wall_time_ceiling_seconds": 3600,
        "cost_ceiling_usd": 1.0,
        "output_root": "artifacts/synthetic-authorization-output-not-created",
    }


def _registry(private_key: Any, *, enabled: bool = True) -> dict[str, Any]:
    _, _, serialization, _ = crypto_types()
    public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    body = {
        "schema": REGISTRY_SCHEMA,
        "environment": "synthetic_benchmark_only",
        "authorities": [
            {
                "authority_id": "synthetic-authority",
                "algorithm": "ed25519",
                "public_key_base64": base64.b64encode(public).decode("ascii"),
                "enabled": enabled,
                "synthetic_fixture_only": True,
                "allowed_actions": [ACTION],
                "maximum_scope": {
                    "maximum_compute_jobs": 64,
                    "maximum_batch_forwards_at_batch_size_4": 41_352_192,
                    "maximum_episode_forward_equivalents": 164_900_736,
                    "maximum_float32_effect_payload_bytes": 4_046_192_640,
                },
            }
        ],
        "revoked_receipt_ids": [],
        "consumed_nonces": [],
    }
    return {**body, "registry_id": content_sha256(body)}


def _receipt_body(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "synthetic_fixture_only",
        "synthetic_fixture_only": True,
        "authority_id": "synthetic-authority",
        "issued_at": "2030-01-01T00:00:00Z",
        "not_before": "2030-01-01T00:00:00Z",
        "expires_at": "2030-01-01T01:00:00Z",
        "nonce": "4" * 64,
        "explicit_user_authorization_statement": (
            "synthetic benchmark only; no real execution is authorized"
        ),
        "source": {
            "preflight_id": preflight["preflight_id"],
            "preflight_sha256": content_sha256(preflight),
        },
        "identity_bindings": preflight["immutable_bindings"],
        "scope": _required_scope(preflight),
    }


def _address_registry(registry: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value for key, value in registry.items() if key != "registry_id"
    }
    return {**body, "registry_id": content_sha256(body)}


def build_report(project_root: Path) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    template_v1 = json.loads(
        (
            project_root
            / "artifacts/vpd_cluster_to_sae_empirical_preflight_v1/preflight.json"
        ).read_text(encoding="utf-8")
    )
    template_v2 = build_preflight(project_root=project_root)
    bound = build_preflight(project_root=project_root, bindings=_bindings())
    Ed25519PrivateKey, _, _, _ = crypto_types()
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
    registry = _registry(key)
    receipt = sign_authorization_receipt(_receipt_body(bound), key)
    clean = verify_authorization_receipt(
        receipt=receipt,
        registry=registry,
        preflight=bound,
        as_of=AS_OF,
    )
    after = MODEL_FRAMEWORKS & set(sys.modules)

    def signed_mutation(mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        body = _receipt_body(bound)
        mutator(body)
        return sign_authorization_receipt(body, key)

    wrong_preflight = signed_mutation(
        lambda body: body["source"].__setitem__("preflight_id", "f" * 64)
    )
    wrong_identity = signed_mutation(
        lambda body: body["identity_bindings"].__setitem__(
            "model_revision", "different-revision"
        )
    )
    broad_sae = signed_mutation(
        lambda body: body["scope"].__setitem__(
            "sae_dictionary_training_authorized", True
        )
    )
    broad_final = signed_mutation(
        lambda body: body["scope"].__setitem__("final_evaluation_authorized", True)
    )
    high_ceiling = signed_mutation(
        lambda body: body["scope"].__setitem__(
            "maximum_compute_jobs", body["scope"]["maximum_compute_jobs"] + 1
        )
    )
    expired = signed_mutation(
        lambda body: body.__setitem__("expires_at", "2030-01-01T00:15:00Z")
    )
    future = signed_mutation(
        lambda body: body.__setitem__("not_before", "2030-01-01T00:45:00Z")
    )
    bad_signature = copy.deepcopy(receipt)
    bad_signature["signature"]["value_base64"] = base64.b64encode(b"0" * 64).decode(
        "ascii"
    )
    disabled_registry = _registry(key, enabled=False)
    consumed_registry = copy.deepcopy(registry)
    consumed_registry["consumed_nonces"] = [receipt["nonce"]]
    consumed_registry = _address_registry(consumed_registry)
    revoked_registry = copy.deepcopy(registry)
    revoked_registry["revoked_receipt_ids"] = [receipt["authorization_receipt_id"]]
    revoked_registry = _address_registry(revoked_registry)
    controls = {
        "v1_content_address_cycle_is_recorded": len(
            template_v1["missing_required_bindings"]
        )
        == 13
        and "authorization_receipt_id" in template_v1["immutable_bindings"],
        "v2_removes_receipt_from_preflight_inputs": len(
            template_v2["missing_required_bindings"]
        )
        == 10
        and set(template_v2["immutable_bindings"]) == set(REQUIRED_EXECUTION_BINDINGS)
        and all(
            field not in template_v2["immutable_bindings"]
            for field in (
                "authorization_receipt_id",
                "authorization_receipt_sha256",
                "explicit_user_authorization_statement",
            )
        ),
        "identity_complete_preflight_awaits_one_way_receipt": bound["status"]
        == "identity_complete_awaiting_signed_authorization_receipt"
        and not bound["missing_required_bindings"],
        "clean_synthetic_signature_and_scope_validate_without_authority": clean[
            "status"
        ]
        == "pass_authenticated_synthetic_fixture_not_authorized"
        and clean["signature_valid"]
        and clean["execution_authorized"] is False,
        "wrong_preflight_binding_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=wrong_preflight,
                registry=registry,
                preflight=bound,
                as_of=AS_OF,
            )
        ),
        "wrong_identity_binding_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=wrong_identity,
                registry=registry,
                preflight=bound,
                as_of=AS_OF,
            )
        ),
        "sae_training_scope_broadening_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=broad_sae, registry=registry, preflight=bound, as_of=AS_OF
            )
        ),
        "final_evaluation_scope_broadening_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=broad_final,
                registry=registry,
                preflight=bound,
                as_of=AS_OF,
            )
        ),
        "resource_ceiling_broadening_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=high_ceiling,
                registry=registry,
                preflight=bound,
                as_of=AS_OF,
            )
        ),
        "expired_and_not_yet_active_receipts_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=expired, registry=registry, preflight=bound, as_of=AS_OF
            )
        )
        and _rejects(
            lambda: verify_authorization_receipt(
                receipt=future, registry=registry, preflight=bound, as_of=AS_OF
            )
        ),
        "invalid_signature_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=bad_signature,
                registry=registry,
                preflight=bound,
                as_of=AS_OF,
            )
        ),
        "disabled_authority_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=receipt,
                registry=disabled_registry,
                preflight=bound,
                as_of=AS_OF,
            )
        ),
        "consumed_nonce_replay_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=receipt,
                registry=consumed_registry,
                preflight=bound,
                as_of=AS_OF,
            )
        ),
        "revoked_receipt_rejected": _rejects(
            lambda: verify_authorization_receipt(
                receipt=receipt,
                registry=revoked_registry,
                preflight=bound,
                as_of=AS_OF,
            )
        ),
        "validator_imports_no_model_frameworks": before == after,
        "all_paths_remain_nonexecuting": all(
            clean[field] is False
            for field in (
                "execution_authorized",
                "sae_training_authorized",
                "final_evaluation_authorized",
                "real_actions_authorized",
            )
        ),
    }
    if not all(controls.values()):
        raise ValueError(
            "cluster-to-SAE capture authorization benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-to-sae-capture-authorization-benchmark.v1",
        "status": "pass",
        "results": {
            "preflight_v1_id": template_v1["preflight_id"],
            "preflight_v2_template_id": template_v2["preflight_id"],
            "synthetic_bound_preflight_id": bound["preflight_id"],
            "synthetic_authority_registry_id": registry["registry_id"],
            "synthetic_authorization_receipt_id": receipt[
                "authorization_receipt_id"
            ],
            "clean_verification": clean,
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "source": {
            "preflight_v2_builder_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/build_vpd_cluster_to_sae_empirical_preflight_v2.py"
            ),
            "authorization_validator_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/verify_vpd_cluster_to_sae_capture_authorization_v1.py"
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "synthetic_keys_and_identities_only": True,
            "private_key_persisted": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "release_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = observed.get("benchmark_id")
    body = {key: value for key, value in observed.items() if key != "benchmark_id"}
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("cluster-to-SAE capture authorization benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-to-sae-capture-authorization-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "preflight_v2_template_id": observed["results"]["preflight_v2_template_id"],
        "controls": observed["summary"]["controls"],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark signed cluster-to-SAE capture authorization."
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
