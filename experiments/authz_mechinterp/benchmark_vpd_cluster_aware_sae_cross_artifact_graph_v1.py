from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.verify_cluster_aware_sae_cross_artifact_graph_v1 import (
    ALLOWED_GRAPH_PARENT,
    EDGE_SPECS,
    MANIFEST_SCHEMA,
    NODE_SPECS,
    REQUIRED_EXECUTION_BINDINGS,
    canonical_inputs,
    validate_graph_manifest,
    verify_graph_contract,
)


BENCHMARK_SCHEMA = "sch.vpd-cluster-aware-sae-cross-artifact-graph-benchmark.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _digest(label: str) -> str:
    return hashlib.sha256(f"vpd-synthetic-cross-graph:{label}".encode()).hexdigest()


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, OSError, TypeError, ValueError):
        return True
    return False


def _set_field(document: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = document
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set synthetic fixture field {dotted_path}")
        current = child
    current[parts[-1]] = value


def _readdress_document(
    document: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    changed = copy.deepcopy(document)
    body = {
        key: value
        for key, value in changed.items()
        if key not in set(spec["address_excluded_fields"])
    }
    changed[spec["id_field"]] = content_sha256(body)
    return changed


def _readdress_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(manifest)
    body = {key: value for key, value in changed.items() if key != "graph_manifest_id"}
    changed["graph_manifest_id"] = content_sha256(body)
    return changed


def _synthetic_execution_bindings(project_root: Path) -> dict[str, Any]:
    return {
        "model_id": "synthetic-model-id-not-opened",
        "model_revision": "synthetic-immutable-revision-not-opened",
        "model_files_manifest_sha256": _digest("model-files-manifest"),
        "adapter_path": str(
            (project_root / "artifacts/synthetic-adapter-fixture-not-opened").resolve()
        ),
        "adapter_weights_sha256": _digest("adapter-weights"),
        "adapter_config_sha256": _digest("adapter-config"),
        "device": "cpu",
        "wall_time_ceiling_seconds": 1,
        "cost_ceiling_usd": 0.01,
        "output_root": str(
            (
                project_root
                / "artifacts/vpd_cluster_to_sae_empirical_capture_runs/synthetic-cross-graph-fixture-never-created"
            ).resolve()
        ),
    }


def _build_fixture(
    *, project_root: Path, graph_root: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    bindings = _synthetic_execution_bindings(project_root)
    descriptors: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    specs = {row["node"]: row for row in NODE_SPECS}
    for spec in NODE_SPECS:
        name = spec["node"]
        document: dict[str, Any] = {
            "schema": spec["schema"],
            "synthetic_fixture_only": True,
            "source": {},
            "fixture_label": f"synthetic-{name}-not-empirical",
        }
        if name == "empirical_preflight_v2":
            document["immutable_bindings"] = copy.deepcopy(bindings)
        for edge in (row for row in EDGE_SPECS if row["source_node"] == name):
            if edge["target_node"] == "__budget_policy__":
                target = {
                    "content_id": contract["source"]["budget_policy_id"],
                    "file_sha256": contract["source"]["budget_policy_sha256"],
                }
            else:
                target = descriptors[edge["target_node"]]
            _set_field(document, edge["source_field_path"], target[edge["target_value"]])
        if "signature" in spec["address_excluded_fields"]:
            document["signature"] = "synthetic-signature-never-authenticated"
        document = _readdress_document(document, spec)
        path = graph_root / f"{name}.json"
        write_json(path, document)
        descriptor = {
            "node": name,
            "path": path.relative_to(project_root).as_posix(),
            "file_sha256": sha256_file(path),
            "content_id": document[spec["id_field"]],
        }
        documents[name] = document
        descriptors[name] = descriptor
    body = {
        "schema": MANIFEST_SCHEMA,
        "graph_contract_id": contract["graph_contract_id"],
        "graph_kind": "synthetic_benchmark_fixture",
        "graph_root": graph_root.relative_to(project_root).as_posix(),
        "execution_identity_bindings": bindings,
        "nodes": [descriptors[row["node"]] for row in NODE_SPECS],
    }
    manifest = {**body, "graph_manifest_id": content_sha256(body)}
    return manifest, documents, descriptors


def _replace_node(
    *,
    project_root: Path,
    graph_root: Path,
    manifest: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    node: str,
    label: str,
    mutator: Callable[[dict[str, Any]], None],
    readdress: bool,
) -> dict[str, Any]:
    spec = next(row for row in NODE_SPECS if row["node"] == node)
    document = copy.deepcopy(documents[node])
    mutator(document)
    if readdress:
        document = _readdress_document(document, spec)
    path = graph_root / "variants" / f"{label}.json"
    write_json(path, document)
    replacement = {
        "node": node,
        "path": path.relative_to(project_root).as_posix(),
        "file_sha256": sha256_file(path),
        "content_id": document[spec["id_field"]],
    }
    changed = copy.deepcopy(manifest)
    index = next(i for i, row in enumerate(changed["nodes"]) if row["node"] == node)
    changed["nodes"][index] = replacement
    return _readdress_manifest(changed)


def _manifest_mutation(
    manifest: dict[str, Any], mutator: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    changed = copy.deepcopy(manifest)
    mutator(changed)
    return _readdress_manifest(changed)


def _exercise_fixture(
    *, project_root: Path, graph_root: Path, contract: dict[str, Any]
) -> dict[str, bool]:
    manifest, documents, descriptors = _build_fixture(
        project_root=project_root, graph_root=graph_root, contract=contract
    )
    validate = lambda candidate: validate_graph_manifest(
        contract=contract, manifest=candidate, project_root=project_root
    )
    clean = validate(manifest)

    missing_node = _manifest_mutation(manifest, lambda value: value["nodes"].pop())
    duplicate_node = _manifest_mutation(
        manifest, lambda value: value["nodes"].append(copy.deepcopy(value["nodes"][0]))
    )
    unknown_node = _manifest_mutation(
        manifest, lambda value: value["nodes"][0].__setitem__("node", "unknown")
    )
    wrong_hash = _manifest_mutation(
        manifest, lambda value: value["nodes"][0].__setitem__("file_sha256", "f" * 64)
    )
    wrong_content_id = _manifest_mutation(
        manifest, lambda value: value["nodes"][0].__setitem__("content_id", "f" * 64)
    )
    schema_tamper = _replace_node(
        project_root=project_root,
        graph_root=graph_root,
        manifest=manifest,
        documents=documents,
        node="empirical_routing_bundle",
        label="schema-tamper",
        mutator=lambda value: value.__setitem__("schema", "sch.wrong"),
        readdress=True,
    )
    content_tamper = _replace_node(
        project_root=project_root,
        graph_root=graph_root,
        manifest=manifest,
        documents=documents,
        node="routing_stability_manifest",
        label="unaddressed-content-tamper",
        mutator=lambda value: value.__setitem__("fixture_label", "tampered"),
        readdress=False,
    )
    edge_tamper = _replace_node(
        project_root=project_root,
        graph_root=graph_root,
        manifest=manifest,
        documents=documents,
        node="execution_blueprint",
        label="readdressed-edge-tamper",
        mutator=lambda value: value["source"].__setitem__("materialization_id", "e" * 64),
        readdress=True,
    )
    policy_tamper = _replace_node(
        project_root=project_root,
        graph_root=graph_root,
        manifest=manifest,
        documents=documents,
        node="budget_proposal",
        label="readdressed-policy-tamper",
        mutator=lambda value: value["source"].__setitem__("budget_policy_id", "e" * 64),
        readdress=True,
    )
    preflight_mismatch = _replace_node(
        project_root=project_root,
        graph_root=graph_root,
        manifest=manifest,
        documents=documents,
        node="empirical_preflight_v2",
        label="preflight-binding-mismatch",
        mutator=lambda value: value["immutable_bindings"].__setitem__(
            "model_revision", "different-synthetic-revision"
        ),
        readdress=True,
    )
    missing_binding = _manifest_mutation(
        manifest,
        lambda value: value["execution_identity_bindings"].pop("adapter_config_sha256"),
    )
    absolute_path = _manifest_mutation(
        manifest,
        lambda value: value["nodes"][0].__setitem__(
            "path", str((project_root / value["nodes"][0]["path"]).resolve())
        ),
    )
    traversal_path = _manifest_mutation(
        manifest,
        lambda value: value["nodes"][0].__setitem__(
            "path", f"{manifest['graph_root']}/../{graph_root.name}/{graph_root.name}.json"
        ),
    )
    root_escape = _manifest_mutation(
        manifest, lambda value: value.__setitem__("graph_root", ALLOWED_GRAPH_PARENT)
    )
    missing_file = _manifest_mutation(
        manifest,
        lambda value: value["nodes"][0].update(
            {
                "path": f"{manifest['graph_root']}/missing-node.json",
                "file_sha256": "0" * 64,
            }
        ),
    )
    wrong_contract = _manifest_mutation(
        manifest, lambda value: value.__setitem__("graph_contract_id", "0" * 64)
    )
    manifest_content_tamper = copy.deepcopy(manifest)
    manifest_content_tamper["graph_kind"] = "empirical_candidate"
    mixed_boundary = _replace_node(
        project_root=project_root,
        graph_root=graph_root,
        manifest=manifest,
        documents=documents,
        node="global_pair_split",
        label="mixed-boundary",
        mutator=lambda value: value.__setitem__("synthetic_fixture_only", False),
        readdress=True,
    )
    kind_mismatch = _manifest_mutation(
        manifest, lambda value: value.__setitem__("graph_kind", "empirical_candidate")
    )
    invalid_kind = _manifest_mutation(
        manifest, lambda value: value.__setitem__("graph_kind", "unknown")
    )

    link_path = graph_root / "variants" / "linked-preflight.json"
    reparse_path = link_path
    reparse_created = False
    try:
        os.symlink(
            graph_root / "empirical_preflight_v2.json",
            link_path,
            target_is_directory=False,
        )
        reparse_created = True
    except (NotImplementedError, OSError):
        junction_path = graph_root / "variants" / "linked-graph-root"
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction_path), str(graph_root)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            reparse_created = True
            reparse_path = junction_path / "empirical_preflight_v2.json"
    if reparse_created:
        linked_node = _manifest_mutation(
            manifest,
            lambda value: value["nodes"][0].update(
                {
                    "path": reparse_path.relative_to(project_root).as_posix(),
                    "file_sha256": descriptors["empirical_preflight_v2"]["file_sha256"],
                }
            ),
        )
        reparse_rejected = _rejects(lambda: validate(linked_node))
    else:
        reparse_rejected = False

    controls = {
        "clean_synthetic_graph_structurally_verifies": clean["status"]
        == "synthetic_graph_structurally_verified_non_authorizing",
        "clean_graph_has_exactly_20_nodes": clean["verified"]["nodes"] == 20,
        "clean_graph_has_exactly_37_edges": clean["verified"]["edges"] == 37,
        "clean_graph_has_exactly_10_identity_bindings": clean["verified"]
        ["execution_identity_bindings"]
        == 10,
        "all_node_file_hashes_verify": clean["verified"]["file_hashes"] == 20,
        "all_node_content_addresses_verify": clean["verified"]["content_addresses"]
        == 20,
        "missing_node_rejected": _rejects(lambda: validate(missing_node)),
        "duplicate_node_rejected": _rejects(lambda: validate(duplicate_node)),
        "unknown_node_rejected": _rejects(lambda: validate(unknown_node)),
        "wrong_node_file_hash_rejected": _rejects(lambda: validate(wrong_hash)),
        "wrong_descriptor_content_id_rejected": _rejects(
            lambda: validate(wrong_content_id)
        ),
        "readdressed_schema_tamper_rejected": _rejects(lambda: validate(schema_tamper)),
        "unaddressed_node_content_tamper_rejected": _rejects(
            lambda: validate(content_tamper)
        ),
        "readdressed_cross_artifact_edge_tamper_rejected": _rejects(
            lambda: validate(edge_tamper)
        ),
        "readdressed_budget_policy_substitution_rejected": _rejects(
            lambda: validate(policy_tamper)
        ),
        "preflight_execution_binding_mismatch_rejected": _rejects(
            lambda: validate(preflight_mismatch)
        ),
        "missing_execution_binding_rejected": _rejects(lambda: validate(missing_binding)),
        "absolute_node_path_rejected": _rejects(lambda: validate(absolute_path)),
        "node_path_traversal_rejected": _rejects(lambda: validate(traversal_path)),
        "graph_root_escape_rejected": _rejects(lambda: validate(root_escape)),
        "missing_node_file_rejected": _rejects(lambda: validate(missing_file)),
        "symbolic_link_or_junction_node_rejected": reparse_rejected,
        "wrong_contract_binding_rejected": _rejects(lambda: validate(wrong_contract)),
        "manifest_content_tamper_rejected": _rejects(
            lambda: validate(manifest_content_tamper)
        ),
        "mixed_synthetic_empirical_nodes_rejected": _rejects(
            lambda: validate(mixed_boundary)
        ),
        "graph_kind_node_boundary_mismatch_rejected": _rejects(
            lambda: validate(kind_mismatch)
        ),
        "unknown_graph_kind_rejected": _rejects(lambda: validate(invalid_kind)),
        "structural_pass_authenticates_no_signature_or_authority": all(
            clean["scientific_boundary"][field] is False
            for field in (
                "node_specific_semantic_verifiers_invoked",
                "receipt_signatures_verified",
                "authority_status_verified",
                "nonce_freshness_verified",
                "output_reservation_exclusivity_verified",
            )
        ),
        "structural_pass_authorizes_no_execution_or_training": all(
            clean["scientific_boundary"][field] is False
            for field in (
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
        "synthetic_output_root_not_created": not Path(
            manifest["execution_identity_bindings"]["output_root"]
        ).exists(),
    }
    return controls


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    contract_path = (
        project_root
        / "artifacts/vpd_cluster_aware_sae_cross_artifact_graph_contract_v1/contract.json"
    )
    verification = verify_graph_contract(contract_path, **canonical_inputs(project_root))
    contract = _load(contract_path)
    allowed_parent = project_root / ALLOWED_GRAPH_PARENT
    allowed_parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="benchmark-", dir=allowed_parent) as value:
        temporary_path = Path(value)
        controls = _exercise_fixture(
            project_root=project_root,
            graph_root=temporary_path,
            contract=contract,
        )
    controls["temporary_synthetic_graph_removed"] = bool(
        temporary_path is not None and not temporary_path.exists()
    )
    controls["contract_reproduces"] = verification["status"] == "pass"
    controls["contract_shape_is_exact"] = (
        verification["required_nodes"],
        verification["required_edges"],
        verification["execution_identity_bindings"],
    ) == (20, 37, 10)
    controls["benchmark_imports_no_model_frameworks"] = before_frameworks == (
        MODEL_FRAMEWORKS & set(sys.modules)
    )
    if not all(controls.values()):
        raise ValueError(
            "cross-artifact graph benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": BENCHMARK_SCHEMA,
        "status": "pass",
        "source": {
            "graph_contract_path": contract_path.relative_to(project_root).as_posix(),
            "graph_contract_sha256": sha256_file(contract_path),
            "graph_contract_id": contract["graph_contract_id"],
            "verifier_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/verify_cluster_aware_sae_cross_artifact_graph_v1.py"
            ),
        },
        "controls": controls,
        "summary": {
            "passed": sum(controls.values()),
            "controls": len(controls),
            "required_nodes": len(NODE_SPECS),
            "required_edges": len(EDGE_SPECS),
            "required_execution_identity_bindings": len(REQUIRED_EXECUTION_BINDINGS),
            "synthetic_graph_persisted": False,
        },
        "scientific_boundary": {
            "temporary_synthetic_fixture_only": True,
            "empirical_graph_present": False,
            "node_specific_semantic_verifiers_invoked": False,
            "receipt_signatures_verified": False,
            "authority_status_verified": False,
            "nonce_freshness_verified": False,
            "output_reservation_exclusivity_verified": False,
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
        raise ValueError("cross-artifact graph benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-cross-artifact-graph-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "graph_contract_id": observed["source"]["graph_contract_id"],
        "controls": observed["summary"]["controls"],
        "required_nodes": observed["summary"]["required_nodes"],
        "required_edges": observed["summary"]["required_edges"],
        "required_execution_identity_bindings": observed["summary"]
        ["required_execution_identity_bindings"],
        "synthetic_graph_persisted": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the model-free cross-artifact graph benchmark V1."
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
                "graph_contract_id": result.get(
                    "graph_contract_id", result.get("source", {}).get("graph_contract_id")
                ),
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
