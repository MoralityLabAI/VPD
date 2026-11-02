from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    OUTPUT_PARENT,
    _canonical_absolute,
    _existing_ancestor_has_reparse_point,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_runtime_admission_v1 import (
    _atomic_reserve_output_root,
    verify_runtime_admission,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-cluster-to-sae-output-reservation-transition.v1"
RECEIPT_NAME = "output_root_reservation_receipt.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_admission_address(admission_path: Path) -> dict[str, Any]:
    admission = _load(admission_path)
    if admission.get("schema") != "sch.vpd-cluster-to-sae-runtime-admission.v1":
        raise ValueError("output reservation admission schema differs")
    body = {key: value for key, value in admission.items() if key != "admission_id"}
    if admission.get("admission_id") != content_sha256(body):
        raise ValueError("output reservation admission address differs")
    joined = admission.get("joined_checks", {})
    required = (
        "preflight_reproducible",
        "canonical_bindings_complete",
        "output_root_nonexistent",
        "exact_model_tree_verified",
        "exact_adapter_bytes_verified",
        "signature_valid",
        "scope_exact",
        "authorization_active",
        "nonce_unconsumed",
        "all_checks_join_same_preflight",
    )
    if any(joined.get(field) is not True for field in required):
        raise ValueError("output reservation admission joined checks differ")
    if admission.get("reservation", {}).get("reservation_performed") is not False:
        raise ValueError("output reservation admission transition was already consumed")
    for source in admission.get("source", {}).values():
        path = Path(str(source.get("path", "")))
        if not path.is_absolute() or not path.is_file():
            raise ValueError("output reservation admission source path differs")
        if sha256_file(path) != str(source.get("sha256", "")):
            raise ValueError("output reservation admission source bytes differ")
    return admission


def _reverify_real_admission(
    *, admission_path: Path, admission: dict[str, Any], project_root: Path
) -> None:
    source = admission["source"]
    verification = verify_runtime_admission(
        admission_path,
        preflight_path=Path(source["preflight"]["path"]),
        model_manifest_path=Path(source["model_manifest"]["path"]),
        authorization_receipt_path=Path(source["authorization_receipt"]["path"]),
        authority_registry_path=Path(source["authority_registry"]["path"]),
        as_of=str(admission["as_of"]),
        project_root=project_root,
    )
    if (
        verification["admission_id"] != admission["admission_id"]
        or verification["synthetic_authorization_fixture_only"] is not False
    ):
        raise ValueError("output reservation real admission verification differs")


def reserve_admitted_output_root(
    *,
    admission_path: Path,
    project_root: Path,
    simulation_only: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    admission_path = admission_path.resolve()
    admission = _verify_admission_address(admission_path)
    synthetic = bool(
        admission.get("scientific_boundary", {}).get(
            "synthetic_authorization_fixture_only"
        )
    )
    if simulation_only is not synthetic:
        raise ValueError("output reservation simulation boundary differs")
    if not simulation_only:
        _reverify_real_admission(
            admission_path=admission_path,
            admission=admission,
            project_root=project_root,
        )
    output_root = _canonical_absolute(
        admission["reservation"]["output_root"], field="output_root"
    )
    reservation = _atomic_reserve_output_root(
        output_root=output_root, project_root=project_root
    )
    body = {
        "schema": SCHEMA,
        "status": (
            "reserved_simulation_only_capture_not_started"
            if simulation_only
            else "reserved_real_admission_capture_not_started"
        ),
        "source": {
            "runtime_admission": {
                "path": str(admission_path),
                "sha256": sha256_file(admission_path),
                "admission_id": admission["admission_id"],
                "preflight_id": admission["source"]["preflight"]["preflight_id"],
            }
        },
        "reservation": reservation,
        "simulation_only": simulation_only,
        "runtime_admission_real": not simulation_only,
        "output_reserved": True,
        "capture_started": False,
        "model_loaded": False,
        "model_inference_performed": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    receipt = {**body, "reservation_transition_id": content_sha256(body)}
    receipt_path = output_root / RECEIPT_NAME
    if receipt_path.exists():
        raise ValueError("output reservation transition receipt already exists")
    write_json(receipt_path, receipt)
    return receipt


def verify_output_reservation(
    receipt_path: Path, *, project_root: Path
) -> dict[str, Any]:
    project_root = project_root.resolve()
    receipt_path = receipt_path.resolve()
    if receipt_path.name != RECEIPT_NAME or not receipt_path.is_file():
        raise ValueError("output reservation transition receipt path differs")
    receipt = _load(receipt_path)
    if receipt.get("schema") != SCHEMA:
        raise ValueError("output reservation transition schema differs")
    body = {
        key: value for key, value in receipt.items() if key != "reservation_transition_id"
    }
    if receipt.get("reservation_transition_id") != content_sha256(body):
        raise ValueError("output reservation transition address differs")
    admission_record = receipt["source"]["runtime_admission"]
    admission_path = Path(str(admission_record["path"]))
    if (
        not admission_path.is_absolute()
        or sha256_file(admission_path) != admission_record["sha256"]
    ):
        raise ValueError("output reservation transition admission bytes differ")
    admission = _verify_admission_address(admission_path)
    if (
        admission["admission_id"] != admission_record["admission_id"]
        or admission["source"]["preflight"]["preflight_id"]
        != admission_record["preflight_id"]
    ):
        raise ValueError("output reservation transition admission identity differs")
    reservation = receipt["reservation"]
    reservation_body = {
        key: value for key, value in reservation.items() if key != "reservation_id"
    }
    if reservation.get("reservation_id") != content_sha256(reservation_body):
        raise ValueError("output reservation subreceipt address differs")
    output_root = Path(str(reservation["output_root"])).resolve()
    expected_parent = (project_root / OUTPUT_PARENT).resolve()
    if (
        output_root != receipt_path.parent
        or output_root.parent != expected_parent
        or not output_root.is_dir()
        or _existing_ancestor_has_reparse_point(output_root, stop=project_root)
    ):
        raise ValueError("output reservation current root differs")
    simulation_only = receipt.get("simulation_only") is True
    if (
        receipt.get("runtime_admission_real") is not (not simulation_only)
        or receipt.get("output_reserved") is not True
        or any(
            receipt.get(field) not in (False, 0)
            for field in (
                "capture_started",
                "model_loaded",
                "model_inference_performed",
                "execution_authorized",
                "sae_training_authorized",
            )
        )
    ):
        raise ValueError("output reservation scientific boundary differs")
    return {
        "schema": "sch.vpd-cluster-to-sae-output-reservation-transition-verification.v1",
        "status": "pass",
        "reservation_transition_id": receipt["reservation_transition_id"],
        "admission_id": admission["admission_id"],
        "preflight_id": admission_record["preflight_id"],
        "output_root": str(output_root),
        "simulation_only": simulation_only,
        "runtime_admission_real": not simulation_only,
        "output_reserved": True,
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reserve or verify a VPD admitted output root.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reserve-admission", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--simulation-only", action="store_true")
    args = parser.parse_args()
    if args.reserve_admission is not None:
        result = reserve_admitted_output_root(
            admission_path=args.reserve_admission,
            project_root=args.project_root,
            simulation_only=args.simulation_only,
        )
    else:
        if args.simulation_only:
            parser.error("--simulation-only is valid only with --reserve-admission")
        result = verify_output_reservation(
            args.verify, project_root=args.project_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
