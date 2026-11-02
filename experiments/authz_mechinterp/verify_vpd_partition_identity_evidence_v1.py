from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_activation_category_cluster_partition_identity_v4 import (
    verify_benchmark as verify_partition_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_partition_identity_admission_v4 import (
    verify_benchmark as verify_admission_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


def verify_partition_identity_evidence(
    *,
    partition_report: Path,
    admission_report: Path,
    project_root: Path,
) -> dict[str, object]:
    partition = verify_partition_benchmark(partition_report)
    admission = verify_admission_benchmark(
        admission_report, project_root=project_root.resolve()
    )
    body: dict[str, object] = {
        "schema": "sch.vpd-partition-identity-evidence-verification.v1",
        "status": "pass",
        "partition_identity_benchmark_id": partition["benchmark_id"],
        "partition_negative_controls": partition["negative_controls"],
        "production_admission_benchmark_id": admission["benchmark_id"],
        "production_scenarios": admission["scenarios"],
        "execution_authorized": False,
    }
    return {**body, "verification_id": content_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify V4 partition and production admission evidence without using "
            "their legacy CLI summary paths."
        )
    )
    parser.add_argument(
        "--partition-report",
        type=Path,
        default=Path(
            "artifacts/vpd_activation_category_cluster_partition_identity_benchmark_v4/report.json"
        ),
    )
    parser.add_argument(
        "--admission-report",
        type=Path,
        default=Path(
            "artifacts/vpd_cluster_candidate_partition_identity_admission_benchmark_v4/report.json"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = verify_partition_identity_evidence(
        partition_report=args.partition_report,
        admission_report=args.admission_report,
        project_root=args.project_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
