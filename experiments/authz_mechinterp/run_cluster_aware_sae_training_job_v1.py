from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1 import (
    JOB_SCHEMA,
    SCHEMA,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


SIMULATION_RELEASE_SCHEMA = (
    "sch.vpd-cluster-aware-sae-scientific-release-verification.v1"
)
SIMULATION_COMPUTE_SCHEMA = (
    "sch.vpd-cluster-aware-sae-compute-authorization-verification.v1"
)
SIMULATION_STATUS = "pass_synthetic_benchmark_only_not_authorized"
REAL_STATUS = "pass"
REQUIRED_ACTION = "cluster_aware_sae_dictionary_training"
WORKER_RECEIPT_SCHEMA = "sch.vpd-cluster-aware-sae-runtime-worker-receipt.v1"


def _content_id(value: Any, *, field_name: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise ValueError(f"{field_name} is not a lowercase SHA-256 content ID")
    return observed


def _addressed(value: dict[str, Any], *, id_field: str, label: str) -> str:
    body = dict(value)
    identifier = _content_id(body.pop(id_field, ""), field_name=f"{label} ID")
    if identifier != content_sha256(body):
        raise ValueError(f"{label} content address differs")
    return identifier


def validate_execution_blueprint_content(
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    if blueprint.get("schema") != SCHEMA or blueprint.get("status") != (
        "complete_manifest_only_cluster_routed_execution_blueprint"
    ):
        raise ValueError("cluster-aware SAE execution blueprint contract differs")
    blueprint_id = _addressed(
        blueprint,
        id_field="execution_blueprint_id",
        label="cluster-aware SAE execution blueprint",
    )
    boundary = blueprint.get("boundary", {})
    if (
        boundary.get("manifest_only") is not True
        or any(
            boundary.get(field_name) is not False
            for field_name in (
                "scientific_release_receipt_present",
                "explicit_compute_authorization_present",
                "model_loaded",
                "model_inference_performed",
                "weights_loaded_or_mutated",
                "sae_training_authorized",
                "execution_authorized",
            )
        )
    ):
        raise ValueError("cluster-aware SAE execution blueprint boundary differs")
    source = blueprint.get("source", {})
    global_split = blueprint.get("global_pair_split", {})
    global_split_id = _content_id(
        source.get("global_pair_split_id", ""), field_name="global pair split ID"
    )
    if (
        global_split.get("global_pair_split_id") != global_split_id
        or content_sha256(
            {
                key: value
                for key, value in global_split.items()
                if key != "global_pair_split_id"
            }
        )
        != global_split_id
        or global_split.get("pair_sets_disjoint") is not True
        or global_split.get(
            "one_global_assignment_across_all_category_compartments"
        )
        is not True
    ):
        raise ValueError("cluster-aware SAE global pair split differs")
    global_training_pairs = set(global_split.get("training_pair_ids", []))
    global_validation_pairs = set(global_split.get("validation_pair_ids", []))
    if not global_training_pairs or not global_validation_pairs or (
        global_training_pairs & global_validation_pairs
    ):
        raise ValueError("cluster-aware SAE global pair split is invalid")

    jobs = blueprint.get("execution_jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("cluster-aware SAE execution jobs are absent")
    job_ids: set[str] = set()
    output_dirs: set[str] = set()
    for job in jobs:
        if job.get("schema") != JOB_SCHEMA:
            raise ValueError("cluster-aware SAE execution job schema differs")
        job_id = _addressed(
            job,
            id_field="execution_job_id",
            label="cluster-aware SAE execution job",
        )
        output_dir = str(job.get("output_dir", ""))
        if (
            job_id in job_ids
            or output_dir in output_dirs
            or not output_dir
            or output_dir.startswith(("/", "\\"))
            or len(output_dir) >= 2
            and output_dir[1] == ":"
            or ".." in output_dir.replace("\\", "/").split("/")
        ):
            raise ValueError("cluster-aware SAE job identity or output repeats")
        job_ids.add(job_id)
        output_dirs.add(output_dir)
        sample = job.get("exact_sample_contract", {})
        episode_ids = sample.get("episode_ids", [])
        pair_ids = sample.get("pair_ids", [])
        training_pairs = sample.get("training_pair_ids", [])
        validation_pairs = sample.get("validation_pair_ids", [])
        training_episodes = sample.get("training_episode_ids", [])
        validation_episodes = sample.get("validation_episode_ids", [])
        if (
            not episode_ids
            or not pair_ids
            or not training_pairs
            or not validation_pairs
            or set(training_pairs) & set(validation_pairs)
            or set(training_pairs) | set(validation_pairs) != set(pair_ids)
            or not set(training_pairs).issubset(global_training_pairs)
            or not set(validation_pairs).issubset(global_validation_pairs)
            or set(training_episodes) & set(validation_episodes)
            or set(training_episodes) | set(validation_episodes) != set(episode_ids)
        ):
            raise ValueError("cluster-aware SAE exact sample partition differs")
        for values_field, digest_field in (
            ("episode_ids", "episode_id_set_sha256"),
            ("pair_ids", "pair_id_set_sha256"),
            ("training_pair_ids", "training_pair_id_set_sha256"),
            ("validation_pair_ids", "validation_pair_id_set_sha256"),
            ("training_episode_ids", "training_episode_id_set_sha256"),
            ("validation_episode_ids", "validation_episode_id_set_sha256"),
        ):
            if content_sha256(sample[values_field]) != sample.get(digest_field):
                raise ValueError("cluster-aware SAE exact sample hash differs")
        rows = int(sample.get("sae_activation_rows", 0))
        rows_per_episode = int(sample.get("sae_activation_rows_per_episode", 0))
        profile = job.get("profile", {})
        if (
            rows <= 0
            or rows_per_episode <= 0
            or rows != len(episode_ids) * rows_per_episode
            or int(profile.get("steps", 0)) <= 0
            or int(profile.get("batch_size", 0)) <= 0
            or int(job.get("dictionary_features", 0)) <= 0
            or int(profile.get("dictionary_features", -1))
            != int(job["dictionary_features"])
            or int(job.get("initialization_seed", -1))
            not in list(profile.get("initialization_seeds", []))
            or job.get("global_pair_split_id") != global_split_id
            or sample.get("cross_compartment_pair_assignment_globally_consistent")
            is not True
            or not job.get("routing_receipt_ids")
            or not job.get("routing_stability_ids")
        ):
            raise ValueError("cluster-aware SAE job resource or lineage contract differs")
        for evidence_id in job["routing_receipt_ids"] + job[
            "routing_stability_ids"
        ]:
            _content_id(evidence_id, field_name="routing evidence ID")
        job_boundary = job.get("boundary", {})
        if (
            job_boundary.get("manifest_only") is not True
            or job_boundary.get("sae_training_authorized") is not False
            or job_boundary.get("execution_authorized") is not False
        ):
            raise ValueError("cluster-aware SAE execution job boundary differs")
    summary = blueprint.get("summary", {})
    if (
        int(summary.get("execution_jobs", -1)) != len(jobs)
        or int(summary.get("unique_output_dirs", -1)) != len(output_dirs)
        or summary.get("one_global_pair_split") is not True
        or summary.get("exact_cluster_routed_episode_sets") is not True
    ):
        raise ValueError("cluster-aware SAE execution blueprint summary differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-runtime-blueprint-verification.v1",
        "status": "pass",
        "execution_blueprint_id": blueprint_id,
        "materialization_id": _content_id(
            source.get("materialization_id", ""), field_name="materialization ID"
        ),
        "cluster_aware_sae_plan_id": _content_id(
            source.get("cluster_aware_sae_plan_id", ""),
            field_name="cluster-aware SAE plan ID",
        ),
        "global_pair_split_id": global_split_id,
        "evidence_kind": str(source.get("evidence_kind", "")),
        "execution_jobs": len(jobs),
        "execution_authorized": False,
    }


def select_execution_job(
    blueprint: dict[str, Any], *, execution_job_id: str
) -> dict[str, Any]:
    matches = [
        job
        for job in blueprint.get("execution_jobs", [])
        if str(job.get("execution_job_id")) == execution_job_id
    ]
    if len(matches) != 1:
        raise ValueError("cluster-aware SAE blueprint does not identify one job")
    return matches[0]


def _verify_runtime_authority(
    *,
    blueprint_check: dict[str, Any],
    job: dict[str, Any],
    release_verification: dict[str, Any],
    compute_authorization_verification: dict[str, Any],
    simulation_only: bool,
) -> tuple[str, str]:
    expected_status = SIMULATION_STATUS if simulation_only else REAL_STATUS
    if release_verification.get("schema") != SIMULATION_RELEASE_SCHEMA or (
        release_verification.get("status") != expected_status
    ):
        raise ValueError("cluster-aware SAE scientific release verification differs")
    release_id = _content_id(
        release_verification.get("scientific_release_receipt_id", ""),
        field_name="scientific release receipt ID",
    )
    _content_id(
        release_verification.get("scientific_release_receipt_sha256", ""),
        field_name="scientific release receipt SHA-256",
    )
    if (
        release_verification.get("execution_blueprint_id")
        != blueprint_check["execution_blueprint_id"]
        or release_verification.get("materialization_id")
        != blueprint_check["materialization_id"]
        or release_verification.get("cluster_aware_sae_plan_id")
        != blueprint_check["cluster_aware_sae_plan_id"]
        or release_verification.get("global_pair_split_id")
        != blueprint_check["global_pair_split_id"]
        or release_verification.get("compute_authorized") is not False
    ):
        raise ValueError("cluster-aware SAE scientific release scope differs")
    if compute_authorization_verification.get("schema") != SIMULATION_COMPUTE_SCHEMA or (
        compute_authorization_verification.get("status") != expected_status
    ):
        raise ValueError("cluster-aware SAE compute authorization verification differs")
    compute_id = _content_id(
        compute_authorization_verification.get("compute_authorization_id", ""),
        field_name="compute authorization ID",
    )
    _content_id(
        compute_authorization_verification.get("compute_authorization_sha256", ""),
        field_name="compute authorization SHA-256",
    )
    if (
        compute_authorization_verification.get("execution_blueprint_id")
        != blueprint_check["execution_blueprint_id"]
        or compute_authorization_verification.get("required_job_id")
        != job["execution_job_id"]
        or compute_authorization_verification.get("required_action")
        != REQUIRED_ACTION
        or compute_authorization_verification.get("scientific_release_receipt_id")
        != release_id
        or compute_authorization_verification.get("base_model_weights_mutable")
        is not False
        or compute_authorization_verification.get("real_actions_authorized")
        is not False
        or compute_authorization_verification.get(
            "sae_dictionary_weight_training_authorized"
        )
        is not (not simulation_only)
        or compute_authorization_verification.get("execution_authorized")
        is not (not simulation_only)
    ):
        raise ValueError("cluster-aware SAE compute authorization scope differs")
    if simulation_only:
        if blueprint_check["evidence_kind"] != "synthetic_benchmark":
            raise ValueError("simulation requires synthetic benchmark evidence")
    elif blueprint_check["evidence_kind"] != "empirical_guarded_v3_capture":
        raise ValueError("real training requires empirical guarded V3 evidence")
    return release_id, compute_id


@dataclass(frozen=True)
class ClusterAwareSAETrainingAllocation:
    execution_job_id: str
    planned_steps: int
    planned_activation_rows: int
    dictionary_features: int


@dataclass
class ClusterAwareSAETrainingLedger:
    execution_blueprint_id: str
    reservations: dict[str, str] = field(default_factory=dict)
    planned_steps: int = 0
    planned_activation_rows: int = 0
    trainer_calls: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "execution_blueprint_id": self.execution_blueprint_id,
            "reservations": dict(self.reservations),
            "jobs_reserved": len(self.reservations),
            "planned_steps": self.planned_steps,
            "planned_activation_rows": self.planned_activation_rows,
            "trainer_calls": self.trainer_calls,
        }

    def reserve(self, allocation: ClusterAwareSAETrainingAllocation) -> None:
        if allocation.execution_job_id in self.reservations:
            raise ValueError("cluster-aware SAE execution job was already reserved")
        self.reservations[allocation.execution_job_id] = "reserved"
        self.planned_steps += allocation.planned_steps
        self.planned_activation_rows += allocation.planned_activation_rows

    def start(self, execution_job_id: str) -> None:
        if self.reservations.get(execution_job_id) != "reserved":
            raise ValueError("cluster-aware SAE job cannot start from this state")
        self.reservations[execution_job_id] = "started"
        self.trainer_calls += 1

    def finish(self, execution_job_id: str, *, failed: bool) -> None:
        if self.reservations.get(execution_job_id) != "started":
            raise ValueError("cluster-aware SAE job cannot finish from this state")
        self.reservations[execution_job_id] = "failed" if failed else "completed"


TrainingCallback = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ClusterAwareSAETrainingRuntime:
    train_job: TrainingCallback
    close: Callable[[], None]
    runtime_kind: str
    actual_model_loaded: bool
    actual_training_enabled: bool
    persistent_output_enabled: bool


@dataclass
class ClusterAwareSAETrainingSession:
    blueprint_check: dict[str, Any]
    job: dict[str, Any]
    allocation: ClusterAwareSAETrainingAllocation
    ledger: ClusterAwareSAETrainingLedger
    runtime: ClusterAwareSAETrainingRuntime
    scientific_release_receipt_id: str
    compute_authorization_id: str
    simulation_only: bool
    before_reservation: dict[str, Any]
    invoked: bool = False
    closed: bool = False

    def run(self) -> dict[str, Any]:
        if self.invoked:
            raise ValueError("cluster-aware SAE trainer callback is single-use")
        self.invoked = True
        self.ledger.start(self.job["execution_job_id"])
        try:
            callback_result = self.runtime.train_job(self.job)
            self._validate_callback_result(callback_result)
        except Exception:
            self.ledger.finish(self.job["execution_job_id"], failed=True)
            raise
        self.ledger.finish(self.job["execution_job_id"], failed=False)
        body = {
            "schema": WORKER_RECEIPT_SCHEMA,
            "status": (
                "complete_synthetic_callback_no_training"
                if self.simulation_only
                else "complete_authorized_cluster_aware_sae_training"
            ),
            "execution_blueprint_id": self.blueprint_check[
                "execution_blueprint_id"
            ],
            "materialization_id": self.blueprint_check["materialization_id"],
            "cluster_aware_sae_plan_id": self.blueprint_check[
                "cluster_aware_sae_plan_id"
            ],
            "global_pair_split_id": self.blueprint_check["global_pair_split_id"],
            "execution_job_id": self.job["execution_job_id"],
            "source_training_job_id": self.job["source_training_job_id"],
            "compartment_id": self.job["compartment_id"],
            "compartment_contract_id": self.job["compartment_contract_id"],
            "exact_sample_contract_sha256": content_sha256(
                self.job["exact_sample_contract"]
            ),
            "routing_receipt_ids": list(self.job["routing_receipt_ids"]),
            "routing_stability_ids": list(self.job["routing_stability_ids"]),
            "allocation": {
                "planned_steps": self.allocation.planned_steps,
                "planned_activation_rows": self.allocation.planned_activation_rows,
                "dictionary_features": self.allocation.dictionary_features,
            },
            "scientific_release_receipt_id": self.scientific_release_receipt_id,
            "compute_authorization_id": self.compute_authorization_id,
            "runtime_kind": self.runtime.runtime_kind,
            "callback_result": callback_result,
            "ledger_after": self.ledger.snapshot(),
            "scientific_boundary": {
                "simulation_only": self.simulation_only,
                "model_loaded": self.runtime.actual_model_loaded,
                "training_performed": self.runtime.actual_training_enabled,
                "persistent_output_written": self.runtime.persistent_output_enabled,
                "base_model_weights_mutated": False,
                "real_actions": 0,
                "receipt_authorizes_additional_execution": False,
            },
        }
        return {**body, "runtime_worker_receipt_id": content_sha256(body)}

    def _validate_callback_result(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            raise TypeError("cluster-aware SAE trainer callback result must be an object")
        if result.get("execution_job_id") != self.job["execution_job_id"]:
            raise ValueError("cluster-aware SAE trainer callback job identity differs")
        if self.simulation_only:
            if (
                result.get("status") != "synthetic_completed_no_training"
                or result.get("model_loaded") is not False
                or result.get("weights_mutated") is not False
                or result.get("persistent_output_written") is not False
                or result.get("dictionary_id") is not None
            ):
                raise ValueError("synthetic cluster-aware SAE result weakened boundary")
        else:
            if (
                result.get("status") != "completed_authorized_training"
                or result.get("weights_mutated") is not True
                or result.get("persistent_output_written") is not True
            ):
                raise ValueError("real cluster-aware SAE result is incomplete")
            _content_id(result.get("dictionary_id", ""), field_name="dictionary ID")
            _content_id(
                result.get("dictionary_manifest_sha256", ""),
                field_name="dictionary manifest SHA-256",
            )
            _content_id(
                result.get("dictionary_artifact_sha256", ""),
                field_name="dictionary artifact SHA-256",
            )

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.runtime.close()


@dataclass
class ClusterAwareSAETrainingReservation:
    blueprint_check: dict[str, Any]
    job: dict[str, Any]
    allocation: ClusterAwareSAETrainingAllocation
    ledger: ClusterAwareSAETrainingLedger
    scientific_release_receipt_id: str
    compute_authorization_id: str
    simulation_only: bool
    before_reservation: dict[str, Any]
    runtime_attached: bool = False

    def attach_runtime(
        self, runtime: ClusterAwareSAETrainingRuntime
    ) -> ClusterAwareSAETrainingSession:
        if self.runtime_attached:
            raise ValueError("cluster-aware SAE reservation already attached a runtime")
        if self.ledger.reservations.get(self.job["execution_job_id"]) != "reserved":
            raise ValueError("cluster-aware SAE reservation is no longer attachable")
        if self.simulation_only:
            if (
                runtime.actual_model_loaded
                or runtime.actual_training_enabled
                or runtime.persistent_output_enabled
            ):
                raise ValueError("synthetic cluster-aware SAE runtime is not inert")
        elif (
            not runtime.actual_training_enabled
            or not runtime.persistent_output_enabled
        ):
            raise ValueError("authorized cluster-aware SAE runtime is incomplete")
        self.runtime_attached = True
        return ClusterAwareSAETrainingSession(
            blueprint_check=self.blueprint_check,
            job=self.job,
            allocation=self.allocation,
            ledger=self.ledger,
            runtime=runtime,
            scientific_release_receipt_id=self.scientific_release_receipt_id,
            compute_authorization_id=self.compute_authorization_id,
            simulation_only=self.simulation_only,
            before_reservation=self.before_reservation,
        )


def reserve_cluster_aware_sae_job_before_runtime_attach(
    *,
    blueprint: dict[str, Any],
    execution_job_id: str,
    release_verification: dict[str, Any],
    compute_authorization_verification: dict[str, Any],
    ledger: ClusterAwareSAETrainingLedger,
    simulation_only: bool,
) -> ClusterAwareSAETrainingReservation:
    blueprint_check = validate_execution_blueprint_content(blueprint)
    if ledger.execution_blueprint_id != blueprint_check["execution_blueprint_id"]:
        raise ValueError("cluster-aware SAE ledger belongs to another blueprint")
    job = select_execution_job(blueprint, execution_job_id=execution_job_id)
    release_id, compute_id = _verify_runtime_authority(
        blueprint_check=blueprint_check,
        job=job,
        release_verification=release_verification,
        compute_authorization_verification=compute_authorization_verification,
        simulation_only=simulation_only,
    )
    allocation = ClusterAwareSAETrainingAllocation(
        execution_job_id=job["execution_job_id"],
        planned_steps=int(job["profile"]["steps"]),
        planned_activation_rows=int(
            job["exact_sample_contract"]["sae_activation_rows"]
        ),
        dictionary_features=int(job["dictionary_features"]),
    )
    before = ledger.snapshot()
    ledger.reserve(allocation)
    return ClusterAwareSAETrainingReservation(
        blueprint_check=blueprint_check,
        job=job,
        allocation=allocation,
        ledger=ledger,
        scientific_release_receipt_id=release_id,
        compute_authorization_id=compute_id,
        simulation_only=simulation_only,
        before_reservation=before,
    )
