"""CPU development puzzles from exactly two reviewed recovered source modules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import subprocess
import sys
import types

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.authz_mechinterp.nonogram_text_adapter_v1 import adapt_example, grade_answer

SOURCE = Path("E:/Recovered_C_projects/Remote_clones/prime_intellect_research_environments/environments/reasoning/prolog/prolog/generators")
PINS = {"base.py": "6257dca4bab8914c31621fddfa160c05bd89cef4a71df07cd0835f51bbd9cbd4",
        "nonogram.py": "817998a6dc21c16ed68fc14d767d9e7ed9cb8fc3005b10812a6dd79bb6129e52"}
NAMESPACE = "_vpd_reviewed_nonogram_canary"


def load_generator(directory):
    contents = {name: (directory/name).read_bytes() for name in PINS}
    if any(hashlib.sha256(contents[name]).hexdigest() != expected for name, expected in PINS.items()):
        raise ValueError("reviewed source hash mismatch; no source may execute")
    package = types.ModuleType(NAMESPACE)
    package.__path__ = []
    sys.modules[NAMESPACE] = package
    for name in PINS:
        module_name = f"{NAMESPACE}.{Path(name).stem}"
        module = types.ModuleType(module_name)
        module.__file__, module.__package__ = str(directory/name), NAMESPACE
        sys.modules[module_name] = module
        exec(compile(contents[name], str(directory/name), "exec"), module.__dict__)
    return sys.modules[f"{NAMESPACE}.nonogram"].NonogramGenerator


def run_worker():
    generator = load_generator(SOURCE)()
    questions, graders, checks = [], [], []
    for difficulty in ("medium", "hard", "expert"):
        for seed in (0, 1, 2, 3):
            first = generator.generate(Random(seed), difficulty)
            second = generator.generate(Random(seed), difficulty)
            reference = json.loads(first.expected_answer)
            q, g = adapt_example(first.metadata, reference)
            if (q, g) != adapt_example(second.metadata, json.loads(second.expected_answer)):
                raise ValueError("non-deterministic generator/adapter")
            if not grade_answer(q["constraints"], json.dumps(reference)):
                raise ValueError("source reference rejected")
            invalid = [row.copy() for row in reference]
            invalid[0][0] = 1-invalid[0][0]
            if grade_answer(q["constraints"], json.dumps(invalid)):
                raise ValueError("single-cell constraint violation accepted")
            id = f"nonogram-development-{difficulty}-seed-{seed}"
            questions.append({"id": id, "role": "development_only", **q})
            graders.append({"id": id, **g})
            checks.append({"id": id, "difficulty": difficulty, "seed": seed,
                           "rows": q["constraints"]["num_rows"], "cols": q["constraints"]["num_cols"],
                           "deterministic": True, "reference_accepted": True, "single_cell_error_rejected": True})
    hashes = [q["public_instance_sha256"] for q in questions]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate public clue set in development canary")
    return {"questions": questions, "graders": graders, "checks": checks, "python_version": sys.version}


def generate_report():
    worker = subprocess.run([sys.executable, "-B", "-m", "experiments.authz_mechinterp.check_nonogram_generator_canary_v1", "--worker"],
                            capture_output=True, text=True, check=True, timeout=20)
    result = json.loads(worker.stdout)
    body = {"schema": "sch.vpd-nonogram-generator-canary.v1", "source_directory": str(SOURCE), "source_pins": PINS,
            "checks": result["checks"], "python_version": result["python_version"],
            "questions_content_sha256": content_sha256(result["questions"]),
            "graders_content_sha256": content_sha256(result["graders"]),
            "provenance": {"runner_sha256": sha256_file(Path(__file__)),
                           "adapter_sha256": sha256_file(Path(__file__).with_name("nonogram_text_adapter_v1.py"))},
            "limitations": ["Twelve development puzzles, not a fresh holdout or semantic family admission.",
                            "Exact public-clue hashes do not detect symmetry equivalence or establish prior-exposure absence.",
                            "Any clue-satisfying grid is accepted; reference uniqueness and generator difficulty claims are not certified.",
                            "New text/JSON task interface, not execution of the official Prolog/container task."],
            "boundary": {"puzzle_generation_performed": True, "worker_timeout_seconds": 20,
                         "model_inference_performed": False, "sae_training_performed": False,
                         "paid_compute_launched": False, "additional_families_admitted": 0}}
    return {**body, "report_id": content_sha256(body)}, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--output-directory", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_worker(), allow_nan=False))
        return
    if args.output_directory and args.output_directory.exists():
        parser.error("refuse an existing output directory")
    report, result = generate_report()
    if args.verify:
        if json.loads(args.verify.read_text(encoding="utf-8")) != report:
            raise ValueError("nonogram canary does not reproduce")
        for name in ("questions", "graders"):
            if json.loads((args.verify.parent/f"{name}.json").read_text(encoding="utf-8")) != result[name]:
                raise ValueError(f"{name} do not reproduce")
    else:
        args.output_directory.mkdir(parents=True, exist_ok=False)
        for name in ("questions", "graders"):
            write_json(args.output_directory/f"{name}.json", result[name])
        write_json(args.output_directory/"report.json", report)
    print(json.dumps({"report_id": report["report_id"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
